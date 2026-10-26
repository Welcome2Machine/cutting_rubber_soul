"""Local, evidence-gated material maps for contact measurements (all lengths px).

These routines consume saved trajectories; they never alter or predict tracks.
``supported`` must already exclude positions without current image support.
The caller must choose a single physical material branch near a crack. A good
affine fit or small surface mismatch does not establish that branch's identity.

For row-vector inputs the forward map is ``reference @ A.T + b``. Thresholds
are explicit measurement gates, not calibrated confidence intervals. In
particular, held-out trajectory errors cannot expose coherent tracking drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _failure(reason: str, **fields: Any) -> dict[str, Any]:
    return {"valid": False, "reason": reason, "mapped_point": None,
            "A": None, "b": None, "count": 0, **fields}


def _side_mask(reference: np.ndarray, current: np.ndarray, side: Any) -> np.ndarray:
    if side is None:
        return np.ones(len(reference), dtype=bool)
    if not isinstance(side, dict):
        mask = np.asarray(side, dtype=bool)
        if mask.shape != (len(reference),):
            raise ValueError("side mask must have shape (N,)")
        return mask
    coords = reference if side.get("space", "reference") == "reference" else current
    if side.get("space", "reference") not in ("reference", "current"):
        raise ValueError("side space must be reference or current")
    point = np.asarray(side["point"], dtype=float)
    normal = np.asarray(side["normal"], dtype=float)
    norm = np.linalg.norm(normal)
    if point.shape != (2,) or normal.shape != (2,) or norm < 1e-12:
        raise ValueError("side needs a 2D point and nonzero normal")
    sign = float(side.get("sign", 1.0))
    if sign not in (-1.0, 1.0):
        raise ValueError("side sign must be +1 or -1")
    return sign * ((coords - point) @ (normal / norm)) >= float(side.get("margin", 0.0))


def _fit_affine(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Huber IRLS with centered, scaled predictors; no regularizing to identity."""
    center = np.average(x, axis=0, weights=weights)
    scale = max(float(np.sqrt(np.average(np.sum((x-center)**2, axis=1), weights=weights))), 1e-9)
    design = np.column_stack(((x - center) / scale, np.ones(len(x))))
    effective = weights.copy()
    beta = None
    for _ in range(12):
        root_w = np.sqrt(effective)
        weighted_design = design * root_w[:, None]
        if np.linalg.matrix_rank(weighted_design) < 3:
            raise np.linalg.LinAlgError("collinear support")
        beta = np.linalg.lstsq(weighted_design, y * root_w[:, None], rcond=None)[0]
        residual = np.linalg.norm(design @ beta - y, axis=1)
        # Median radial residual / sqrt(2 log 2) estimates per-coordinate sigma.
        sigma = max(float(np.median(residual) / 1.17741), 0.05)
        huber = np.minimum(1.0, (2.5 * sigma) / np.maximum(residual, 1e-12))
        next_weights = weights * huber
        if np.max(np.abs(next_weights - effective)) < 1e-6:
            effective = next_weights
            break
        effective = next_weights
    A = beta[:2].T / scale
    b = beta[2] - A @ center
    return A, b, residual, effective / np.maximum(weights, 1e-12)


def _hull_distance(points: np.ndarray, query: np.ndarray) -> float:
    """Euclidean distance to convex support hull, zero for interpolation."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        return float("inf")
    if np.all(hull.equations[:, :2] @ query + hull.equations[:, 2] <= 1e-7):
        return 0.0
    vertices = points[hull.vertices]
    a, b = vertices, np.roll(vertices, -1, axis=0)
    delta = b - a
    alpha = np.clip(np.sum((query-a)*delta, axis=1) / np.maximum(np.sum(delta**2, axis=1), 1e-15), 0, 1)
    return float(np.min(np.linalg.norm(query - (a + alpha[:, None]*delta), axis=1)))


def local_material_map(
    reference, current, supported, query, query_space="current", radius=48.0,
    min_points=8, side=None, *, max_extrapolation_px=12.0,
    max_heldout_residual_px=3.0, max_fit_residual_px=3.0,
    max_condition=20.0, min_stretch=0.10, max_stretch=8.0,
) -> dict[str, Any]:
    """Fit a same-branch affine map and forward/inverse-map a virtual point.

    ``query_space='current'`` returns a reference material address; ``reference``
    returns its current position. ``radius`` is measured in the query space.
    ``side`` is None, an N-element mask, or a half-plane dictionary containing
    point, normal, sign (+1/-1), space ('reference'/'current'), and margin (px).
    Both coordinate spaces must support the query within the extrapolation
    allowance. All results include a validity flag; invalid coordinates must
    not be used for a measurement. Returned arrays are numpy arrays.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    query = np.asarray(query, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or current.shape != reference.shape:
        raise ValueError("reference and current must have matching (N,2) shapes")
    if supported.shape != (len(reference),) or query.shape != (2,):
        raise ValueError("supported must be (N,) and query must be (2,)")
    if query_space not in ("current", "reference"):
        raise ValueError("query_space must be current or reference")
    if radius <= 0 or min_points < 4:
        raise ValueError("positive radius and at least four min_points required")
    if not np.isfinite(query).all():
        return _failure("nonfinite_query")
    coords = current if query_space == "current" else reference
    distance = np.linalg.norm(coords - query, axis=1)
    mask = (supported & np.isfinite(reference).all(axis=1) & np.isfinite(current).all(axis=1)
            & _side_mask(reference, current, side) & (distance <= radius))
    indices = np.flatnonzero(mask)
    count = len(indices)
    common = {"count": count, "radius_px": float(radius), "query_space": query_space,
              "support_indices": indices, "side_applied": side is not None}
    if count < min_points:
        return _failure("insufficient_supported_points", **common)
    x, y = reference[mask], current[mask]
    weights = np.exp(-0.5 * (distance[mask] / (0.55*radius))**2)
    try:
        A, b, residual, robust_weights = _fit_affine(x, y, weights)
    except np.linalg.LinAlgError:
        return _failure("rank_deficient_support", **common)
    singular = np.linalg.svd(A, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    determinant = float(np.linalg.det(A))
    common.update(A=A, b=b, singular_values=singular, condition=condition, determinant=determinant)
    if determinant <= 0 or singular[-1] < min_stretch or singular[0] > max_stretch or condition > max_condition:
        return _failure("noninvertible_or_implausible_local_map", **common)
    inliers = robust_weights >= 0.25
    if np.count_nonzero(inliers) < min_points:
        return _failure("insufficient_robust_support", **common)
    support_eigenvalues = np.linalg.eigvalsh(np.cov(x[inliers].T))
    support_shape_condition = float(np.sqrt(support_eigenvalues[-1] / max(support_eigenvalues[0], 1e-15)))
    common["support_shape_condition"] = support_shape_condition
    if support_shape_condition > 20.0:
        return _failure("nearly_collinear_support", **common)
    fit_residual = float(np.sqrt(np.average(residual[inliers]**2, weights=weights[inliers])))

    # Hold out actual points, rather than report only training residuals. Use
    # angular blocks about the local support center; extrapolation into a held
    # sector intentionally tests whether one affine model explains the patch.
    angles = np.arctan2(x[:, 1]-np.mean(x[:, 1]), x[:, 0]-np.mean(x[:, 0]))
    order = np.argsort(angles, kind="stable")
    folds = np.array_split(order, 4)
    heldout = np.full(count, np.nan)
    for test in folds:
        train = np.ones(count, dtype=bool)
        train[test] = False
        if np.count_nonzero(train) < 4:
            continue
        try:
            cv_A, cv_b, _, _ = _fit_affine(x[train], y[train], weights[train])
            heldout[test] = np.linalg.norm(x[test] @ cv_A.T + cv_b - y[test], axis=1)
        except np.linalg.LinAlgError:
            pass
    finite_cv = np.isfinite(heldout)
    cv_inliers = finite_cv & inliers
    if np.count_nonzero(cv_inliers) < min_points:
        return _failure("heldout_fit_unavailable", **common)
    cv_residual = float(np.sqrt(np.average(heldout[cv_inliers]**2, weights=weights[cv_inliers])))
    mapped = np.linalg.solve(A, query-b) if query_space == "current" else A @ query+b
    reference_query = mapped if query_space == "current" else query
    current_query = query if query_space == "current" else mapped
    reference_extrapolation = _hull_distance(x[inliers], reference_query)
    current_extrapolation = _hull_distance(y[inliers], current_query)
    extrapolation = current_extrapolation if query_space == "current" else reference_extrapolation
    fields = {**common, "mapped_point": mapped, "reference_xy": reference_query,
              "current_xy": current_query, "fit_residual_px": fit_residual,
              "heldout_residual_px": cv_residual,
              "heldout_residual_all_p90_px": float(np.nanpercentile(heldout, 90)),
              "reference_extrapolation_px": reference_extrapolation,
              "current_extrapolation_px": current_extrapolation,
              "extrapolation_px": extrapolation,
              "inlier_count": int(np.count_nonzero(inliers)),
              "outlier_count": int(np.count_nonzero(~inliers))}
    if max(reference_extrapolation, current_extrapolation) > max_extrapolation_px + 1e-7:
        return _failure("extrapolation_exceeds_allowance", **fields)
    if fit_residual > max_fit_residual_px or cv_residual > max_heldout_residual_px:
        return _failure("local_map_residual_too_large", **fields)
    return {"valid": True, "reason": "ok", **fields}


class LocalFitGeometry:
    """Immutable reference design with a bounded rank cache for missing nodes."""
    def __init__(self,reference,origin,axes,quadratic=False):
        ref=np.asarray(reference,float);uv=(ref-origin)@axes/12.
        self.X=np.column_stack([np.ones(len(ref)),uv])
        if quadratic:self.X=np.column_stack([self.X,uv[:,0]**2])
        self.finite=np.isfinite(ref).all(1)
        self.ranks={}

    def rank(self,rows):
        key=rows.tobytes()
        if key not in self.ranks:
            if len(self.ranks)>=64:self.ranks.clear()
            self.ranks[key]=np.linalg.matrix_rank(self.X[rows])
        return self.ranks[key]

def fit_local(reference,current,weight,origin,axes,quadratic=False,check_cv=False,prepared=None):
    """Small material-coordinate map, including curvature along the surface."""
    cur=np.asarray(current,float);w=np.asarray(weight,float)
    geometry=prepared if prepared is not None else LocalFitGeometry(reference,origin,axes,quadratic)
    X=geometry.X
    use=np.isfinite(cur).all(1)&geometry.finite&(w>0)
    empty=dict(xy=np.full(2,np.nan),F=np.full((2,2),np.nan),rms=np.inf,cv=np.inf,valid=False)
    if use.sum()<X.shape[1]+2 or geometry.rank(use)<X.shape[1]:return empty
    x,y,basew=X[use],cur[use],w[use];rw=basew.copy()
    def solve(a,b,weights):
        root=np.sqrt(weights)
        return np.linalg.lstsq(a*root[:,None],b*root[:,None],rcond=None)[0]
    for _ in range(3):
        beta=solve(x,y,rw);error=np.linalg.norm(x@beta-y,axis=1)
        sigma=max(.2,float(np.median(error))*1.4826)
        rw=basew*np.minimum(1,2.5*sigma/np.maximum(error,1e-9))
    F=(beta[1:3].T/12)@axes.T
    rms=float(np.sqrt(np.average(error**2,weights=basew)))
    cv=rms
    if check_cv:
        residual=[]
        for parity in (0,1):
            test=np.arange(len(x))%2==parity;train=~test
            original_train=use.copy();original_train[use]=train
            if train.sum()<X.shape[1] or geometry.rank(original_train)<X.shape[1]:continue
            b=solve(x[train],y[train],basew[train])
            residual.extend(np.linalg.norm(x[test]@b-y[test],axis=1))
        cv=float(np.sqrt(np.mean(np.square(residual)))) if residual else np.inf
    sv=np.linalg.svd(F,compute_uv=False)
    valid=np.linalg.det(F)>0 and sv[-1]>.08 and sv[0]<10 and sv[0]/sv[-1]<30
    return dict(xy=beta[0],F=F,rms=rms,cv=cv,valid=bool(valid))
