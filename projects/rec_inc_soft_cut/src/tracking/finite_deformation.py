"""Research-only, topology-aware global registration of a material mesh.

Piecewise-affine image registration uses local intensity normalization and a
robust loss. Optional regularization compares adjacent right Cauchy--Green
tensors, C = F.T @ F, provides objective kinematic smoothing. A separate opt-in
finite-strain equilibrium gap uses a conditional plane-strain Neo-Hookean law,
with unknown boundary forces excluded. Neither branch makes hidden material an
observation. No boundary traction, incompressibility, or contact sticking is imposed.

Coordinates are native image pixels (x right, y down). Node order is immutable;
the caller retains its IDs. This local refinement needs a reasonable initial
map and does not discover a crack or recover arbitrary occluded deformation.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay, QhullError

from .material_patch_tracker import _gray, masked_gray


@dataclass(frozen=True)
class FiniteDeformationConfig:
    sample_side: int = 5
    illumination_sigma: float = 8.
    texture_floor: float = .008
    min_texture: float = .008
    max_photometric_residual: float = .65
    robust_scale: float = .3
    kinematic_weight: float = .03
    min_jacobian: float = .08
    jacobian_weight: float = 10.
    max_update_px: float = 5.
    max_nfev: int = 35
    smoothing_levels: tuple = (1.2, .55)
    max_reference_condition: float = 40.
    feature_mode: str = 'legacy'
    min_feature_support: float = .25
    equilibrium_weight: float = 0.
    equilibrium_bulk_shear_ratio: float = 10.
    equilibrium_reference: str = 'unspecified'
    feasible_backtracking: bool = False

    def __post_init__(self):
        if not (self.sample_side >= 2 and self.illumination_sigma > 0
                and self.texture_floor > 0 and self.min_texture > 0
                and self.max_photometric_residual > 0 and self.robust_scale > 0
                and self.kinematic_weight >= 0 and 0 < self.min_jacobian < 1
                and self.jacobian_weight > 0 and self.max_update_px > 0
                and self.max_nfev > 0 and len(self.smoothing_levels)
                and min(self.smoothing_levels) >= 0
                and self.max_reference_condition > 1
                and self.feature_mode in ('legacy', 'material-masked')
                and 0 < self.min_feature_support <= 1
                and np.isfinite(self.equilibrium_weight) and self.equilibrium_weight >= 0
                and np.isfinite(self.equilibrium_bulk_shear_ratio)
                and self.equilibrium_bulk_shear_ratio > 2/3
                and self.equilibrium_reference in ('unspecified', 'stress-free-assumed')
                and (self.equilibrium_weight == 0
                     or self.equilibrium_reference == 'stress-free-assumed')
                and isinstance(self.feasible_backtracking, bool)):
            raise ValueError('Invalid finite-deformation configuration')


def _sample(image, xy):
    return map_coordinates(image, [xy[..., 1], xy[..., 0]], order=1,
                           mode='constant', cval=0., prefilter=False)


def _sample_gradient(image, xy):
    """Exact spatial derivative of the same bilinear interpolant as _sample."""
    h, w = image.shape
    x, y = xy[:, 0], xy[:, 1]
    ix = np.clip(np.floor(x).astype(int), 0, w-2)
    iy = np.clip(np.floor(y).astype(int), 0, h-2)
    dx, dy = x-ix, y-iy
    a, b, c, d = image[iy, ix], image[iy, ix+1], image[iy+1, ix], image[iy+1, ix+1]
    result = np.column_stack(((1-dy)*(b-a)+dy*(d-c), (1-dx)*(c-a)+dx*(d-b)))
    result[(x < 0) | (y < 0) | (x > w-1) | (y > h-1)] = 0
    return result


def _mask_pair(mask, shape, *, reference_only=False):
    if mask is None:
        return None, None
    a, b = mask if isinstance(mask, tuple) else ((mask, None) if reference_only else (None, mask))
    for value in (a, b):
        if value is not None and np.asarray(value).shape != shape:
            raise ValueError('Mask and image shapes differ')
    return (None if a is None else np.asarray(a, bool),
            None if b is None else np.asarray(b, bool))


def _geometry(points, triangles):
    edges = np.stack((points[triangles[:, 1]]-points[triangles[:, 0]],
                      points[triangles[:, 2]]-points[triangles[:, 0]]), axis=2)
    return edges


def _gradients(points, triangles):
    inv = np.linalg.inv(_geometry(points, triangles))
    return np.concatenate((-inv.sum(axis=1)[:, None], inv), axis=1)


def deformation_diagnostics(reference_points, positions, triangles):
    """Return F, det(F), condition(F), and objective Green--Lagrange strain."""
    tri = np.asarray(triangles, np.int64).reshape(-1, 3)
    F = np.einsum('tna,tnb->tab', np.asarray(positions)[tri],
                  _gradients(np.asarray(reference_points), tri))
    C = np.einsum('tka,tkb->tab', F, F)
    return dict(deformation_gradient=F, jacobian=np.linalg.det(F),
                condition=np.linalg.cond(F), green_lagrange_strain=(C-np.eye(2))/2)


def _clear_triangles(points, triangles, material, blade, shape):
    """Conservative pixel raster test, including element interiors and edges."""
    h, w = shape
    clear = np.ones(len(triangles), bool)
    for i, tri in enumerate(triangles):
        xy = points[tri]
        if (xy[:, 0].min() < 1 or xy[:, 1].min() < 1
                or xy[:, 0].max() >= w-1 or xy[:, 1].max() >= h-1):
            clear[i] = False
            continue
        if material is None and blade is None:
            continue
        lo = np.floor(xy.min(0)).astype(int)
        hi = np.ceil(xy.max(0)).astype(int)+1
        raster = np.zeros((hi[1]-lo[1], hi[0]-lo[0]), np.uint8)
        cv2.fillConvexPoly(raster, np.rint((xy-lo)*256).astype(np.int32), 1, shift=8)
        inside = raster.astype(bool)
        cut = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        if material is not None and np.any(~material[cut] & inside):
            clear[i] = False
        if blade is not None and np.any(blade[cut] & inside):
            clear[i] = False
    return clear


def _masked_smooth(image, valid, sigma, kernel=(0, 0)):
    """Normalized convolution; an invalid pixel contributes no intensity."""
    if sigma <= 0:
        return np.where(valid, image, 0), valid.astype(np.float32)
    mass = cv2.GaussianBlur(valid.astype(np.float32), kernel, sigma)
    total = cv2.GaussianBlur(np.where(valid, image, 0), kernel, sigma)
    value = np.divide(total, mass, out=np.zeros_like(total), where=mass > 1.e-8)
    return value, mass


def _material_gray(frame, valid, config):
    """Mask the tracker-compatible initial blur as well as later features."""
    return masked_gray(frame, valid, config.min_feature_support)


def _features(gray, config, sigma, valid_mask=None, *, return_valid=False):
    valid = (np.ones(gray.shape, bool) if valid_mask is None
             else np.asarray(valid_mask, bool))
    if valid.shape != gray.shape:
        raise ValueError('Feature mask and image shapes differ')
    if config.feature_mode == 'material-masked' and not valid.all():
        image, smooth_mass = _masked_smooth(gray, valid, sigma)
        statistics_mask = valid & (smooth_mass >= config.min_feature_support)
        mean, mass = _masked_smooth(image, statistics_mask, config.illumination_sigma)
        second, _ = _masked_smooth(image*image, statistics_mask, config.illumination_sigma)
        variance = np.maximum(second-mean*mean, 0)
        supported = statistics_mask & (mass >= config.min_feature_support)
        feature = (image-mean)/np.sqrt(variance+config.texture_floor**2)
        feature = np.where(supported, np.clip(feature, -4., 4.), 0).astype(np.float64)
        return (feature, supported) if return_valid else feature
    image = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
    mean = cv2.GaussianBlur(image, (0, 0), config.illumination_sigma)
    variance = cv2.GaussianBlur(image*image, (0, 0), config.illumination_sigma)-mean*mean
    feature = (image-mean)/np.sqrt(np.maximum(variance, 0)+config.texture_floor**2)
    feature = np.clip(feature, -4., 4.).astype(np.float64)
    return (feature, np.ones(gray.shape, bool)) if return_valid else feature


def _feature_mask(material, blade, shape):
    valid = np.ones(shape, bool) if material is None else material.copy()
    if blade is not None:
        valid &= ~blade
    return valid


def _interpolation_support(valid):
    # Avoid turning the zero-filled invalid region into a synthetic image edge.
    # A one-pixel erosion keeps every bilinear value/gradient stencil in support.
    return cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _neighbors(triangles):
    edges, pairs = {}, []
    for i, tri in enumerate(triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((tri[a], tri[b])))
            if key in edges:
                pairs.append((edges[key], i))
            else:
                edges[key] = i
    return np.asarray(pairs, int).reshape(-1, 2)


def _neo_hookean(F, bulk_shear_ratio, floor):
    """P/mu and d(P/mu)/dF for F3D=diag(F2D,1), K/mu supplied explicitly.

    W/mu=(tr(C3D)-3)/2-ln(J)+(lambda/mu)*ln(J)**2/2,
    lambda/mu=K/mu-2/3. Below the positive-J floor, only optimizer trial
    evaluations use a quadratic continuation of volumetric energy; accepted
    maps must remain above the floor. Continuation values are not stresses.
    Source: https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html
    """
    J = np.linalg.det(F)
    cof = np.stack((F[:, 1, 1], -F[:, 1, 0], -F[:, 0, 1],
                    F[:, 0, 0]), axis=1).reshape(-1, 2, 2)
    lam = bulk_shear_ratio-2/3
    safe = np.maximum(J, floor)
    h = (lam*np.log(safe)-1)/safe
    dh = (lam+1-lam*np.log(safe))/safe**2
    h += dh*np.minimum(J-floor, 0)
    P = F+h[:, None, None]*cof
    eye, eps = np.eye(2), np.array([[0., 1.], [-1., 0.]])
    tangent = (np.einsum('ik,jl->ijkl', eye, eye)[None]
               + dh[:, None, None, None, None]*np.einsum('tij,tkl->tijkl', cof, cof)
               + h[:, None, None, None, None]*np.einsum('ik,jl->ijkl', eps, eps)[None])
    return P, tangent


class _EquilibriumGap:
    """Genet (2023), Eq.14-15 bulk term, with consistent P1 mass whitening.

    R_i=sum_e A_e P_e grad(N_i), with every boundary row set to zero.
    r=L^-1 R/mu, M=L L.T, so ||r||²=R.T M^-1 R/mu². No boundary
    traction smoothness term from that paper is implemented. All topology cuts,
    holes and nonmanifold edges are treated as unknown-force boundaries.
    No body force, inertia, prestress, damage or out-of-plane motion is modeled.
    """
    def __init__(self, points, triangles, config):
        self.tri, self.c = triangles, config
        self.active = np.unique(triangles)
        self.local = np.searchsorted(self.active, triangles)
        self.G = _gradients(points, triangles)
        self.area = np.abs(np.linalg.det(_geometry(points, triangles)))/2
        edges = np.sort(self.local[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.interior = np.ones(len(self.active), bool)
        self.interior[np.unique(unique[counts != 2])] = False
        mass = np.zeros((len(self.active), len(self.active)))
        for a in range(3):
            for b in range(3):
                np.add.at(mass, (self.local[:, a], self.local[:, b]),
                          self.area*(2 if a == b else 1)/12)
        self.cholesky = np.linalg.cholesky(mass)

    def evaluate(self, positions, jacobian=False):
        F = np.einsum('tna,tnb->tab', positions[self.tri], self.G)
        P, tangent = _neo_hookean(F, self.c.equilibrium_bulk_shear_ratio,
                                  self.c.min_jacobian)
        n = len(self.active)
        if not jacobian:
            residual = np.zeros((n, 2))
            force = self.area[:, None, None]*np.einsum('tij,tnj->tni', P, self.G)
            np.add.at(residual, self.local.ravel(), force.reshape(-1, 2))
            residual[~self.interior] = 0
            return solve_triangular(self.cholesky, residual, lower=True).ravel()
        element = self.area[:, None, None, None, None]*np.einsum(
            'taj,tijkl,tbl->taibk', self.G, tangent, self.G)
        dofs = self.local[:, :, None]*2+np.arange(2)
        rows = np.broadcast_to(dofs[:, :, :, None, None], element.shape)
        cols = np.broadcast_to(dofs[:, None, None, :, :], element.shape)
        matrix = coo_matrix((element.ravel(), (rows.ravel(), cols.ravel())),
                            shape=(n*2, n*2)).toarray().reshape(n, -1)
        matrix[~self.interior] = 0
        return solve_triangular(self.cholesky, matrix, lower=True).reshape(n*2, n*2)


class _Objective:
    """Sparse analytic Jacobian; each image sample depends on three vertices."""
    def __init__(self, reference, current, points, initial, triangles, bary, config,
                 mechanical_triangles=None):
        self.initial, self.tri, self.bary, self.c = initial, triangles, bary, config
        self.equilibrium = (_EquilibriumGap(points, triangles if mechanical_triangles is None
                                            else mechanical_triangles, config)
                            if config.equilibrium_weight > 0 else None)
        self.active = np.unique(triangles if self.equilibrium is None else
                                np.concatenate((triangles.ravel(), self.equilibrium.active)))
        self.lookup = np.full(len(initial), -1, int)
        self.lookup[self.active] = np.arange(len(self.active))
        self.local = self.lookup[triangles]
        self.G = _gradients(points, triangles)
        self.pairs = _neighbors(triangles)
        self.sample_nodes = np.repeat(self.local, len(bary), axis=0)
        self.sample_weights = np.tile(bary, (len(triangles), 1))
        xy = np.einsum('sn,tnd->tsd', bary, points[triangles]).reshape(-1, 2)
        self.target = _sample(reference, xy)
        self.current = current

    def loss(self, z):
        """Keep the equilibrium norm quadratic and objective under rotations."""
        root = np.sqrt(1+z)
        rho = np.array([2*(root-1), 1/root, -.5/root**3])
        if self.equilibrium is not None:
            size = 2*len(self.equilibrium.active)
            rho[:, -size:] = np.array([z[-size:], np.ones(size), np.zeros(size)])
        return rho

    def cost(self, x):
        residual = self.evaluate(x)/self.c.robust_scale
        if self.equilibrium is None:
            return np.sum(self.c.robust_scale**2*(np.sqrt(1+residual**2)-1))
        return .5*self.c.robust_scale**2*self.loss(residual**2)[0].sum()

    def unpack(self, x):
        q = self.initial.copy()
        q[self.active] = x.reshape(-1, 2)
        return q

    def evaluate(self, x, jacobian=False):
        q = self.unpack(x)
        xy = np.einsum('sn,tnd->tsd', self.bary, q[self.tri]).reshape(-1, 2)
        F = np.einsum('tna,tnb->tab', q[self.tri], self.G)
        J = np.linalg.det(F)
        C = np.einsum('tka,tkb->tab', F, F)
        scale = np.sqrt(self.c.kinematic_weight*len(self.bary))
        barrier = np.sqrt(self.c.jacobian_weight*len(self.bary))
        photo = _sample(self.current, xy)-self.target
        smooth = scale*(C[self.pairs[:, 0]]-C[self.pairs[:, 1]])
        penalty = barrier*np.minimum(J-self.c.min_jacobian, 0)
        eq_scale = np.sqrt(self.c.equilibrium_weight*len(self.bary))
        if not jacobian:
            eq = [] if self.equilibrium is None else eq_scale*self.equilibrium.evaluate(q)
            return np.concatenate((photo, smooth.ravel(), penalty, eq))
        rows, cols, values = [], [], []
        image_gradient = _sample_gradient(self.current, xy)
        rows.append(np.repeat(np.arange(len(photo)), 6))
        cols.append((self.sample_nodes[:, :, None]*2+np.arange(2)).ravel())
        values.append((self.sample_weights[:, :, None]*image_gradient[:, None]).ravel())
        offset = len(photo)
        for side, sign in ((0, 1), (1, -1)):
            tid = self.pairs[:, side]
            # dC_ab / dq_nc = G_na F_cb + F_ca G_nb.
            dC = (np.einsum('tna,tcb->tabnc', self.G[tid], F[tid])
                  + np.einsum('tca,tnb->tabnc', F[tid], self.G[tid]))
            rows.append(np.repeat(offset+np.arange(len(tid)*4), 6))
            cols.append(np.broadcast_to(self.local[tid, None, None, :, None]*2
                                        + np.arange(2), dC.shape).ravel())
            values.append((scale*sign*dC).ravel())
        offset += len(self.pairs)*4
        cofactor = np.stack((F[:, 1, 1], -F[:, 1, 0], -F[:, 0, 1], F[:, 0, 0]), axis=1).reshape(-1, 2, 2)
        dJ = np.einsum('tab,tnb->tna', cofactor, self.G)
        rows.append(np.repeat(offset+np.arange(len(self.tri)), 6))
        cols.append((self.local[:, :, None]*2+np.arange(2)).ravel())
        values.append((barrier*dJ*(J < self.c.min_jacobian)[:, None, None]).ravel())
        offset += len(self.tri)
        if self.equilibrium is not None:
            eq = self.equilibrium
            tangent = coo_matrix(eq_scale*eq.evaluate(q, True))
            mapping = (self.lookup[eq.active, None]*2+np.arange(2)).ravel()
            rows.append(offset+tangent.row)
            cols.append(mapping[tangent.col])
            values.append(tangent.data)
            offset += 2*len(eq.active)
        return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(offset, len(x))).tocsr()


def fit_material_mesh(reference_image, current_image, reference_points,
                      current_initial_points, observed_mask, *, material_mask=None,
                      blade_mask=None, excluded_edges=None, config=None):
    """Refine an existing material map without changing its identity/evidence.

    ``material_mask`` is a reference-frame mask; ``blade_mask`` is a current-
    frame mask. Either accepts ``(reference_mask, current_mask)``. Excluded
    edges are integer node-index pairs: any triangle carrying that coupling is
    removed. Masks and explicit edge cuts are the caller's topology evidence;
    this routine cannot infer a hidden crack. Whole elements touching a masked
    region are excluded conservatively, so their boundary coverage is limited.

    ``supported`` requires incoming observed_mask AND current image checks.
    Completion is always ``inferred``. Folded proposals are rejected, never
    repaired by moving points onto a tool boundary. No parameters are learned.

    ``feature_mode='material-masked'`` excludes nonmaterial/tool pixels from
    every blur and intensity statistic, including the initial gray conversion.
    A mask is required for each frame to exclude that frame's hidden region;
    a missing mask means all pixels are allowed there. Low-support regions and
    interpolation stencils crossing a feature boundary are rejected. This mode
    is an optional ablation; the default ``legacy`` preserves previous results.

    ``equilibrium_weight>0`` requires an explicit stress-free reference
    assumption. Only internal nodes of the eligible topology are constrained;
    tool, masked, excluded-edge and outer boundaries have unknown forces.
    This conditional plane-strain constitutive prior is not a fitted rubber
    law, does not constrain det(F2D)=1, and does not measure traction or stress.

    ``feasible_backtracking=True`` may shorten an unsafe optimizer proposal by
    successive halves (at most 12). Every accepted trial must pass the SAME
    element/J/feature checks and not increase the SAME objective. Raw proposals
    and their failures are retained. A shortened step is not a converged
    constrained solution. This behavior is off by default to freeze old results.
    """
    c = config or FiniteDeformationConfig()
    shape = np.asarray(reference_image).shape[:2]
    material_ref, material_cur = _mask_pair(material_mask, shape, reference_only=True)
    blade_ref, blade_cur = _mask_pair(blade_mask, shape)
    if np.asarray(current_image).shape[:2] != shape:
        raise ValueError('Corresponding image shapes differ')
    ref_valid = _feature_mask(material_ref, blade_ref, shape)
    cur_valid = _feature_mask(material_cur, blade_cur, shape)
    if c.feature_mode == 'material-masked':
        ref, ref_valid = _material_gray(reference_image, ref_valid, c)
        cur, cur_valid = _material_gray(current_image, cur_valid, c)
    else:
        ref, cur = _gray(reference_image), _gray(current_image)
    p = np.asarray(reference_points, float)
    initial = np.asarray(current_initial_points, float)
    observed = np.asarray(observed_mask, bool)
    if (ref.shape != cur.shape or p.ndim != 2 or p.shape[1] != 2
            or initial.shape != p.shape or observed.shape != (len(p),)
            or len(p) < 3 or not np.isfinite(p).all() or not np.isfinite(initial).all()):
        raise ValueError('Finite corresponding images and at least three 2D nodes required')
    if len(np.unique(p, axis=0)) != len(p):
        raise ValueError('Reference nodes must be unique')
    try:
        triangles = Delaunay(p).simplices
    except QhullError as error:
        raise ValueError('Reference nodes must span a nondegenerate 2D mesh') from error
    eligible = (_clear_triangles(p, triangles, material_ref, blade_ref, ref.shape)
                & _clear_triangles(initial, triangles, material_cur, blade_cur, ref.shape))
    ref_condition = np.linalg.cond(_geometry(p, triangles))
    initial_diag = deformation_diagnostics(p, initial, triangles)
    eligible &= (ref_condition <= c.max_reference_condition) & (initial_diag['jacobian'] > c.min_jacobian)
    if excluded_edges is not None:
        cuts = np.asarray(excluded_edges, int).reshape(-1, 2)
        if np.any(cuts < 0) or np.any(cuts >= len(p)):
            raise ValueError('Excluded edges must index the original nodes')
        forbidden = {tuple(sorted(edge)) for edge in cuts}
        eligible &= np.array([not any(tuple(sorted((t[a], t[b]))) in forbidden
                                     for a, b in ((0, 1), (1, 2), (2, 0))) for t in triangles])
    # Uniform interior quadrature; raw texture gates prevent local normalization
    # from promoting a constant or almost black region into image evidence.
    n = c.sample_side
    bary = np.array([(a+1/3, b+1/3, n-a-b+1/3)
                     for a in range(n+1) for b in range(n-a+1)], float)/(n+1)
    sample_ref = np.einsum('sn,tnd->tsd', bary, p[triangles])
    sample_cur = np.einsum('sn,tnd->tsd', bary, initial[triangles])
    texture = np.minimum(_sample(ref, sample_ref).std(1), _sample(cur, sample_cur).std(1))
    photo_eligible = eligible & (texture >= c.min_texture) & observed[triangles].any(1)
    feature_levels = []
    feature_eligible = np.ones(len(triangles), bool)
    for sigma in c.smoothing_levels:
        rf, rv = _features(ref, c, sigma, ref_valid, return_valid=True)
        cf, cv = _features(cur, c, sigma, cur_valid, return_valid=True)
        if c.feature_mode == 'material-masked':
            rv, cv = _interpolation_support(rv), _interpolation_support(cv)
            feature_eligible &= (_clear_triangles(p, triangles, rv, None, ref.shape)
                                 & _clear_triangles(initial, triangles, cv, None, ref.shape))
        feature_levels.append((sigma, rf, cf, rv, cv))
    reference_feature_valid = np.logical_and.reduce([level[3] for level in feature_levels])
    current_feature_valid = np.logical_and.reduce([level[4] for level in feature_levels])
    photo_eligible &= feature_eligible
    chosen = triangles[photo_eligible]
    positions = initial.copy()
    steps, proposals, accepted = [], [], False
    objective = None

    def safety(q):
        # Geometry/topology and feature support stay fixed during the search;
        # no removal of failed elements or relaxation of acceptance thresholds.
        J = deformation_diagnostics(p, q, triangles)['jacobian'][eligible]
        material_clear = _clear_triangles(q, triangles, material_cur, blade_cur, ref.shape)
        feature_clear = (_clear_triangles(q, chosen, current_feature_valid, None, ref.shape)
                         if c.feature_mode == 'material-masked' else np.ones(len(chosen), bool))
        return dict(jacobian=int(np.sum(~(J > c.min_jacobian))),
                    material=int(np.sum(~material_clear[eligible])),
                    feature=int(np.sum(~feature_clear)))

    if len(chosen):
        for sigma, rf, cf, rv, cv in feature_levels:
            objective = _Objective(rf, cf, p, initial, chosen, bary, c,
                                   mechanical_triangles=triangles[eligible])
            x0 = positions[objective.active].ravel()
            anchor = initial[objective.active].ravel()
            fit = least_squares(objective.evaluate, x0,
                jac=lambda x: objective.evaluate(x, True),
                bounds=(anchor-c.max_update_px, anchor+c.max_update_px),
                loss=objective.loss if c.equilibrium_weight > 0 else 'soft_l1',
                f_scale=c.robust_scale, max_nfev=c.max_nfev,
                ftol=1.e-5, xtol=1.e-5, gtol=1.e-5, x_scale='jac')
            proposal = objective.unpack(fit.x)
            proposals.append(proposal.copy())
            initial_cost = float(objective.cost(x0))
            initial_failures = safety(positions)
            trials, accepted_cost, step_scale = [], None, 0.
            for halving in range(13 if c.feasible_backtracking else 1):
                fraction = 2.**-halving
                trial_x = fit.x if halving == 0 else x0+fraction*(fit.x-x0)
                trial = proposal if halving == 0 else objective.unpack(trial_x)
                failures = safety(trial)
                trial_cost = float(fit.cost if halving == 0 else objective.cost(trial_x))
                safe = not any(failures.values())
                improved = np.isfinite(trial_cost) and trial_cost <= initial_cost+1.e-10
                trials.append(dict(step_scale=fraction, cost=trial_cost, failures=failures,
                                   safe=safe, objective_not_increased=bool(improved)))
                if safe and improved:
                    positions, accepted_cost, step_scale = trial, trial_cost, fraction
                    accepted = True
                    break
            steps.append(dict(sigma=float(sigma), success=bool(fit.success), status=int(fit.status),
                              nfev=int(fit.nfev), cost=float(fit.cost),
                              optimality=float(fit.optimality), accepted=step_scale > 0,
                              initial_cost=initial_cost, candidate_cost=float(fit.cost),
                              accepted_cost=accepted_cost, step_scale=step_scale,
                              accepted_full_optimizer_solution=step_scale == 1.,
                              initial_failures=initial_failures,
                              candidate_failures=trials[0]['failures'], line_search=trials))
    diagnostics = deformation_diagnostics(p, positions, triangles)
    residual = np.full(len(p), np.nan)
    image_support = np.zeros(len(p), bool)
    sample_count = np.zeros(len(p), int)
    if objective is not None:
        values = objective.evaluate(positions[objective.active].ravel())[:len(chosen)*len(bary)]
        per_triangle = np.sqrt(np.mean(values.reshape(len(chosen), -1)**2, axis=1))
        totals = np.zeros(len(p))
        for node in range(3):
            np.add.at(totals, chosen[:, node], per_triangle)
            np.add.at(sample_count, chosen[:, node], 1)
        np.divide(totals, sample_count, out=residual, where=sample_count > 0)
        image_support = (sample_count > 0) & (residual <= c.max_photometric_residual)
    supported = observed & image_support
    # Separate quadrature locations never supplied to the optimizer. They are
    # nearby pixels from the same images, not an independent accuracy bound.
    hn = n+1
    hold_bary = np.array([(a+.25, b+.4, hn-a-b+.35)
                         for a in range(hn+1) for b in range(hn-a+1)], float)/(hn+1)
    holdout = dict(before_normalized_rms=None, after_normalized_rms=None,
                   sample_count=0, improvement_fraction=None,
                   interpretation='held-out sample locations, not independent measurement truth')
    if len(chosen):
        _, rf, cf, _, _ = feature_levels[-1]
        target = _sample(rf, np.einsum('sn,tnd->tsd', hold_bary, p[chosen]))
        before = _sample(cf, np.einsum('sn,tnd->tsd', hold_bary, initial[chosen]))-target
        after = _sample(cf, np.einsum('sn,tnd->tsd', hold_bary, positions[chosen]))-target
        a, b = float(np.sqrt(np.mean(before**2))), float(np.sqrt(np.mean(after**2)))
        holdout.update(before_normalized_rms=a, after_normalized_rms=b,
                       sample_count=int(target.size), improvement_fraction=(a-b)/max(a, 1.e-12))
    eq = None if objective is None else objective.equilibrium
    before_eq = np.array([]) if eq is None else eq.evaluate(initial)
    after_eq = np.array([]) if eq is None else eq.evaluate(positions)
    eq_metadata = dict(enabled=c.equilibrium_weight > 0, weight=c.equilibrium_weight,
        bulk_shear_ratio=c.equilibrium_bulk_shear_ratio,
        bulk_shear_ratio_definition='3D bulk modulus K/mu; lambda/mu=K/mu-2/3',
        reference_assumption=c.equilibrium_reference,
        constitutive_assumption='homogeneous compressible Neo-Hookean, plane strain; not calibrated',
        interior_node_count=0 if eq is None else int(eq.interior.sum()),
        boundary_node_count=0 if eq is None else int((~eq.interior).sum()),
        before_normalized_rms=None if not before_eq.size else float(np.sqrt(np.mean(before_eq**2))),
        after_normalized_rms=None if not after_eq.size else float(np.sqrt(np.mean(after_eq**2))),
        normalization='first Piola divided by mu; consistent P1 mass matrix inverse square root',
        boundary_policy='all topology boundaries have unknown forces; no traction residual',
        interpretation='conditional mechanics regularization; no independent measurement confidence',
        source='https://doi.org/10.5802/crmeca.228, Eq.14-15 bulk term only')
    return dict(positions=positions.astype(np.float32), triangles=triangles,
                eligible_triangles=eligible, image_eligible_triangles=photo_eligible,
                feature_eligible_triangles=feature_eligible,
                reference_condition=ref_condition, photometric_residual=residual,
                incident_image_elements=sample_count, input_observed=observed.copy(),
                supported=supported, inferred=~supported, optimizer_steps=steps,
                optimizer_candidate_positions=np.asarray(proposals, dtype=np.float32).reshape(-1, len(p), 2),
                feasible_backtracking=dict(enabled=c.feasible_backtracking, maximum_halvings=12,
                    acceptance='unchanged J/material/feature support and non-increasing objective',
                    interpretation='shortened feasible trial is not a converged constrained optimum'),
                optimizer_converged=bool(steps and steps[-1]['success']),
                accepted=accepted, holdout=holdout, supported_fraction=float(supported.mean()),
                input_observed_fraction=float(observed.mean()), **diagnostics,
                equilibrium_gap=eq_metadata,
                equilibrium_interior_nodes=np.array([], int) if eq is None else eq.active[eq.interior],
                equilibrium_normalized_residual=after_eq,
                feature_normalization=dict(mode=c.feature_mode,
                    minimum_kernel_support=float(c.min_feature_support),
                    reference_valid_fraction=float(reference_feature_valid.mean()),
                    current_valid_fraction=float(current_feature_valid.mean()),
                    reference_mask_provided=material_ref is not None or blade_ref is not None,
                    current_mask_provided=material_cur is not None or blade_cur is not None,
                    interpretation='mask-qualified intensity statistics, not physical measurement confidence'),
                regularization=('objective neighboring C=F.T@F kinematic smoothing'
                                + ('; conditional finite-strain equilibrium gap' if c.equilibrium_weight > 0
                                   else '; not equilibrium')),
                evidence_definition='incoming observed AND local image residual; no physical accuracy claim')
