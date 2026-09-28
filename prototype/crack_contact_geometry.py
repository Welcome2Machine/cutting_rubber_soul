"""Oriented two-dimensional contact geometry for explicit, closed crack faces.

The smooth signed gap belongs to the supporting line of an oriented segment.
Candidate topology, outward normal choice, projection admissibility, endpoint
contact, and switching between segments are the caller's responsibility. In
particular, geometric proximity alone does not define a valid contact pair.
There is no positive gap offset: coincident, already separated faces may have
exactly zero gap. These primitives do not prescribe a crack nucleation rule.
"""

import numpy as np


_ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
_IDENTITY = np.eye(2)
# Map (point, a, b) increments to (r = point - a, e = b - a).
_RELATIVE_MAP = np.block([
    [_IDENTITY, -_IDENTITY, np.zeros((2, 2))],
    [np.zeros((2, 2)), -_IDENTITY, _IDENTITY],
])


def _point(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain two finite coordinates")
    return value


def signed_gap_derivatives(point, a, b):
    """Return ``(gap, gradient[6], Hessian[6, 6])`` with analytic derivatives.

    The coordinate order is ``(point_x, point_y, a_x, a_y, b_x, b_y)``.
    With ``e = b - a``, ``n = rotate_ccw(e) / |e|``, the signed gap is
    ``(point - a) @ n``. Positive gap lies to the left of ``a -> b``.
    Zero gap is regular, including a point coincident with an endpoint.

    This function differentiates the supporting-line gap even when the
    orthogonal projection is outside the finite segment. It must not be used
    there as an endpoint distance derivative. A zero-length segment or
    nonfinite input raises ``ValueError``; no artificial length is inserted.

    For ``r = point - a`` and ``s = r @ R @ e``, differentiating ``s / |e|``
    first in independent ``(r, e)`` coordinates gives H_rr = 0,
    H_re = R (I - t t.T) / |e|, and
    H_ee = (-w t.T - t w.T - g I + 3 g t t.T) / |e|**2,
    where t = e / |e| and w = R.T @ r. A constant linear map then gives
    derivatives with respect to all three nodes, including normal rotation.
    """
    point, a, b = (_point(value, name) for value, name in
                   ((point, "point"), (a, "a"), (b, "b")))
    r, e = point - a, b - a
    length = float(np.hypot(e[0], e[1]))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("contact segment must have finite, nonzero length")
    tangent = e / length
    normal = _ROTATION @ tangent
    gap = float(r @ normal)
    scaled_w = _ROTATION.T @ (r / length)
    scaled_gap = gap / length
    gradient_e = scaled_w - scaled_gap * tangent
    gradient = _RELATIVE_MAP.T @ np.concatenate((normal, gradient_e))

    tangent_outer = np.outer(tangent, tangent)
    hessian_re = _ROTATION @ (_IDENTITY - tangent_outer) / length
    hessian_ee = (
        -np.outer(scaled_w, tangent) - np.outer(tangent, scaled_w)
        - scaled_gap * _IDENTITY + 3.0 * scaled_gap * tangent_outer
    ) / length
    hessian_relative = np.block([
        [np.zeros((2, 2)), hessian_re],
        [hessian_re.T, hessian_ee],
    ])
    hessian = _RELATIVE_MAP.T @ hessian_relative @ _RELATIVE_MAP
    return gap, gradient, hessian


def nearest_segment(point, vertices, edges):
    """Return ``(edge_index, clamped_parameter)`` for the nearest finite edge.

    ``vertices`` has shape (n, 2); integer ``edges`` has shape (m, 2).
    Euclidean point-to-segment distance, not signed gap or distance to an
    infinite line, selects the edge. The returned parameter is in [0, 1],
    with the closest point ``(1 - parameter) * a + parameter * b``. A value
    of 0 or 1 indicates an endpoint projection; it does not distinguish an
    exact endpoint projection from an out-of-range projection after clamping.
    Ties choose the first input edge. Invalid inputs, empty candidate lists,
    and any degenerate candidate edge raise ``ValueError``.
    """
    point = _point(point, "point")
    vertices = np.asarray(vertices, dtype=float)
    edges = np.asarray(edges)
    if (vertices.ndim != 2 or vertices.shape[1] != 2
            or not np.all(np.isfinite(vertices))):
        raise ValueError("vertices must have shape (n, 2) and be finite")
    if (edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0
            or not np.issubdtype(edges.dtype, np.integer)):
        raise ValueError("edges must be a nonempty integer array of shape (m, 2)")
    if np.any(edges < 0) or np.any(edges >= len(vertices)):
        raise ValueError("edge vertex index is out of bounds")

    starts = vertices[edges[:, 0]]
    vectors = vertices[edges[:, 1]] - starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    if np.any(~np.isfinite(lengths)) or np.any(lengths == 0.0):
        raise ValueError("contact segments must have finite, nonzero length")
    tangents = vectors / lengths[:, None]
    parameters = np.sum((point - starts) * tangents, axis=1) / lengths
    parameters = np.clip(parameters, 0.0, 1.0)
    residuals = point - starts - parameters[:, None] * vectors
    distances = np.hypot(residuals[:, 0], residuals[:, 1])
    index = int(np.argmin(distances))
    return index, float(parameters[index])
