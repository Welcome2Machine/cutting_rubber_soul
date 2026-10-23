"""Material coordinates of a rounded blade's numerical contact, without solving.

The input is an accepted, intact P1 state. Distances below IPC's ``dhat``
identify a geometric barrier support, not zero-gap contact or contact pressure.
Original top-boundary edge identities, and the knife pole identity, stay fixed.
"""

from __future__ import annotations

import numpy as np


def _merge_intervals(intervals, tolerance=0.0):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1] + tolerance:
            result[-1][1] = max(result[-1][1], float(hi))
        else:
            result.append([float(lo), float(hi)])
    return result


def _capsule_intervals(a, b, c, d, radius):
    """Return closures of nonempty open u-intervals with dist(a+u(b-a), cd)<r.

    The segment projection is affine in u. Its endpoint/interior changes split
    [0, 1] into at most three pieces, each with a convex quadratic distance.
    No quadrature or whole-edge inflation is used. Isolated tangencies vanish.
    """
    v, w = b - a, d - c
    w2 = float(w @ w)
    if w2 <= 0 or float(v @ v) <= 0:
        raise ValueError("Degenerate contact segment")
    r = a - c
    intercept, slope = float(r @ w / w2), float(v @ w / w2)
    cuts = [0.0, 1.0]
    if slope != 0:
        cuts.extend(t for t in (-intercept / slope, (1 - intercept) / slope) if 0 < t < 1)
    cuts.sort()
    intervals = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        projection = intercept + slope * ((lo + hi) / 2)
        if projection <= 0:
            p, q = r, v
        elif projection >= 1:
            p, q = a - d, v
        else:
            # Factor the one-dimensional perpendicular quadratic. This avoids
            # subtracting almost equal vectors for nearly parallel segments.
            normal = np.array([-w[1], w[0]]) / np.sqrt(w2)
            constant, linear = float(r @ normal), float(v @ normal)
            if linear == 0:
                if abs(constant) < radius:
                    intervals.append((lo, hi))
            else:
                left, right = sorted(((-radius - constant) / linear, (radius - constant) / linear))
                lower, upper = max(lo, left), min(hi, right)
                if lower < upper:
                    intervals.append((lower, upper))
            continue
        quadratic = float(q @ q)
        if quadratic == 0:
            if float(p @ p) < radius * radius:
                intervals.append((lo, hi))
            continue
        center = -float(p @ q) / quadratic
        nearest = p + center * q
        available = radius * radius - float(nearest @ nearest)
        if available <= 0:
            continue
        halfwidth = np.sqrt(available / quadratic)
        lower, upper = max(lo, center - halfwidth), min(hi, center + halfwidth)
        if lower < upper:
            intervals.append((lower, upper))
    return _merge_intervals(intervals)


class TipContactTracker:
    """Report fixed-pole contact and the full faceted nose's material intervals.

    ``config`` is a FEMConfig-like object (or a mapping) containing inclination,
    specimen height/length and drop_s. The reference knife must use IPCProblem's
    oblique polygon ordering: two left flank vertices, nose_facets+1 circular
    vertices, two right flank vertices. Flat-punch patch cases are rejected.
    """

    def __init__(self, reference_xy, triangles, body_nodes, config, dhat, nose_facets):
        self.reference = np.array(reference_xy, dtype=float, copy=True)
        tri = np.asarray(triangles)
        if self.reference.ndim != 2 or self.reference.shape[1] != 2 or not np.all(np.isfinite(self.reference)):
            raise ValueError("Reference positions must be finite (n, 2)")
        if (isinstance(body_nodes, (bool, np.bool_)) or not isinstance(body_nodes, (int, np.integer))
                or not 3 <= body_nodes < len(self.reference)):
            raise ValueError("Invalid body node count")
        if (isinstance(nose_facets, (bool, np.bool_)) or not isinstance(nose_facets, (int, np.integer))
                or nose_facets < 4 or nose_facets % 2):
            raise ValueError("An even nose facet count >= 4 is required")
        canonical=nose_facets+5
        if len(self.reference) - body_nodes < canonical:
            raise ValueError("Unsupported blade topology; flat-punch patch cases have no rounded tip")
        if len(self.reference)-body_nodes>canonical:
            # Explicit straight-edge subdivision keeps canonical nose indices.
            # Extra off-outline points cannot silently enter this observer.
            blade=self.reference[body_nodes:body_nodes+canonical]
            points=self.reference[body_nodes+canonical:]
            distance=np.full(len(points),np.inf)
            for i,j in ((0,1),(1,2),(canonical-3,canonical-2),(canonical-2,canonical-1)):
                a,b=blade[[i,j]];edge=b-a
                t=np.clip((points-a)@edge/(edge@edge),0.,1.)
                distance=np.minimum(distance,np.linalg.norm(points-a-t[:,None]*edge,axis=1))
            if np.max(distance)>1e-9:raise ValueError('Extra blade nodes must subdivide canonical straight edges')
        if (tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0
                or not np.issubdtype(tri.dtype, np.integer)
                or np.any(tri < 0) or np.any(tri >= body_nodes)
                or np.any(np.diff(np.sort(tri, axis=1), axis=1) == 0)):
            raise ValueError("Invalid body triangles")
        self.dhat = float(dhat)
        if not np.isfinite(self.dhat) or self.dhat <= 0:
            raise ValueError("dhat must be finite and positive")
        value = (lambda key: config[key]) if isinstance(config, dict) else (lambda key: getattr(config, key))
        inclination, height, length, self.origin = map(float, (
            value("inclination_deg"), value("height_mm"), value("length_mm"), value("drop_s_mm")))
        if (not all(np.isfinite(k) for k in (inclination, height, length, self.origin))
                or not 0 <= inclination < 90 or height <= 0 or not 0 < self.origin < length):
            raise ValueError("Invalid specimen coordinates")
        angle = np.deg2rad(inclination)
        es, en = np.array([np.cos(angle), -np.sin(angle)]), np.array([np.sin(angle), np.cos(angle)])
        self.tolerance = 1e-10 * max(1.0, length, height)
        self.s = self.reference[:body_nodes] @ es
        normal = self.reference[:body_nodes] @ en
        edges = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Nonmanifold body boundary")
        edges = edges[(counts == 1) & np.all(abs(normal[edges] - height) <= self.tolerance, axis=1)]
        if len(edges) == 0:
            raise ValueError("No original top boundary was found")
        reverse = self.s[edges[:, 0]] > self.s[edges[:, 1]]
        edges[reverse] = edges[reverse, ::-1]
        self.edges = edges[np.argsort(self.s[edges[:, 0]], kind="stable")]
        if np.any(np.diff(self.s[self.edges], axis=1) <= self.tolerance):
            raise ValueError("Degenerate original top edge")
        self.nose_nodes = np.arange(body_nodes + 2, body_nodes + 3 + nose_facets)
        self.tip_node = int(body_nodes + 2 + nose_facets // 2)
        arc = self.reference[self.nose_nodes]
        if np.argmin(self.reference[body_nodes:, 1]) + body_nodes != self.tip_node:
            raise ValueError("Reference knife pole is inconsistent with rounded blade ordering")
        # The fit checks topology and records the faceted versus ideal-circle
        # first-touch baseline; it does not replace the collision polygon.
        offset = arc[[0, -1]] - arc[nose_facets // 2]
        try:
            relative_center = np.linalg.solve(2 * offset, np.sum(offset * offset, axis=1))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Rounded nose is not circular") from exc
        radius = float(np.linalg.norm(relative_center))
        center = arc[nose_facets // 2] + relative_center
        if (not np.isfinite(radius) or radius <= 0 or not np.all(np.isfinite(center))
                or not np.allclose(np.linalg.norm(arc - center, axis=1), radius, rtol=1e-8, atol=self.tolerance)):
            raise ValueError("Rounded nose is not circular")
        approach = float((np.min(arc @ en) - height) / en[1])
        support = np.abs(arc @ en - np.min(arc @ en)) <= self.tolerance
        first_s = (arc[support] - np.array([0.0, approach])) @ es
        rest_depth = float((height + radius - center @ en) / en[1])
        if not np.all(np.isfinite(np.r_[approach, first_s, rest_depth])):
            raise ValueError("Nonfinite initial contact geometry")
        self.metadata = dict(
            method="fixed_knife_pole_projection_and_exact_capsule_intervals_v1",
            interpretation="geometric IPC barrier support; not pressure or zero-gap physical contact",
            unique_status_meaning="unique closest-pole material projection within the distance tie tolerance; other farther material branches may also lie inside dhat",
            reference_origin_s_mm=self.origin,
            reference_distance="signed undeformed top arclength s - configured drop_s",
            positive_direction="increasing material s (right/down along inclined original top)",
            dhat_mm=self.dhat, knife_tip_node=self.tip_node,
            original_top_edges=len(self.edges), nose_facets=int(nose_facets),
            tip_radius_mm=radius,
            interval_endpoints="closures of distance < dhat intervals; equality alone is not contact",
            ambiguity_tolerance_mm=self.tolerance,
            ideal_circle_initial_contact_s_mm=self.origin,
            faceted_first_contact_depth_mm=rest_depth + approach,
            faceted_first_contact_s_interval_mm=[float(first_s.min()), float(first_s.max())],
            faceted_first_contact_offset_interval_mm=[float(first_s.min() - self.origin), float(first_s.max() - self.origin)],
            depth_origin="unchanged solver depth; not re-zeroed at positive-gap IPC onset",
        )

    def observe(self, positions, depth_mm):
        x = np.asarray(positions, dtype=float)
        depth = float(depth_mm)
        if x.shape != self.reference.shape or not np.all(np.isfinite(x)) or not np.isfinite(depth):
            raise ValueError("Current positions/depth must be finite and match reference topology")
        a, b = x[self.edges[:, 0]], x[self.edges[:, 1]]
        v = b - a
        lengths2 = np.einsum("ij,ij->i", v, v)
        if np.any(lengths2 <= np.finfo(float).eps ** 2) or not np.all(np.isfinite(lengths2)):
            raise ValueError("Degenerate current top edge")
        tip = x[self.tip_node]
        alpha = np.clip(np.einsum("ij,ij->i", tip - a, v) / lengths2, 0, 1)
        projected = a + alpha[:, None] * v
        distances = np.linalg.norm(projected - tip, axis=1)
        if not np.all(np.isfinite(distances)):
            raise ValueError("Nonfinite current contact geometry")
        minimum = float(distances.min())
        nearest = []
        for index in np.flatnonzero(distances <= minimum + self.tolerance):
            edge, t = self.edges[index], float(alpha[index])
            s = float((1 - t) * self.s[edge[0]] + t * self.s[edge[1]])
            # Shared endpoints are one material point, even with two edges.
            if any(abs(item["reference_s_mm"] - s) <= self.tolerance for item in nearest):
                continue
            nearest.append(dict(
                reference_s_mm=s, reference_offset_mm=s - self.origin,
                absolute_reference_offset_mm=abs(s - self.origin),
                reference_point_mm=((1 - t) * self.reference[edge[0]] + t * self.reference[edge[1]]).tolist(),
                current_point_mm=projected[index].tolist(),
                edge_nodes=edge.tolist(), edge_alpha=t, gap_mm=float(distances[index]),
            ))
        nearest.sort(key=lambda item: item["reference_s_mm"])
        candidates = [item for item in nearest if item["gap_mm"] < self.dhat]
        status = "no_tip_contact" if not candidates else "unique" if len(candidates) == 1 else "ambiguous"
        result = dict(depth_mm=depth, status=status, knife_tip_mm=tip.tolist(), gap_mm=minimum,
                      reference_offset_mm=None, absolute_reference_offset_mm=None, reference_s_mm=None,
                      reference_point_mm=None, current_point_mm=None, edge_nodes=None, edge_alpha=None,
                      candidates=candidates, nearest_projection_candidates=nearest)
        if status == "unique":
            result.update(candidates[0])
        arc = x[self.nose_nodes]
        c, d = arc[:-1], arc[1:]
        if np.any(np.linalg.norm(d - c, axis=1) <= np.finfo(float).eps):
            raise ValueError("Degenerate current nose edge")
        # AABB rejection is conservative. Exact capsule clipping is done only
        # for the remaining pairs; the full upper surface is still considered.
        near = np.all((np.maximum(a, b) >= arc.min(axis=0) - self.dhat)
                      & (np.minimum(a, b) <= arc.max(axis=0) + self.dhat), axis=1)
        intervals = []
        for index in np.flatnonzero(near):
            pair_mask = np.all((np.maximum(c, d) >= np.minimum(a[index], b[index]) - self.dhat)
                               & (np.minimum(c, d) <= np.maximum(a[index], b[index]) + self.dhat), axis=1)
            s0, s1 = self.s[self.edges[index]]
            for j in np.flatnonzero(pair_mask):
                for lo, hi in _capsule_intervals(a[index], b[index], c[j], d[j], self.dhat):
                    intervals.append((s0 + lo * (s1 - s0), s0 + hi * (s1 - s0)))
        intervals = _merge_intervals(intervals, tolerance=self.tolerance)
        result.update(
            nose_status="no_nose_contact" if not intervals else "connected" if len(intervals) == 1 else "disconnected",
            nose_reference_intervals_mm=intervals,
            nose_offset_intervals_mm=[[lo - self.origin, hi - self.origin] for lo, hi in intervals],
        )
        return result

    def observe_footprint(self, positions, depth_mm):
        """Measure original top-surface support across the entire rigid blade.

        The distance threshold is the existing IPC ``dhat``. These intervals
        describe geometric barrier proximity, not contact pressure, exact
        zero-gap contact, or accumulated material slip. All segment/capsule
        intersections are clipped analytically, including support wholly
        inside a long material edge. The existing fixed-pole observable is
        deliberately independent and unchanged.

        Part supports can overlap near shared blade vertices. The whole-blade
        lengths are unions on each material edge, so overlapping supports are
        counted only once. Current lengths follow the deformed material edge,
        not the blade's arclength. The closing cap is included for completeness.
        """
        x = np.asarray(positions, dtype=float)
        depth = float(depth_mm)
        if x.shape != self.reference.shape or not np.all(np.isfinite(x)) or not np.isfinite(depth):
            raise ValueError("Current positions/depth must be finite and match reference topology")
        a, b = x[self.edges[:, 0]], x[self.edges[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        if np.any(lengths <= np.finfo(float).eps) or not np.all(np.isfinite(lengths)):
            raise ValueError("Degenerate current top edge")
        first, last = int(self.nose_nodes[0]), int(self.nose_nodes[-1])
        parts = {
            "left_shank": [(first - 2, first - 1)],
            "left_flank": [(first - 1, first)],
            "nose": list(zip(self.nose_nodes[:-1], self.nose_nodes[1:])),
            "right_flank": [(last, last + 1)],
            "right_shank": [(last + 1, last + 2)],
            "top_cap": [(last + 2, first - 2)],
        }
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        whole_support = {}
        summaries = {}

        def summarize(support):
            intervals, current_length, edge_count = [], 0.0, 0
            for index, pieces in support.items():
                pieces = _merge_intervals(pieces)
                if not pieces:
                    continue
                s0, s1 = self.s[self.edges[index]]
                intervals.extend((s0 + lo * (s1 - s0), s0 + hi * (s1 - s0)) for lo, hi in pieces)
                current_length += lengths[index] * sum(hi - lo for lo, hi in pieces)
                edge_count += 1
            intervals = _merge_intervals(intervals, tolerance=self.tolerance)
            return dict(
                status="no_support" if not intervals else "connected" if len(intervals) == 1 else "disconnected",
                reference_s_intervals_mm=intervals,
                reference_offset_intervals_mm=[[lo - self.origin, hi - self.origin] for lo, hi in intervals],
                supported_reference_length_mm=float(sum(hi - lo for lo, hi in intervals)),
                supported_current_length_mm=float(current_length),
                supporting_top_edge_count=edge_count,
            )

        for name, segments in parts.items():
            support = {}
            for i, j in segments:
                c, d = x[i], x[j]
                if np.linalg.norm(d - c) <= np.finfo(float).eps:
                    raise ValueError("Degenerate current blade edge")
                # One segment-sized temporary per pass; never a dense
                # material-edge by blade-edge allocation.
                near = np.all((amax >= np.minimum(c, d) - self.dhat)
                              & (amin <= np.maximum(c, d) + self.dhat), axis=1)
                for index in np.flatnonzero(near):
                    pieces = _capsule_intervals(a[index], b[index], c, d, self.dhat)
                    if pieces:
                        support.setdefault(int(index), []).extend(pieces)
                        whole_support.setdefault(int(index), []).extend(pieces)
            summaries[name] = summarize(support)
        return dict(
            depth_mm=depth,
            method="whole_blade_exact_capsule_intervals_v1",
            interpretation="original upper surface within IPC dhat of blade; geometric support, not pressure, exact zero-gap contact or accumulated slip",
            dhat_mm=self.dhat,
            part_lengths_may_overlap=True,
            **summarize(whole_support),
            parts=summaries,
        )
