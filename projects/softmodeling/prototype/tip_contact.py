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

    def __init__(self, reference_xy, triangles, body_nodes, config, dhat, nose_facets, landmarks=None):
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
        self.body_nodes = int(body_nodes)
        self.es, self.en = es, en
        if np.any(np.diff(self.s[self.edges], axis=1) <= self.tolerance):
            raise ValueError("Degenerate original top edge")
        self.nose_nodes = np.arange(body_nodes + 2, body_nodes + 3 + nose_facets)
        pole = int(landmarks["pole_local"])-2 if landmarks else nose_facets//2
        if not 0 < pole < nose_facets:raise ValueError("Invalid blade pole")
        self.tip_node = int(body_nodes + 2 + pole)
        self.top_edge_map = {tuple(sorted(edge)): edge for edge in self.edges}
        self.top_node_edge = {int(node): edge for edge in self.edges for node in edge}
        arc = self.reference[self.nose_nodes]
        self.nose_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
        self.anchor_arc = self.nose_length[pole]
        self.blade_part_cache = {}
        self.arc_resolution = float(np.max(np.diff(self.nose_length)))
        if np.argmin(self.reference[body_nodes:, 1]) + body_nodes != self.tip_node:
            raise ValueError("Reference knife pole is inconsistent with rounded blade ordering")
        # The fit checks topology and records the faceted versus ideal-circle
        # first-touch baseline; it does not replace the collision polygon.
        offset = arc[[0, -1]] - arc[pole]
        try:
            relative_center = np.linalg.solve(2 * offset, np.sum(offset * offset, axis=1))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Rounded nose is not circular") from exc
        radius = float(np.linalg.norm(relative_center))
        center = arc[pole] + relative_center
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
            initial_depth_mm=rest_depth,
            mechanical_first_contact="not independently identified by geometric proximity",
            footprint_method="whole_blade_exact_capsule_intervals_v1",
            velocity_method="same_current_edge_weights_in_both_accepted_states_v1",
            relative_motion="raw lab-frame difference and finite rigid-blade-frame transport; proximity diagnostics, not loaded-interface slip",
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

    @staticmethod
    def _projection(point, a, b):
        edge = b-a
        return float(np.clip((point-a) @ edge / (edge @ edge), 0., 1.))

    def _blade_part(self, nodes):
        key = tuple(sorted(map(int,nodes)))
        if key not in self.blade_part_cache:
            first,last = self.nose_nodes[[0,-1]]
            if all(first <= n <= last for n in nodes):
                part = 'nose'
            else:
                midpoint = self.reference[nodes].mean(axis=0)
                parts = [('left_shank',first-2,first-1),('left_flank',first-1,first),
                         ('right_flank',last,last+1),('right_shank',last+1,last+2),
                         ('top_cap',last+2,first-2)]
                def distance(item):
                    a,b = self.reference[list(item[1:])]
                    return np.linalg.norm(midpoint-a-self._projection(midpoint,a,b)*(b-a))
                part = min(parts,key=distance)[0]
            self.blade_part_cache[key] = part
        return self.blade_part_cache[key]

    def native_accounting(self, problem, x, step):
        """Signed stencil contributions. Shared vertices follow fixed feature ownership."""
        du = problem.step_increment(x)
        rows = {}
        for key,collection,potential,values in (
                ('normal',problem.collisions(x),problem.barrier,x),
                ('friction',problem.lag(x),problem.friction,du)):
            if collection is None:
                continue
            for stencil in [*collection.ev_collisions,*collection.vv_collisions]:
                local = np.asarray(stencil.vertex_ids(problem.cmesh.edges,problem.cmesh.faces))
                local = local[local>=0]
                nodes = problem.surface[local]
                tool = nodes >= self.body_nodes
                if tool.any():
                    name = self._blade_part(nodes[tool])
                else:
                    adjacent = len(nodes)==2 and tuple(sorted(map(int,local))) in problem._connected_boundary_pairs
                    name = 'connected_body_vv' if adjacent else 'body_self_other'
                row = rows.setdefault(name,dict(normal_fx_n=0.,normal_fz_n=0.,friction_fx_n=0.,
                    friction_fz_n=0.,friction_work_n_mm=0.))
                gradient = potential.gradient(stencil,values[nodes].ravel()).reshape(-1,2)
                reaction = -gradient[tool].sum(axis=0)
                row[key+'_fx_n'] += float(reaction[0])
                row[key+'_fz_n'] += float(reaction[1])
                if key == 'friction':
                    row['friction_work_n_mm'] += float(np.sum(gradient*du[nodes]))
        force = [sum(r['normal_'+c]+r['friction_'+c] for r in rows.values()) for c in ('fx_n','fz_n')]
        closure = (np.asarray(force)-[step['reaction_x_n'],step['reaction_fz_n']]).tolist()
        return dict(parts=rows,force_closure_fx_fz_n=closure,
            friction_work_closure_n_mm=sum(r['friction_work_n_mm'] for r in rows.values())-step['friction_work_n_mm'],
            shared_vertex_rule='nose endpoints belong to nose for vertex contact; other shared vertices use first incident canonical segment')

    def native_nose_candidates(self, problem, x):
        """Retain repulsive EV/VV terms on the original top and circular nose."""
        c = problem.collisions(x)
        rows = []
        for kind, stencils in (("EV", c.ev_collisions), ("VV", c.vv_collisions)):
            for stencil in stencils:
                ids = np.asarray(stencil.vertex_ids(problem.cmesh.edges, problem.cmesh.faces))
                nodes = problem.surface[ids[ids >= 0]]
                body, tool = nodes[nodes < self.body_nodes], nodes[nodes >= self.body_nodes]
                if not len(body) or not len(tool) or not np.all(np.isin(tool, self.nose_nodes)):
                    continue
                if len(body) == 1 and int(body[0]) in self.top_node_edge:
                    edge = self.top_node_edge[int(body[0])]
                    u = float(body[0] == edge[1])
                elif len(body) == 2 and tuple(sorted(body)) in self.top_edge_map:
                    edge = self.top_edge_map[tuple(sorted(body))]
                    u = self._projection(x[tool[0]], *x[edge])
                else:
                    continue
                material = (1-u)*x[edge[0]] + u*x[edge[1]]
                arc = self.nose_length[tool-self.nose_nodes[0]]
                t = self._projection(material, *x[tool]) if len(tool)==2 else 0.
                blade = (1-t)*x[tool[0]] + t*x[tool[-1]]
                delta = blade-material
                gap = float(np.linalg.norm(delta))
                if not 0 < gap < self.dhat:
                    continue
                grad = problem.barrier.gradient(stencil, x[nodes].ravel()).reshape(-1, 2)
                load = float(-grad[nodes >= self.body_nodes].sum(axis=0) @ delta / gap)
                if load > 0:
                    rows.append(dict(kind=kind, edge_nodes=edge.tolist(), edge_alpha=u,
                        nose_arc_mm=float((1-t)*arc[0]+t*arc[-1]), gap_mm=gap, repulsive_term_n=load,
                        tip_contact_point_mm=material.tolist(), blade_point_mm=blade.tolist()))
        return rows

    def sliding_amount(self, x, contact, native=None):
        """Select by fixed-anchor arc distance. Keep disconnected material support explicit."""
        intervals = contact["nose_reference_intervals_mm"]
        out = dict(method="native_nose_arc_reference_s/2", S_landing_mm=self.origin,
                   status="no_nose_support", S_tip_mm=None, SA_signed_mm=None, SA_abs_mm=None,
                   S_tip_intervals_mm=intervals, SA_intervals_mm=contact["nose_offset_intervals_mm"],
                   selected=None, native_supported=bool(native), branch_id=None, proximity_estimate=None)
        if not intervals:
            return out
        if len(intervals) > 1 and not native:
            out["status"] = "ambiguous_material_branches"
            return out
        pool = list(native or [])
        if not pool:
            # Closest supported arc coordinate to the anchor on each facet.
            # A capsule endpoint is a one-sided limit, not pressure evidence.
            a, b = x[self.edges[:, 0]], x[self.edges[:, 1]]
            for i, (j,k) in enumerate(zip(self.nose_nodes[:-1], self.nose_nodes[1:])):
                c, d = x[[j,k]]
                near = np.all((np.maximum(a,b) >= np.minimum(c,d)-self.dhat) &
                              (np.minimum(a,b) <= np.maximum(c,d)+self.dhat),axis=1)
                for idx in np.flatnonzero(near):
                    for lo,hi in _capsule_intervals(c,d,a[idx],b[idx],self.dhat):
                        t = hi if j < self.tip_node else lo
                        pt = c+t*(d-c)
                        u = self._projection(pt,a[idx],b[idx])
                        mat = a[idx]+u*(b[idx]-a[idx])
                        pool.append(dict(edge_nodes=self.edges[idx].tolist(),edge_alpha=u,
                            nose_arc_mm=float(self.nose_length[i]+t*(self.nose_length[i+1]-self.nose_length[i])),
                            gap_mm=float(np.linalg.norm(pt-mat)),tip_contact_point_mm=mat.tolist(),
                            blade_point_mm=pt.tolist()))
        if not pool:
            out["status"] = "unresolved_geometry"
            return out
        distance = lambda r: abs(r["nose_arc_mm"]-self.anchor_arc)
        pool.sort(key=distance)
        best = pool[0]
        def address(row):
            a,b = self.s[row["edge_nodes"]]
            return float(a+row["edge_alpha"]*(b-a))
        s = address(best)
        tied = [r for r in pool if distance(r)<=distance(best)+self.arc_resolution*.5]
        def branch(sm):
            return next((i for i,(lo,hi) in enumerate(intervals)
                         if lo-self.tolerance<=sm<=hi+self.tolerance),None)
        out["branch_candidates"]=[]
        for idx,(lo,hi) in enumerate(intervals):
            members=[r for r in pool if lo-self.tolerance<=address(r)<=hi+self.tolerance]
            if members:
                row=min(members,key=distance)
                out["branch_candidates"].append(dict(branch_id=idx,S_tip_mm=address(row),
                    SA_signed_mm=address(row)-self.origin,arc_distance_mm=distance(row),
                    candidate_count=len(members),representative=row))
        out["resolution_tied_addresses_mm"]=sorted({address(r) for r in tied})
        out["multiple_material_branches"]=len(intervals)>1
        close=any(abs(address(r)-s)>self.tolerance for r in tied)
        exact=any(abs(address(r)-s)>self.tolerance and distance(r)<=distance(best)+self.tolerance for r in tied)
        if exact or (not native and close):
            out["status"]="ambiguous_within_facet_resolution"
        elif native:
            status="conditional_nearest_nose" if close or len(intervals)>1 else "native_nose_contact"
            out.update(status=status,selected=best,branch_id=branch(s),
                       S_tip_mm=s,SA_signed_mm=s-self.origin,SA_abs_mm=abs(s-self.origin))
        else:
            out.update(status="proximity_only",proximity_estimate=dict(best,S_tip_mm=s,SA_signed_mm=s-self.origin))
        return out

    def pole_diagnostics(self, observation):
        """Expose the closest projection even when strict pole M is undefined.

        A nearest address outside dhat is diagnostic geometry, never a valid
        migration sample. Keep this separate so historical online observations
        retain their exact schema and equality checks during offline replay.
        """
        nearest = observation['nearest_projection_candidates']
        return dict(
            reason=('gap_not_below_dhat' if observation['gap_mm'] >= self.dhat else
                    'multiple_nearest_material_points' if observation['status'] == 'ambiguous' else
                    'unique_nearest_within_dhat'),
            dhat_mm=self.dhat,
            gap_minus_dhat_mm=observation['gap_mm']-self.dhat,
            nearest_projection_count=len(nearest),
            nearest_projections=nearest,
            nearest_is_migration=observation['status'] == 'unique',
        )

    def nose_support_evidence(self, sa, native, x, previous, dt, prior_intervals=(), prior_evidence=None):
        """Keep geometric bands and sampled repulsive addresses distinct in replay."""
        bands = sa['S_tip_intervals_mm']
        branches = []
        for i, (lo, hi) in enumerate(bands):
            parents = [j for j, (a, b) in enumerate(prior_intervals)
                       if max(a, lo) <= min(b, hi) + self.tolerance]
            branches.append(dict(branch_id=i, parent_indices=parents,
                                 geometric_S_mm=[lo, hi], native_S_mm=[]))
        for row in native:
            a, b = row['edge_nodes']
            s = float(self.s[a] + row['edge_alpha'] * (self.s[b] - self.s[a]))
            for branch in branches:
                lo, hi = branch['geometric_S_mm']
                if lo-self.tolerance <= s <= hi+self.tolerance:
                    branch['native_S_mm'].append(s)
                    break
            else:
                raise ValueError('Native nose address outside every geometric band')
        for branch in branches:
            branch['native_S_mm'] = sorted(set(branch['native_S_mm']))
        selected = sa['selected']
        sample = None
        before = (prior_evidence or {}).get('selected_material')
        transition = 'lost' if before else 'unselected'
        if selected is not None:
            a, b = selected['edge_nodes']
            w = selected['edge_alpha']
            weights = np.array([1-w, w])
            ref = weights @ self.reference[[a, b]]
            current = weights @ x[[a, b]]
            if (abs(ref @ self.es - sa['S_tip_mm']) > self.tolerance or
                    not np.allclose(current, selected['tip_contact_point_mm'], atol=self.tolerance, rtol=0)):
                raise ValueError('Saved representative differs from its material edge interpolation')
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError('Positive physical dt required for nose kinematics')
            vm = weights @ (x[[a,b]]-previous[[a,b]]) / dt
            branch_id = sa['branch_id']
            parents = branches[branch_id]['parent_indices']
            transition = 'acquired'
            if before:
                parent_id = before['branch_id']
                if parent_id not in parents:
                    transition = 'switched'
                elif len(parents) != 1 or sum(parent_id in g['parent_indices'] for g in branches) != 1:
                    transition = 'split_or_merge'
                else:
                    transition = 'continued'
            sample = dict(reference_xy_mm=ref.tolist(), material_velocity_mm_s=vm.tolist(),
                          branch_id=branch_id)
            if 'blade_point_mm' in selected:
                # Pull back the same point on the rigid nose facet, including for
                # a rotating tool. The selected body weights are also held fixed.
                arc = selected['nose_arc_mm']
                j = np.clip(self.nose_length.searchsorted(arc)-1, 0, len(self.nose_nodes)-2)
                u = (arc-self.nose_length[j])/(self.nose_length[j+1]-self.nose_length[j])
                nodes = self.nose_nodes[j:j+2]
                blade = (1-u)*x[nodes[0]] + u*x[nodes[1]]
                if not np.allclose(blade, selected['blade_point_mm'], atol=self.tolerance, rtol=0):
                    raise ValueError('Selected blade point differs from reference facet interpolation')
                vb = np.array([1-u,u]) @ (x[nodes]-previous[nodes]) / dt
                relative = vm-vb
                gap = blade-current
                normal = gap/np.linalg.norm(gap)
                pair = np.array([normal[1],-normal[0]])
                # The current material edge keeps its original increasing-S order,
                # even after rotating through 90 degrees in laboratory coordinates.
                left,right = self.top_edge_map[tuple(sorted((a,b)))]
                edge = x[right]-x[left]
                edge = edge/np.linalg.norm(edge)
                direction = edge
                if transition == 'continued' and before.get('pair_tangent') is not None:
                    # Transport the prior oriented pair in the local material frame.
                    old_edge = np.asarray(before['material_tangent'])
                    old_pair = np.asarray(before['pair_tangent'])
                    direction = ((old_pair @ old_edge)*edge +
                                 (old_pair @ [-old_edge[1],old_edge[0]])*np.array([-edge[1],edge[0]]))
                alignment = float(pair @ direction)
                signed_pair = None if abs(alignment)<=1e-10 else pair*np.sign(alignment)
                sample.update(blade_velocity_mm_s=vb.tolist(), relative_body_minus_tool_mm_s=relative.tolist(),
                              material_tangent=edge.tolist(), relative_material_tangential_mm_s=float(relative @ edge),
                              pair_tangent=None if signed_pair is None else signed_pair.tolist(),
                              relative_tangential_mm_s=None if signed_pair is None else float(relative @ signed_pair))
        return dict(branches=branches, selected_material=sample, selected_branch_transition=transition, dhat_mm=self.dhat)

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

    def relative_motion(self, positions, previous, observation, dt):
        """Same material identity in both states; exact finite rigid transport.

        The transported increment compares the current material position to
        its previous position carried by the blade's rigid motion. This removes
        the omega*gap artifact (and finite-rotation chord error), but supplies
        neither contact traction nor proof of contact throughout the interval.
        """
        result = dict(status=observation['status'], valid=False,
                      material_velocity_mm_s=None, blade_velocity_mm_s=None,
                      tangent=None, local_edge_stretch=None, raw_mm_s=None,
                      transported_mm_s=None, transport_correction_mm_s=None)
        if observation['status'] != 'unique':
            return result
        x, old = np.asarray(positions), np.asarray(previous)
        if x.shape != self.reference.shape or old.shape != x.shape or not np.isfinite(old).all():
            raise ValueError('Incomplete kinematic parent')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive physical dt required')
        edge = np.asarray(observation['edge_nodes'], dtype=int)
        a = float(observation['edge_alpha'])
        vector = x[edge[1]]-x[edge[0]]
        tangent = vector/np.linalg.norm(vector)
        # A corner can have a unique material identity and two different tangents.
        endpoint = edge[0] if a <= 1e-10 else edge[1] if a >= 1-1e-10 else None
        if endpoint is not None:
            for other in self.edges[np.any(self.edges == endpoint, axis=1)]:
                direction = x[other[1]]-x[other[0]]
                direction /= np.linalg.norm(direction)
                if np.linalg.norm(direction-tangent) > 1e-6:
                    result['status'] = 'nonunique_tangent'
                    return result
        p = (1-a)*x[edge[0]]+a*x[edge[1]]
        pold = (1-a)*old[edge[0]]+a*old[edge[1]]
        blade, prior_blade = x[self.body_nodes:], old[self.body_nodes:]
        c, cold = blade.mean(axis=0), prior_blade.mean(axis=0)
        u, _, vh = np.linalg.svd((prior_blade-cold).T@(blade-c))
        rotation = u@vh
        if np.linalg.det(rotation) <= 0 or not np.allclose(
                (prior_blade-cold)@rotation+c, blade, atol=self.tolerance, rtol=0):
            raise ValueError('Blade motion is not rigid')
        vm = (p-pold)/dt
        vb = (x[self.tip_node]-old[self.tip_node])/dt
        raw = float((vb-vm)@tangent)
        transported = float(((pold-cold)@rotation+c-p)@tangent/dt)
        result.update(valid=True, status='valid_proximity', edge_nodes=edge.tolist(), edge_alpha=a,
                      material_velocity_mm_s=vm.tolist(), blade_velocity_mm_s=vb.tolist(),
                      tangent=tangent.tolist(), local_edge_stretch=float(np.linalg.norm(vector)/
                          np.linalg.norm(self.reference[edge[1]]-self.reference[edge[0]])),
                      raw_mm_s=raw, transported_mm_s=transported,
                      transport_correction_mm_s=transported-raw)
        return result

    def fixed_material_definition(self):
        """A priori top ROIs and gauges, fixed in reference S, in millimetres."""
        offsets = self.s-self.origin
        top = np.unique(self.edges)
        ranges = dict(left_far=(-3., -1.), left_near=(-1., 0.),
                      right_near=(0., 1.), right_far=(1., 3.))
        rois = {name: top[(offsets[top] >= lo) & (offsets[top] < hi)].tolist()
                for name, (lo, hi) in ranges.items()}
        ordered = top[np.argsort(self.s[top])]
        gauges = {}
        for name, limits in dict(near=(-.5, .5), left=(-2., -1.), right=(1., 2.)).items():
            # Save actual endpoints: nearest reference nodes, never refit later.
            nodes = [int(ordered[np.argmin(abs(offsets[ordered]-v))]) for v in limits]
            if nodes[0] != nodes[1]:
                gauges[name] = nodes
        # Integrate the linear shape functions on each clipped reference edge.
        # The legacy node mean remains available for replay of older reports.
        weights = {}
        ss = self.s[self.edges] - self.origin
        for name, (lo, hi) in ranges.items():
            left = np.maximum(ss[:, 0], lo)
            right = np.minimum(ss[:, 1], hi)
            length = np.maximum(0., right - left)
            midpoint_alpha = ((left + right) / 2 - ss[:, 0]) / (ss[:, 1] - ss[:, 0])
            nodal = np.zeros(self.body_nodes)
            np.add.at(nodal, self.edges[:, 0], length * (1 - midpoint_alpha))
            np.add.at(nodal, self.edges[:, 1], length * midpoint_alpha)
            ids = np.flatnonzero(nodal > 0)
            weights[name] = dict(nodes=ids.tolist(), length_weights_mm=nodal[ids].tolist(),
                                 covered_reference_mm=float(length.sum()))
        markers={}
        for offset in (-2.,-1.,-.5,-.25,-.1,-.05,0.,.05,.1,.25,.5,1.,2.):
            target=self.origin+offset
            if self.s[ordered[0]]<=target<=self.s[ordered[-1]]:
                index=int(np.clip(np.searchsorted(self.s[ordered],target),1,len(ordered)-1))
                i,j=ordered[index-1:index+1]
                markers[str(offset)]=dict(nodes=[int(i),int(j)],weight=float((target-self.s[i])/(self.s[j]-self.s[i])))
        return dict(fixed_offset_markers=markers,reference='undeformed material top', axes=dict(s=self.es.tolist(), n=self.en.tolist()),
                    roi_offset_ranges_mm=ranges, roi_nodes=rois, gauge_nodes=gauges,
                    roi_reference_integrals=weights,
                    gauge_measure='Euclidean distance between fixed material endpoints divided by initial distance; not surface arclength',
                    roi_statistic='unweighted mean over fixed reference node IDs; mesh dependent')

    def fixed_material_motion(self, positions, definition):
        x = np.asarray(positions)
        displacement = x[:self.body_nodes]-self.reference[:self.body_nodes]
        rois = {}
        for name, nodes in definition['roi_nodes'].items():
            u = displacement[nodes].mean(axis=0) if nodes else None
            rois[name] = dict(count=len(nodes), u_s_mm=float(u@self.es) if u is not None else None,
                              u_n_mm=float(u@self.en) if u is not None else None)
        gauges = {name: float(np.linalg.norm(x[j]-x[i])/
                  np.linalg.norm(self.reference[j]-self.reference[i]))
                  for name, (i, j) in definition['gauge_nodes'].items()}
        points={}
        positions={}
        for key, marker in definition.get('fixed_offset_markers',{}).items():
            i,j=marker['nodes'];w=marker['weight']
            u=(1-w)*displacement[i]+w*displacement[j]
            positions[key]=(1-w)*x[i]+w*x[j]
            points[key]=dict(u_s_mm=float(u@self.es),u_n_mm=float(u@self.en),
                lambda_s=float(np.linalg.norm(x[j]-x[i])/(self.s[j]-self.s[i])))
        stretches={name:float(np.linalg.norm(positions[str(b)]-positions[str(a)])/(b-a))
            for name,a,b in [('near',-.5,.5),('left',-2.,-1.),('right',1.,2.)]
            if str(a) in positions and str(b) in positions}
        integrals = {}
        for name, rule in definition.get('roi_reference_integrals', {}).items():
            lo, hi = definition['roi_offset_ranges_mm'][name]
            if abs(rule['covered_reference_mm'] - (hi - lo)) > self.tolerance:
                integrals[name] = None
                continue
            mean = np.asarray(rule['length_weights_mm']) @ displacement[rule['nodes']] / (hi - lo)
            integrals[name] = dict(u_s_mm=float(mean @ self.es), u_n_mm=float(mean @ self.en))
        return dict(rois=rois,gauge_stretch=gauges,fixed_offset_points=points,fixed_offset_stretch=stretches,
                    reference_arclength_mean=integrals)

    def projection_continuity(self, previous, positions, prior_contact, contact):
        """Resolve fast crossings of adjacent reference edges on a geometric chord.

        Interior samples are NOT equilibrium states or new physical observations.
        They support a sampled projection branch only, not continuous pressure
        contact throughout the physical interval. Genuine gaps/ambiguities stay
        invalid; an unresolved jump never receives a cumulative contribution.
        """
        x, old = np.asarray(positions), np.asarray(previous)
        last = None
        for intervals in (2, 4, 8, 16, 32):
            samples = [dict(fraction=0., **prior_contact)]
            try:
                for i in range(1, intervals):
                    f = i/intervals
                    depth = (1-f)*prior_contact['depth_mm']+f*contact['depth_mm']
                    samples.append(dict(fraction=f, **self.observe((1-f)*old+f*x, depth)))
            except ValueError as exc:
                return dict(continuous=False, status='invalid_chord_geometry', error=str(exc),
                            intermediate_geometry_only=True)
            samples.append(dict(fraction=1., **contact))
            last = [dict(fraction=s['fraction'], status=s['status'], edge_nodes=s['edge_nodes'],
                         reference_s_mm=s['reference_s_mm'], gap_mm=s['gap_mm']) for s in samples]
            if any(s['status'] != 'unique' for s in samples):
                return dict(continuous=False, status='chord_gap_or_ambiguous_projection',
                            intervals=intervals, samples=last, intermediate_geometry_only=True)
            if all(a['edge_nodes'] == b['edge_nodes'] or set(a['edge_nodes']).intersection(b['edge_nodes'])
                   for a, b in zip(samples[:-1], samples[1:])):
                return dict(continuous=True, status='sampled_continuous_projection',
                            intervals=intervals, samples=last, intermediate_geometry_only=True)
        return dict(continuous=False, status='unresolved_projection_change', intervals=32,
                    samples=last, intermediate_geometry_only=True)
