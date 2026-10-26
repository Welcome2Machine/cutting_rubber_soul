"""Infer original-surface addresses at the registered blade tip from saved IDs.

The current marker is always the image-registered tip. A separate reference
marker denotes a material address; it is never a distant current point joined
to the tip. Local block maps provide conditional inference when surface texture
is absent. They cannot establish that contact belongs to the original free
surface rather than a new crack face, nor turn coherent tracking drift into
ground truth. All uncertainty envelopes below are sensitivity bounds, not
statistical confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_material_mapping import _hull_distance, LocalFitGeometry, fit_local


STATE_LABELS = {0: "unknown", 1: "conditional_inference", 2: "direct_supported"}


def extend_initial_surface_reference(surface_reference, initial_material_mask):
    """Extend the measured initial sloping interface, preserving saved stations.

    Only air-to-material transitions present in the initial semantic image are
    added. Nothing is extrapolated beyond the visible boundary or through an
    excluded image strip. The saved coordinate origin and all its old samples
    remain unchanged.
    """
    from .original_surface_cohort import original_surface_stations
    old = {key: np.asarray(value).copy() for key, value in surface_reference.items()}
    boundary, addresses = old["boundary_xy"], old["surface_s_px"]
    tangent = old["tangent"].astype(float); tangent /= np.linalg.norm(tangent)
    center = np.mean(boundary - addresses[:, None] * tangent, axis=0)
    contact_s = float(old["contact_s_px"])
    pitch = float(np.median(np.diff(addresses)))
    contact_x = center[0] + contact_s * tangent[0]
    expanded = original_surface_stations(np.asarray(initial_material_mask), center, tangent,
        contact_x, half_span=1200., pitch=pitch)
    use = (expanded["surface_s_px"] < addresses[0] - .5 * pitch) | (expanded["surface_s_px"] > addresses[-1] + .5 * pitch)
    all_s = np.r_[addresses, expanded["surface_s_px"][use]]
    all_boundary = np.vstack([boundary, expanded["boundary_xy"][use]])
    order = np.argsort(all_s)
    old["boundary_xy"] = all_boundary[order]
    old["surface_s_px"] = all_s[order]
    old["saved_station_indices"] = np.flatnonzero(order < len(addresses))
    old["initial_mask_extension"] = np.array(True)
    return old


@dataclass(frozen=True)
class TipAddressConfig:
    station_half_windows: tuple[int, ...] = (2, 3)
    maximum_depth_layer: int = 8
    maximum_candidate_stations: int = 3
    current_radius_px: float = 64.
    minimum_points: int = 8
    maximum_fit_residual_px: float = 3.
    maximum_heldout_residual_px: float = 3.
    maximum_condition: float = 20.
    maximum_extrapolation_px: float = 12.
    maximum_reference_normal_px: float = 12.
    maximum_surface_gap_px: float = 12.
    direct_surface_gap_px: float = 4.
    direct_reference_normal_px: float = 4.
    maximum_candidate_spread_px: float = 12.
    tip_location_sensitivity_px: float = 2.


def _unknown(tip, reason, **fields):
    result = dict(state=0, label=STATE_LABELS[0], reason=reason, route="unavailable",
        current_marker_xy=np.asarray(tip, float).copy(), reference_marker_xy=np.full(2, np.nan),
        s_px=np.nan, s_um=np.nan, reference_s_px=np.nan, low_px=np.nan, high_px=np.nan,
        low_um=np.nan, high_um=np.nan, surface_normal_offset_px=np.nan,
        surface_gap_px=np.nan, fit_residual_px=np.nan, heldout_residual_px=np.nan,
        current_extrapolation_px=np.nan, reference_extrapolation_px=np.nan,
        tangent_current=np.full(2, np.nan), surface_turn_deg=np.nan,
        support_ids=np.empty(0, np.int64), candidate_count=0, branch_label=0,
        mapped_surface_proxy_xy=np.full(2, np.nan), F=np.full((2, 2), np.nan),
        map_origin=np.full(2, np.nan), map_xy=np.full(2, np.nan),
        raw_inverse_material_xy=np.full(2, np.nan), raw_reference_s_px=np.nan,
        raw_reference_s_um=np.nan, raw_surface_normal_offset_px=np.nan,
        raw_low_um=np.nan, raw_high_um=np.nan, raw_candidate_reason="unavailable",
        address_inside_initial_surface=False, original_surface_identity_proven=False)
    result.update(fields)
    return result


class TipSurfaceAddressEstimator:
    """Small bank of fixed original material blocks, evaluated independently.

    ``supported`` must mean current image support, not a forecast or a held
    coordinate. ``blocked`` excludes blade-occupied/crossing IDs. Optional
    ``branch_labels`` partitions material (0 excludes); blocks never mix its
    nonzero branches. Without a branch partition, results remain conditional.
    The caller retains responsibility for correctly identifying material sides.
    """
    def __init__(self, seed_points, point_ids, depth_layer, surface_station,
                 surface_reference, um_per_px, config=None, *, initial_material_mask=None):
        self.reference = np.asarray(seed_points, float).copy()
        self.ids = np.asarray(point_ids, np.int64).copy()
        self.layers = np.asarray(depth_layer, int).copy()
        self.stations = np.asarray(surface_station, int).copy()
        n = len(self.reference)
        if (self.reference.shape != (n, 2) or self.ids.shape != (n,)
                or self.layers.shape != (n,) or self.stations.shape != (n,)
                or not np.isfinite(self.reference).all() or len(np.unique(self.ids)) != n):
            raise ValueError("Invalid fixed material identities")
        if not np.isfinite(um_per_px) or um_per_px <= 0:
            raise ValueError("A positive physical pixel calibration is required")
        self.scale = float(um_per_px)
        self.config = config or TipAddressConfig()
        self.initial_mask_extended = initial_material_mask is not None
        saved_surface = surface_reference
        if self.initial_mask_extended:
            surface_reference = extend_initial_surface_reference(surface_reference, initial_material_mask)
            mapped = np.asarray(surface_reference["saved_station_indices"], int)
            old_valid = (self.stations >= 0) & (self.stations < len(mapped))
            self.stations[old_valid] = mapped[self.stations[old_valid]]
        self.boundary = np.asarray(surface_reference["boundary_xy"], float).copy()
        self.addresses = np.asarray(surface_reference["surface_s_px"], float).copy()
        if (self.boundary.shape != (len(self.addresses), 2) or len(self.addresses) < 4
                or not np.isfinite(self.boundary).all() or not np.isfinite(self.addresses).all()
                or np.any(np.diff(self.addresses) <= 0)):
            raise ValueError("Initial surface stations must be finite and strictly ordered")
        self.tangent = np.asarray(surface_reference["tangent"], float).copy()
        self.tangent /= np.linalg.norm(self.tangent)
        normal = np.asarray(surface_reference["inward_normal"], float)
        self.normal = normal / np.linalg.norm(normal)
        self.axes = np.column_stack([self.tangent, self.normal])
        if not np.isfinite(self.axes).all() or abs(np.linalg.det(self.axes)) < .95:
            raise ValueError("Initial material surface axes must be orthonormal")
        self.contact_s = float(surface_reference["contact_s_px"])
        self.pitch = float(np.median(np.diff(self.addresses)))
        self.surface_texture = self.layers == 0
        self.original_dense = self.layers >= 0
        self.core = ((self.layers >= 0) & (self.layers <= self.config.maximum_depth_layer)
                     & (self.stations >= 0) & (self.stations < len(self.addresses)))
        self.outer = np.zeros(n, bool)
        if self.initial_mask_extended:
            # Sparse outer points retain their actual original IDs. Membership
            # in an interior material block does not make them surface nodes.
            for index in np.flatnonzero(self.layers < 0):
                address = self._address(self.reference[index])
                if not np.isfinite(address):
                    continue
                normal_depth = float((self.reference[index] - self.surface_point(address)) @ self.normal)
                if 0 <= normal_depth <= 90:
                    self.outer[index] = True
                    self.stations[index] = int(np.argmin(abs(self.addresses - address)))
                    self.layers[index] = int(np.floor(normal_depth / 6))
            self.core |= self.outer
        dense_min = float(np.min(saved_surface["surface_s_px"]))
        dense_max = float(np.max(saved_surface["surface_s_px"]))
        self.windows = {station: self.config.station_half_windows if dense_min <= s <= dense_max else (6, 8)
                        for station, s in enumerate(self.addresses)}
        self.bank = {}
        for station in range(len(self.addresses)):
            for half in self.windows[station]:
                indices = np.flatnonzero(self.core & (abs(self.stations - station) <= half))
                if len(indices) >= self.config.minimum_points:
                    origin = self.boundary[station]
                    self.bank[station, half] = (indices, LocalFitGeometry(self.reference[indices], origin, self.axes))

    @staticmethod
    def _segment_clear(a, b, tip, blade_mask):
        if blade_mask is None:
            return True
        mask = np.asarray(blade_mask)
        blade = mask if mask.dtype == bool else mask == 2
        samples = a + np.linspace(0., 1., max(2, int(np.ceil(np.linalg.norm(b - a))) + 1))[:, None] * (b - a)
        pixels = np.rint(samples).astype(int)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0])
        if not inside.all():
            return False
        # A 2-px apex-contact tolerance is not permission to span a blade face.
        forbidden = blade[pixels[:, 1], pixels[:, 0]] & (np.linalg.norm(samples - tip, axis=1) > 2.)
        return not forbidden.any()

    def _surface_curve_candidates(self, current, valid, tip, branches, branch_verified, blade_mask):
        """Track original normal columns; only extrapolate their known surface offset."""
        columns = {}
        for station in np.unique(self.stations[valid & self.original_dense & (self.layers <= 3)]):
            for branch in np.unique(branches[valid & (self.stations == station)]):
                ids = np.flatnonzero(valid & self.original_dense & (self.stations == station)
                                     & (self.layers <= 3) & (branches == branch))
                if len(ids) < 2:
                    continue
                depth = (self.reference[ids] - self.boundary[station]) @ self.normal
                if depth.min() > 8. or np.ptp(depth) < 5.:
                    continue
                x = np.column_stack([np.ones(len(ids)), depth])
                beta = np.linalg.lstsq(x, current[ids], rcond=None)[0]
                residual = float(np.sqrt(np.mean(np.sum((x @ beta - current[ids]) ** 2, axis=1))))
                if residual > 2. or not .1 < np.linalg.norm(beta[1]) < 8.:
                    continue
                columns[station, branch] = dict(xy=beta[0], normal=beta[1], residual=residual,
                    direct=bool(np.any(self.surface_texture[ids])), ids=ids, depth_min=float(depth.min()))
        candidates = []
        for (station, branch), first in columns.items():
            second = columns.get((station + 1, branch))
            if second is None:
                continue
            ds = self.addresses[station + 1] - self.addresses[station]
            if ds > 1.5 * self.pitch:
                continue
            a, b = first["xy"], second["xy"]
            delta = b - a; length = float(np.linalg.norm(delta))
            if not .1 * ds < length < 8 * ds:
                continue
            alpha = float((tip - a) @ delta / (length * length))
            if not 0 <= alpha <= 1:
                continue
            proxy = a + alpha * delta
            gap = float(np.linalg.norm(proxy - tip))
            if gap > self.config.direct_surface_gap_px or not self._segment_clear(a, b, tip, blade_mask):
                continue
            address = float(self.addresses[station] + alpha * ds)
            foot = self.surface_point(address)
            normal = (1 - alpha) * first["normal"] + alpha * second["normal"]
            F = np.column_stack([delta / ds, normal]) @ self.axes.T
            sv = np.linalg.svd(F, compute_uv=False)
            if np.linalg.det(F) <= 0 or sv[-1] < .08 or sv[0] / sv[-1] > self.config.maximum_condition:
                continue
            inverse = np.linalg.inv(F)
            initial = foot + inverse @ (tip - proxy)
            ndepth = float((initial - foot) @ self.normal)
            if abs(ndepth) > self.config.maximum_reference_normal_px:
                continue
            error = max(first["residual"], second["residual"])
            current_extrapolation = max(first["depth_min"] * np.linalg.norm(first["normal"]),
                                        second["depth_min"] * np.linalg.norm(second["normal"]))
            if current_extrapolation > self.config.maximum_extrapolation_px:
                continue
            radius = max(3., float(np.linalg.norm(self.tangent @ inverse)) * (self.config.tip_location_sensitivity_px + error + gap))
            ids = np.r_[first["ids"], second["ids"]]
            direct = bool(first["direct"] and second["direct"] and branch_verified and
                          blade_mask is not None and abs(ndepth) <= self.config.direct_reference_normal_px)
            tangent = delta / length
            candidates.append(dict(address_px=address, initial_xy=initial, reference_xy=foot,
                normal_depth_px=ndepth, surface_proxy_xy=proxy, surface_gap_px=gap, radius_px=radius,
                fit_residual_px=error, heldout_residual_px=np.nan,
                current_extrapolation_px=current_extrapolation,
                reference_extrapolation_px=max(first["depth_min"], second["depth_min"]),
                nearest_material_px=float(np.linalg.norm(current[ids] - tip, axis=1).min()), count=len(ids),
                support_ids=self.ids[ids], branch_label=int(branch), direct=direct,
                tangent=tangent, surface_turn_deg=float(np.degrees(np.arctan2(
                    self.tangent[0] * tangent[1] - self.tangent[1] * tangent[0], self.tangent @ tangent))),
                F=F, map_xy=proxy, map_origin=foot, route="original_surface_normal_columns"))
        return candidates

    def surface_point(self, address):
        if not self.addresses[0] <= address <= self.addresses[-1]:
            return np.full(2, np.nan)
        return np.array([np.interp(address, self.addresses, self.boundary[:, d]) for d in range(2)])

    def _address(self, point):
        along = self.boundary @ self.tangent
        u = float(point @ self.tangent)
        if not along[0] <= u <= along[-1]:
            return np.nan
        return float(np.interp(u, along, self.addresses))

    def _candidate(self, station, half, current, valid, distance, tip, branch, branch_verified, blade_mask=None):
        cfg = self.config
        bank = self.bank.get((station, half))
        if bank is None:
            return None, "insufficient_fixed_patch"
        indices, prepared = bank
        chosen = valid[indices] & (distance[indices] <= cfg.current_radius_px)
        use_ids = indices[chosen]
        if (len(use_ids) < cfg.minimum_points or len(np.unique(self.stations[use_ids])) < 2
                or len(np.unique(self.layers[use_ids])) < 2):
            return None, "insufficient_two_dimensional_patch"
        origin = self.boundary[station]
        weight = np.exp(-.5 * (distance[indices] / (cfg.current_radius_px * .55)) ** 2)
        weight[~chosen] = 0
        fit = fit_local(self.reference[indices], current[indices], weight, origin, self.axes,
                        check_cv=True, prepared=prepared)
        if not fit["valid"] or not np.isfinite(fit["cv"]):
            return None, "noninvertible_patch"
        if np.linalg.cond(fit["F"]) > cfg.maximum_condition:
            return None, "patch_residual_or_condition"
        inverse = np.linalg.inv(fit["F"])
        initial = origin + inverse @ (tip - fit["xy"])
        address = self._address(initial)
        if not np.isfinite(address):
            return dict(raw_only=True, raw_inverse_material_xy=initial,
                        nearest_material_px=float(distance[use_ids].min()), heldout_residual_px=float(fit["cv"])), "outside_tracked_original_surface"
        foot = self.surface_point(address)
        normal_depth = float((initial - foot) @ self.normal)
        surface_proxy = fit["xy"] + fit["F"] @ (foot - origin)
        gap = float(np.linalg.norm(surface_proxy - tip))
        sensitivity = max(3., float(np.linalg.norm(self.tangent @ inverse)) *
                          (cfg.tip_location_sensitivity_px + fit["cv"]))
        raw = dict(raw_inverse_material_xy=initial, raw_reference_s_px=address,
            raw_reference_s_um=(address - self.contact_s) * self.scale,
            raw_surface_normal_offset_px=normal_depth,
            raw_low_um=(address - self.contact_s - sensitivity) * self.scale,
            raw_high_um=(address - self.contact_s + sensitivity) * self.scale,
            raw_candidate_reason="ok", raw_only=True,
            nearest_material_px=float(distance[use_ids].min()), heldout_residual_px=float(fit["cv"]))
        if fit["cv"] > cfg.maximum_heldout_residual_px or fit["rms"] > cfg.maximum_fit_residual_px:
            return raw, "high_residual_material_inverse_only"
        if (abs(normal_depth) > cfg.maximum_reference_normal_px or gap > cfg.maximum_surface_gap_px):
            return raw, "original_surface_mismatch_possible_new_crack_face"
        ext_ref = max(_hull_distance(self.reference[use_ids], initial), _hull_distance(self.reference[use_ids], foot))
        ext_cur = max(_hull_distance(current[use_ids], tip), _hull_distance(current[use_ids], surface_proxy))
        if max(ext_ref, ext_cur) > cfg.maximum_extrapolation_px:
            return raw, "extrapolation_exceeds_allowance"
        supported_stations = self.stations[use_ids]
        covered = self.addresses[supported_stations]
        bracketed = bool(covered.min() <= address <= covered.max())
        if not bracketed:
            return raw, "no_material_station_bracket"
        ft = fit["F"] @ self.tangent
        stretch = float(np.linalg.norm(ft))
        tangent = ft / stretch
        # Surface texture lives 1.5 px inside the mathematical boundary.
        # Extrapolate that known small offset with this same material map;
        # do not use a distant current point as the contact marker.
        first = use_ids[self.surface_texture[use_ids]]
        direct = False
        direct_gap = np.inf
        for a, b in zip(first[:-1], first[1:]):
            if self.stations[b] != self.stations[a] + 1:
                continue
            pa = current[a] - fit["F"] @ (self.reference[a] - self.boundary[self.stations[a]])
            pb = current[b] - fit["F"] @ (self.reference[b] - self.boundary[self.stations[b]])
            delta = pb - pa
            if np.dot(delta, delta) < 1e-8:
                continue
            alpha = float(np.dot(tip - pa, delta) / np.dot(delta, delta))
            if 0 <= alpha <= 1:
                segment_gap = float(np.linalg.norm(tip - (pa + alpha * delta)))
                direct_gap = min(direct_gap, segment_gap)
                direct |= (blade_mask is not None and self._segment_clear(pa, pb, tip, blade_mask)
                           and segment_gap <= cfg.direct_surface_gap_px and max(distance[a], distance[b]) <= 24
                           and abs(normal_depth) <= cfg.direct_reference_normal_px)
        direct = bool(direct and branch_verified)
        return dict(address_px=address, initial_xy=initial, reference_xy=foot,
            normal_depth_px=normal_depth, surface_proxy_xy=surface_proxy, surface_gap_px=gap,
            radius_px=sensitivity, fit_residual_px=float(fit["rms"]), heldout_residual_px=float(fit["cv"]),
            current_extrapolation_px=float(ext_cur), reference_extrapolation_px=float(ext_ref),
            nearest_material_px=float(distance[use_ids].min()), count=len(use_ids),
            support_ids=self.ids[use_ids], station=station, half_window=half, branch_label=int(branch),
            direct=direct, direct_surface_gap_px=direct_gap, tangent=tangent,
            surface_turn_deg=float(np.degrees(np.arctan2(self.tangent[0] * tangent[1] - self.tangent[1] * tangent[0], self.tangent @ tangent))),
            F=fit["F"], map_xy=fit["xy"], map_origin=origin), "ok"

    def estimate_frame(self, current_points, supported, tip_xy, *, geometry_supported=True,
                       blocked=None, branch_labels=None, blade_mask=None):
        current = np.asarray(current_points, float)
        tip = np.asarray(tip_xy, float)
        valid = np.asarray(supported, bool).copy()
        if current.shape != self.reference.shape or valid.shape != (len(self.reference),) or tip.shape != (2,):
            raise ValueError("Current coordinates, support, and tip shape differ from fixed query identities")
        if not geometry_supported or not np.isfinite(tip).all():
            return _unknown(tip, "unsupported_blade_tip")
        if blocked is not None:
            excluded = np.asarray(blocked, bool)
            if excluded.shape != valid.shape:
                raise ValueError("Blocked material mask must have shape (N,)")
            valid &= ~excluded
        valid &= self.core & np.isfinite(current).all(axis=1)
        distance = np.linalg.norm(current - tip, axis=1)
        valid &= distance <= self.config.current_radius_px
        if branch_labels is None:
            branches = np.ones(len(valid), int)
        else:
            branches = np.asarray(branch_labels, int)
            if branches.shape != valid.shape:
                raise ValueError("Branch labels must have shape (N,)")
            valid &= branches != 0
        curve = self._surface_curve_candidates(current, valid, tip, branches, branch_labels is not None, blade_mask)
        if curve:
            best = min(curve, key=lambda c: c["surface_gap_px"] + c["fit_residual_px"])
            comparable = [c for c in curve if c["surface_gap_px"] <= best["surface_gap_px"] + 2.]
            if np.ptp([c["address_px"] for c in comparable]) > self.config.maximum_candidate_spread_px:
                return _unknown(tip, "ambiguous_original_surface_curve", candidate_count=len(comparable))
            low = min(c["address_px"] - c["radius_px"] for c in comparable)
            high = max(c["address_px"] + c["radius_px"] for c in comparable)
            return self._result(tip, comparable, best, best["address_px"], low, high, 2 if best["direct"] else 1)
        candidates, failures, raw_candidates = [], [], []
        for branch in np.unique(branches[valid]):
            use = valid & (branches == branch)
            order = np.flatnonzero(use)
            order = order[np.argsort(distance[order], kind="stable")]
            stations = list(dict.fromkeys(self.stations[order].tolist()))[:self.config.maximum_candidate_stations]
            for station in stations:
                for half in self.windows[station]:
                    candidate, reason = self._candidate(station, half, current, use, distance, tip,
                                                        branch, branch_labels is not None, blade_mask)
                    if candidate is not None and not candidate.get("raw_only", False):
                        candidates.append(candidate)
                    else:
                        failures.append(reason)
                        if candidate is not None:
                            candidate["raw_candidate_reason"] = reason
                            raw_candidates.append(candidate)
        if not candidates:
            reason = max(dict.fromkeys(failures), key=failures.count) if failures else "no_nearby_supported_material"
            raw = min(raw_candidates, key=lambda c: c["heldout_residual_px"] + .05 * c["nearest_material_px"]) if raw_candidates else {}
            fields = {k: v for k, v in raw.items() if k.startswith("raw_")}
            comparable = [c for c in raw_candidates if c["nearest_material_px"] <= raw.get("nearest_material_px", 0) + 12
                          and c["heldout_residual_px"] <= max(8., raw.get("heldout_residual_px", 0) + 3)
                          and np.isfinite(c.get("raw_low_um", np.nan))]
            if comparable:
                fields["raw_low_um"] = min(c["raw_low_um"] for c in comparable)
                fields["raw_high_um"] = max(c["raw_high_um"] for c in comparable)
            return _unknown(tip, reason, candidate_count=len(raw_candidates), **fields)
        # Two scales/station blocks must independently support a surface
        # address. Never average incompatible branches into a fictitious point.
        if len(candidates) < 2:
            return _unknown(tip, "insufficient_independent_patch_support")
        values = np.array([c["address_px"] for c in candidates])
        if np.ptp(values) > self.config.maximum_candidate_spread_px:
            return _unknown(tip, "ambiguous_material_addresses", candidate_count=len(candidates))
        selected = min(candidates, key=lambda c: c["heldout_residual_px"] + .05 * c["nearest_material_px"] + .1 * c["current_extrapolation_px"])
        compatible = [c for c in candidates if c["branch_label"] == selected["branch_label"]]
        if len(compatible) < 2:
            return _unknown(tip, "insufficient_same_branch_patch_support")
        address = float(np.median([c["address_px"] for c in compatible]))
        low = min(c["address_px"] - c["radius_px"] for c in candidates)
        high = max(c["address_px"] + c["radius_px"] for c in candidates)
        state = 2 if sum(c["direct"] for c in compatible) >= 2 else 1
        return self._result(tip, candidates, selected, address, low, high, state)

    def _result(self, tip, candidates, selected, address, low, high, state):
        result = _unknown(tip, "ok")
        result.update(state=state, label=STATE_LABELS[state],
            s_px=address - self.contact_s, s_um=(address - self.contact_s) * self.scale,
            reference_s_px=address, low_px=low - self.contact_s, high_px=high - self.contact_s,
            low_um=(low - self.contact_s) * self.scale, high_um=(high - self.contact_s) * self.scale,
            reference_marker_xy=self.surface_point(address), surface_normal_offset_px=selected["normal_depth_px"],
            surface_gap_px=selected["surface_gap_px"], fit_residual_px=selected["fit_residual_px"],
            heldout_residual_px=selected["heldout_residual_px"],
            current_extrapolation_px=selected["current_extrapolation_px"],
            reference_extrapolation_px=selected["reference_extrapolation_px"],
            tangent_current=selected["tangent"], surface_turn_deg=selected["surface_turn_deg"],
            support_ids=selected["support_ids"], candidate_count=len(candidates), branch_label=selected["branch_label"],
            mapped_surface_proxy_xy=selected["surface_proxy_xy"], F=selected["F"],
            map_origin=selected["map_origin"], map_xy=selected["map_xy"],
            address_inside_initial_surface=True,
            raw_inverse_material_xy=selected["initial_xy"], raw_reference_s_px=selected["address_px"],
            raw_reference_s_um=(selected["address_px"] - self.contact_s) * self.scale,
            raw_surface_normal_offset_px=selected["normal_depth_px"],
            raw_low_um=(low - self.contact_s) * self.scale, raw_high_um=(high - self.contact_s) * self.scale,
            raw_candidate_reason="surface_compatible", route=selected.get("route", "fixed_block_inverse"))
        return result


def analyze_tip_surface_series(estimator, positions, supported, tips_px, times_s, *,
                               geometry_supported=None, blocked=None, branch_labels=None,
                               contact_frame=0, blade_masks=None):
    """Per-frame estimates and two distinct, gap-safe motion measures.

    ``address_rate_um_s`` is migration through the initial surface labels.
    ``relative_tangent_velocity_um_s`` evaluates the same midpoint material
    address in adjacent local maps, subtracts blade-tip translation, and
    projects onto the current surface tangent. It is conditional kinematics;
    frictional slip additionally needs actual persistent contact. Blade angular
    velocity is not available here. No gaps or branch changes are integrated.
    """
    positions, supported, tips = np.asarray(positions), np.asarray(supported, bool), np.asarray(tips_px, float)
    times = np.asarray(times_s, float)
    count = len(positions)
    if (supported.shape != positions.shape[:2] or tips.shape != (count, 2)
            or times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Inconsistent sequence arrays or nonincreasing physical sample times")
    geometry = np.isfinite(tips).all(axis=1) if geometry_supported is None else np.asarray(geometry_supported, bool)
    if geometry.shape != (count,):
        raise ValueError("Geometry support must have shape (T,)")
    records = []
    for frame in range(count):
        records.append(estimator.estimate_frame(positions[frame], supported[frame], tips[frame],
            geometry_supported=geometry[frame], blocked=None if blocked is None else blocked[frame],
            branch_labels=None if branch_labels is None else branch_labels[frame],
            blade_mask=None if blade_masks is None else blade_masks[frame]))
    keys = ("state", "label", "reason", "route", "s_px", "s_um", "reference_s_px", "low_px", "high_px", "low_um", "high_um",
            "current_marker_xy", "reference_marker_xy", "surface_normal_offset_px", "surface_gap_px",
            "fit_residual_px", "heldout_residual_px", "current_extrapolation_px", "reference_extrapolation_px",
            "tangent_current", "surface_turn_deg", "candidate_count", "branch_label", "F", "map_origin", "map_xy",
            "raw_inverse_material_xy", "raw_reference_s_px", "raw_reference_s_um", "raw_surface_normal_offset_px",
            "raw_low_um", "raw_high_um", "raw_candidate_reason")
    result = {key: np.asarray([r[key] for r in records]) for key in keys}
    result["support_ids"] = [r["support_ids"] for r in records]
    result["times_s"] = times.copy()
    result["after_first_contact_frame"] = np.arange(count) >= contact_frame
    result["address_rate_um_s"] = np.full(count, np.nan)
    result["relative_tangent_step_um"] = np.full(count, np.nan)
    result["relative_tangent_velocity_um_s"] = np.full(count, np.nan)
    result["motion_step_state"] = np.zeros(count, np.uint8)
    result["migration_from_initial_contact_um"] = np.full(count, np.nan)
    if 0 <= contact_frame < count and result["state"][contact_frame] > 0:
        result["migration_from_initial_contact_um"] = result["s_um"] - result["s_um"][contact_frame]
    for f in range(1, count):
        before, after = records[f - 1], records[f]
        if (min(before["state"], after["state"]) == 0 or before["branch_label"] != after["branch_label"]):
            continue
        # Shared observed material IDs are needed even when local fit windows
        # change. A newly selected different block must not create a slip step.
        minimum_common = 4 if before.get("route") == after.get("route") == "original_surface_normal_columns" else estimator.config.minimum_points
        if len(np.intersect1d(before["support_ids"], after["support_ids"])) < minimum_common:
            continue
        midpoint = estimator.surface_point(.5 * (before["reference_s_px"] + after["reference_s_px"]))
        q0 = before["map_xy"] + before["F"] @ (midpoint - before["map_origin"])
        q1 = after["map_xy"] + after["F"] @ (midpoint - after["map_origin"])
        tangent = before["tangent_current"] + after["tangent_current"]
        norm = np.linalg.norm(tangent)
        if not np.isfinite(norm) or norm < .1:
            continue
        step = float((q1 - q0 - (tips[f] - tips[f - 1])) @ (tangent / norm) * estimator.scale)
        dt = times[f] - times[f - 1]
        result["address_rate_um_s"][f] = (after["s_um"] - before["s_um"]) / dt
        result["relative_tangent_step_um"][f] = step
        result["relative_tangent_velocity_um_s"][f] = step / dt
        result["motion_step_state"][f] = min(before["state"], after["state"])
    result["metadata"] = dict(schema="tip_original_surface_address_v1", um_per_px=estimator.scale,
        current_marker="registered blade tip, never nearest material point",
        reference_marker="original surface material address estimated by inverse fixed-block map",
        state_labels=STATE_LABELS, uncertainty="sensitivity envelope, not statistical confidence interval",
        physical_truth_validated=False, new_crack_surface_identity_available=False,
        branch_partition_supplied=branch_labels is not None, blade_rotation_compensated=False,
        initial_surface_extended_from_mask=estimator.initial_mask_extended,
        initial_surface_address_range_px=estimator.addresses[[0, -1]].tolist(),
        address_rate="convective original-surface address change; not material velocity",
        relative_velocity="same-address material motion relative to translating blade, projected along current surface",
        precontact_or_withdrawal_contact_not_assumed=True)
    return result
