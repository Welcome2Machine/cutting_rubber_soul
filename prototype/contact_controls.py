"""Explicit, separately selectable contact diagnostics; no default changes.

Velocity regularization and exclusion of directly connected body VV stencils
are independent controls. Full native CCD/intersection checking is inherited.
Neither control repairs finite-gap friction objectivity or nose feature changes.
Direct VV exclusion FAILS a feature-crossing continuity test. It is retained
ONLY to reproduce the rejection evidence, and is forbidden in the long runner.
The local objective-slip kernel below is NOT yet a general surface friction law:
it requires a persistent oriented master edge, including through load history.
"""
import json
import numpy as np

POLICY = "contact-controls-v1"


def projected_surface_slip(current, previous):
    """Signed material-coordinate slip on ONE persistent master edge in 2D.

    Node order: slave point, master endpoint a, master endpoint b. Old and new
    closest coordinates are UNCLAMPED: extending this across vertices needs
    continuous surface charts/history transport and is deliberately not done.
    s = L * (beta - beta_old). Return s and its exact current-position gradient.
    Common rigid motion, edge dilation at fixed beta, and pure normal motion
    produce zero slip, also at a finite normal gap.
    """
    x, old = np.asarray(current, float), np.asarray(previous, float)
    if x.shape != (3, 2) or old.shape != (3, 2) or not (
            np.isfinite(x).all() and np.isfinite(old).all()):
        raise ValueError("Finite (3,2) point/edge configurations required")
    e, e0 = x[2]-x[1], old[2]-old[1]
    length, length0 = np.linalg.norm(e), np.linalg.norm(e0)
    if min(length, length0) <= 1e-14:
        raise ValueError("Degenerate master edge")
    t = e/length
    beta0 = float((old[0]-old[1])@e0/(length0*length0))
    v = x[0]-x[1]
    tangential = float(t@v)
    db = (v-t*tangential)/length-beta0*t
    gradient = np.stack((t, -t-db, db))
    return tangential-beta0*length, gradient


def objective_edge_contact(current, previous, *, weight, dhat, stiffness, mu, epsilon):
    """Work-conjugate positive-weight persistent EV contact candidate, NOT FE.

    Complete fresh force/Jacobian for the local objective slip, including the
    finite-gap moment corrections and normal-capacity derivatives. Clamping
    its Hessian would not be a physical tangent. No branch/history transfer is
    implemented: both projections must remain strictly inside the SAME edge.
    Negative improved-max correction weights cannot be passed as contact areas.
    These restrictions deliberately prevent promotion by replacing native EVs
    while silently retaining incompatible VV terms.
    """
    from .curve_contact import physical_barrier, friction_mollifier
    x, old = np.asarray(current, float), np.asarray(previous, float)
    slip, ds = projected_surface_slip(x, old)
    if not all(np.isfinite(z) for z in (weight,dhat,stiffness,mu,epsilon)) or min(
            weight,dhat,stiffness,epsilon)<=0 or mu<0:
        raise ValueError('Positive quadrature weight/scales and nonnegative friction required')
    e, e0 = x[2]-x[1], old[2]-old[1]
    L, L0 = np.linalg.norm(e), np.linalg.norm(e0)
    v, v0 = x[0]-x[1], old[0]-old[1]
    beta, beta0 = float(v@e/L**2), float(v0@e0/L0**2)
    if not (0 < beta < 1 and 0 < beta0 < 1):
        raise ValueError('Cross-edge/VV history requires a qualified surface chart')
    t=e/L;P=np.eye(2)-np.outer(t,t)
    rot=np.array([[0.,-1.],[1.,0.]])
    side=np.sign((rot@(e0/L0))@v0)
    n=side*rot@t;gap=float(n@v)
    if not side or gap<=0:
        raise ValueError('Persistent oriented contact requires a positive gap')
    Bv=np.hstack((np.eye(2),-np.eye(2),np.zeros((2,2))))
    Be=np.hstack((np.zeros((2,2)),-np.eye(2),np.eye(2)))
    def radial_hessian(w):
        pw=P@w
        return -((t@w)*P+np.outer(t,pw)+np.outer(pw,t))/L**2
    hs=Bv.T@(P/L)@Be+Be.T@(P/L)@Bv+Be.T@(radial_hessian(v)-beta0*P/L)@Be
    dg=Bv.T@n+Be.T@(P@(side*rot.T@v)/L)
    cross=side*rot@P/L
    hg=Bv.T@cross@Be+Be.T@cross.T@Bv+Be.T@radial_hessian(side*rot.T@v)@Be
    b,bp,bpp=(float(a[0]) for a in physical_barrier(np.array([gap]),dhat,stiffness))
    f0,f1,f2=(float(a[0]) for a in friction_mollifier(np.array([slip]),epsilon))
    ds=ds.ravel();normal=bp*dg;N=-bp
    friction=mu*N*f1*ds
    hn=bpp*np.outer(dg,dg)+bp*hg
    jf=mu*(-bpp*f1*np.outer(ds,dg)+N*f2*np.outer(ds,ds)+N*f1*hs)
    return dict(gap_mm=gap,slip_mm=float(slip),normal_capacity_n=weight*N,
        normal_gradient=weight*normal,friction_gradient=weight*friction,
        gradient=weight*(normal+friction),jacobian=weight*(hn+jf),
        slip_gradient=ds,slip_hessian=hs,
        friction_dissipation_nmm=float(weight*mu*N*f1*slip),
        normal_energy_nmm=weight*b,
        scope='positive-weight persistent EV ONLY; no native stencil substitution or full FE qualification')


class ContactControlMixin:
    """P1/Maxwell-only isolated controls with exact checkpoint policy checks."""
    def enable_contact_controls(self, *, velocity_epsilon=None, exclude_adjacent=False):
        if self.element != "p1" or self.enrichment is not None:
            raise ValueError("Contact control qualified only for P1")
        if self.contact_formulation != "improved-max":
            raise ValueError("Contact control requires native improved-max")
        if velocity_epsilon is not None and (
                not np.isfinite(velocity_epsilon) or velocity_epsilon <= 0):
            raise ValueError("Positive finite velocity epsilon required")
        self.contact_control = dict(policy=POLICY,
            velocity_epsilon_mm_s=velocity_epsilon,
            exclude_adjacent_body_vv=bool(exclude_adjacent),
            initial_rest_to_zero="legacy displacement epsilon",
            objectivity="native finite-gap limitation retained")
        self._body_connected_pairs = {
            pair for pair in self._connected_boundary_pairs
            if np.all(self.surface[list(pair)] < self.n)}
        self._contact_initial_preparation = False

    def collisions(self, x):
        c = super().collisions(x)
        if self.contact_control["exclude_adjacent_body_vv"]:
            c.vv_collisions = [
                s for s in c.vv_collisions
                if tuple(sorted((s.vertex0_id, s.vertex1_id)))
                not in self._body_connected_pairs]
        return c

    def _sync_friction_scale(self):
        from .ipc_reference import BASELINE
        ev = self.contact_control["velocity_epsilon_mm_s"]
        eps = BASELINE["friction_smoothing_mm"]
        if ev is not None and not self._contact_initial_preparation:
            if self.material_dt is None or not np.isfinite(self.material_dt) or self.material_dt <= 0:
                raise ValueError("Velocity regularization requires real material dt")
            eps = ev*self.material_dt
        self.friction.eps_v = eps  # toolkit name; its argument here is displacement
        return eps

    def lag(self, x):
        self._sync_friction_scale()
        return super().lag(x)

    def step(self, x, old, depth, deadline):
        # All comparison paths use the identical original rest-to-zero preparation.
        # This is POST-ZERO refinement, not convergence of initial preparation.
        self._contact_initial_preparation = bool(abs(depth) < 1e-12 and old < 0)
        return super().step(x, old, depth, deadline)

    def acceptance(self, x, depth, iterations, outer_iterations):
        row = super().acceptance(x, depth, iterations, outer_iterations)
        if self.contact_control["exclude_adjacent_body_vv"]:
            raw = super().collisions(x)
            removed = [s for s in raw.vv_collisions
                if tuple(sorted((s.vertex0_id, s.vertex1_id))) in self._body_connected_pairs]
            raw_min = float(np.sqrt(raw.compute_minimum_distance(self.cmesh, x[self.surface])))
            if raw_min <= 0:
                raise RuntimeError("Nonpositive full native clearance")
            row["contact_diagnostics"].update(
                excluded_adjacent_body_vv_count=len(removed),
                all_native_minimum_distance_mm=raw_min if np.isfinite(raw_min) else None)
        row["contact_control"] = dict(self.contact_control,
            displacement_epsilon_mm=float(self.friction.eps_v),
            initial_preparation=self._contact_initial_preparation)
        return row

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),
            contact_control_json=np.array(json.dumps(self.contact_control, sort_keys=True)),
            friction_displacement_epsilon_mm=float(self.friction.eps_v),
            contact_initial_preparation=self._contact_initial_preparation)

    def audit_state(self, state, record):
        if json.loads(str(state.get("contact_control_json", "{}"))) != self.contact_control:
            raise ValueError("Contact policy missing or mismatched in checkpoint")
        previous = self._contact_initial_preparation
        self._contact_initial_preparation = bool(state["contact_initial_preparation"])
        if self._contact_initial_preparation != (abs(float(state["depth_mm"])) < 1e-12):
            raise ValueError("Initial preparation flag inconsistent")
        try:
            row = super().audit_state(state, record)
            # Super restores dt after assembly, so compare the saved numerical scale.
            np.testing.assert_allclose(self.friction.eps_v,
                float(state["friction_displacement_epsilon_mm"]), rtol=1e-12, atol=1e-18)
            np.testing.assert_allclose(self.friction.eps_v,
                record["contact_control"]["displacement_epsilon_mm"], rtol=1e-12, atol=1e-18)
            return row
        finally:
            self._contact_initial_preparation = previous

