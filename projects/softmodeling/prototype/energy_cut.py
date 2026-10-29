"""Separate energy-work cutting candidate, never the intact baseline.

Theta kinematics, C-space bulk discrete gradient, exact cohesive work,
endpoint-inclusive native IPC barrier and midpoint Coulomb friction. Maxwell
relaxes at fixed final deformation over the actual dt (Lie split). Spatial
contact, splitting and temporal accuracy still require independent studies.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from mechanics.discrete_gradient import discrete_response, frozen_energy, cohesive_discrete_response
from .zero_gap_cut import ZeroGapCutProblem
from .ipc_reference import LoadStepFailure


def barrier_time_quadrature(points, rule):
    if points not in (2, 3, 5) or rule not in ('gauss', 'lobatto'):
        raise ValueError('Explicit Gauss/Lobatto rule with 2, 3 or 5 points required')
    if rule == 'gauss':
        nodes, weights = np.polynomial.legendre.leggauss(points)
        return .5*(nodes+1), .5*weights
    poly = np.polynomial.legendre.Legendre.basis(points-1)
    nodes = np.r_[-1., poly.deriv().roots(), 1.]
    return .5*(nodes+1), 1/(points*(points-1)*poly(nodes)**2)


def nonlocal_tensile_mask(traction, released, y, area, strength, length, *, intact_only=False):
    """Dual local + reference-neighbourhood tensile-strength condition.

    A regularized hypothesis, NOT the coupled finite-crack energy criterion.
    Gc governs the subsequent local cohesive work. Length must be identified
    independently; positive-part averaging never lets compression drive damage.
    """
    if length <= 0 or not np.isfinite(length):
        raise ValueError('Positive finite independent length required')
    weight = np.maximum(0., 1-np.abs(y[:, None]-y[None, :])/length)*area[None, :]
    if intact_only:
        # Constraint reaction is a traction measurement only on intact ties.
        # Released free nodes have zero assembled residual, not zero bulk
        # stress; they must not dilute the intact-ligament strength average.
        weight *= ~released[None, :]
    average = np.divide(weight@np.maximum(traction, 0.), weight.sum(axis=1),
                        out=np.zeros(len(y)), where=weight.sum(axis=1)>0)
    return (~released) & (y > 0) & (traction > strength) & (average > strength)


def connected_patch_mask(traction, released, y, area, strength, length):
    """Surface-connected nonlocal tensile-strength hypothesis, not Leguillon FFM.

    Signed traction is averaged over the next reference-length patch. Only its
    surface-most virgin node may be released; the local extrinsic graph still
    resists opening. Released != damaged != fully open. No depth/force trigger.
    """
    if length <= 0 or not np.isfinite(length):
        raise ValueError('Explicit positive process-neighbourhood length required')
    order = np.argsort(y)[::-1]
    virgin = order[(~released[order]) & (y[order] > 0)]
    mask = np.zeros_like(released)
    if not len(virgin):
        return mask
    front = virgin[0]
    patch = virgin[y[virgin] >= y[front]-length-1e-12]
    # Reference nodal tributaries truncated to the physical patch interval.
    # A smooth compact kernel avoids counting an entire terminal half-cell.
    weight = area[patch]*np.maximum(0., 1-(y[front]-y[patch])/length)
    average = float(weight@traction[patch]/weight.sum())
    if average > strength:
        mask[front] = True
    return mask


class EnergyCutProblem(ZeroGapCutProblem):
    def __init__(self, *, barrier_quadrature=3, barrier_quadrature_rule='lobatto', insertion_rule='local_strength',
                 nucleation_length_mm=None, kinematic_theta=.5, **kwargs):
        super().__init__(**kwargs)
        if self.cohesive_unloading != 'secant':
            raise ValueError('Discrete cohesive work currently requires secant unloading')
        if self.nonlinear_solver not in ('newton_lm', 'increment_ptc', 'friction_continuation'):
            raise ValueError('Energy-work candidate requires a primary-increment force solver')
        if barrier_quadrature not in (2, 3, 5):
            raise ValueError('Choose 2, 3 or 5 barrier quadrature points')
        if insertion_rule not in ('local_strength', 'connected_patch', 'nonlocal_tensile', 'nonlocal_intact'):
            raise ValueError('Unknown insertion rule')
        if insertion_rule != 'local_strength' and (nucleation_length_mm is None or
                not np.isfinite(nucleation_length_mm) or nucleation_length_mm <= 0):
            raise ValueError('Neighbourhood rule requires an independent length')
        self.insertion_rule, self.nucleation_length = insertion_rule, nucleation_length_mm
        if not np.isfinite(kinematic_theta) or not .5 <= kinematic_theta <= 1:
            raise ValueError('Kinematic theta must be in [0.5,1]')
        self.kinematic_theta = float(kinematic_theta)
        self.barrier_quadrature_rule = barrier_quadrature_rule
        self.barrier_nodes, self.barrier_weights = barrier_time_quadrature(barrier_quadrature, barrier_quadrature_rule)

    def insertion_mask(self, traction):
        if self.insertion_rule == 'local_strength':
            return super().insertion_mask(traction)
        if self.insertion_rule in ('nonlocal_tensile', 'nonlocal_intact'):
            return nonlocal_tensile_mask(traction, self.released, self.rest[self.seam, 1],
                self.seam_area, self.strength, self.nucleation_length,
                intact_only=self.insertion_rule == 'nonlocal_intact')
        return connected_patch_mask(traction, self.released, self.rest[self.seam, 1],
            self.seam_area, self.strength, self.nucleation_length)

    def endpoint_velocity(self, x):
        theta = self.kinematic_theta
        velocity = (self.step_increment(x)/self.material_dt-(1-theta)*self.previous_velocity)/theta
        velocity[self.n:] = 0.
        return velocity

    def deformation(self, x):
        return np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (x[:self.n]-self.mesh.reference_xy)[self.mesh.triangles], self.grad)

    def frozen(self, x, metric=None):
        return frozen_energy(self.deformation(x), self.previous_material_metric if metric is None else metric,
            self.config.shear_modulus_n_mm2, self.material.mu_ov[0], self.config.bulk_modulus_n_mm2)

    def material_bulk(self, u, *, with_tangent=True):
        areas, gradients, dofs = self.mesh.bulk_data
        f1 = np.eye(2)+np.einsum('eai,eaJ->eiJ', u.reshape(-1, 2)[self.mesh.triangles], gradients)
        q = discrete_response(self.deformation(self.previous), f1, self.previous_material_metric,
            self.config.shear_modulus_n_mm2, self.material.mu_ov[0], self.config.bulk_modulus_n_mm2,
            tangent=with_tangent)
        volume = areas*self.config.width_mm
        r = np.einsum('e,eiJ,eaJ->eai', volume, q['P'], gradients).reshape(-1, 6)
        k = None
        if with_tangent:
            k = np.einsum('e,eaJ,eiJkL,ebL->eaibk', volume, gradients,
                q['tangent'].reshape(-1, 2, 2, 2, 2), gradients, optimize=True).reshape(-1, 6, 6)
        return r, k, dofs, float(volume@q['energy']), float(q['J'].min())

    def lag(self, x):
        return super().lag(self.previous+.5*self.step_increment(x))

    def normal_response(self, x, *, tangent=False, projected=False):
        delta = self.step_increment(x)
        g = np.zeros(2*len(self.surface))
        h = None
        projection = self.ipc.PSDProjectionMethod.CLAMP if projected else self.ipc.PSDProjectionMethod.NONE
        for node, weight in zip(self.barrier_nodes, self.barrier_weights):
            value = self.previous+node*delta
            c = self.collisions(value)
            surface = value[self.surface]
            g += weight*self.barrier.gradient(c, self.cmesh, surface)
            if tangent:
                term = weight*node*self.barrier.hessian(c, self.cmesh, surface, projection)
                h = term if h is None else h+term
        return g, h

    def cohesive_algorithmic(self, x):
        return cohesive_discrete_response(2*(self.previous[self.seam, 0]-self.config.drop_s_mm),
            2*(x[self.seam, 0]-self.config.drop_s_mm), self.previous_history,
            self.strength, self.toughness)

    def evaluate(self, x, lag=None, tangent=True, projected=True, *, with_metric=False):
        if with_metric and not tangent:
            raise ValueError('Paired metric requires tangent')
        r, k, dofs, energy, minj = self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(), with_tangent=tangent)
        g = np.zeros(self.total)
        np.add.at(g, dofs.ravel(), r.ravel())
        normal, nh = self.normal_response(x, tangent=tangent, projected=projected)
        du = self.step_increment(x)
        fg = np.zeros_like(normal) if lag is None else self.friction.gradient(lag, self.cmesh, du[self.surface])
        g[self.surface_dofs] += normal+fg
        coh = self.cohesive_algorithmic(x)
        ids = 2*self.seam
        g[ids] += self.seam_area*self.released*coh['traction']
        inertia = self.mass_dof/(self.kinematic_theta*self.material_dt**2)
        g += inertia*(du-self.material_dt*self.previous_velocity).ravel()
        h = metric = None
        if tangent:
            if projected or with_metric:
                eig, vectors = np.linalg.eigh(.5*(k+k.transpose(0, 2, 1)))
                kp = np.einsum('eik,ek,ejk->eij', vectors, np.maximum(eig, 0.), vectors)
            rows = np.broadcast_to(dofs[:, :, None], k.shape).ravel()
            cols = np.broadcast_to(dofs[:, None, :], k.shape).ravel()
            def assemble(local, clamp, contact):
                m = coo_matrix((local.ravel(), (rows, cols)), shape=(self.total, self.total)).tocsc()
                proj = self.ipc.PSDProjectionMethod.CLAMP if clamp else self.ipc.PSDProjectionMethod.NONE
                if lag is not None:
                    contact = contact+self.friction.hessian(lag, self.cmesh, du[self.surface], proj)
                c = contact.tocoo()
                m += coo_matrix((c.data, (self.surface_dofs[c.row], self.surface_dofs[c.col])), shape=m.shape).tocsc()
                extra = inertia.copy()
                extra[ids] += 2*self.seam_area*self.released*coh['tangent']
                return m+diags(np.maximum(extra, inertia) if clamp else extra)
            h = assemble(kp if projected else k, projected, nh)
            if with_metric:
                metric = h if projected else assemble(kp, True, self.normal_response(x, tangent=True, projected=True)[1])
        return (energy, g, h, minj, metric) if with_metric else (energy, g, h, minj)

    def coupled_friction_tangent(self, x, lag, frozen_hessian):
        if lag is None:
            return frozen_hessian
        midpoint = self.previous+.5*self.step_increment(x)
        jac = -.5*self.friction.force_jacobian(lag, self.cmesh, self.rest[self.surface],
            (midpoint-self.rest)[self.surface], self.step_increment(x)[self.surface], self.barrier,
            self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS).tocoo()
        return frozen_hessian+coo_matrix((jac.data, (self.surface_dofs[jac.row], self.surface_dofs[jac.col])),
            shape=frozen_hessian.shape).tocsc()

    def acceptance(self, x, depth, iterations, outer_iterations):
        try:
            rec = super().acceptance(x, depth, iterations, outer_iterations)
        except RuntimeError as error:
            if str(error) != 'Negative IPC friction incremental work':
                raise
            # Reject and cut back; NEVER accept or relabel a negative friction
            # contribution. A minimum-step repeat remains an explicit failure.
            raise LoadStepFailure('Negative IPC friction incremental work; retry smaller physical step') from error
        ng = self.normal_response(x)[0].reshape(-1, 2)
        rec['reaction_normal_fz_n'] = -2*float(ng[self.surface >= self.n, 1].sum())
        rec['reaction_friction_fz_n'] = rec['reaction_fz_n']-rec['reaction_normal_fz_n']
        rec['reaction_sampling'] = 'algorithmic interval-average; not instantaneous endpoint force'
        rec['time_integrator'] = 'C_discrete_gradient_theta_kinematics_Lie_Maxwell'
        rec['material']['dissipation_quadrature'] = 'fixed-F Maxwell energy decrease; Lie split; time convergence required'
        rec['material']['viscous_dissipation_increment_n_mm'] = .5*rec['energy']['viscous_endpoint_dissipation_n_mm']
        rec['insertion_rule'] = self.insertion_rule
        rec['barrier_quadrature_rule'] = self.barrier_quadrature_rule
        rec['kinematic_theta'] = self.kinematic_theta
        rec['physical_force_solver'] = self.nonlinear_solver
        if self.nonlinear_solver == 'friction_continuation' and hasattr(self, 'friction_continuation_summary'):
            rec['force_continuation'] = self.friction_continuation_summary
        rec['numerical_inertia_dissipation'] = 'zero at theta=0.5; (2*theta-1)*sum(m*dv^2) for the reflected full body; not physical damping'
        return rec

    def energy_ledger(self, x):
        du, v = self.step_increment(x), self.endpoint_velocity(x)
        volume = self.material_integration_weights()
        old, frozen = self.frozen(self.previous), self.frozen(x)
        relaxed = self.material_response(x)['energy']
        viscous = 2*float(volume@(frozen-relaxed))
        if viscous < -1e-9:
            raise RuntimeError('Maxwell split increased fixed-F energy')
        stored = 2*float(volume@(relaxed-old))
        bulk_r, _, dofs, _, _ = self.material_bulk((x[:self.n]-self.mesh.reference_xy).ravel(), with_tangent=False)
        bw = 2*float(np.sum(bulk_r*du.ravel()[dofs]))
        coh, oldc = self.cohesive_response(x), self.cohesive_response(self.previous, self.previous_history)
        mask = self.released.astype(float)
        cs = float(self.seam_area@(mask*(coh['stored']-oldc['stored'])))
        cd = float(self.seam_area@(mask*(coh['dissipated']-oldc['dissipated'])))
        cw = 2*float((self.seam_area*mask*self.cohesive_algorithmic(x)['traction'])@du[self.seam, 0])
        b1 = float(self.barrier(self.collisions(x), self.cmesh, x[self.surface]))
        b0 = float(self.barrier(self.collisions(self.previous), self.cmesh, self.previous[self.surface]))
        jump = 2*(b0-self.previous_barrier_energy)
        bd = 2*(b1-self.previous_barrier_energy)
        bg = self.normal_response(x)[0]
        bwork = 2*float(bg@du[self.surface].ravel())
        fg = self.friction.gradient(self.lag(x), self.cmesh, du[self.surface])
        friction = 2*float(fg@du[self.surface].ravel())
        kinetic = float(self.mass@np.sum(v[:self.n]**2, axis=1))
        oldk = float(self.mass@np.sum(self.previous_velocity[:self.n]**2, axis=1))
        g = self.evaluate(x, self.lag(x), tangent=False)[1].reshape(-1, 2)
        external = 2*float(np.sum(g[self.n:]*du[self.n:]))
        contact = self.active_contact.copy()
        contact[self.seam[~self.released]] = False
        contact_loss = -2*float(g[:self.n, 0][contact]@du[:self.n, 0][contact])
        # Legacy ledger keys are retained for schema compatibility. Here the
        # 'endpoint' bulk/cohesive/barrier terms mean algorithmic-work defects.
        inertia_loss = (2*self.kinematic_theta-1)*float(self.mass@np.sum((v[:self.n]-self.previous_velocity[:self.n])**2, axis=1))
        defects = dict(backward_euler_inertia_loss_n_mm=inertia_loss,
            bulk_endpoint_defect_n_mm=bw-stored-viscous,
            cohesive_endpoint_defect_n_mm=cw-cs-cd,
            barrier_endpoint_defect_n_mm=bwork-bd+jump,
            contact_topology_defect_n_mm=-jump,
            hard_contact_endpoint_loss_n_mm=contact_loss)
        physical = stored+cs+cd+viscous+bd+friction+kinetic-oldk
        reservoir = (float(volume@(relaxed+old))+.5*float(self.seam_area@(mask*(coh['stored']+oldc['stored'])))
            +b1+self.previous_barrier_energy+.5*(kinetic+oldk))
        return dict(external_work_n_mm=external, bulk_stored_delta_n_mm=stored,
            mechanical_reservoir_n_mm=reservoir, cohesive_stored_delta_n_mm=cs,
            fracture_dissipation_n_mm=cd, viscous_endpoint_dissipation_n_mm=viscous,
            friction_work_n_mm=friction, barrier_delta_n_mm=bd, kinetic_delta_n_mm=kinetic-oldk,
            kinetic_energy_n_mm=kinetic, physical_balance_defect_n_mm=external-physical,
            algebraic_balance_residual_n_mm=external-physical-sum(defects.values()), **defects)
