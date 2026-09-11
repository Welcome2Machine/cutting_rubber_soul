"""Two-dimensional non-associated Coulomb primal-dual interior-point solver.

R21 motivates the cone formulation; this is not its ANM implementation.
An affine mechanical subproblem has fixed geometry and fixed physical history.
Traction is (pressure, shear resisting positive slip), in N/mm^2.
The nonlinear outer loop alone may accept a physical increment.
"""
import time
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags, kron
from scipy.sparse.linalg import splu


class InteriorFailure(RuntimeError):
    """A bounded inner solve failed; its iterate is not a physical solution."""
    def __init__(self,message,state=None):
        super().__init__(message)
        self.state=state


def cone_slack(gap_slip, mu):
    """de Saxce dual cone eigenvalues; gap is positive for penetration.

    r=(p-t/mu,p+t/mu), s=(-gap+mu*(|v|+v),-gap+mu*(|v|-v)).
    r>=0, s>=0, r*s=0 implies Signorini and Coulomb WITHOUT dilatancy.
    """
    gap, slip = np.asarray(gap_slip).T
    if mu==0:return -gap
    return np.column_stack((-gap+mu*(abs(slip)+slip),
                            -gap+mu*(abs(slip)-slip))).ravel()


def cone_derivative(gap_slip, kinematic, mu):
    if mu==0:return -kinematic[::2].tocsc()
    sign = np.sign(np.asarray(gap_slip)[:, 1])
    normal, tangent = kinematic[::2], kinematic[1::2]
    # Interleave the two eigenvalue rows.
    rows = np.arange(len(sign)*2).reshape(-1, 2)
    from scipy.sparse import coo_matrix
    plus = coo_matrix((np.ones(len(sign)), (rows[:, 0], np.arange(len(sign)))),
                      shape=(2*len(sign), len(sign))).tocsr()
    minus = coo_matrix((np.ones(len(sign)), (rows[:, 1], np.arange(len(sign)))),
                       shape=plus.shape).tocsr()
    return (plus@(-normal+diags(mu*(sign+1))@tangent)
            + minus@(-normal+diags(mu*(sign-1))@tangent)).tocsc()


def boundary_fraction(value, direction, fraction=1.):
    negative = direction < 0
    return min(1., fraction*float(np.min(-value[negative]/direction[negative]))) if np.any(negative) else 1.


def solve_affine(stiffness, force, kinematic, offset, weights, mu, *,
                 initial_traction=None, force_tolerance=1e-6,
                 traction_tolerance=1e-7, kinematic_tolerance=1e-9,
                 max_iterations=80, seconds=60., observer=None):
    """Infeasible-start predictor/corrector, strictly interior in both cones.

    Solve K*x + f + J.T*W*lambda=0 and Coulomb(offset+J*x,lambda).
    mu=0 solves the normal-only complementarity system with exactly zero shear.
    K must be a symmetric positive-definite mechanical model. Geometry and
    kinematic offsets are immutable during this solve. All contact rows remain.
    No diagonal compliance is added to the interface equations.
    """
    k = csc_matrix(stiffness); j = csc_matrix(kinematic)
    f = np.array(force, dtype=float, copy=True)
    offset = np.array(offset, dtype=float, copy=True)
    weights = np.array(weights, dtype=float, copy=True)
    m = len(weights); n = len(f)
    if (m == 0 or k.shape != (n,n) or j.shape != (2*m,n) or offset.shape != (m,2)
        or not np.all(np.isfinite(np.r_[k.data,j.data,f,offset.ravel(),weights]))
        or np.any(weights <= 0) or not np.isfinite(mu) or mu < 0
        or max_iterations < 1 or not np.all(np.isfinite([seconds,force_tolerance,traction_tolerance,kinematic_tolerance]))
        or min(seconds,force_tolerance,traction_tolerance,kinematic_tolerance) <= 0):
        raise ValueError('Invalid affine Coulomb problem or budget')
    skew = k-k.T
    if skew.nnz and np.max(abs(skew.data)) > 1e-10*max(1.,np.max(abs(k.data))):
        raise ValueError('Mechanical model must be symmetric')
    # Exact eigenvalue check for small independent benchmarks. Large callers
    # construct a certified positive metric with positive_metric below.
    if n <= 256 and np.linalg.eigvalsh(k.toarray())[0] <= 0:
        raise ValueError('Mechanical model must be positive definite')
    # Exactly frictionless contact has ONE normal complementary pair. Do not
    # approximate mu=0 by a narrow cone with divisions by a small coefficient.
    components=1 if mu==0 else 2
    block=[[1.],[0.]] if mu==0 else [[.5,.5],[-.5*mu,.5*mu]]
    transform = kron(diags(np.ones(m)), csc_matrix(block), format='csc')
    b = j.T@diags(np.repeat(weights,2))
    c = (b@transform).tocsc()
    x = np.zeros(n)
    if initial_traction is None:
        lam = np.zeros((m,2))
    else:
        lam = np.array(initial_traction,dtype=float,copy=True)
        if lam.shape != (m,2) or not np.all(np.isfinite(lam)):
            raise ValueError('Invalid initial traction')
    r = lam[:,0].copy() if mu==0 else np.column_stack((lam[:,0]-lam[:,1]/mu,lam[:,0]+lam[:,1]/mu)).ravel()
    # Preserve a good warm start. The maximum pressure at one active node must
    # not be imposed on every open node, nor the largest remote gap on contact.
    traction_floor=max(10*traction_tolerance,1e-6)
    slack_floor=max(10*kinematic_tolerance,1e-8)
    r = np.maximum(r,traction_floor)
    s = np.maximum(cone_slack(offset,mu),slack_floor)
    traction_scale=np.repeat(np.maximum(np.max(abs(lam[:,:components]),axis=1),traction_floor),components)
    slack_scale=np.repeat(np.maximum(np.max(abs(offset[:,:components]),axis=1),slack_floor),components)
    # Fixed, dimensionless residual units; no changing norm along a line search.
    force_scale = max(np.linalg.norm(f),np.linalg.norm(c@r),force_tolerance)
    product_scale = traction_scale*slack_scale
    deadline = time.monotonic()+seconds

    def residual(xx,rr,ss,target):
        kin = offset+(j@xx).reshape(m,2)
        rf = k@xx+f+c@rr
        re = ss-cone_slack(kin,mu)
        merit = np.linalg.norm(np.r_[rf/force_scale,re/slack_scale,(rr*ss-target)/product_scale])
        return rf,re,kin,float(merit)

    for iteration in range(max_iterations):
        if time.monotonic() > deadline:raise InteriorFailure('IPM wall budget')
        rf,re,kin,_ = residual(x,r,s,0.)
        lam = (transform@r).reshape(m,2)
        # Independent nonsmoothed return map: scales carry physical units.
        kn = traction_tolerance/kinematic_tolerance
        pn = np.maximum(0.,lam[:,0]+kn*kin[:,0])
        tn = np.clip(lam[:,1]+kn*kin[:,1],-mu*pn,mu*pn)
        projection = float(np.max(abs(np.column_stack((pn,tn))-lam)))
        force_norm = float(np.linalg.norm(rf))
        mean_product = float(np.mean(r*s))
        entry = dict(iteration=iteration,force_n=force_norm,projection_mpa=projection,
                     equality_mm=float(np.max(abs(re))),complementarity=float(np.max(r*s)),
                     barrier=mean_product,minimum_primal=float(r.min()),minimum_dual=float(s.min()))
        if observer is not None:observer(entry)
        if (force_norm <= force_tolerance and projection <= traction_tolerance
            and np.max(abs(re)) <= kinematic_tolerance
            and np.max(r*s) <= traction_tolerance*kinematic_tolerance):
            return x,lam,entry
        d = cone_derivative(kin,j,mu)
        # Eliminate independent dual slacks, retaining nonsymmetric mechanics.
        matrix = bmat([[k,c],[diags(r)@d,diags(s)]],format='csc')
        # Row equilibration is algebraic; it changes neither equations nor step.
        row = 1/np.maximum(np.asarray(abs(matrix).max(axis=1).toarray()).ravel(),1e-30)
        try:lu = splu((diags(row)@matrix).tocsc())
        except RuntimeError as exc:raise InteriorFailure('IPM singular Newton system') from exc
        def direction(target,cross=None):
            rhs = -r*s+target+r*re
            if cross is not None:rhs -= cross
            step = lu.solve(np.r_[-rf,rhs]*row)
            dx,dr = step[:n],step[n:]
            ds = d@dx-re
            return dx,dr,ds
        dx,dr,ds = direction(0.)
        affine = min(boundary_fraction(r,dr),boundary_fraction(s,ds))
        after = float(np.mean((r+affine*dr)*(s+affine*ds)))
        sigma = np.clip((after/mean_product)**3,.001,.9)
        target = sigma*mean_product
        cross = dr*ds
        accepted = False
        # Try Mehrotra's cross term, then a pure centered Newton direction if
        # the correction is not a descent direction for the actual |slip|.
        for corrector in (cross,None):
            dx,dr,ds = direction(target,corrector)
            alpha = min(boundary_fraction(r,dr,.995),boundary_fraction(s,ds,.995))
            old_merit = residual(x,r,s,target)[3]
            for cut in range(20):
                step = alpha*.5**cut
                xx,rr,ss = x+step*dx,r+step*dr,s+step*ds
                if np.min(rr) <= 0 or np.min(ss) <= 0:continue
                if residual(xx,rr,ss,target)[3] < old_merit*(1-1e-4*step):
                    x,r,s = xx,rr,ss;accepted=True;break
            if accepted:break
        if not accepted:
            raise InteriorFailure(f'IPM line search failed at iteration {iteration}',
                dict(inner_displacement=x,inner_primal=r,inner_dual=s,inner_direction=np.r_[dx,dr,ds],
                     inner_target=np.array(target),inner_iteration=np.array(iteration)))
    raise InteriorFailure('IPM iteration budget',dict(inner_displacement=x,inner_primal=r,inner_dual=s))


def positive_metric(stiffness, proximal=0.):
    """Positive body model, checked by symmetric LU/LDL pivots.

    Try the true Hessian before adding a diagonal. A scaled Gershgorin bound
    supplies a guaranteed fallback if the numerical factor certificate fails.
    This modifies only the step model, never the physical residual.
    """
    k = csc_matrix(stiffness)
    if not np.isfinite(proximal) or proximal < 0:raise ValueError('Invalid proximal coefficient')
    skew = k-k.T
    if skew.nnz and np.max(abs(skew.data)) > 1e-10*max(1.,np.max(abs(k.data))):
        raise ValueError('Body Hessian is not symmetric; do not silently symmetrize')
    scale = np.maximum(abs(k.diagonal()),1.)
    inverse = diags(1/np.sqrt(scale))
    normalized = inverse@k@inverse
    lower = float(np.min(normalized.diagonal()-(np.asarray(abs(normalized).sum(axis=1)).ravel()-abs(normalized.diagonal()))))
    bound = max(proximal,1e-3-lower,0.)
    shift=proximal;certificate='Gershgorin'
    while shift<bound:
        candidate=(normalized+diags(np.full(k.shape[0],shift))).tocsc()
        try:
            factor=splu(candidate,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
            pivot=factor.U.diagonal()
            mismatch=factor.U-diags(pivot)@factor.L.T
            error=0. if not mismatch.nnz else float(np.max(abs(mismatch.data)))
            if (np.array_equal(factor.perm_r,factor.perm_c) and np.all(pivot>0)
                and error<=1e-9*max(1.,float(np.max(abs(factor.U.data))))):
                certificate='positive LDL pivots';break
        except RuntimeError:pass
        shift=min(bound,max(1e-3,10*shift))
    return (k+diags(shift*scale)).tocsc(),dict(metric_lower_bound=lower,metric_shift=shift,metric_certificate=certificate)


def affine_model(p,u,previous,depth,old_depth,dt,lam,proximal=0.):
    """Freeze one work-conjugate frame; retain the exact nonlinear anchor force."""
    frozen=p.surface_contact.freeze(u,depth,lam,affine_normal=True)
    r,k,q,_=p.assemble(u,previous,depth,old_depth,dt,lam[:,0],lam[:,1],mixed=True,
                       frozen_contact=frozen,roundoff_guard=False)
    body=(k-q['global_K'])[p.free][:,p.free]
    metric,info=positive_metric(body,proximal)
    j=q['global_kinematic_u'][:,p.free].tocsc()
    b=j.T@diags(np.repeat(p.weights,2))
    error=b-q['global_B'][p.free]
    if error.nnz and np.max(abs(error.data))>1e-10*max(1.,np.max(abs(b.data))):
        raise ValueError('Frozen contact force and displacement are not work conjugate')
    force=r[p.free]-b@lam.ravel()
    offset=np.column_stack((q['gap'],q['slip']))
    return metric,force,j,offset,info


def interior_step(p,previous,old_depth,depth,dt,old_state,*,initial=None,
                  initial_multipliers=None,seconds=180.,max_outer=30,max_inner=80,
                  trust_radius=.02):
    """Experimental finite-strain outer loop with affine Coulomb IPM steps.

    A ratio of actual to predicted ORIGINAL residual reduction controls updates.
    Positive mechanics is a local step model, not a replacement constitutive law.
    Never advances physical time/history inside either nonlinear iteration.
    """
    from .initiation import ContactSolveError,ContactBudgetError,ContactResolutionError
    if (p.surface_contact is None or p.options.volumetric_discretization!='quadratic_full'
        or p.config.adhesive_shear_strength_n_mm2 or p.config.friction_coefficient<0):
        raise ValueError('IPM outer loop requires nonadhesive frictional P2 surface contact')
    if (max_outer<1 or max_inner<1 or not np.isfinite(seconds) or seconds<=0
        or not np.isfinite(trust_radius) or trust_radius<=0):raise ValueError('Invalid IPM outer budget')
    c=p.config;start=time.monotonic();deadline=start+seconds
    u=np.array(previous if initial is None else initial,copy=True)
    lam=np.array(old_state if initial_multipliers is None else initial_multipliers,copy=True)
    radius=trust_radius;proximal=0.;best=None;inner_count=0;reason='outer iteration budget';inner_state={}
    force_goal=c.newton_tolerance_n;projection_goal=c.contact_multiplier_tolerance_n_mm2;polishes=0
    def norm(force,kin,traction):
        pn=np.maximum(0,traction[:,0]+c.contact_surface_penalty_n_mm3*kin[:,0])
        tn=np.clip(traction[:,1]+p.options.tangential_stiffness_mpa_per_mm*kin[:,1],
                   -c.friction_coefficient*pn,c.friction_coefficient*pn)
        defect=np.column_stack((pn,tn))-traction
        return float(np.linalg.norm(np.r_[force/c.newton_tolerance_n,
            defect.ravel()/c.contact_multiplier_tolerance_n_mm2]))
    def evaluate(v,l):
        r,_,q,_=p.assemble(v,previous,depth,old_depth,dt,l[:,0],l[:,1],with_tangent=False,roundoff_guard=False)
        score=norm(r[p.free],np.column_stack((q['gap'],q['slip'])),l)
        return score,float(np.linalg.norm(r[p.free])),float(np.max(abs(np.column_stack((q['pressure'],q['shear']))-l)))
    def verify(v,l):
        # This is the existing independent original-equation/whole-edge audit.
        result=p._mixed_step(previous,old_depth,depth,dt,old_state,initial=v,initial_multipliers=l,verify_only=True)
        result[2].update(contact_solver='finite-strain affine Coulomb IPM outer iteration',
                         contact_inner_iterations=inner_count,contact_trust_radius_mm=radius,
                         geometry_polish_attempts=polishes,solver_force_goal_n=force_goal,
                         solver_projection_goal_mpa=projection_goal)
        return result
    for outer in range(max_outer):
        score,force_error,projection=evaluate(u,lam)
        if best is None or score<best[0]:best=(score,u.copy(),lam.copy())
        print(f'IPM OUTER iteration={outer} force={force_error:.8g} projection={projection:.8g} merit={score:.8g} radius={radius:.8g}',flush=True)
        if force_error<=force_goal and projection<=projection_goal:
            try:return verify(u,lam)
            except ContactResolutionError:
                # Force acceptance does not bound displacement error tightly
                # enough to classify a small geometric excess as under-resolution.
                # Refine the equilibrium before requesting a different mesh.
                while polishes<3 and force_error<=force_goal and projection<=projection_goal:
                    force_goal=max(c.newton_tolerance_n*1e-3,force_goal*.1)
                    projection_goal=max(c.contact_multiplier_tolerance_n_mm2*1e-3,projection_goal*.1)
                    polishes+=1
                if force_error<=force_goal and projection<=projection_goal:raise
                print(f'IPM GEOMETRY POLISH attempt={polishes} force_goal={force_goal:.8g} projection_goal={projection_goal:.8g}; physical tolerances unchanged',flush=True)
            except ContactSolveError:pass
        if time.monotonic()>deadline:reason='outer wall budget';break
        metric,force,j,offset,info=affine_model(p,u,previous,depth,old_depth,dt,lam,proximal)
        # The trust ratio must use the true residual derivative, not the
        # positive surrogate used to compute a direction. Otherwise a harmless
        # fixed diagonal shift can shrink the radius forever near a linear root.
        original_r,original_k,original_q,_=p.assemble(u,previous,depth,old_depth,dt,
            lam[:,0],lam[:,1],mixed=True,roundoff_guard=False)
        print('IPM MODEL '+str(info),flush=True)
        def observer(entry):
            nonlocal inner_count
            inner_count+=1
            print('IPM INNER '+str(entry),flush=True)
        try:
            du,new_lam,_=solve_affine(metric,force,j,offset,p.weights,c.friction_coefficient,
                initial_traction=lam,force_tolerance=.05*force_goal,
                traction_tolerance=.05*projection_goal,
                kinematic_tolerance=.05*projection_goal/max(c.contact_surface_penalty_n_mm3,p.options.tangential_stiffness_mpa_per_mm),
                max_iterations=max_inner,seconds=max(1e-9,deadline-time.monotonic()),observer=observer)
        except InteriorFailure as exc:
            if exc.state is not None:
                inner_state=dict(exc.state,inner_anchor=u.copy(),inner_anchor_traction=lam.copy(),inner_proximal=np.array(proximal))
            reason=str(exc);break
        full=np.zeros_like(u);full[p.free]=du
        max_motion=float(np.max(np.linalg.norm(full.reshape(-1,2),axis=1)))
        alpha=min(1.,radius/max(max_motion,1e-30));accepted=False
        for cut in range(12):
            if time.monotonic()>deadline:reason='outer wall budget';break
            step=alpha*.5**cut
            candidate=u+step*full;traction=lam+step*(new_lam-lam)
            predicted_force=original_r[p.free]+original_k[p.free][:,p.free]@(step*du)+original_q['global_B'][p.free]@(traction-lam).ravel()
            predicted_kin=offset+(original_q['global_kinematic_u'][:,p.free]@(step*du)).reshape(-1,2)
            predicted=norm(predicted_force,predicted_kin,traction)
            reduction=score-predicted
            if reduction<=0:continue
            try:actual=evaluate(candidate,traction)[0]
            except FloatingPointError:continue
            ratio=(score-actual)/reduction
            print(f'IPM TRIAL alpha={step:.8g} predicted={predicted:.8g} actual={actual:.8g} ratio={ratio:.8g}',flush=True)
            if ratio>=.05 and actual<score:
                u,lam=candidate,traction;accepted=True
                if actual<best[0]:best=(actual,u.copy(),lam.copy())
                if ratio>.75 and cut==0:radius=min(trust_radius,2*radius)
                elif ratio<.25:radius*=.5
                proximal=max(0.,proximal*.5)
                break
        if not accepted:
            radius*=.25;proximal=max(1.,info['metric_shift']*2)
            print('IPM REJECT original residual did not justify the model step',flush=True)
            if radius<1e-8:reason='outer trust radius below minimum';break
    _,u,lam=best
    score,force_error,projection=evaluate(u,lam)
    if force_error<=c.newton_tolerance_n and projection<=c.contact_multiplier_tolerance_n_mm2:
        try:return verify(u,lam)
        except ContactSolveError:pass
    kind=ContactBudgetError if 'budget' in reason else ContactSolveError
    error=kind(f'{reason}: depth={depth:.8g}, residual={force_error:.8g} N, projection={projection:.8g} MPa',u,lam)
    error.inner_state=inner_state
    raise error
