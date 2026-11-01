"""Prospective finite-precision stopping certificate, not a root error bound.

The preferred force goal is tried first. At a roundoff plateau we evaluate real
Newton probes, verify the tangent at TWO resolvable amplitudes, and measure
observable changes and residual impulse/work. No auxiliary force is accepted.
An unreliable sub-ulp J^-1 r is never by itself a stopping certificate.
"""
import hashlib,time,warnings
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu,onenormest,LinearOperator


def validate_policy(q):
    required={'version','time_error_fraction','nonlinear_fraction','force_change_n',
              'equation_work_rate_n_mm_s','impulse_horizon_s','resolved_prediction_rtol','condition_limit'}
    if set(q)!=required or q['version']!='resolved_probe_budget_v1':raise ValueError('Explicit nonlinear error policy required')
    if not all(np.isfinite(v) and v>0 for k,v in q.items() if k!='version'):raise ValueError('Finite positive accuracy controls required')
    if q['time_error_fraction']>.05 or q['nonlinear_fraction']>.1 or q['force_change_n']>1e-6 or q['resolved_prediction_rtol']>.05 or q['condition_limit']>1e10:raise ValueError('Nonlinear allocation exceeds registered reliability limits')
    return dict(q)


def binding(x,du):
    h=hashlib.sha256()
    for a in (x,du):h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def signature(p,x,g):
    c=p.collisions(x);contacts=[]
    for kind in ('vv_collisions','ev_collisions'):
        contacts.extend((kind,*map(int,q.vertex_ids(p.cmesh.edges,p.cmesh.faces))) for q in getattr(c,kind))
    active=(x[:p.n,0]-p.config.drop_s_mm<=1e-12)&(g.reshape(-1,2)[:p.n,0]>=0)
    active[p.seam[~p.released]]=True
    lag=p.lag(x);branches=[];du=p.step_increment(x)
    if lag is not None:
        for kind in ('vv_collisions','ev_collisions'):
            for q in getattr(lag,kind):
                ids=np.array([i for i in q.vertex_ids(p.cmesh.edges,p.cmesh.faces) if i>=0]);nodes=p.surface[ids]
                v=q.relative_velocity((du[nodes]/p.material_dt).ravel())
                branches.append((kind,*map(int,ids),bool(np.linalg.norm(np.asarray(q.tangent_basis).T@v)<p.friction_velocity_epsilon)))
    return sorted(contacts),active.tolist(),sorted(branches)


def residual_budget(p,x,g):
    q=p.nonlinear_error_policy;h=p.material_dt;idx=p.free
    massnorm=lambda v:float(np.sqrt(p.mass@np.sum(v[:p.n]**2,axis=1)))
    scale=max(massnorm(p.endpoint_velocity(x)),massnorm(p.previous_velocity),p.loading_rate*np.sqrt(p.mass.sum()))
    velocity_allow=q['nonlinear_fraction']*q['time_error_fraction']*scale
    return dict(full_force_residual_n=float(np.sqrt(2)*np.linalg.norm(g[idx])),
        velocity_scale_sqrt_nmm=scale,velocity_change_allowance_sqrt_nmm=velocity_allow,
        residual_impulse_n_s=float(np.sqrt(2)*h*np.linalg.norm(g[idx])),
        residual_impulse_mass_norm_sqrt_nmm=float(h*np.sqrt(np.sum(g[idx]**2/p.mass_dof[idx]))),
        residual_impulse_allowance_sqrt_nmm=velocity_allow*h/q['impulse_horizon_s'],
        absolute_equation_work_n_mm=abs(float(2*g[idx]@p.step_increment(x).ravel()[idx])),
        equation_work_allowance_n_mm=q['equation_work_rate_n_mm_s']*h)


def inspect(p,x,deadline=float('inf')):
    """Actual fixed-parent trials; no history commit, physical dt or forces change.

    Every probe retains its own increment. The unamplified tangent mismatch is
    recorded even when roundoff dominates it. Resolved probes must independently
    support a common tangent/active set; this is empirical reliability, no bound.
    """
    q=p.nonlinear_error_policy;du=p.step_increment(x).copy();idx=p.free.copy();old_trial=getattr(p,'trial',None)
    history={k:getattr(p,k).copy() for k in ('velocity','history','material_metric','active_faces','active_contact','released','previous_history','previous_material_metric')}
    report=dict(policy=q,binding_sha256=binding(x,du),passed=False,scope='actual safe correction and resolved tangent probes; empirical local certificate, not a rigorous error bound',probes=[])
    try:
        if getattr(p,'_friction_homotopy_factor',1.)!=1.:raise ValueError('Original friction required')
        _,g,H,j=p.evaluate(x,p.lag(x),tangent=True,projected=False)
        report.update(residual_budget(p,x,g));r=g[idx]
        if not np.isfinite(g).all() or report['full_force_residual_n']>1e-4 or j<=0:raise ValueError('Original physical gate failed')
        if not p.accepted_chord_is_safe(x):raise ValueError('Original trajectory unsafe')
        J=p.coupled_friction_tangent(x,p.lag(x),H)[idx,:][:,idx].tocsc()
        scale=1/np.sqrt(np.maximum(np.asarray(abs(J).sum(axis=1)).ravel(),1e-300));B=(diags(scale)@J@diags(scale)).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter('error');lu=splu(B)
        op=LinearOperator(B.shape,matvec=lu.solve,rmatvec=lambda z:lu.solve(z,'T'),dtype=float)
        # onenormest's randomized probes must not change the simulation RNG.
        rng=np.random.get_state()
        try:np.random.seed(0);condition=float(onenormest(B)*onenormest(op))
        finally:np.random.set_state(rng)
        d=scale*lu.solve(-scale*r);relative=float(np.linalg.norm(J@d+r)/max(np.linalg.norm(r),1e-300))
        report.update(scaled_condition_estimate=condition,linear_relative_residual=relative)
        if not np.isfinite(d).all() or condition>q['condition_limit'] or relative>1e-6:raise ValueError('Unreliable tangent solve')
        direction=np.zeros(p.total);direction[idx]=d;direction=direction.reshape(x.shape)
        sig=signature(p,x,g);f0=float(g.reshape(-1,2)[p.n:,1].sum());v0=p.endpoint_velocity(x);k0=p.cohesive_response(x)['history'];A0=p.material_response(x)['metric']
        for amp in (1.,16.,256.):
            if time.monotonic()>=deadline:raise TimeoutError('Nonlinear error verification budget')
            delta=amp*direction;inc=du+delta;y=p.previous+inc
            if p.safe_step(x,delta)<1-1e-10 or not p.accepted_chord_is_safe(y):raise ValueError('Correction trajectory unsafe')
            if np.any(y[:p.n,0]<p.config.drop_s_mm) or abs(delta).max()>1e-6:raise ValueError('Correction leaves local boundary neighbourhood')
            p.set_primary_increment(y,inc);_,gy,_,jy=p.evaluate(y,p.lag(y),tangent=False)
            mismatch=float(np.linalg.norm(gy[idx]-r-amp*(J@d))/max(amp*np.linalg.norm(r),1e-300))
            dv=p.endpoint_velocity(y)-v0;vchange=float(np.sqrt(p.mass@np.sum(dv[:p.n]**2,axis=1)))/amp
            fchange=2*abs(float(gy.reshape(-1,2)[p.n:,1].sum())-f0)/amp
            kh=float(abs(p.cohesive_response(y)['history']-k0).max())/amp
            ma=float(abs(p.material_response(y)['metric']-A0).max())/amp
            record=dict(amplifier=amp,full_force_residual_n=float(np.sqrt(2)*np.linalg.norm(gy[idx])),relative_prediction_error=mismatch,velocity_change_per_amplifier_sqrt_nmm=vchange,force_change_per_amplifier_n=fchange,history_change_per_amplifier_mm=kh,maxwell_change_per_amplifier=ma,min_j=jy,position_changed_dofs=int(np.count_nonzero(y!=x)),increment_changed_dofs=int(np.count_nonzero(inc!=du)),contact_and_mirror_sets_unchanged=signature(p,y,gy)==sig)
            report['probes'].append(record)
            if jy<=0 or not record['contact_and_mirror_sets_unchanged']:raise ValueError('Probe changes contact, slip branch or unilateral active set')
            if vchange>report['velocity_change_allowance_sqrt_nmm'] or fchange>q['force_change_n']:raise ValueError('Observed correction exceeds allocation')
            if kh>1e-6*(2*p.toughness/p.strength) or ma>1e-8:raise ValueError('Correction changes history appreciably')
            if amp>1 and mismatch>q['resolved_prediction_rtol']:raise ValueError('Resolved correction disagrees with tangent')
        if report['residual_impulse_mass_norm_sqrt_nmm']>report['residual_impulse_allowance_sqrt_nmm'] or report['absolute_equation_work_n_mm']>report['equation_work_allowance_n_mm']:raise ValueError('Residual impulse/work exceeds allocation')
        report['passed']=True
    except (RuntimeError,ValueError,FloatingPointError,TimeoutError) as e:report['reason']=str(e)
    finally:
        p.set_primary_increment(x,du)
        if old_trial is not None:p.trial=old_trial
        np.testing.assert_array_equal(p.free,idx)
        for k,v in history.items():np.testing.assert_array_equal(getattr(p,k),v)
    return report


def try_stop(p,x,deadline):
    if not getattr(p,'nonlinear_error_policy',None) or getattr(p,'_friction_homotopy_factor',1.)!=1.:return False
    # Do not spend certificates on a physical nonlinear failure.
    g=p.evaluate(x,p.lag(x),tangent=False)[1]
    if np.sqrt(2)*np.linalg.norm(g[p.free])>1e-4:return False
    report=inspect(p,x,deadline);p.nonlinear_accuracy_attempt=report
    if report['passed']:p.nonlinear_accuracy_certificate=report;return True
    return False
