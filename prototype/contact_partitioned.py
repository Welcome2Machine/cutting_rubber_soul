"""Frozen-frame Tresca subproblems with a finite-strain nonlinear outer loop.

Inspired by IPC lagged friction and stabilized multiphysics splitting (R17-R20).
This is not IPC, LATIN or an implementation of the 2026 IPM-ANM algorithm.
The original geometry, force, cone and dissipation audits alone accept a step.
"""
import time
import numpy as np
from .initiation import ContactSolveError,ContactResolutionError,ContactBudgetError


def anderson_candidate(states, images, indices, scales, memory=5):
    """Fit interface residual secants; apply the same coefficients to all DOFs.

    SVD rank filtering removes dependent columns. No physical history is used.
    """
    if len(states)<2:return None
    x=np.array(states[-memory-1:]);g=np.array(images[-memory-1:])
    f=(g-x)[:,indices]*scales
    df=np.diff(f,axis=0).T
    gamma=np.linalg.lstsq(df,f[-1],rcond=1e-8)[0]
    # Small relaxed updates can require large coefficients even for a benign
    # affine map. Safeguard the resulting physical increment, not its basis-
    # dependent coefficients, in the outer line search.
    if not np.all(np.isfinite(gamma)):return None
    return g[-1]-np.diff(g,axis=0).T@gamma


def partitioned_step(p,previous,old_depth,depth,dt,old_state,*,initial=None,
                     initial_multipliers=None,max_outer=30,max_inner=30,
                     seconds=180.,acceleration=True,affine_normal=False,traction_compliance=None,initial_proximal=0.):
    """Bounded numerical solve, without committing time, slip or accepted state."""
    if p.surface_contact is None or p.options.volumetric_discretization!='quadratic_full':
        raise ValueError('Partitioned contact currently requires full-integration P2')
    if p.config.adhesive_shear_strength_n_mm2:
        raise ValueError('Partitioned contact currently supports nonadhesive Coulomb only')
    if max_outer<1 or max_inner<1 or not np.isfinite(seconds) or seconds<=0:
        raise ValueError('Invalid partitioned solver budget')
    c=p.config;start=time.monotonic();n=len(previous)
    if traction_compliance is None:traction_compliance=1/c.contact_surface_penalty_n_mm3
    if not np.isfinite(traction_compliance) or traction_compliance<0:
        raise ValueError('Traction compliance must be finite and nonnegative')
    if not np.isfinite(initial_proximal) or initial_proximal<0:raise ValueError('Invalid initial proximal scale')
    u=(previous if initial is None else initial).copy()
    lam=(old_state if initial_multipliers is None else initial_multipliers).copy()
    states=[];images=[];best=None;inner_total=0;accelerated=0;prox=initial_proximal
    boundary=(2*p.contact_nodes[:,None]+np.arange(2)).ravel()
    indices=np.r_[boundary,np.arange(n,n+lam.size)]
    scales=np.r_[np.ones(len(boundary)),
        np.tile([1/c.contact_surface_penalty_n_mm3,1/p.options.tangential_stiffness_mpa_per_mm],len(lam))]

    def evaluate(v,l):
        r,_,q,_=p.assemble(v,previous,depth,old_depth,dt,l[:,0],l[:,1],
                           with_tangent=False,roundoff_guard=False)
        force=float(np.linalg.norm(r[p.free]))
        f=np.column_stack((q['pressure'],q['shear']))-l
        merit=float(np.hypot(force/c.newton_tolerance_n,np.linalg.norm(f)/c.contact_multiplier_tolerance_n_mm2))
        return merit,force,float(np.max(abs(f)))

    def verify_original(v,l):
        nonlocal best
        merit,force,projection=evaluate(v,l)
        if best is None or merit<best[0]:best=(merit,v.copy(),l.copy())
        if force>c.newton_tolerance_n or projection>c.contact_multiplier_tolerance_n_mm2:return None
        try:
            result=p._mixed_step(previous,old_depth,depth,dt,old_state,initial=v,
                                 initial_multipliers=l,verify_only=True)
        except ContactSolveError:return None
        result[2]['original_accepted']=True
        return result

    reason='outer iteration budget'
    for outer in range(max_outer):
        merit,force,projection=evaluate(u,lam)
        if best is None or merit<best[0]:best=(merit,u.copy(),lam.copy())
        print(f'PARTITION outer={outer} depth={depth:.8g} original_force={force:.7g} original_projection={projection:.7g} merit={merit:.7g} proximal={prox:g}',flush=True)
        if force<=c.newton_tolerance_n and projection<=c.contact_multiplier_tolerance_n_mm2:
            try:
                result=p._mixed_step(previous,old_depth,depth,dt,old_state,initial=u,
                                     initial_multipliers=lam,verify_only=True)
            except ContactSolveError:pass
            else:
                result[2].update(contact_solver='finite-strain frozen-frame Tresca outer iteration',
                    contact_outer_iterations=outer,contact_inner_iterations=inner_total,
                    contact_accelerated_steps=accelerated,contact_proximal_scale=prox)
                return result
        if time.monotonic()-start>seconds:
            reason='wall budget between subproblems';break
        # Numerical proximal metric is a body stiffness scale, not a change to
        # material or an assertion that the full body Hessian is positive.
        _,k,q,_=p.assemble(u,previous,depth,old_depth,dt,lam[:,0],lam[:,1],mixed=True)
        diagonal=np.maximum(abs((k-q['global_K']).diagonal()),1.)
        frozen=p.surface_contact.freeze(u,depth,lam,affine_normal=affine_normal)
        frozen.update(traction_reference=lam.copy(),compliance=traction_compliance)
        solved=None
        for attempt in range(4):
            model=dict(contact=frozen,anchor=u.copy(),stiffness=prox*diagonal,deadline=start+seconds,
                       verify_original=verify_original)
            print(f'PARTITION INNER outer={outer} attempt={attempt} proximal={prox:g}',flush=True)
            try:
                solved=p._mixed_step(previous,old_depth,depth,dt,old_state,initial=u,
                    initial_multipliers=lam,subproblem=model,iteration_limit=max_inner)
            except (ContactSolveError,FloatingPointError) as exc:
                print(f'PARTITION INNER FAILED {exc}',flush=True)
                states.clear();images.clear()
                prox=1e-3 if prox==0 else prox*10
                if time.monotonic()-start>seconds:break
            else:break
            finally:inner_total+=model.get('iterations',0)
        if solved is None:
            reason='wall budget during inner solve' if time.monotonic()-start>seconds else 'frozen inner problem failed'
            break
        v,l,info=solved
        if info.get('original_accepted'):
            info.update(contact_solver='finite-strain stabilized Tresca (original equations accepted)',
                contact_outer_iterations=outer+1,contact_inner_iterations=inner_total,
                contact_accelerated_steps=accelerated,contact_proximal_scale=prox)
            return v,l,info
        x=np.r_[u,lam.ravel()];g=np.r_[v,l.ravel()]
        states.append(x);images.append(g)
        states=states[-6:];images=images[-6:]
        # Backtracked fixed-point updates remain numerical trials.
        # Check the original equations, including the moving contact frame.
        candidates=[(x+(.5**cut)*(g-x),False) for cut in range(8)]
        aa=anderson_candidate(states,images,indices,scales) if acceleration else None
        if aa is not None and np.all(np.isfinite(aa)):
            step=aa-x
            trial_size=np.linalg.norm(step[indices]*scales)
            map_size=np.linalg.norm((g-x)[indices]*scales)
            if trial_size>10*map_size:step*=10*map_size/trial_size
            candidates.extend((x+(.5**cut)*step,True) for cut in range(8))
        ranked=[]
        for z,is_aa in candidates:
            v=z[:n].copy();v[p.fixed]=previous[p.fixed]
            l=z[n:].reshape(lam.shape).copy()
            # Inner Tresca forces need not satisfy the UPDATED Coulomb bound.
            # Clipping them here destroys their solved body force balance.
            # Original projection residual and final acceptance police the law.
            try:score=evaluate(v,l)[0]
            except FloatingPointError:continue
            if np.isfinite(score):ranked.append((score,v,l,is_aa))
            if is_aa:print(f'PARTITION ANDERSON original_merit={score:.7g}',flush=True)
        if not ranked:
            reason='all outer candidates have invalid deformation';break
        score,v,l,is_aa=min(ranked,key=lambda item:item[0])
        print(f'PARTITION UPDATE original_merit={score:.7g} accelerated={is_aa}',flush=True)
        # A changed metric defines a different fixed-point map. Clear only
        # acceleration history; the physical previous state is always fixed.
        if score>2*merit:
            prox=max(1e-3,prox*10);states.clear();images.clear()
            print('PARTITION REJECT outer growth; retry same anchor with stronger proximal term',flush=True)
            continue
        changed=np.count_nonzero((lam[:,0]>0)!=(l[:,0]>0))
        u,lam=v,l;accelerated+=int(is_aa)
        if changed>max(4,.1*len(lam)):states.clear();images.clear()
        if score<best[0]:best=(score,u.copy(),lam.copy())
    _,u,lam=best
    _,force,projection=evaluate(u,lam)
    if force<=c.newton_tolerance_n and projection<=c.contact_multiplier_tolerance_n_mm2:
        result=p._mixed_step(previous,old_depth,depth,dt,old_state,initial=u,
                             initial_multipliers=lam,verify_only=True)
        result[2].update(contact_solver='finite-strain frozen-frame Tresca outer iteration',
            contact_outer_iterations=outer+1,contact_inner_iterations=inner_total,
            contact_accelerated_steps=accelerated,contact_proximal_scale=prox)
        return result
    error_type=ContactBudgetError if 'budget' in reason else ContactSolveError
    raise error_type(f'Partitioned {reason}: depth={depth:.8g}, residual={force:.7g} N, projection={projection:.7g} MPa',u,lam)


def path_step(p,previous,old_depth,depth,dt,old_state,*,initial=None,
              initial_multipliers=None,seconds=180.,max_steps=100,path_step_size=.05,
              path_metric_ratio=.1,auxiliary_tolerance=1e-6,switch_events=True,resume=None):
    """Track R(x)-(1-a)R(x0)=0 at FIXED physical depth/time/history.

    This is residual homotopy with pseudo-arclength, not an IPM central path.
    All contact multipliers remain unknowns, including initially open nodes.
    The initial residual vector disappears at a=1; the original equations and
    geometry, never the auxiliary residual, alone permit physical acceptance.
    """
    from scipy.sparse import bmat,diags
    from scipy.sparse.linalg import splu
    from .continuation import follow_path,PathFailure
    if p.surface_contact is None or p.options.volumetric_discretization!='quadratic_full':
        raise ValueError('Auxiliary path currently requires full-integration P2 surface contact')
    if p.config.adhesive_shear_strength_n_mm2:
        raise ValueError('Residual-path switching currently supports nonadhesive Coulomb only')
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('Invalid path time budget')
    if not np.isfinite(path_metric_ratio) or path_metric_ratio<=0:raise ValueError('Invalid path metric ratio')
    start=time.monotonic();c=p.config;n=len(p.free)
    u0=(previous if initial is None else initial).copy()
    l0=(old_state if initial_multipliers is None else initial_multipliers).copy()
    if u0.shape!=previous.shape or l0.shape!=old_state.shape or not np.all(np.isfinite(np.r_[u0,l0.ravel()])):
        raise ValueError('Invalid auxiliary path initial state')
    x0=np.r_[u0[p.free],l0.ravel()]
    scale_r=np.r_[np.full(n,1/c.newton_tolerance_n),
                  np.full(l0.size,1/c.contact_multiplier_tolerance_n_mm2)]
    def unpack(x):
        u=previous.copy();u[p.free]=x[:n]
        return u,x[n:].reshape(l0.shape).copy()
    best=None
    def original(x,tangent):
        nonlocal best
        u,l=unpack(x)
        r,k,q,_=p.assemble(u,previous,depth,old_depth,dt,l[:,0],l[:,1],
            mixed=True,with_tangent=tangent,formulation='alart_curnier',roundoff_guard=False)
        f=np.r_[r[p.free],(np.c_[q['pressure'],q['shear']]-l).ravel()]*scale_r
        merit=float(np.linalg.norm(f))
        if np.isfinite(merit) and (best is None or merit<best[0]):best=(merit,u.copy(),l.copy())
        j=None
        if tangent:
            j=diags(scale_r)@bmat([[k[p.free][:,p.free],q['global_B'][p.free]],
                [q['global_C'][:,p.free],q['global_D']]],format='csc')
        return f,j
    f0,j0=original(x0,True)
    # Column equilibration followed by ONE fixed global scale makes the
    # initial state tangent and auxiliary parameter comparable in arc length.
    # Recomputing this metric along the path would change its tangent equation.
    scale_x=1/np.maximum(np.sqrt(np.asarray(j0.power(2).sum(axis=0))).ravel(),1e-20)
    try:direction=splu(j0.tocsc()).solve(-f0)
    except RuntimeError as exc:raise ContactSolveError('Singular initial residual-path tangent',u0,l0) from exc
    scale_x*=max(np.linalg.norm(direction/scale_x),1.)*path_metric_ratio
    print(f'PATH SCALE min={scale_x.min():.7g} max={scale_x.max():.7g} initial_merit={np.linalg.norm(f0):.7g}',flush=True)
    counts=dict(points=0,turns=0,last_sign=1)
    def evaluate(z,tangent):
        f,j=original(x0+scale_x*z[:-1],tangent)
        return f-(1-z[-1])*f0,(j@diags(scale_x) if tangent else None),f0
    def accept(z):
        x=x0+scale_x*z[:-1];u,l=unpack(x)
        f,_=original(x,False)
        force=np.linalg.norm(f[:n]);projection=np.max(abs(f[n:]))
        print(f'PATH ORIGINAL parameter={z[-1]:.8g} force={force*c.newton_tolerance_n:.7g} projection={projection*c.contact_multiplier_tolerance_n_mm2:.7g}',flush=True)
        if force>1 or projection>1:return None
        try:result=p._mixed_step(previous,old_depth,depth,dt,old_state,
                    initial=u,initial_multipliers=l,verify_only=True)
        except ContactSolveError:return None
        result[2].update(contact_solver='fixed-history residual homotopy pseudo-arclength',
            auxiliary_parameter=float(z[-1]),auxiliary_points=counts['points'],auxiliary_turns=counts['turns'])
        return result
    def observer(entry):
        print('PATH '+str(entry),flush=True)
        if entry['event']=='point':
            counts['points']+=1
            sign=int(np.sign(entry['tangent_parameter']))
            if sign and sign!=counts['last_sign']:counts['turns']+=1;counts['last_sign']=sign
    def contact_at(z,tangent=False):
        u,l=unpack(x0+scale_x*z[:-1])
        q=p.surface_contact.evaluate(u,previous,depth,old_depth,l,with_tangent=tangent,
            formulation='alart_curnier',roundoff_guard=False)
        trial=l[:,1]+p.options.tangential_stiffness_mpa_per_mm*q['slip']
        margin=abs(trial)-c.friction_coefficient*q['pressure']
        q['normal_trial']=l[:,0]+c.contact_surface_penalty_n_mm3*q['gap']
        return q,trial,margin
    consumed=None
    def event(a,b):
        from scipy.optimize import brentq
        qa,ta,ha=contact_at(a);qb,tb,hb=contact_at(b)
        pa,pb=qa['normal_trial'],qb['normal_trial']
        normal_crossing=(pa*pb<0)|((pa==0)&(pb!=0))
        friction_crossing=(qa['pressure']>1e-8)&(qb['pressure']>1e-8)&(ta*tb>0)&(ha*hb<0)
        hits=[]
        for kind,mask in (('normal',normal_crossing),('friction',friction_crossing)):
            for i in np.flatnonzero(mask):
                # Suppress only the event just consumed at this EXACT point;
                # a pressure threshold would hide nearly open contact again.
                if consumed is not None and consumed[:2]==(kind,int(i)) and np.array_equal(a,consumed[2]):continue
                def value(s):
                    q,_,h=contact_at(a+s*(b-a))
                    return q['normal_trial'][i] if kind=='normal' else h[i]
                fraction=brentq(value,0.,1.,xtol=1e-12)
                hits.append((fraction,0 if kind=='normal' else 1,int(i)))
        if not hits:return None
        fraction,mode,i=min(hits);kind='normal' if mode==0 else 'friction'
        sign=float(np.sign(ta[i]));outgoing=1 if (pb[i] if mode==0 else hb[i])>0 else -1
        def equation(z):
            q,trial,margin=contact_at(z,True)
            if mode==1 and (q['pressure'][i]<=1e-8 or sign*trial[i]<=0):
                raise FloatingPointError('Friction event left its compressive fixed-sign branch')
            kinetic=q['global_kinematic_u'][:,p.free]
            normal=kinetic[2*i].toarray().ravel()
            if mode==0:
                value=q['normal_trial'][i]
                derivative=np.r_[c.contact_surface_penalty_n_mm3*normal,np.zeros(l0.size)]
                derivative[n+2*i]=1.
            else:
                value=margin[i];slip=kinetic[2*i+1].toarray().ravel()
                derivative=np.r_[sign*p.options.tangential_stiffness_mpa_per_mm*slip-
                    c.friction_coefficient*c.contact_surface_penalty_n_mm3*normal,np.zeros(l0.size)]
                derivative[n+2*i]=-c.friction_coefficient;derivative[n+2*i+1]=sign
            return value/c.contact_multiplier_tolerance_n_mm2,np.r_[derivative*scale_x/c.contact_multiplier_tolerance_n_mm2,0.]
        def side_jacobian(z,j):
            q,trial,_=contact_at(z)
            if mode==0:
                if c.friction_coefficient>0 and (abs(trial[i])<1e-10 or q['stick'][i]):
                    raise RuntimeError('Simultaneous normal/friction apex needs a multi-branch tangent')
                delta=int(outgoing>0)-int(q['pressure'][i]>0)
                if not delta:return j
                gradient=equation(z)[1][:-1];result=j.tolil(copy=True);row=n+2*i
                result[row]=result[row]+delta*gradient
                # The changing normal return also changes the sliding bound.
                result[row+1]=result[row+1]+delta*c.friction_coefficient*np.sign(trial[i])*gradient
                return result.tocsc()
            desired_stick=outgoing<0
            if bool(q['stick'][i])==desired_stick:return j
            gradient=equation(z)[1][:-1]
            result=j.tolil(copy=True);row=n+2*i+1
            result[row]=result[row]+(sign if desired_stick else -sign)*gradient
            return result.tocsc()
        def on_accept(z):
            nonlocal consumed
            consumed=(kind,i,z.copy())
        return dict(fraction=fraction,equation=equation,jacobian=side_jacobian,
                    direction=outgoing,label=f'{kind} node {int(p.contact_nodes[i])}',on_accept=on_accept)
    start_point=np.zeros(len(x0)+1);orientation=None
    if resume is not None:
        start_point=np.r_[(np.r_[resume['displacement'][p.free],resume['multipliers'].ravel()]-x0)/scale_x,resume['parameter']]
        orientation=resume.get('tangent')
        if orientation is None:
            orientation=np.zeros_like(start_point);orientation[-1]=resume['parameter_direction']
        print(f'PATH RESUME parameter={start_point[-1]:.9g}; original residual source and physical history unchanged',flush=True)
    try:
        _,result=follow_path(evaluate,start_point,seconds=max(1e-9,seconds-(time.monotonic()-start)),
            max_steps=max_steps,step=path_step_size,tolerance=auxiliary_tolerance,accept=accept,observer=observer,
            event=event if switch_events else None,initial_tangent=orientation)
    except PathFailure as exc:
        last_x=x0+scale_x*exc.point[:-1]
        last_f,_=original(last_x,False)
        _,u,l=best
        kind=ContactBudgetError if exc.budget else ContactSolveError
        error=kind(f'{exc}; parameter={exc.point[-1]:.8g}; original_best_merit={best[0]:.7g}',u,l)
        pu,pl=unpack(last_x)
        error.path_state=dict(auxiliary_displacement=pu,auxiliary_multipliers=pl,
            auxiliary_parameter=float(exc.point[-1]))
        if exc.tangent is not None:error.path_state['auxiliary_tangent']=exc.tangent
        error.path_diagnostics=dict(points=counts['points'],turns=counts['turns'],
            last_parameter=float(exc.point[-1]),scale_min=float(scale_x.min()),scale_max=float(scale_x.max()),
            last_auxiliary_residual=float(np.linalg.norm(last_f-(1-exc.point[-1])*f0)))
        raise error from exc
    return result


def load_path_resume(folder,p,previous,history,old,depth,dt,trial,metric_ratio):
    """Restore numerical continuation only; reject changed physical history."""
    import ast,json
    from pathlib import Path
    folder=Path(folder)
    report=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    if report['config']!=p.config.to_dict() or report['options']!=vars(p.options):
        raise ValueError('Resume physical configuration differs')
    if report['method']!='path' or report['path_metric_ratio']!=metric_ratio:
        raise ValueError('Resume path definition or metric differs')
    if trial is None or Path(report['trial']).resolve()!=Path(trial).resolve():
        raise ValueError('Resume must retain the original residual source trial')
    for key,value in (('old_depth_mm',old),('depth_mm',depth),('dt_s',dt)):
        if not np.isclose(report[key],value,rtol=0,atol=1e-12):raise ValueError('Resume physical increment differs')
    with np.load(folder/'trial.npz') as z:
        for key,value in (('previous',previous),('previous_multipliers',history),
            ('reference_xy',p.mesh.reference_xy),('triangles',p.mesh.triangles),('contact_nodes',p.contact_nodes)):
            if key not in z or not np.array_equal(z[key],value):raise ValueError(f'Resume {key} differs')
        result=dict(displacement=z['auxiliary_displacement'].copy(),multipliers=z['auxiliary_multipliers'].copy(),
                    parameter=float(z['auxiliary_parameter']))
        if 'auxiliary_tangent' in z:result['tangent']=z['auxiliary_tangent'].copy()
    if 'tangent' not in result:
        # Legacy checkpoints saved only the tangent parameter in the log.
        # A nonzero component fixes orientation; the full tangent is rebuilt.
        points=[ast.literal_eval(line[5:]) for line in (folder/'run.log').read_text(encoding='utf-8').splitlines() if line.startswith('PATH {')]
        last=next(entry for entry in reversed(points) if entry['event']=='point')
        if abs(last['tangent_parameter'])<1e-12:raise ValueError('Cannot reconstruct legacy tangent at a fold')
        result['parameter_direction']=float(np.sign(last['tangent_parameter']))
    return result


def main():
    """Reproduce one physical increment from an unchanged checkpoint."""
    import argparse
    import contextlib
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from run_support import write_json,save_npz,source_hashes
    from .contact_study import load_failed_problem
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--trial',type=Path)
    parser.add_argument('--resume-path',type=Path,help='Continue a saved auxiliary path with its original residual source and physical history')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--depth',type=float,default=.5)
    parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--method',choices=('partitioned','path'),default='partitioned')
    parser.add_argument('--path-steps',type=int,default=100)
    parser.add_argument('--path-step-size',type=float,default=.05)
    parser.add_argument('--path-metric-ratio',type=float,default=.1)
    parser.add_argument('--auxiliary-tolerance',type=float,default=1e-6)
    parser.add_argument('--no-path-switch-events',action='store_true')
    parser.add_argument('--no-acceleration',action='store_true')
    parser.add_argument('--affine-normal',action='store_true',help='Control: also freeze the normal geometry; default keeps exact normal contact')
    parser.add_argument('--traction-compliance',type=float,help='Numerical interface compliance mm/MPa; default 1/normal augmentation, zero disables it')
    parser.add_argument('--initial-proximal',type=float,default=0.,help='Initial body diagonal metric scale; numerical only')
    args=parser.parse_args();out=args.output
    if args.threads<1:parser.error('--threads must be positive')
    if args.resume_path and args.method!='path':parser.error('--resume-path requires --method path')
    if out.exists() and any(out.iterdir()):raise ValueError('Probe output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    p,previous,history,old=load_failed_problem(args.source)
    initial=previous.copy();lam=history.copy()
    if args.trial:
        with np.load(args.trial) as z:
            initial=z['displacement'].copy();lam=z['contact_multipliers'].copy()
            if initial.shape!=previous.shape or lam.shape!=history.shape:
                raise ValueError('Trial mesh/state shape mismatch')
            for name,actual in (('reference_xy',p.mesh.reference_xy),('triangles',p.mesh.triangles),('contact_nodes',p.contact_nodes)):
                if name not in z or not np.array_equal(z[name],actual):
                    raise ValueError(f'Trial {name} differs from the physical history mesh')
            if not np.isclose(float(z['indentation_mm']),args.depth,rtol=0,atol=1e-12):
                raise ValueError('Trial target differs from the requested depth')
    dt=(args.depth-old)/p.config.speed_mm_s
    if dt<=0:raise ValueError('Probe must advance the known physical history')
    resume=None if args.resume_path is None else load_path_resume(args.resume_path,p,previous,history,old,args.depth,dt,args.trial,args.path_metric_ratio)
    report=dict(completed=False,status='running',source=str(args.source),trial=str(args.trial),
        config=p.config.to_dict(),options=vars(p.options),source_sha256=source_hashes(snapshot=out/'source'),
        old_depth_mm=old,depth_mm=args.depth,dt_s=dt,seconds=args.seconds,
        acceleration=not args.no_acceleration,affine_normal=args.affine_normal,
        traction_compliance=(1/p.config.contact_surface_penalty_n_mm3 if args.traction_compliance is None else args.traction_compliance),
        initial_proximal=args.initial_proximal,threads=args.threads,method=args.method,
        path_steps=args.path_steps,path_step_size=args.path_step_size,
        path_metric_ratio=args.path_metric_ratio,auxiliary_tolerance=args.auxiliary_tolerance,
        path_switch_events=not args.no_path_switch_events,resume_path=str(args.resume_path) if args.resume_path else None)
    write_json(out/'result.json',report);start=time.monotonic();path_state={}
    with (out/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),threadpool_limits(limits=args.threads):
        try:
            if args.method=='path':
                u,lam,d=path_step(p,previous,old,args.depth,dt,history,initial=initial,
                    initial_multipliers=lam,seconds=args.seconds,max_steps=args.path_steps,path_step_size=args.path_step_size,
                    path_metric_ratio=args.path_metric_ratio,auxiliary_tolerance=args.auxiliary_tolerance,
                    switch_events=not args.no_path_switch_events,resume=resume)
            else:
                u,lam,d=partitioned_step(p,previous,old,args.depth,dt,history,initial=initial,
                    initial_multipliers=lam,seconds=args.seconds,acceleration=not args.no_acceleration,affine_normal=args.affine_normal,
                    traction_compliance=args.traction_compliance,initial_proximal=args.initial_proximal)
        except ContactSolveError as exc:
            u,lam=exc.u,exc.multipliers
            report.update(status='budget_exhausted' if isinstance(exc,ContactBudgetError) else 'failed',error=str(exc))
            if hasattr(exc,'path_state'):
                path_state=exc.path_state;report['path_diagnostics']=exc.path_diagnostics
        except ContactResolutionError as exc:
            if exc.trial is None:raise
            u,lam,_=exc.trial;report.update(status='geometry_rejected',error=str(exc))
        except Exception as exc:
            report.update(status='error',error=f'{type(exc).__name__}: {exc}')
            write_json(out/'result.json',report)
            raise
        else:report.update(completed=True,status='completed',diagnostics=d)
        r,_,q,d=p.assemble(u,previous,args.depth,old,dt,lam[:,0],lam[:,1],with_tangent=False,roundoff_guard=False)
        report.update(elapsed_s=time.monotonic()-start,residual_n=float(np.linalg.norm(r[p.free])),
            projection_mpa=float(np.max(abs(np.c_[q['pressure'],q['shear']]-lam))),
            min_j=d['min_j'],max_edge_penetration_mm=p.edge_penetration(u,args.depth))
        save_npz(out/'trial.npz',displacement=u,contact_multipliers=lam,
                 previous=previous,previous_multipliers=history,old_depth=old,depth=args.depth,dt=dt,
                 indentation_mm=args.depth,reference_xy=p.mesh.reference_xy,
                 triangles=p.mesh.triangles,contact_nodes=p.contact_nodes,**path_state)
        write_json(out/'result.json',report)
    print(report['status'],report.get('error',''),flush=True)


if __name__=='__main__':main()
