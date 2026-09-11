"""Bounded search of linearized Signorini/Coulomb contact branches.

The condensed kinematics satisfy (gap, slip) = offset + mobility * traction.
Every returned state is only a Newton predictor, never an accepted time step.
Opening is a branch of the contact law, not removal of material or a crack.
"""
import itertools
import numpy as np
from scipy.sparse import coo_matrix,bmat,diags
from scipy.sparse.linalg import splu


def affine_branch_milp(mobility,offset,coefficient,*,pressure_bound,
                       kinematic_lower,kinematic_upper,time_limit=30.,fixed_modes=None):
    """Diagnostic of ALL affine 2D Coulomb branches in explicit finite bounds.

    This is not a nonlinear equilibrium solver or a physical existence test.
    Tractions are scaled by pressure_bound, kinematics by their bounds. Four
    binary labels per node encode open, stick and both slip signs exactly.
    The result is independently checked, including after a MILP time limit.
    """
    from scipy.optimize import milp,Bounds,LinearConstraint
    offset=np.asarray(offset,dtype=float);mobility=np.asarray(mobility,dtype=float)
    lower=np.asarray(kinematic_lower,dtype=float);upper=np.asarray(kinematic_upper,dtype=float)
    n=offset.size//2
    if (offset.shape!=(2*n,) or n<1 or mobility.shape!=(2*n,2*n)
            or lower.shape!=offset.shape or upper.shape!=offset.shape
            or not all(np.all(np.isfinite(a)) for a in (offset,mobility,lower,upper))
            or np.any(lower>=upper) or not np.isfinite(pressure_bound) or pressure_bound<=0
            or not np.isfinite(coefficient) or coefficient<0 or not np.isfinite(time_limit) or time_limit<=0):
        raise ValueError('Finite compatible affine data and positive bounds/budget required')
    # x = [traction/P, kinematics/S, four one-hot labels per node].
    scale=np.maximum(np.maximum(abs(lower),abs(upper)),1e-8)
    size=8*n;lb=np.r_[np.tile([0.,-coefficient],n),lower/scale,np.zeros(4*n)]
    ub=np.r_[np.tile([1.,coefficient],n),upper/scale,np.ones(4*n)]
    for i,mode in (fixed_modes or {}).items():
        if not isinstance(i,(int,np.integer)) or not 0<=i<n or mode not in (2,0,1,-1):
            raise ValueError('Invalid fixed contact label')
        start=4*n+4*i;lb[start:start+4]=ub[start:start+4]=0.
        lb[start+(2,0,1,-1).index(mode)]=ub[start+(2,0,1,-1).index(mode)]=1.
    rows=[];cols=[];values=[];lo=[];hi=[]
    def row(entries,a=-np.inf,b=np.inf):
        j=len(lo)
        for i,v in entries.items():
            if v:rows.append(j);cols.append(i);values.append(v)
        lo.append(a);hi.append(b)
    for j in range(2*n):
        entries={i:float(-pressure_bound*mobility[j,i]/scale[j]) for i in range(2*n)}
        entries[2*n+j]=1.;rhs=offset[j]/scale[j]
        magnitude=max(1.,abs(rhs),max(abs(v) for v in entries.values()))
        row({i:v/magnitude for i,v in entries.items()},rhs/magnitude,rhs/magnitude)
    for i in range(n):
        p,t=2*i,2*i+1;g,s=2*n+p,2*n+t;o,st,ps,ns=4*n+4*i+np.arange(4)
        gl=lower[p]/scale[p];sl=lower[t]/scale[t];su=upper[t]/scale[t]
        row({o:1,st:1,ps:1,ns:1},1,1)
        row({p:1,o:1},b=1)                    # open => p=tau=0
        row({t:1,p:-coefficient},b=0)          # exact 2D Coulomb cone
        row({t:-1,p:-coefficient},b=0)
        row({g:1},b=0)
        row({g:1,o:-gl},a=0)                  # closed => gap=0
        row({s:1,st:sl},a=sl)                 # stick => slip=0
        row({s:1,st:su},b=su)
        row({s:1,ps:sl},a=sl)                 # positive sliding
        row({s:1,ns:su},b=su)                 # negative sliding
        row({t:1,p:-coefficient,ps:-2*coefficient},a=-2*coefficient)
        row({t:1,p:coefficient,ns:2*coefficient},b=2*coefficient)
    matrix=coo_matrix((values,(rows,cols)),shape=(len(lo),size)).tocsc()
    result=milp(np.zeros(size),integrality=np.r_[np.zeros(4*n),np.ones(4*n)],
        bounds=Bounds(lb,ub),constraints=LinearConstraint(matrix,lo,hi),
        options=dict(time_limit=float(time_limit),mip_rel_gap=0.))
    report=dict(status=int(result.status),message=result.message,verified=False,
        pressure_bound_mpa=float(pressure_bound),time_limit_s=float(time_limit),contacts=n,
        fixed_modes_count=len(fixed_modes or {}),
        scope='Bounded frozen affine system only; infeasibility is not nonlinear or physical nonexistence')
    if result.x is None:return None,None,report
    traction=(result.x[:2*n]*pressure_bound).reshape(n,2)
    z=(offset+mobility@traction.ravel()).reshape(n,2)
    binary=result.x[4*n:].reshape(n,4);labels=np.array([2,0,1,-1])[binary.argmax(axis=1)]
    opened=labels==2;stick=labels==0;slide=abs(labels)==1
    maximum=lambda a:float(np.max(a,initial=0.))
    gap_error=max(maximum(z[:,0]),maximum(abs(z[~opened,0])))
    slip_error=max(maximum(abs(z[stick,1])),maximum(-labels[slide]*z[slide,1]))
    traction_error=max(maximum(-traction[:,0]),maximum(traction[:,0]-pressure_bound),
        maximum(abs(traction[:,1])-coefficient*traction[:,0]),maximum(abs(traction[opened])),
        maximum(abs(traction[slide,1]-labels[slide]*coefficient*traction[slide,0])))
    bound_error=max(maximum(lower-z.ravel()),maximum(z.ravel()-upper))
    report.update(gap_error_mm=gap_error,slip_error_mm=slip_error,traction_error_mpa=traction_error,
        kinematic_bound_error_mm=bound_error,
        verified=bool(max(gap_error,slip_error,bound_error)<=1e-8 and traction_error<=1e-6
                      and maximum(abs(binary-np.round(binary)))<=1e-6))
    return (traction,labels,report) if report['verified'] else (None,None,report)


def nearby_modes(choices):
    """Visit label combinations by the number of changes from the warm start."""
    base=tuple(values[0] for values in choices)
    variable=[i for i,values in enumerate(choices) if len(values)>1]
    yield base
    for count in range(1,len(variable)+1):
        for changed in itertools.combinations(variable,count):
            for alternatives in itertools.product(*(choices[i][1:] for i in changed)):
                labels=list(base)
                for i,value in zip(changed,alternatives):labels[i]=value
                yield tuple(labels)


def coulomb_switch_distance(lam,gap,slip,normal_stiffness,tangent_stiffness,coefficient,*,independent=False):
    """Absolute switching-function values in MPa for the chosen projection.

    Zero slip alone is not a stick/slip switching surface. A contact may be
    arbitrarily close to switching with nonzero slip if its current traction
    nearly cancels that slip in the trial return map. In the open region only
    the normal opening boundary matters; tangential clipping there is inactive.
    """
    normal_trial=lam[:,0]+normal_stiffness*gap
    tangent_trial=lam[:,1]+tangent_stiffness*slip
    capacity=np.maximum(lam[:,0] if independent else normal_trial,0)
    tangent_distance=np.where((normal_trial>0)&(coefficient>0),
        abs(abs(tangent_trial)-coefficient*capacity),np.inf)
    return np.minimum(abs(normal_trial),tangent_distance)


def admissible_branches(mobility,offset,coefficient,choices,*,budget=262144):
    """Enumerate at most budget affine subproblems; yield feasible tractions."""
    count=len(choices)
    if coefficient < 0 or budget < 1:
        raise ValueError('Invalid branch search parameters')
    for examined,modes in enumerate(itertools.islice(nearby_modes(choices),budget),1):
        matrix=mobility.copy();rhs=-offset.copy()
        for i,mode in enumerate(modes):
            if mode==2:  # open: both tractions vanish
                matrix[2*i:2*i+2]=0
                matrix[2*i,2*i]=matrix[2*i+1,2*i+1]=1
                rhs[2*i:2*i+2]=0
            elif mode:  # slip: tau = sign(slip) * mu * p
                matrix[2*i+1]=0
                matrix[2*i+1,2*i]=-mode*coefficient
                matrix[2*i+1,2*i+1]=1;rhs[2*i+1]=0
        try:
            traction=np.linalg.solve(matrix,rhs).reshape(count,2)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(traction)) or np.any(traction[:,0]<-1e-8):
            continue
        kinematics=(offset+mobility@traction.ravel()).reshape(count,2)
        modes=np.asarray(modes);opened=modes==2;stick=modes==0;sliding=~(opened|stick)
        if np.any(kinematics[opened,0]>1e-10):continue
        if np.any(np.abs(kinematics[~opened,0])>1e-10):continue
        if np.any(np.abs(kinematics[stick,1])>1e-10):continue
        if np.any(np.abs(traction[stick,1])>coefficient*traction[stick,0]+1e-8):continue
        if np.any(modes[sliding]*kinematics[sliding,1]<-1e-10):continue
        yield traction,modes,examined


def branch_predictors(problem,previous,old_depth,depth,dt,u,multipliers,*,max_predictors=8):
    """Keep physical history fixed; search at most nine uncertain contacts.

All participating nodes remain in the condensed system. For larger contact
patches, only the nine nodes nearest opening/stick-slip switches are enumerated;
other branch labels are held at their current prediction. Failure to find a
predictor is not evidence that the nonlinear physical problem has no solution.
"""
    p=problem;mu=p.config.friction_coefficient
    if p.surface_contact is not None:
        yield from integral_branch_predictors(p,previous,old_depth,depth,dt,u,multipliers,max_predictors=max_predictors)
        return
    if p.config.adhesive_shear_strength_n_mm2:
        return  # this fallback derives the non-adhesive Coulomb cone
    r,k,q,_=p.assemble(u,previous,depth,old_depth,dt,multipliers[:,0],multipliers[:,1],mixed=True)
    if np.linalg.norm(r[p.free])>10*p.config.newton_tolerance_n:
        print('BRANCH SEARCH skipped: trial is far from force equilibrium; use load bisection',flush=True)
        return
    selected=np.flatnonzero((q['pressure']>0)|np.any(multipliers!=0,axis=1))
    count=len(selected)
    if not count:return
    if count>32:
        print(f'BRANCH SEARCH skipped: {count} contacts exceed dense fallback limit 32',flush=True)
        return
    mdofs=2*np.arange(count)[:,None]+np.arange(2);shape=(count,2,2)
    rows_u=np.broadcast_to(p.cdofs[selected,:,None],shape).ravel()
    cols_m=np.broadcast_to(mdofs[:,None,:],shape).ravel()
    b=coo_matrix((-q['dforce_dlambda'][selected].ravel(),(rows_u,cols_m)),
                 shape=(len(u),2*count)).tocsr()[p.free].toarray()
    gradient=np.stack((-q['normal'],q['slip_derivative']),axis=1)[selected]
    rows_m=np.broadcast_to(mdofs[:,:,None],shape).ravel()
    cols_u=np.broadcast_to(p.cdofs[selected,None,:],shape).ravel()
    c=coo_matrix((gradient.ravel(),(rows_m,cols_u)),shape=(2*count,len(u))).tocsr()[:,p.free]
    lu=splu(k[p.free][:,p.free].tocsc())
    free_correction=-lu.solve(r[p.free]);response=-lu.solve(b)
    mobility=c@response
    offset=(np.column_stack((q['gap'],q['slip']))[selected].ravel()
            +c@free_correction-mobility@multipliers[selected].ravel())
    pressure=multipliers[selected,0];shear=multipliers[selected,1]
    margin=np.abs(mu*pressure-np.abs(shear))
    opening=np.abs(pressure)
    uncertain=set(np.argsort(np.minimum(opening,margin),kind='stable')[:9])
    modes=np.where(q['stick'][selected],0,np.where(q['slip'][selected]>=0,1,-1))
    modes=np.where(q['pressure'][selected]>0,modes,2)
    choices=[(int(modes[i]),)+tuple(mode for mode in (0,1,-1,2) if mode!=modes[i])
             if i in uncertain else (int(modes[i]),) for i in range(count)]
    print(f'BRANCH SEARCH contacts={count} enumerated={len(uncertain)} budget=262144',flush=True)
    for found,(traction,labels,examined) in enumerate(admissible_branches(mobility,offset,mu,choices),1):
        correction=free_correction+response@(traction-multipliers[selected]).ravel()
        for cut in range(12):
            fraction=.5**cut;seed=u.copy();warm=multipliers.copy()
            seed[p.free]+=fraction*correction
            warm[selected]+=fraction*(traction-multipliers[selected])
            try:
                p.assemble(seed,previous,depth,old_depth,dt,warm[:,0],warm[:,1],mixed=True,with_tangent=False)
                break
            except FloatingPointError:
                continue
        else:
            print('BRANCH PREDICTOR discarded: no admissible deformation along predictor',flush=True)
            if found>=max_predictors:return
            continue
        print(f'BRANCH PREDICTOR {found} examined={examined} fraction={fraction:g} open_nodes={p.contact_nodes[selected[labels==2]].tolist()}',flush=True)
        yield seed,warm,examined
        if found>=max_predictors:return


def integral_branch_predictors(p,previous,old_depth,depth,dt,u,lam,*,max_predictors=8):
    """Enumerate six uncertain contacts after sparse elimination of the rest.

    Most contact modes and all body DOFs stay in the sparse eliminated system.
    Only a 12-by-12 affine complementarity problem is enumerated. A predictor
    may cross a merit-function barrier; final nonlinear acceptance is unchanged.
    """
    mu=p.config.friction_coefficient
    if p.config.adhesive_shear_strength_n_mm2:return
    r,k,q,_=p.assemble(u,previous,depth,old_depth,dt,lam[:,0],lam[:,1],mixed=True)
    if 'pressure_coupling' in q or np.linalg.norm(r[p.free])>10*p.config.newton_tolerance_n:return
    selected=np.flatnonzero((q['pressure']>0)|np.any(lam!=0,axis=1));count=len(selected)
    if not count:return
    chosen=(2*selected[:,None]+np.arange(2)).ravel();n=len(p.free)
    B=q['global_B'][p.free][:,chosen];C=q['global_kinematic_u'][chosen][:,p.free]
    z=np.column_stack((q['gap'],q['slip']))[selected]
    current=lam[selected]
    modes=np.where(q['stick'][selected],0,np.where(q['slip'][selected]>=0,1,-1))
    modes=np.where(q['pressure'][selected]>0,modes,2)
    defect=np.max(abs(np.column_stack((q['pressure'],q['shear']))[selected]-current),axis=1)
    switch=coulomb_switch_distance(current,z[:,0],z[:,1],
        p.config.contact_surface_penalty_n_mm3,p.options.tangential_stiffness_mpa_per_mm,mu,
        independent=p.options.coulomb_formulation=='jean_moreau')
    nearest=np.argsort(switch,kind='stable')[:6]
    groups=[np.argsort(-defect,kind='stable')[:6],
        np.unique(np.r_[np.argsort(-defect,kind='stable')[:3],nearest[:3]]),nearest]
    seen=set();found=0
    for uncertain in groups:
        key=tuple(sorted(uncertain))
        if key in seen:continue
        seen.add(key);fixed=np.setdiff1d(np.arange(count),uncertain)
        ui=(2*uncertain[:,None]+np.arange(2)).ravel();fi=(2*fixed[:,None]+np.arange(2)).ravel()
        mask=np.ones((count,2));equation=z.copy();D=np.zeros((count,2,2))
        for i,mode in enumerate(modes):
            if mode==2:
                mask[i]=0;D[i]=np.eye(2);equation[i]=current[i]
            elif mode:
                mask[i,1]=0;D[i,1]=[-mode*mu,1.];equation[i,1]=current[i,1]-mode*mu*current[i,0]
        scale=np.where(mask>0,np.array([p.config.contact_surface_penalty_n_mm3,p.options.tangential_stiffness_mpa_per_mm]),1.)
        rows=np.broadcast_to(np.arange(2*count).reshape(count,2,1),D.shape).ravel()
        cols=np.broadcast_to(np.arange(2*count).reshape(count,1,2),D.shape).ravel()
        dm=coo_matrix(((D*scale[:,:,None]).ravel(),(rows,cols)),shape=(2*count,)*2).tocsr()
        cm=diags((mask*scale).ravel())@C;weight=diags(np.repeat(p.weights[selected[fixed]],2))
        matrix=bmat([[k[p.free][:,p.free],B[:,fi]],[weight@cm[fi],weight@dm[fi][:,fi]]],format='csc')
        rhs=-np.r_[r[p.free],np.repeat(p.weights[selected[fixed]],2)*(equation*scale).ravel()[fi]]
        try:
            lu=splu(matrix)
            solved=lu.solve(np.column_stack((rhs,-np.vstack((B[:,ui].toarray(),np.zeros((len(fi),len(ui))))))))
        except RuntimeError:continue
        base,response=solved[:,0],solved[:,1:]
        mobility=C[ui]@response[:n]
        offset=z.ravel()[ui]+C[ui]@base[:n]-mobility@current[uncertain].ravel()
        choices=[(int(modes[i]),)+tuple(v for v in (0,1,-1,2) if v!=modes[i]) for i in uncertain]
        print(f'INTEGRAL BRANCH SEARCH contacts={count} enumerated={len(uncertain)} budget=4096 nodes={p.contact_nodes[selected[uncertain]].tolist()}',flush=True)
        for traction,labels,examined in admissible_branches(mobility,offset,mu,choices,budget=4096):
            delta=base+response@(traction-current[uncertain]).ravel()
            all_lam=current.copy();all_lam[uncertain]=traction;all_lam[fixed]+=delta[n:].reshape(-1,2)
            all_z=z+(C@delta[:n]).reshape(-1,2);all_modes=modes.copy();all_modes[uncertain]=labels
            opened=all_modes==2;stuck=all_modes==0;sliding=~(opened|stuck)
            if (all_lam[:,0].min()<-1e-8 or np.any(all_z[opened,0]>1e-8)
                    or np.any(abs(all_lam[stuck,1])>mu*all_lam[stuck,0]+1e-8)
                    or np.any(all_modes[sliding]*all_z[sliding,1]<-1e-8)):continue
            for cut in range(12):
                fraction=.5**cut;seed=u.copy();seed[p.free]+=fraction*delta[:n]
                warm=lam.copy();warm[selected]+=fraction*(all_lam-current)
                try:p.assemble(seed,previous,depth,old_depth,dt,warm[:,0],warm[:,1],with_tangent=False)
                except FloatingPointError:continue
                found+=1
                print(f'INTEGRAL BRANCH PREDICTOR {found} examined={examined} fraction={fraction:g}',flush=True)
                yield seed,warm,examined
                break
            if found>=max_predictors:return
