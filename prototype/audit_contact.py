"""Audit saved failed states or accepted increments with their physical history."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_support import source_hashes
from scipy.sparse import bmat,coo_matrix,diags
from scipy.sparse.linalg import splu,onenormest,LinearOperator
from .contact_study import load_failed_problem,mesh_metrics,mesh_fingerprint,write_json


def row_switch_certificate(stick,rhs,row,slide_row,switch_gradient,row_scale):
    """Two one-sided linearizations at ONE verified stick/slip boundary.

    h<0 is stick and h>0 slide. Both nonsingular matrices may give steps
    pointing into the opposite branch. This is a local linearization result,
    not proof that the nonlinear equation or the physical experiment has no root.
    The determinant ratio uses a rank-one row replacement, never a large det().
    """
    slide=stick.tolil(copy=True);slide[row]=slide_row;slide=slide.tocsc()
    factor=splu(stick.tocsc());e=np.zeros(len(rhs));e[row]=1.
    null=factor.solve(e);ratio=float((slide_row@null).item())
    merit=float(np.linalg.norm(row_scale*rhs));directions=[]
    for label,matrix,lu in (('stick',stick,factor),('slide',slide,splu(slide))):
        delta=lu.solve(-rhs);dh=float((switch_gradient@delta).item())
        actual=slide if dh>0 else stick
        directions.append(dict(branch=label,
            linear_relative_residual=float(np.linalg.norm(matrix@delta+rhs)/max(np.linalg.norm(rhs),1e-30)),
            switch_direction_mpa=dh,branch_consistent=bool(dh<=0 if label=='stick' else dh>=0),
            merit_directional_derivative=float((row_scale*rhs)@(row_scale*(actual@delta))/max(merit,1e-30))))
    result=dict(determinant_ratio=ratio,merit=merit,directions=directions,
        both_steps_inconsistent=all(not d['branch_consistent'] for d in directions),
        both_steps_ascent=all(d['merit_directional_derivative']>0 for d in directions),
        interpretation='One-switch frozen linearization at a non-root; no physical instability or nonlinear nonexistence claim')
    if ratio<0:
        theta=1/(1-ratio);mixed=(1-theta)*stick+theta*slide
        result.update(singular_mix_fraction=theta,
            null_relative_residual=float(np.linalg.norm(mixed@null)/max(np.linalg.norm(stick@null)+np.linalg.norm(slide@null),1e-30)))
    return result


def friction_switch_audit(p,previous,old,depth,dt,u,lam,delta,selected,node):
    """Find the actual return-map boundary along the saved Newton direction."""
    from scipy.optimize import brentq
    from scipy.sparse import hstack,csr_matrix
    if p.surface_contact is None or p.options.coulomb_formulation!='alart_curnier' or p.config.adhesive_shear_strength_n_mm2:
        raise ValueError('Switch certificate currently requires non-adhesive integral Alart-Curnier contact')
    indices=np.flatnonzero(p.contact_nodes==node)
    if len(indices)!=1:raise ValueError('Switch node is not a contact node')
    i=int(indices[0]);n=len(p.free);kn=p.config.contact_surface_penalty_n_mm3
    kt=p.options.tangential_stiffness_mpa_per_mm;mu=p.config.friction_coefficient
    if mu<=0:raise ValueError('Stick/slip audit requires positive friction')
    du=np.zeros_like(u);du[p.free]=delta[:n];dl=np.zeros_like(lam);dl[selected]=delta[n:].reshape(-1,2)
    def contact(a):
        return p.surface_contact.evaluate(u+a*du,previous,depth,old,lam+a*dl,
            with_tangent=False,formulation='alart_curnier',roundoff_guard=False)
    initial=contact(0.);sign=1 if lam[i,1]+kt*initial['slip'][i]>=0 else -1
    def margin(a):
        q=contact(a)
        if q['pressure'][i]<=0:raise ValueError('Opening reached before selected friction boundary')
        return sign*(lam[i,1]+a*dl[i,1]+kt*q['slip'][i])-mu*q['pressure'][i]
    h0=margin(0.)
    if abs(h0)<1e-12:alpha=0.
    else:
        for upper in 2.**np.arange(-30,1):
            if h0*margin(upper)<=0:break
        else:raise ValueError('No selected friction switch on the saved Newton segment')
        alpha=brentq(margin,0.,upper,xtol=1e-15)
    v=u+alpha*du;m=lam+alpha*dl
    r,k,q,_=p.assemble(v,previous,depth,old,dt,m[:,0],m[:,1],mixed=True,roundoff_guard=False)
    if 'pressure_coupling' in q:raise ValueError('Switch audit currently requires a condensed displacement body')
    sel=np.flatnonzero((q['pressure']>0)|np.any(m!=0,axis=1));chosen=(2*sel[:,None]+np.arange(2)).ravel()
    j=int(np.flatnonzero(sel==i)[0]);row=n+2*j+1;weight=p.weights[i];count=len(chosen)
    W=diags(np.repeat(p.weights[sel],2))
    J=bmat([[k[p.free][:,p.free],q['global_B'][p.free][:,chosen]],
        [W@q['global_C'][chosen][:,p.free],W@q['global_D'][chosen][:,chosen]]],format='csc')
    R=np.r_[r[p.free],(p.weights[sel,None]*(np.column_stack((q['pressure'],q['shear']))-m)[sel]).ravel()]
    slip=hstack([kt*q['global_kinematic_u'][2*i+1][:,p.free],csr_matrix((1,count))],format='csr')
    normal=hstack([kn*q['global_kinematic_u'][2*i][:,p.free],
        csr_matrix(([1.],([0],[2*j])),shape=(1,count))],format='csr')
    tau=csr_matrix(([1.],([0],[n+2*j+1])),shape=(1,J.shape[0]))
    slide=weight*(sign*mu*normal-tau);hrow=sign*(slip+tau)-mu*normal
    stick=J.tolil();stick[row]=weight*slip;stick=stick.tocsc()
    scale=np.r_[np.full(n,1/p.config.newton_tolerance_n),
        1/(np.repeat(p.weights[sel],2)*p.config.contact_multiplier_tolerance_n_mm2)]
    report=row_switch_certificate(stick,R,row,slide,hrow,scale)
    report.update(node=int(node),reference_sn_mm=p.mesh.local_sn[node].tolist(),
        alpha_to_boundary=float(alpha),boundary_margin_mpa=float(margin(alpha)))
    # Check predicted one-sided slopes against the ORIGINAL nonlinear merit.
    check=[];slideJ=stick.tolil();slideJ[row]=slide;slideJ=slideJ.tocsc()
    for label,matrix in (('stick',stick),('slide',slideJ)):
        direction=splu(matrix).solve(-R)
        for fraction in (1e-6,1e-7):
            vv=v.copy();vv[p.free]+=fraction*direction[:n]
            mm=m.copy();mm[sel]+=fraction*direction[n:].reshape(-1,2)
            rr,_,qq,_=p.assemble(vv,previous,depth,old,dt,mm[:,0],mm[:,1],mixed=True,
                with_tangent=False,roundoff_guard=False)
            actual_merit=np.hypot(np.linalg.norm(rr[p.free])/p.config.newton_tolerance_n,
                np.linalg.norm(np.column_stack((qq['pressure'],qq['shear']))-mm)/p.config.contact_multiplier_tolerance_n_mm2)
            others=np.flatnonzero(qq['stick']!=q['stick']);others=others[others!=i]
            check.append(dict(branch=label,fraction=fraction,
                measured_merit_slope=float((actual_merit-report['merit'])/fraction),
                other_friction_switch_nodes=p.contact_nodes[others].tolist(),
                normal_switch_nodes=p.contact_nodes[(qq['pressure']>0)!=(q['pressure']>0)].tolist()))
    report['one_sided_nonlinear_checks']=check
    other=np.flatnonzero(q['stick']!=initial['stick']);other=other[other!=i]
    report['other_switches_before_boundary']=p.contact_nodes[other].tolist()
    report['single_switch_isolated']=bool(not len(other) and all(
        not c['other_friction_switch_nodes'] and not c['normal_switch_nodes'] for c in check))
    return report


def constrained_eigenmodes(matrix,displacement_count,count=3,*,selection='nearest'):
    """Finite modes of J v = eigenvalue * diag(M,0) v.

    Contact rows, including sliding-traction derivatives, have ZERO spectral
    mass and hence remain constraints. M is a positive numerical diagonal, not
    physical inertia. Use nonsymmetric eigs; do not symmetrize Coulomb contact.
    """
    from scipy.sparse.linalg import eigs
    n=displacement_count
    if (matrix.shape[0]!=matrix.shape[1] or not 1<=n<=matrix.shape[0]
            or selection not in ('nearest','negative') or not 1<=count<=n
            or (n>8 and count>=n-1)):
        raise ValueError('Invalid constrained mode selection or count')
    factor=splu(matrix.tocsc())
    mass=np.maximum(abs(matrix.diagonal()[:n]),1e-12)
    def solve(v):
        return factor.solve(v.real)+1j*factor.solve(v.imag) if np.iscomplexobj(v) else factor.solve(v)
    def action(v):return solve(np.r_[mass*v,np.zeros(matrix.shape[0]-n)])[:n]
    if n<=8:
        theta,vectors=np.linalg.eig(np.column_stack([action(v) for v in np.eye(n)]))
        valid=abs(theta)>1e-14;theta,vectors=theta[valid],vectors[:,valid]
    else:
        theta,vectors=eigs(LinearOperator((n,n),matvec=action,dtype=float),k=count,
            which='LM' if selection=='nearest' else 'SR',tol=1e-9,maxiter=100,
            v0=np.random.default_rng(17).normal(size=n))
    order=(np.argsort(abs(1/theta)) if selection=='nearest' else np.argsort(theta.real))[:count]
    values=1/theta[order]
    modes=[];records=[]
    for index,value in zip(order,values):
        v=vectors[:,index]
        full=solve(np.r_[mass*v,np.zeros(matrix.shape[0]-n)])/theta[index]
        full[:n]=v;full/=np.linalg.norm(full[:n])
        residual=matrix@full-np.r_[value*mass*full[:n],np.zeros(matrix.shape[0]-n)]
        records.append(dict(real=float(value.real),imaginary=float(value.imag),
            relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(matrix@full),1e-30)),
            constraint_residual=float(np.linalg.norm(residual[n:]))))
        modes.append(full)
    return records,np.column_stack(modes)


def coupled_modes(matrix,count=3):
    """Small singular triplets of the NONSYMMETRIC coupled contact tangent.

    Row/column infinity equilibration removes gross unit disparities. Results
    remain scaling dependent and describe a frozen branch, not dynamic stability.
    The inverse is applied by sparse solves; neither J^-1 nor J.T@J is stored.
    """
    from scipy.sparse.linalg import svds
    if matrix.shape[0]!=matrix.shape[1] or not 1<=count<matrix.shape[0]:
        raise ValueError('Coupled SVD requires a square matrix and 0 < count < size')
    row=1/np.maximum(abs(matrix).max(axis=1).toarray().ravel(),1e-30)
    scaled=diags(row)@matrix
    column=1/np.maximum(abs(scaled).max(axis=0).toarray().ravel(),1e-30)
    scaled=(scaled@diags(column)).tocsc();factor=splu(scaled)
    inverse=LinearOperator(scaled.shape,matvec=factor.solve,
        rmatvec=lambda x:factor.solve(x,trans='T'),dtype=float)
    left_inverse,singular_inverse,right_inverse=svds(inverse,k=count,which='LM',
        tol=1e-10,maxiter=1000,random_state=17)
    order=np.argsort(1/singular_inverse)
    singular=1/singular_inverse[order]
    right=left_inverse[:,order];left=right_inverse[order].T
    residuals=[dict(right=float(np.linalg.norm(scaled@right[:,i]-singular[i]*left[:,i])),
                    left=float(np.linalg.norm(scaled.T@left[:,i]-singular[i]*right[:,i])))
               for i in range(count)]
    return dict(singular_values=singular.tolist(),triplet_residuals=residuals,
        scaled_asymmetry=float(np.linalg.norm((scaled-scaled.T).data)/np.linalg.norm(scaled.data)),
        scaling='row infinity, then column infinity; mixed force/traction coordinates',
        interpretation='Frozen coupled contact branch; singular values are not physical stability eigenvalues'),column[:,None]*right


def perturb_modes(directory,archive,output,amplitudes,*,iterations=30):
    """Bounded signed mode probes; every solve uses the SAME physical history."""
    import contextlib,time,os
    from dataclasses import replace
    from .contact_study import save_state
    if iterations<1 or not amplitudes or any(a<=0 or not np.isfinite(a) for a in amplitudes):
        raise ValueError('Positive finite amplitudes and an iteration budget are required')
    p,previous,history,old=load_failed_problem(directory)
    source_config=p.config.to_dict()
    p.config=replace(p.config,max_contact_iterations=iterations)
    output.mkdir(parents=True,exist_ok=False)
    records=[];started=time.monotonic()
    report=dict(source=str(directory),config=p.config.to_dict(),options=vars(p.options),
        source_sha256=source_hashes(snapshot=output/'source_snapshot'),
        scope='signed frozen-history mode probes, not a loading path',trials=records,
        status='running',pid=os.getpid(),mode_archive=str(archive))
    saved_previous=previous.copy();saved_history=history.copy()
    with np.load(archive) as z:
        if (json.loads(str(z['config_json']))!=source_config
                or json.loads(str(z['options_json']))!=vars(p.options)):
            raise ValueError('Mode archive configuration differs from source')
        if not np.array_equal(z['previous_displacement'],previous) or not np.array_equal(z['previous_multipliers'],history):
            raise ValueError('Mode archive physical history differs from source')
        depth=float(z['depth_mm']);dt=float(z['dt_s'])
        if not np.isfinite(dt) or dt<=0 or float(z['old_depth_mm'])!=old:
            raise ValueError('Mode archive has an invalid physical increment')
        write_json(output/'result.json',report)
        for mode in range(len(z['displacement_modes'])):
            for amplitude in amplitudes:
                for sign in (-1,1):
                    record=dict(mode=mode,amplitude_mm=sign*amplitude,accepted=False)
                    u=z['base_displacement']+sign*amplitude*z['displacement_modes'][mode]
                    lam=z['base_multipliers']+sign*amplitude*z['traction_modes'][mode]
                    with (output/'run.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                        print('PERTURB',record,flush=True)
                        try:
                            u,lam,d=p._mixed_step(previous,old,depth,dt,history,initial=u,initial_multipliers=lam)
                            record.update(accepted=True,diagnostics=d)
                        except (RuntimeError,FloatingPointError,ValueError) as exc:
                            record.update(error_type=type(exc).__name__,error=str(exc))
                    records.append(record);report['elapsed_s']=time.monotonic()-started
                    if not np.array_equal(previous,saved_previous) or not np.array_equal(history,saved_history):
                        raise AssertionError('Mode probe modified physical history')
                    if record['accepted']:
                        report['status']='accepted'
                        save_state(output/'accepted_state.npz',p,u,lam,depth)
                    write_json(output/'result.json',report)
                    print(json.dumps(record),flush=True)
                    if record['accepted']:return report
    report['status']='exhausted';write_json(output/'result.json',report)
    return report


def audit(directory,*,curvature=False,modes=0,mode_output=None,mode_kind='singular',increment=None,
          branch_seconds=0.,switch_node=None):
    if modes<0 or mode_kind not in ('singular','negative') or not np.isfinite(branch_seconds) or branch_seconds<0:
        raise ValueError('Invalid mode count or kind')
    p,previous,state,old=load_failed_problem(directory)
    bounded=not (directory/'status.json').exists()
    status=json.loads((directory/('result.json' if bounded else 'status.json')).read_text(encoding='utf-8'))
    if increment is not None:
        with np.load(increment) as z:
            if (str(z['mesh_sha256'])!=mesh_fingerprint(p)
                    or json.loads(str(z['config_json']))!=p.config.to_dict()
                    or json.loads(str(z['options_json']))!=vars(p.options)):
                raise ValueError('Increment mesh or configuration differs from source')
            previous,state=z['previous_displacement'],z['previous_multipliers']
            u,lam=z['displacement'],z['contact_multipliers']
            old,depth,dt=float(z['old_depth_mm']),float(z['depth_mm']),float(z['dt_s'])
    else:
        depth=status['failed_depth_mm'] if bounded else status['failed_attempt']['depth_mm']
        dt=(depth-old)/p.config.speed_mm_s if bounded else status['failed_attempt']['dt_s']
        with np.load(directory/('failed_trial.npz' if bounded else 'failed_attempt.npz')) as z:
            u,lam=z['displacement' if bounded else 'trial_displacement'],z['contact_multipliers' if bounded else 'trial_contact_multipliers']
    r,k,q,d=p.assemble(u,previous,depth,old,dt,lam[:,0],lam[:,1],mixed=True)
    curvature_result=None
    if curvature:
        if p.options.volumetric_discretization!='quadratic_full':
            raise ValueError('Curvature audit currently requires the full-integration P2 body')
        from scipy.sparse.linalg import eigsh
        bulk=(k-q['global_K'])[p.free][:,p.free].tocsc()
        scale=diags(1/np.sqrt(np.maximum(abs(bulk.diagonal()),1e-30)))
        normalized=scale@bulk@scale
        values,vectors=eigsh(normalized,k=3,which='SA',tol=1e-5,maxiter=1000,
                            v0=np.random.default_rng(17).normal(size=len(p.free)))
        curvature_result=dict(scaled_bulk_eigenvalues=values.tolist(),
            eigen_residuals=[float(np.linalg.norm(normalized@vectors[:,i]-values[i]*vectors[:,i])) for i in range(3)],
            interpretation='Body tangent with fixed boundary DOFs only; contact constraints are NOT projected out. Negative curvature is not proof of physical instability or the unique failure cause.')
    selected=np.flatnonzero((q['pressure']>0)|np.any(lam!=0,axis=1))
    mdofs=2*np.arange(len(selected))[:,None]+np.arange(2)
    shape=(len(selected),2,2)
    ru=np.broadcast_to(p.cdofs[selected,:,None],shape).ravel()
    cm=np.broadcast_to(mdofs[:,None,:],shape).ravel()
    rm=np.broadcast_to(mdofs[:,:,None],shape).ravel()
    cu=np.broadcast_to(p.cdofs[selected,None,:],shape).ravel()
    if p.surface_contact is not None:
        chosen=(2*selected[:,None]+np.arange(2)).ravel()
        B=q['global_B'][p.free][:,chosen]
        C=q['global_C'][chosen][:,p.free]
        D=q['global_D'][chosen][:,chosen]
    else:
        B=coo_matrix((-q['dforce_dlambda'][selected].ravel(),(ru,cm)),shape=(len(u),2*len(selected))).tocsr()[p.free]
        C=coo_matrix((q['projection_u'][selected].ravel(),(rm,cu)),shape=(2*len(selected),len(u))).tocsr()[:,p.free]
        D=coo_matrix((q['projection_lambda'][selected].ravel(),(rm,cm)),shape=(2*len(selected),2*len(selected))).tocsr()
    W=diags(np.repeat(p.weights[selected],2))
    jac=bmat([[k[p.free][:,p.free],B],[W@C,W@D]],format='csc')
    def evaluate(displacement,multiplier):
        rr,_,contact,_=p.assemble(displacement,previous,depth,old,dt,multiplier[:,0],multiplier[:,1],mixed=True)
        defect=np.column_stack((contact['pressure'],contact['shear']))-multiplier
        return np.r_[rr[p.free],(p.weights[selected,None]*defect[selected]).ravel()],contact
    rhs,_=evaluate(u,lam);lu=splu(jac);delta=lu.solve(-rhs)
    result=dict(source=str(directory),source_status=status,mesh=mesh_metrics(p),
        increment_source=str(increment) if increment is not None else None,
        old_depth_mm=old,depth_mm=depth,dt_s=dt,
        trial_residual_n=float(np.linalg.norm(r[p.free])),selected_contact_nodes=len(selected),
        linear_relative_residual=float(np.linalg.norm(jac@delta+rhs)/np.linalg.norm(rhs)),
        linear_residual_definition='norm2(J*delta+R)/norm2(R)',
        condition_estimate_note='1-norm estimate depends on the displacement/traction units; not proof of causation')
    if curvature_result is not None:result['curvature']=curvature_result
    if modes:
        from run_support import save_npz
        spectral,vectors=coupled_modes(jac,modes)
        branch,branch_vectors=constrained_eigenmodes(jac,len(p.free),modes)
        spectral['constrained_branch_eigenvalues']=branch
        from scipy.sparse.linalg import ArpackNoConvergence
        try:
            negative,negative_vectors=constrained_eigenmodes(jac,len(p.free),modes,selection='negative')
            spectral['negative_search_status']='converged'
        except ArpackNoConvergence:
            negative=[];negative_vectors=np.empty((jac.shape[0],0))
            spectral['negative_search_status']='iteration_budget_exhausted; no conclusion about absence of negative modes'
        for record,vector in zip(negative,negative_vectors.T):
            displacement=np.zeros(len(u),dtype=complex);displacement[p.free]=vector[:len(p.free)]
            node=int(np.linalg.norm(displacement.reshape(-1,2),axis=1).argmax())
            record.update(peak_node=node,reference_sn_mm=p.mesh.local_sn[node].tolist())
        spectral['negative_branch_search']=negative
        spectral['negative_search_note']='Smallest real inverse eigenvalues; inspect returned signs. Not an exhaustive stability proof, and the numerical metric is not physical mass.'
        if mode_kind=='negative':
            if len(negative)!=modes or any(abs(v['imaginary'])>1e-8 or v['real']>=0 for v in negative):
                raise ValueError('Signed real perturbations require real negative branch modes')
            vectors=negative_vectors.real
        spectral['perturbation_mode_kind']=mode_kind
        spectral['branch_eigenvalue_note']='Positive numerical displacement metric; contact rows kept algebraic. Spectrum alone does not certify an equilibrium bifurcation or physical dynamic stability; see original-equation verification.'
        displacements=np.zeros((modes,len(u)));tractions=np.zeros((modes,*lam.shape))
        locations=[]
        current=p.mesh.reference_xy+u.reshape(-1,2)
        for i in range(modes):
            displacements[i,p.free]=vectors[:len(p.free),i]
            tractions[i,selected]=vectors[len(p.free):,i].reshape(-1,2)
            amplitude=np.linalg.norm(displacements[i].reshape(-1,2),axis=1)
            scale=amplitude.max()
            displacements[i]/=scale;tractions[i]/=scale
            node=int(amplitude.argmax())
            locations.append(dict(peak_node=node,reference_sn_mm=p.mesh.local_sn[node].tolist(),
                current_xz_mm=current[node].tolist(),
                traction_mpa_per_mm=float(abs(tractions[i]).max()),
                displacement_effective_nodes=float(amplitude.sum()**2/np.sum(amplitude**2))))
        spectral['mode_locations']=locations;result['coupled_modes']=spectral
        if mode_output is not None:
            save_npz(mode_output,displacement_modes=displacements,traction_modes=tractions,
                singular_values=np.asarray(spectral['singular_values']),
                free_dofs=p.free,selected_contact_indices=selected,
                base_displacement=u,base_multipliers=lam,previous_displacement=previous,
                previous_multipliers=state,old_depth_mm=old,depth_mm=depth,
                dt_s=dt,
                config_json=json.dumps(p.config.to_dict()),options_json=json.dumps(vars(p.options)),
                constrained_branch_vectors=branch_vectors,negative_branch_vectors=negative_vectors,
                mode_kind=mode_kind)
    inverse=LinearOperator(jac.shape,matvec=lu.solve,rmatvec=lambda x:lu.solve(x,trans='T'))
    # scipy's estimator samples signs from NumPy's legacy RNG. Preserve the
    # caller's state while making repeated audits of this state reproducible.
    random_state=np.random.get_state()
    try:
        np.random.seed(17)
        result['jacobian_condition_1_estimate']=float(onenormest(jac)*onenormest(inverse))
    finally:
        np.random.set_state(random_state)
    result['condition_estimator_seed']=17
    row_scale=1/np.maximum(np.abs(jac).max(axis=1).toarray().ravel(),1e-30)
    row_scaled=diags(row_scale)@jac
    col_scale=1/np.maximum(np.abs(row_scaled).max(axis=0).toarray().ravel(),1e-30)
    scaled=(row_scaled@diags(col_scale)).tocsc();scaled_lu=splu(scaled)
    scaled_delta=col_scale*scaled_lu.solve(-row_scale*rhs)
    result['equilibrated_linear_solve']=dict(relative_direction_difference=float(np.linalg.norm(scaled_delta-delta)/np.linalg.norm(delta)),
        original_relative_residual=float(np.linalg.norm(jac@scaled_delta+rhs)/np.linalg.norm(rhs)))
    directions=[]
    rng=np.random.default_rng(17)
    for kind in ('bulk_and_contact','contact_local','newton'):
        if kind=='newton':v=delta/max(np.linalg.norm(delta),1e-30)
        else:
            du=rng.normal(size=len(u));du[p.fixed]=0
            if kind=='contact_local':
                keep=np.unique(p.cdofs[selected]);du[np.setdiff1d(np.arange(len(u)),keep)]=0
            du/=max(np.linalg.norm(du),1e-30)
            dl=rng.normal(size=(len(selected),2));dl*=250/max(np.linalg.norm(dl),1e-30)
            v=np.r_[du[p.free],dl.ravel()]
        du=np.zeros_like(u);du[p.free]=v[:len(p.free)]
        dl=np.zeros_like(lam);dl[selected]=v[len(p.free):].reshape(-1,2)
        expected=jac@v
        for h in ((1e-3,1e-4,1e-5,1e-7) if kind=='newton' else (1e-7,1e-8,1e-9)):
            a,qa=evaluate(u+h*du,lam+h*dl);b,qb=evaluate(u-h*du,lam-h*dl)
            fd=(a-b)/(2*h)
            directions.append(dict(kind=kind,epsilon=h,
                relative_error=float(np.linalg.norm(expected-fd)/max(np.linalg.norm(fd),1e-30)),
                absolute_error=float(np.linalg.norm(expected-fd)),analytic_action_norm=float(np.linalg.norm(expected)),
                normal_branch_changes=int(np.sum((qa['pressure']>0)!=(qb['pressure']>0))),
                friction_branch_changes=int(np.sum(qa['stick']!=qb['stick']))))
    result['directional_checks']=directions
    # Audit the actual Newton step in the SAME all-contact, tolerance-scaled
    # merit used by surface-contact globalization. A good linear solve alone
    # does not imply its full nonlinear step stays on the linearized branches.
    def metrics(displacement,multiplier):
        rr,_,contact,diagnostics=p.assemble(displacement,previous,depth,old,dt,
            multiplier[:,0],multiplier[:,1],mixed=True,with_tangent=False)
        defect=np.column_stack((contact['pressure'],contact['shear']))-multiplier
        return dict(residual_n=float(np.linalg.norm(rr[p.free])),
            projection_mpa=float(np.max(np.abs(defect))),
            normalized_merit=float(np.hypot(np.linalg.norm(rr[p.free])/p.config.newton_tolerance_n,
                np.linalg.norm(defect)/p.config.contact_multiplier_tolerance_n_mm2)),
            min_J=float(diagnostics['min_j']),
            normal_branch_changes=int(np.sum((contact['pressure']>0)!=(q['pressure']>0))),
            friction_branch_changes=int(np.sum(contact['stick']!=q['stick'])),
            friction_switch_nodes=p.contact_nodes[contact['stick']!=q['stick']].tolist())
    baseline=metrics(u,lam);backtracking=[]
    for cut in range(max(24,p.config.max_line_search_iterations)):
        alpha=.5**cut;trial=u.copy();trial[p.free]+=alpha*delta[:len(p.free)]
        multiplier=lam.copy();multiplier[selected]+=alpha*delta[len(p.free):].reshape(-1,2)
        multiplier[abs(multiplier)<1e-12]=0  # match the production backtrack
        try:
            values=metrics(trial,multiplier)
            values['armijo_pass']=bool(values['normalized_merit']<baseline['normalized_merit']*(1-1e-4*alpha))
        except FloatingPointError as exc:
            values=dict(admissible_domain=False,error=str(exc),armijo_pass=False)
        backtracking.append(dict(alpha=alpha,**values))
    result['newton_backtracking']=dict(baseline=baseline,trials=backtracking,
        configured_trial_budget=p.config.max_line_search_iterations,
        note='Diagnostic only; no physical history advanced and no converged state certified')
    if switch_node is not None:
        result['friction_switch_certificate']=friction_switch_audit(p,previous,old,depth,dt,u,lam,delta,selected,switch_node)
    if p.surface_contact is not None and 'pressure_coupling' not in q:
        result['contact_localization']=localize_contact(p,r,k,q,lam,branch_seconds=branch_seconds)
    physical=p.assemble(previous,previous,old,old,.01,state[:,0],state[:,1])
    fields=p.fields(previous);loads=p.weights*state[:,0]
    result['accepted_state']=dict(residual_n=float(np.linalg.norm(physical[0][p.free])),
        residual_note='Zero-increment replay, not the original accepted loading increment',
        min_J=float(fields['J'].min()),max_J=float(fields['J'].max()),
        local_nominal_strength_ratio=float(fields['nominal_tensile_mpa'].max()/p.config.tensile_strength_n_mm2),
        max_reference_pressure_mpa=float(state[:,0].max()),
        max_node_normal_load_fraction=float(loads.max()/loads.sum()),
        max_boundary_penetration_mm=p.edge_penetration(previous,old))
    xy=p.mesh.reference_xy[p.mesh.triangles];v=np.roll(xy,-1,axis=1)-xy;l=np.linalg.norm(v,axis=2)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(v*(-np.roll(v,1,axis=1)),axis=2)/(l*np.roll(l,1,axis=1)),-1,1)))
    bad=np.argsort(angles.min(axis=1))[:5]
    result['worst_triangles']=[dict(element=int(i),min_angle_deg=float(angles[i].min()),
        local_sn_mm=p.mesh.local_sn[p.mesh.triangles[i]].tolist()) for i in bad]
    if hasattr(p.mesh,'elements'):
        field=p.bulk_response(u);stretches=np.linalg.svd(field['F'],compute_uv=False)
        condition=(stretches[...,0]/stretches[...,-1]).max(axis=1)
        worst=np.argsort(-condition)[:10]
        result['deformation_resolution']=dict(max_F_condition=float(condition.max()),
            worst_elements=worst.tolist(),reference_sn_mm=p.mesh.local_sn[p.mesh.triangles[worst]].mean(axis=1).tolist(),
            min_J=d['min_j'])
        if p.options.contact_discretization=='quadratic_integral':
            nodes=np.column_stack((p.edges,p.mesh.boundary_midpoints))
            pressure=lam[np.searchsorted(p.contact_nodes,nodes),0]
            contrast=pressure[:,2]-.5*(pressure[:,0]+pressure[:,1])
            worst_pressure=np.argsort(-abs(contrast))[:10]
            result['deformation_resolution']['pressure_midpoint_contrast']=dict(
                definition='P1-half-edge midpoint pressure minus average endpoint pressure; a local resolution indicator, not proof of spurious oscillation',
                max_abs_mpa=float(abs(contrast).max()),edges=worst_pressure.tolist(),
                contrast_mpa=contrast[worst_pressure].tolist(),
                reference_sn_mm=p.mesh.local_sn[p.edges[worst_pressure]].mean(axis=1).tolist())
            current=p.mesh.reference_xy+u.reshape(-1,2)
            a,b=current[p.edges].transpose(1,0,2);mid=current[p.mesh.boundary_midpoints]
            A=2*(a+b-2*mid);B=4*mid-3*a-b
            length=np.linalg.norm(np.diff(p.mesh.reference_xy[p.edges],axis=1)[:,0],axis=1)
            # Exact extrema of |x'(t)| on each quadratic boundary edge.
            t=np.clip(-np.einsum('ei,ei->e',A,B)/np.maximum(2*np.sum(A*A,axis=1),1e-30),0,1)
            minimum=np.linalg.norm(2*A*t[:,None]+B,axis=1)/length
            maximum=np.maximum(np.linalg.norm(B,axis=1),np.linalg.norm(2*A+B,axis=1))/length
            variation=maximum/np.maximum(minimum,1e-30);worst_surface=np.argsort(-variation)[:10]
            result['deformation_resolution']['surface_stretch']=dict(
                definition='Exact min/max of deformed tangent length divided by reference edge length; no homogeneous half-space critical strain is assumed',
                minimum=float(minimum.min()),maximum=float(maximum.max()),
                edges=worst_surface.tolist(),min_by_edge=minimum[worst_surface].tolist(),
                max_by_edge=maximum[worst_surface].tolist(),variation=variation[worst_surface].tolist(),
                reference_sn_mm=p.mesh.local_sn[p.edges[worst_surface]].mean(axis=1).tolist())
    result['audit_source_sha256']=source_hashes()
    if increment is not None:
        rr,_,qq,dd=p.assemble(u,previous,depth,old,dt,lam[:,0],lam[:,1],
            with_tangent=False,formulation='alart_curnier',roundoff_guard=False)
        defect=float(abs(np.column_stack((qq['pressure'],qq['shear']))-lam).max())
        force=float(np.linalg.norm(rr[p.free]));gap=float(p.edge_penetration(u,depth))
        cone=bool(np.all(lam[:,0]>=0) and np.all(abs(lam[:,1])<=p.config.friction_coefficient*lam[:,0]))
        dissipation_tolerance=float(np.sum(p.weights*abs(qq['shear']))*
            p.config.contact_multiplier_tolerance_n_mm2/p.options.tangential_stiffness_mpa_per_mm+1e-12)
        result['original_increment_verification']=dict(residual_n=force,projection_mpa=defect,
            edge_penetration_mm=gap,min_J=dd['min_j'],cone_feasible=cone,
            friction_dissipation_n_mm=dd['friction_dissipation_n_mm'],
            passed=bool(force<=p.config.newton_tolerance_n and defect<=p.config.contact_multiplier_tolerance_n_mm2
                and gap<=p.config.contact_penetration_tolerance_mm and cone and dd['min_j']>0
                and dd['friction_dissipation_n_mm']>=-dissipation_tolerance))
    return result


def localize_contact(p,r,k,q,lam,*,branch_seconds=0.):
    """Frozen tangent ablations; never change the physical residual or history."""
    defect=np.max(abs(np.column_stack((q['pressure'],q['shear']))-lam),axis=1)
    selected=np.argsort(-defect,kind='stable')[:6]
    chosen=(2*selected[:,None]+np.arange(2)).ravel()
    B=q['global_B'][p.free];C=q['global_kinematic_u'][:,p.free]
    result=dict(nodes=p.contact_nodes[selected].tolist(),
        reference_sn_mm=p.mesh.local_sn[p.contact_nodes[selected]].tolist(),
        projection_defect_mpa=defect[selected].tolist(),
        interpretation='Unconstrained body compliance at a failed trial, not a physical stability certificate',
        compliance={})
    for name,matrix in (('full',k),('without_contact_geometric',k-q['global_K'])):
        matrix=matrix[p.free][:,p.free].tocsc();factor=splu(matrix)
        rhs=B[:,chosen].toarray();response=-factor.solve(rhs)
        mobility=C[chosen]@response
        result['compliance'][name]=dict(normal_diagonal_mm_per_mpa=mobility.diagonal()[::2].tolist(),
            relative_linear_residual=float(np.linalg.norm(matrix@response+rhs)/max(np.linalg.norm(rhs),1e-30)))
    if branch_seconds:
        from .contact_branches import affine_branch_milp
        factor=splu(k[p.free][:,p.free].tocsc())
        response=-factor.solve(B.toarray());base=-factor.solve(r[p.free])
        mobility=C@response;z=np.column_stack((q['gap'],q['slip'])).ravel()
        offset=z+C@base-mobility@lam.ravel()
        traction,labels,report=affine_branch_milp(mobility,offset,p.config.friction_coefficient,
            pressure_bound=100.,kinematic_lower=z-.02,kinematic_upper=z+.02,time_limit=branch_seconds)
        report['kinematic_halfwidth_mm']=.02
        if traction is not None:
            report.update(tractions_mpa=traction.tolist(),labels=labels.tolist(),
                max_displacement_correction_mm=float(abs(base+response@(traction-lam).ravel()).max()),
                nonlinear_acceptance='Not attempted; affine feasibility alone does not accept a time step')
        result['all_contact_branches']=report
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--increment',type=Path,help='Accepted increment NPZ containing the actual previous state and dt')
    parser.add_argument('--curvature',action='store_true',help='Also check the P2 body tangent; not a constrained stability test')
    parser.add_argument('--modes',type=int,default=0,help='Compute smallest singular modes of the coupled tangent')
    parser.add_argument('--mode-kind',choices=('singular','negative'),default='singular',help='Modes exported for signed probes; both spectral searches are reported')
    parser.add_argument('--mode-output',type=Path,help='Local NPZ for bounded signed perturbation tests')
    parser.add_argument('--perturb-mm',type=float,nargs='+',help='Positive mode amplitudes; solve both signs at the same physical increment')
    parser.add_argument('--perturb-output',type=Path)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--branch-seconds',type=float,default=0.,
        help='Optional time budget for all-node bounded affine Coulomb feasibility; 0 disables it')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--switch-node',type=int,help='Locate and certify a selected stick/slip switch on the actual Newton segment')
    args=parser.parse_args()
    if args.modes<0 or args.iterations<1 or args.threads<1 or not np.isfinite(args.branch_seconds) or args.branch_seconds<0:
        parser.error('Mode count must be nonnegative and iteration budget positive')
    if args.perturb_mm and (args.modes<1 or not args.mode_output or not args.perturb_output
                           or any(a<=0 or not np.isfinite(a) for a in args.perturb_mm)):
        parser.error('Positive perturbations require --modes, --mode-output and --perturb-output')
    if args.increment and args.perturb_mm:parser.error('Accepted increment audits do not use failed-state perturbation probes')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(args.threads):
        result=audit(args.source,curvature=args.curvature,modes=args.modes,mode_output=args.mode_output,
            mode_kind=args.mode_kind,increment=args.increment,branch_seconds=args.branch_seconds,switch_node=args.switch_node)
    write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('linear_relative_residual','jacobian_condition_1_estimate','equilibrated_linear_solve')},indent=2))
    if args.perturb_mm:
        perturb_modes(args.source,args.mode_output,args.perturb_output,args.perturb_mm,iterations=args.iterations)


if __name__=='__main__':main()
