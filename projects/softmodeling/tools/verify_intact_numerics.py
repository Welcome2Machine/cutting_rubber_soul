"""Independent numerical benchmarks for the intact plane-strain model. Explicit runs only."""
import argparse
import csv
import hashlib
import inspect
import io
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from mechanics.finite_strain import Material, elastic_response, response
from prototype import material as body_material
from prototype.contact_trace import record_digest
from prototype.ipc_reference import IPCProblem, LoadStepFailure, boundary_balance
from prototype.material_case import make_problem
from prototype.model_config import FEMConfig, FEMMesh, local_to_lab

GE, GV, K, TAU, WIDTH = .1064, .9424, 23.3433333333333, 1.7, 20.
WINDOWS = ((0., .25), (.25, .5), (.5, 1.), (1., 2.))


def read(path):
    return json.loads(path.read_bytes())


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def elastic_config(shear):
    return replace(FEMConfig(), shear_modulus_n_mm2=shear, poisson_ratio=(3*K-2*shear)/(6*K+2*shear),
                   width_mm=WIDTH, ogden_alpha=2.)


def stored_stress(F, A, ge, gv, bulk):
    """Independent first Piola stress of the two committed quadratic isochoric branches."""
    J = np.linalg.det(F)
    inv_t = np.linalg.inv(F).transpose(0, 2, 1)
    FA = F @ A[:, :2, :2]
    tr = ge*(np.sum(F*F, axis=(1,2))+1.) + gv*(np.sum(FA*F, axis=(1,2))+A[:,2,2])
    return J[:,None,None]**(-2/3)*(ge*F+gv*FA-tr[:,None,None]*inv_t/3) + (bulk*J*(J-1))[:,None,None]*inv_t


def history(folder, output, label):
    """Reconstruct physical nodal loads and the external-work balance from accepted checkpoint bytes."""
    report = read(folder/'result.json')
    audit = read(folder/'audit.json')
    result_hash = hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()
    assert audit['passed'] and audit['trajectory_checked'] and audit['source_result_sha256']==result_hash
    settings = report['settings']
    p = make_problem(**{k:v for k,v in settings.items() if k in inspect.signature(make_problem).parameters})
    p.ipc.set_num_threads(2)
    tri = p.mesh.triangles
    areas, grad, dofs = p.mesh.bulk_data
    origin = local_to_lab(np.array([[p.config.drop_s_mm,p.config.height_mm]]),p.config)[0]
    ge,bulk=p.material.moduli()
    gv=p.material.moduli(True)[0]
    initial_force=np.zeros(2); prior_depth=p.start
    old_stored=old_barrier=work=visc=friction=0.
    out=[]
    for i,(step,proof) in enumerate(zip(report['steps'],audit['rows'])):
        path=folder/'states'/f'{i:04d}.npz'
        raw=path.read_bytes(); checkpoint_hash=hashlib.sha256(raw).hexdigest()
        assert proof['passed'] and checkpoint_hash==proof['checkpoint_sha256'] and record_digest(step)==proof['accepted_record_sha256']
        with np.load(io.BytesIO(raw), allow_pickle=False) as state:
            x=state['positions'];prev=state['previous_positions'];A=state['material_metric']
            dt=float(state['material_dt_s'])
            if i==0:
                np.testing.assert_array_equal(state['reference_xy'],p.rest);np.testing.assert_array_equal(state['triangles'],tri)
        p.set_friction_dt(dt)
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:p.n]-p.rest[:p.n])[tri],grad)
        P=stored_stress(F,A,ge,gv,bulk)
        local=(grad@P.transpose(0,2,1))*(areas*WIDTH)[:,None,None]
        g=np.zeros(len(x)*2);np.add.at(g,dofs.ravel(),local.ravel())
        sx=x[p.surface];delta=(x-prev)[p.surface]
        c=p.collisions(x);gn=p.barrier.gradient(c,p.cmesh,sx)
        lag=p.lag(x)
        gf=p.friction.gradient(lag,p.cmesh,delta) if lag is not None else np.zeros_like(gn)
        g[p.surface_dofs]+=gn+gf
        arm=sx-origin; gfxy=gf.reshape(-1,2)
        couple=np.sum(arm[:,0]*gfxy[:,1]-arm[:,1]*gfxy[:,0])
        bal=boundary_balance(x,g,p.n,p.free,origin,couple)
        force=-np.asarray(bal['actuator_on_tool_N'])
        np.testing.assert_allclose(force,[step['reaction_x_n'],step['reaction_fz_n']],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(bal['free_L2_N'],step['residual_n'],rtol=2e-5,atol=2e-8)
        df=float(gf @ delta.ravel()); barrier=float(p.barrier(c,p.cmesh,sx))
        np.testing.assert_allclose([df,barrier],[step['friction_work_n_mm'],step['barrier_energy_n_mm']],rtol=1e-8,atol=1e-10)
        stored=step['material']['stored_energy_n_mm'];dv=step['material']['viscous_dissipation_increment_n_mm']
        depth=step['depth_mm'];dw=(depth-prior_depth)*(initial_force[1]+force[1])*.5
        defect=dw-(stored-old_stored)-(barrier-old_barrier)-dv-df
        work+=dw; visc+=dv; friction+=df
        scale=max(np.linalg.norm(force),.5)
        out.append(dict(index=i,depth_mm=depth,source_checkpoint_sha256=checkpoint_hash,accepted_record_sha256=proof['accepted_record_sha256'],
                        Fz_N=force[1],Fx_N=force[0],support_x_N=bal['support_on_body_N'][0],support_z_N=bal['support_on_body_N'][1],
                        support_moment_Nmm=bal['support_moment_Nmm'],actuator_moment_Nmm=bal['actuator_moment_Nmm'],
                        external_net_x_N=bal['net_external_force_N'][0],external_net_z_N=bal['net_external_force_N'][1],
                        external_net_moment_Nmm=bal['net_external_moment_Nmm'],friction_couple_Nmm=couple,
                        normalized_force=np.linalg.norm(bal['net_external_force_N'])/scale,
                        normalized_moment=abs(bal['net_external_moment_Nmm'])/(5*scale),
                        free_L2_N=bal['free_L2_N'],max_free_node_N=bal['max_free_node_N'],max_free_node=bal['max_free_node'],
                        dt_s=dt,increment_external_work_Nmm=dw,increment_viscous_Nmm=dv,increment_friction_Nmm=df,
                        increment_energy_defect_Nmm=defect,stored_Nmm=stored,barrier_Nmm=barrier,external_work_Nmm=work,
                        viscous_Nmm=visc,friction_Nmm=friction,cumulative_defect_Nmm=work-stored-barrier-visc-friction))
        prior_depth=depth;initial_force=force;old_barrier=barrier;old_stored=stored
    assert len(out)==audit['states']
    table(output/(label+'.csv'),out)
    results=[]
    for lo,hi in WINDOWS:
        rr=[r for r in out if lo-1e-10<=r['depth_mm']<=hi+1e-10]
        if len(rr)<2 or abs(rr[-1]['depth_mm']-hi)>1e-10:continue
        start=rr[0]; end=rr[-1]; work_scale=max(abs(end['external_work_Nmm']-start['external_work_Nmm']),.005)
        drift=[r['cumulative_defect_Nmm']-start['cumulative_defect_Nmm'] for r in rr]
        results.append(dict(window_mm=[lo,hi],samples=len(rr),max_free_L2_N=max(r['free_L2_N'] for r in rr),
            max_force_fraction=max(r['normalized_force'] for r in rr),max_moment_fraction=max(r['normalized_moment'] for r in rr),
            max_window_energy_fraction=max(abs(x) for x in drift)/work_scale,end_window_energy_fraction=abs(drift[-1])/work_scale,
            window_external_work_Nmm=end['external_work_Nmm']-start['external_work_Nmm'],window_signed_defect_Nmm=drift[-1],
            end_cumulative_energy_fraction=abs(end['cumulative_defect_Nmm'])/max(abs(end['external_work_Nmm']),.005)))
    return dict(result_sha256=result_hash, source_audit_sha256=hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest(),
                states=len(out),origin_mm=origin.tolist(),preparation=out[0],windows=results)

def square_mesh(n, diagonals=0):
    grid=np.linspace(0.,1.,n+1)
    X,Y=np.meshgrid(grid,grid);points=np.column_stack((X.ravel(),Y.ravel()))
    cells=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i;b=a+1;c=a+n+1;d=c+1
            flip=(i+j)%2 if diagonals==2 else diagonals
            cells.extend(((a,b,c),(b,d,c)) if flip else ((a,b,d),(a,d,c)))
    return FEMMesh(points,points.copy(),np.asarray(cells),np.empty((0,2),int),np.empty(0),
                   np.flatnonzero(points[:,1]==0),np.empty(0,int),np.zeros(len(points),int))


def reference_energy(F,G,bulk):
    j=np.linalg.det(F)
    return G*.5*(j**(-2/3)*(np.sum(F*F,axis=(-2,-1))+1)-3)+bulk*.5*(j-1)**2


def material_points(refinement):
    cfg=elastic_config(GE)
    pure=body_material.from_config(cfg)
    F=np.array([np.eye(2),[[1.,.72],[0.,1.]],[[.78,0.],[0.,.82]],[[1.38,.12],[0.,1.04]],
                [[np.cos(.53),-np.sin(.53)],[np.sin(.53),np.cos(.53)]]])
    unit=np.eye(4).reshape(4,2,2)
    expected_p=np.stack([reference_energy(F+1e-25j*D,GE,K).imag/1e-25 for D in unit],axis=-1).reshape(-1,2,2)
    A=np.broadcast_to(np.eye(3),(len(F),3,3))
    expected_c=np.stack([stored_stress(F+1e-25j*D,A,GE,0.,K).imag/1e-25 for D in unit],axis=-1).reshape(-1,4,4)
    prod=elastic_response(F,pure)
    scale=lambda z: max(GE,np.linalg.norm(z))
    Perrors=[np.linalg.norm(p-e)/scale(e) for p,e in zip(prod['P'],expected_p)]
    Cerrors=[np.linalg.norm(p-e)/scale(e) for p,e in zip(prod['tangent'],expected_c)]
    maxwell=replace(pure,mu_ov=(GV,),alpha_ov=(2.,),viscous_flow='isochoric_maxwell',relaxation_s=TAU)
    relaxation=[]
    gamma=1e-5
    for n in (5,10,20,40):
        dt=TAU/n;metric=np.eye(3)[None];errors=[]
        for step in range(1,n+1):
            q=response(np.array([[[1.,gamma],[0.,1.]]]),metric,dt,maxwell,tangent=False)
            metric=q['metric']
            expected=GE*gamma+GV*gamma*(1+dt/TAU)**(-step)
            errors.append(abs(q['P'][0,0,1]-expected)/(GV*gamma))
        stress=q['P'][0,0,1]
        relaxation.append(dict(steps=n,dt_s=dt,BE_max_relative_error=max(errors),
            continuum_end_relative_error=abs(stress-(GE+GV*np.exp(-1))*gamma)/(GV*gamma),end_stress_MPa=stress))
    extreme=[]
    for a,depths in ((33,(.15,.5,2.)),(57,(.15,.25,2.))):
        case=refinement/'stageD_2mm'/f'BladeA_R20_{a}'
        audit=read(case/'audit.json');report=read(case/'result.json')
        assert audit['passed'] and audit['source_result_sha256']==hashlib.sha256((case/'result.json').read_bytes()).hexdigest()
        for d in depths:
            i=next(i for i,r in enumerate(report['steps']) if abs(r['depth_mm']-d)<1e-11)
            raw=(case/'states'/f'{i:04d}.npz').read_bytes();digest=hashlib.sha256(raw).hexdigest()
            assert digest==audit['rows'][i]['checkpoint_sha256']
            with np.load(io.BytesIO(raw)) as st:
                t=st['triangles'];X=st['reference_xy'][t];x=st['positions'][t]
                dm=(X[:,1:]-X[:,:1]).transpose(0,2,1);inv=np.linalg.inv(dm)
                Fs=(x[:,1:]-x[:,:1]).transpose(0,2,1)@inv
                metrics=st['previous_material_metric'];dt=float(st['material_dt_s'])
                for criterion,e in [('min_J',int(np.linalg.det(Fs).argmin())),('max_condition',int(np.linalg.cond(Fs).argmax()))]:
                    f=Fs[e:e+1];metric=metrics[e:e+1];direction=np.outer([.6,.8],inv[e,0])
                    length=np.linalg.norm(direction);D=direction/length
                    q=response(f,metric,dt,maxwell);action=(q['tangent'][0]@D.ravel()).reshape(2,2)
                    samples=[]
                    for eps in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8):
                        high=response(f+eps*D,metric,dt,maxwell,tangent=False)['P'][0]
                        low=response(f-eps*D,metric,dt,maxwell,tangent=False)['P'][0]
                        err=np.linalg.norm((high-low)/(2*eps)-action)/max(GE,np.linalg.norm(action))
                        samples.append(dict(perturbation_F=eps,node_motion_mm=eps/length,relative_tangent_error=err))
                    extreme.append(dict(angle=a,depth_mm=d,criterion=criterion,element=e,J=np.linalg.det(f[0]),
                        input_metric_eigenvalues=np.linalg.eigvalsh(metric[0]).tolist(),physical_dt_s=dt,
                        source_sha256=digest,accept_record_sha256=audit['rows'][i]['accepted_record_sha256'],samples=samples))
    return dict(elastic_stress_relative=Perrors,elastic_tangent_relative=Cerrors,
        elastic_energy_max_abs_MPa=float(max(abs(prod['energy']-reference_energy(F,GE,K)))),
        relaxation=relaxation,real_material_tangents=extreme)


def affine_patch():
    rows=[]
    for diag,theta in ((0,0.),(1,.31),(2,-.46)):
        m=square_mesh(5,diag)
        Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        m.reference_xy=m.reference_xy@Q.T
        X=m.reference_xy;tri=m.triangles;edges=np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))
        un,ix,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
        edges=edges[ix[count==1]]
        for kind,A in [('shear',np.array([[1.,.45],[-.1,.92]])),('compression',np.diag([1.,.81])),('rigid',np.array([[.8,-.6],[.6,.8]]))]:
            F=Q@A@Q.T;u=X@(F-np.eye(2)).T + np.array([.07,-.11])
            P=np.array([reference_energy(F+1e-25j*E,GE,K).imag/1e-25 for E in np.eye(4).reshape(-1,2,2)]).reshape(2,2)
            rr,_,dofs,energy,_=body_material.bulk(m,u,elastic_config(GE),with_tangent=False)
            numerical=np.zeros_like(X);np.add.at(numerical.ravel(),dofs.ravel(),rr.ravel())
            exact=np.zeros_like(X);line=X[edges[:,1]]-X[edges[:,0]]
            tr=np.column_stack((line[:,1],-line[:,0]))@P.T*WIDTH/2
            for col in (0,1):np.add.at(exact,edges[:,col],tr)
            Fh=np.eye(2)+np.einsum('eai,eaJ->eiJ',u[tri],m.bulk_data[1])
            rows.append(dict(pattern=diag,rotation_rad=theta,mode=kind,max_F_error=float(np.max(abs(Fh-F))),
                force_relative=np.linalg.norm(numerical-exact)/max(GE*WIDTH,np.linalg.norm(exact)),
                energy_error_Nmm=abs(energy-WIDTH*reference_energy(F,GE,K))))
    return rows


def manufactured(X):
    k=2*np.pi
    x,y=X[...,0],X[...,1];sx=.045*np.sin(k*y);sy=.035*np.sin(k*(x+sx))
    a=.045*k*np.cos(k*y);b=.035*k*np.cos(k*(x+sx))
    F=np.stack((np.ones_like(a),a,b,1+a*b),axis=-1).reshape(X.shape[:-1]+(2,2))
    return np.stack((sx,sy),axis=-1),F


def exact_piola(X,G):
    _,F=manufactured(X)
    # det F = 1 identically for the composed shear, including its spatial derivatives.
    return G*(F-(np.sum(F*F,axis=(-2,-1))+1)[...,None,None]*np.swapaxes(np.linalg.inv(F),-1,-2)/3)


def quadrature(order):
    t,w=np.polynomial.legendre.leggauss(order);t=(t+1)/2;w=w/2
    a,b=np.meshgrid(t,t,indexing='ij')
    return np.stack(((1-a)*(1-b),a,(1-a)*b),axis=-1).reshape(-1,3),(2*(1-a)*np.outer(w,w)).ravel()


def manufactured_case(n,G):
    mesh=square_mesh(n,2);X=mesh.reference_xy;t=mesh.triangles;area,gr,dofs=mesh.bulk_data
    cfg=elastic_config(G);reference_u,_=manufactured(X)
    bdy=np.any((X==0)|(X==1),axis=1);free=np.flatnonzero(np.repeat(~bdy,2))
    loads=[]
    for order in (4,6):
        bary,w=quadrature(order);pos=np.einsum('qa,eai->eqi',bary,X[t]);b=np.zeros_like(pos)
        for j in range(2):
            perturb=np.zeros(2,dtype=complex);perturb[j]=1.e-24j
            b-=exact_piola(pos+perturb,G)[...,j].imag/1.e-24
        local=WIDTH*np.einsum('e,q,qa,eqi->eai',area,w,bary,b)
        force=np.zeros_like(X);np.add.at(force,t,local);loads.append(force)
    load=loads[-1].ravel()
    u=reference_u.copy()
    rridx=np.broadcast_to(dofs[:,:,None],(len(t),6,6)).ravel()
    ccidx=np.broadcast_to(dofs[:,None,:],(len(t),6,6)).ravel()
    for it in range(15):
        r,k,_,_,_=body_material.bulk(mesh,u,cfg)
        g=-load.copy();np.add.at(g,dofs.ravel(),r.ravel())
        if np.linalg.norm(g[free])<2e-10:break
        Kff=coo_matrix((k.ravel(),(rridx,ccidx)),shape=(u.size,u.size)).tocsc()[free][:,free]
        u.ravel()[free]-=spsolve(Kff,g[free])
    else:raise RuntimeError('Manufactured Newton solution did not converge')
    gx,gw=np.polynomial.legendre.leggauss(64);s=(gx+1)/2
    edge=np.column_stack((s,np.ones_like(s)));weight=np.sin(np.pi*s)**2
    expected=WIDTH*.5*(gw @ (exact_piola(edge,G)[:,0,1]*weight))
    top=np.flatnonzero(X[:,1]==1);result=float(g.reshape(-1,2)[top,0]@np.sin(np.pi*X[top,0])**2)
    bary,w=quadrature(6);Z=np.einsum('qa,eai->eqi',bary,X[t]);analytic,DF=manufactured(Z)
    interpolation=np.einsum('qa,eai->eqi',bary,u[t])
    weighted=area[:,None]*w
    F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u[t],gr);J=np.linalg.det(F)
    p=K*(J-1);err=interpolation-analytic
    edges=np.sort(np.vstack((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1)
    order=np.lexsort((edges[:,1],edges[:,0]));pair=np.flatnonzero(np.all(edges[order][1:]==edges[order][:-1],axis=1))
    jump=p[order[pair]%len(t)]-p[order[pair+1]%len(t)]
    length=np.linalg.norm(X[edges[order[pair],1]]-X[edges[order[pair],0]],axis=1)
    pressure_jump=float(np.sqrt(np.average(jump**2,weights=length)))
    eL2=np.sqrt(np.sum(weighted*np.sum(err**2,axis=2))/np.sum(weighted*np.sum(analytic**2,axis=2)))
    return dict(n=n,h_mm=1/n,shear_MPa=G,K_over_G=K/G,free_L2_N=float(np.linalg.norm(g[free])),
        quadrature_load_rel=np.linalg.norm(loads[0]-loads[1])/np.linalg.norm(loads[1]),iterations=it,
        displacement_L2_relative=eL2,weighted_top_shear_N=result,exact_weighted_top_shear_N=expected,
        reaction_relative=abs(result-expected)/abs(expected),gradient_RMS=float(np.sqrt(np.sum(weighted*np.sum((F[:,None]-DF)**2,axis=(-2,-1))))),
        J_abs_rms=float(np.sqrt(area@((J-1)**2))),pressure_rms_MPa=float(np.sqrt(area@(p*p))),
        pressure_rms_over_G=float(np.sqrt(area@(p*p))/G),pressure_face_jump_rms_MPa=pressure_jump,pressure_p01_p05_p50_p95_p99_MPa=np.percentile(p,[1,5,50,95,99]).tolist(),
        pressure_min_max_MPa=[float(p.min()),float(p.max())])

def hertz_case(h, halfwidth, *, dhat=.0002, tool_h=.025, extent=3., stiffness=6.6):
    """Native IPC contact on an independent infinite plane-strain surface compliance (line loading)."""
    from scipy.linalg import solve, toeplitz
    from prototype.ipc_reference import toolkit, weighted_collisions
    ipc=toolkit();ipc.set_num_threads(2)
    radius=100.;E_star=4.;L=12.;a=halfwidth
    axis=np.arange(-round(extent/h),round(extent/h)+1)*h;n=len(axis)
    # A consistent log displacement datum L is used both for the imposed approach and the compliance.
    analytic_force=np.pi*E_star*a*a/(4*radius)*WIDTH
    indent=a*a/(2*radius)*(np.log(2*L/a)+.5)
    def primitive(z):
        return z*(np.log(L/np.abs(z))+1)
    center=np.arange(n)*h
    compliance=toeplitz((2/(np.pi*E_star))*(primitive(center+h/2)-primitive(center-h/2)))
    H=WIDTH*h*solve(compliance,np.eye(n),assume_a='pos')
    tool_x=(np.arange(-round((extent+.4)/tool_h),round((extent+.4)/tool_h)+1)+.31)*tool_h
    profile=tool_x**2/(2*radius)
    body=np.column_stack((axis,np.zeros(n)));tool=np.column_stack((tool_x,profile+.025))
    rest=np.vstack((body,tool))
    edges=np.vstack((np.column_stack((np.arange(n-1),np.arange(1,n))),
                     n+np.column_stack((np.arange(len(tool)-1),np.arange(1,len(tool))))))
    mesh=ipc.CollisionMesh(rest,edges);mesh.can_collide=ipc.make_static_obstacle_filter(n)
    barrier=ipc.BarrierPotential(dhat,WIDTH*stiffness,True)
    x=rest.copy();x[n:,1]=profile-indent
    v=np.maximum(indent-axis**2/(2*radius)+.7*dhat,h*h/(8*radius)+.01*dhat)
    grid=np.union1d(axis,tool_x[(tool_x>=axis[0])&(tool_x<=axis[-1])])
    rigid=np.interp(grid,tool_x,x[n:,1])
    def gap(z):return rigid+np.interp(grid,axis,z)
    def evaluate(z,tangent):
        x[:n,1]=-z
        contacts=weighted_collisions(ipc,mesh,x,dhat)
        grad=barrier.gradient(contacts,mesh,x).reshape(-1,2)
        g=H@z-grad[:n,1]
        value=.5*z@(H@z)+barrier(contacts,mesh,x)
        if not tangent:return value,g,grad,None
        matrix=barrier.hessian(contacts,mesh,x,ipc.PSDProjectionMethod.CLAMP).tocsc()
        dof=2*np.arange(n)+1
        return value,g,grad,H+matrix[dof][:,dof].toarray()
    for it in range(130):
        E,g,gc,M=evaluate(v,True)
        if np.linalg.norm(g)<1e-8:break
        direction=solve(M,-g,assume_a='pos')
        gaps=gap(v);change=np.interp(grid,axis,direction);mask=change<0
        alpha=min(1.,.9*min(gaps[mask]/-change[mask])) if np.any(mask) else 1.
        for _ in range(35):
            trial=v+alpha*direction
            Et,gt,_,_=evaluate(trial,False)
            if Et<=E+1.e-4*alpha*(g@direction)+1.e-15 and gap(trial).min()>0:
                v=trial;break
            alpha*=.5
        else:raise RuntimeError('Hertz energy line search failed')
    else:raise RuntimeError(f'Hertz equilibrium failed at h={h}, residual={np.linalg.norm(g)}')
    p=gc[:n,1]/(WIDTH*h);F=-gc[n:,1].sum()
    peak=E_star*a/(2*radius)
    def int_sqrt(xp):
        y=np.clip(xp/a,-1.,1.)
        return peak*a*.5*(y*np.sqrt(np.maximum(0.,1-y*y))+np.arcsin(y))
    pref=(int_sqrt(axis+h/2)-int_sqrt(axis-h/2))/h
    active=np.flatnonzero(p>1e-9*peak)
    last=(axis[active[-1]]-axis[active[0]])/2
    measured=last+h/2
    moment_a=2*np.sqrt(np.sum(p*axis**2)/np.sum(p))
    return dict(surface_h_mm=h,tool_h_mm=tool_h,extent_mm=extent,dhat_mm=dhat,kappa=stiffness,E_star_MPa=E_star,
        target_halfwidth_mm=a,prescribed_approach_mm=indent,log_datum_mm=L,total_force_N=F,reference_force_N=analytic_force,
        line_load_N_mm=F/WIDTH,force_relative=abs(F/analytic_force-1),halfwidth_from_support_mm=measured,
        support_resolution_half_interval_mm=h/2,halfwidth_relative=abs(measured/a-1),moment_halfwidth_mm=moment_a,
        pressure_relative_L2=np.linalg.norm(p-pref)/np.linalg.norm(pref),load_width_relative=abs(F/(np.pi*E_star*measured**2*WIDTH/(4*radius))-1),
        peak_pressure_MPa=float(p.max()),exact_peak_MPa=peak,gap_min_mm=float(gap(v).min()),free_L2_N=float(np.linalg.norm(g)),
        iterations=it,axis_mm=axis.tolist(),pressure_MPa=p.tolist(),cell_average_exact_MPa=pref.tolist(),displacement_mm=v.tolist())


def friction_reference():
    from prototype.ipc_reference import toolkit
    ipc=toolkit();ipc.set_num_threads(2)
    rows=[]
    for mu in (0.,.35):
        p=IPCProblem('patch',mu=mu,local_spacing=.2,surface_spacing=.1,dhat=.01)
        pos=p.rest.copy();pos[p.n:,1]-=2*p.dhat-.004
        c=p.collisions(pos);normal=p.barrier.gradient(c,p.cmesh,pos[p.surface]).reshape(-1,2)
        N=-normal[p.surface>=p.n,1].sum()
        assert N>0
        lag=p.lag(pos)
        for dt in (.0125,.025):
            eps=.0004*dt
            pot=ipc.FrictionPotential(eps)
            for ratio in (-3.,-1.,-.4,0.,.4,1.,3.):
                delta=np.zeros_like(pos[p.surface]);delta[p.surface>=p.n,0]=ratio*eps
                g=np.zeros_like(delta) if lag is None else pot.gradient(lag,p.cmesh,delta).reshape(-1,2)
                T=-g[p.surface>=p.n,0].sum();work=float(g.ravel()@delta.ravel())
                r=abs(ratio);smooth=2*r-r*r if r<1 else 1.
                expected=-np.sign(ratio)*mu*N*smooth
                rows.append(dict(mu=mu,normal_N=N,dt_s=dt,epsilon_v_mm_s=.0004,epsilon_u_mm=eps,
                    ratio=ratio,tangent_N=T,expected_tangent_N=expected,normalized_error=abs(T-expected)/max(N*.35,1e-12),
                    friction_work_Nmm=work,is_request_stick=False))
    return rows

def failed_increment(folder, *, use_production_backend=True):
    """Restore the actual parent, predictor pair and failed material increment."""
    from prototype.material_performance import enable_performance

    report = read(folder/'result.json')
    audit = read(folder/'audit.json')
    rows = report['steps']
    if (report['status'] not in ('failed', 'budget_exhausted') or not report.get('failed_attempt')
            or len(rows) < 2 or audit['states'] != len(rows)
            or not audit['trajectory_checked'] or audit['qualified_prefix_states'] != len(rows)):
        raise ValueError('A failed run with a complete independently audited parent is required')
    if not audit['passed'] or audit['source_result_sha256'] != hashlib.sha256((folder/'result.json').read_bytes()).hexdigest():
        raise ValueError('Audit and failed result differ')
    root = Path(__file__).resolve().parents[1]
    for name, digest in report['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Production source identity differs: '+name)
    settings = report['settings']
    problem = make_problem(**{k:v for k,v in settings.items() if k in inspect.signature(make_problem).parameters})
    if use_production_backend:
        enable_performance(problem, report['performance_policy'])
    problem.solver_diagnostics = settings['diagnostics']
    return restore_failed_pair(problem, folder, report, audit)


def restore_failed_pair(problem, folder, report, audit):
    """Bind the two predictor states and both physical histories for a diagnostic."""
    rows = report['steps']
    history = []
    for i in range(len(rows)-2, len(rows)):
        path = folder/'states'/f'{i:04d}.npz'
        proof = audit['rows'][i]
        if not proof['passed'] or hashlib.sha256(path.read_bytes()).hexdigest() != proof['checkpoint_sha256'] or record_digest(rows[i]) != proof['accepted_record_sha256']:
            raise ValueError('Predictor checkpoint or record mismatch')
        with np.load(path, allow_pickle=False) as state:
            np.testing.assert_array_equal(state['reference_xy'], problem.rest)
            np.testing.assert_array_equal(state['triangles'], problem.mesh.triangles)
            if history:
                np.testing.assert_array_equal(state['previous_positions'], history[-1][1])
                np.testing.assert_array_equal(state['previous_material_metric'], problem.material_metric)
            history.append((float(state['depth_mm']), state['positions']))
            problem.material_metric = state['material_metric']
            problem.material_dissipation = float(state['cumulative_viscous_dissipation_n_mm'])
    problem.seed_predictor_history(history)
    old, parent = history[-1]
    attempt = report['failed_attempt']
    target = attempt['attempted_depth_mm']
    if old != attempt['from_depth_mm'] or attempt.get('history_pairing') != 'actual_attempt_start':
        raise ValueError('Last checkpoint and failed attempt have different parents')
    with np.load(folder/'failed_trial.npz', allow_pickle=False) as trial:
        np.testing.assert_array_equal(trial['previous_positions'], parent)
        for key in ('material_metric', 'previous_material_metric'):
            np.testing.assert_array_equal(trial[key], problem.material_metric)
        dt = float(trial['material_dt_s'])
        if dt != (target-old)/problem.loading_rate or float(trial['cumulative_viscous_dissipation_n_mm']) != problem.material_dissipation:
            raise ValueError('Failed material time or cumulative history differs')
        endpoint = trial['positions']
    problem.previous = parent.copy()
    problem.previous_material_metric = problem.material_metric.copy()
    problem.material_dt = dt
    problem.set_friction_dt(dt)
    return problem, report, parent, endpoint


@contextmanager
def study_runtime(settings, output, label):
    from numerical_runtime import numerical_runtime
    from run_support import compute_process_policy
    from threadpoolctl import threadpool_limits

    with numerical_runtime(settings['threads'],1,settings['physical_core_affinity']) as runtime, threadpool_limits(limits=1,user_api='blas'), compute_process_policy() as process:
        try:
            yield
        finally:
            dump(output/(label+'_runtime.json'),dict(threads=runtime,process=process))
    dump(output/(label+'_runtime.json'),dict(threads=runtime,process=process))


def replay_failed_increment(folder, output):
    from contextlib import redirect_stdout
    from prototype.ipc_reference import adaptive_load_steps, LoadStepFailure
    from prototype.material_performance import close_performance

    settings = read(folder/'result.json')['settings']
    with study_runtime(settings,output,'replay'), (output/'replay.log').open('x', encoding='utf-8') as log, redirect_stdout(log):
        p, r, parent, original = failed_increment(folder)
        attempt = r['failed_attempt']
        actual = []

        def rejected(event):
            positions = p.trial.copy()
            np.savez_compressed(output/('replay_'+event['predictor']+'.npz'), positions=positions)
            actual.append(dict(**event, endpoint_max_abs_difference_from_saved_mm=float(np.max(abs(positions-original))),
                               endpoint_bitwise_equal_to_saved=bool(np.array_equal(positions, original))))
            dump(output/'replay.json', dict(attempt=attempt, rejected=actual, finished=False))

        try:
            accepted = list(adaptive_load_steps(p, parent, attempt['from_depth_mm'], attempt['attempted_depth_mm'],
                time.monotonic()+1800, settings['minimum_increment_mm'], rejected,
                nominal_target=attempt['nominal_target_mm'], level=attempt['subdivision_level']))
            status = 'accepted'
        except (LoadStepFailure,TimeoutError) as error:
            status, accepted = f'{type(error).__name__}: {error}', []
        finally:
            close_performance(p)
        dump(output/'replay.json', dict(attempt=attempt, rejected=actual, finished=True,
             status=status, accepted=[record for _,record in accepted], dt_s=p.material_dt))


def bisect_failed_increment(folder, output):
    """One event-local time comparison, with two genuine material updates at most."""
    from contextlib import redirect_stdout
    from prototype.ipc_reference import adaptive_load_steps, LoadStepFailure
    from prototype.material_performance import close_performance

    settings = read(folder/'result.json')['settings']
    review = make_problem(**{k:v for k,v in settings.items() if k in inspect.signature(make_problem).parameters})
    with study_runtime(settings,output,'event_half'), (output/'event_half.log').open('x', encoding='utf-8') as log, redirect_stdout(log):
        p, report, parent, _ = failed_increment(folder)
        attempt = report['failed_attempt']
        old, end = attempt['from_depth_mm'], attempt['attempted_depth_mm']
        half = (end-old)/2
        depth, x = old, parent
        result = dict(original_parent_depth_mm=old, target_mm=end, half_increment_mm=half,
                      accepted=[], rejected=[], status='running', production_history_changed=False)
        deadline = time.monotonic()+720
        try:
            for target in (old+half,end):
                metric = p.material_metric.copy()
                dissipated = p.material_dissipation

                def reject(event):
                    np.testing.assert_array_equal(p.material_metric,metric)
                    if p.material_dissipation != dissipated:
                        raise ValueError('Rejected substep committed dissipation')
                    proof = p.physical_chord_certificate(x,p.trial)
                    result['rejected'].append(dict(**event, trial_chord=proof))
                    dump(output/'event_half.json', result)

                for value, rec in adaptive_load_steps(p,x,depth,target,deadline,half,reject):
                    actual = rec['depth_mm']
                    blade = x[p.n:].copy()
                    blade[:,1] -= actual-depth
                    state = dict(reference_xy=p.rest, triangles=p.mesh.triangles, body_nodes=p.n,
                        positions=value, previous_positions=p.previous, depth_mm=actual, **p.checkpoint_data())
                    np.testing.assert_array_equal(state['previous_positions'],x)
                    np.testing.assert_array_equal(state['previous_material_metric'],metric)
                    np.testing.assert_allclose(p.material_dissipation,dissipated+rec['material']['viscous_dissipation_increment_n_mm'],rtol=0,atol=1.e-14)
                    checked = review.audit_state(state,rec,incremental_blade_positions=blade)
                    if not (checked['passed'] and checked['trajectory_passed']):
                        raise ValueError('Independent substep audit failed')
                    path = output/f'event_half_{len(result["accepted"])+1}.npz'
                    np.savez_compressed(path,**state)
                    result['accepted'].append(dict(record=rec, audit=checked, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    x,depth = value,actual
                    dump(output/'event_half.json', result)
            result['status']='completed'
        except (LoadStepFailure,TimeoutError) as error:
            path = output/'event_half_rejected.npz'
            np.savez_compressed(path, positions=p.trial, previous_positions=x, **p.checkpoint_data())
            result.update(status='stopped', error=f'{type(error).__name__}: {error}',
                          final_trial_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        finally:
            close_performance(p)
        result['last_audited_depth_mm'] = depth
        dump(output/'event_half.json',result)


def diagnose_failed_chord(folder, output):
    from prototype.material_performance import close_performance
    from prototype.tip_contact import TipContactTracker
    from tools.audit_virtual_chord import separated_ev_prefix, verify_witness
    from virtual_path_geometry import ev_crossing_witnesses

    p, report, x, y = failed_increment(folder, use_production_backend=False)
    try:
        p.solver_diagnostics = True
        delta = y-x
        certificate = p.physical_chord_certificate(x, y)
        a = x[p.surface]
        b = y[p.surface]
        candidates = p.ipc.Candidates()
        candidates.build(p.cmesh, a, b, 0.)
        edges, faces = p.cmesh.edges, p.cmesh.faces
        limits = []
        for lo,hi in [(0.,1.), *([certificate['unresolved_interval']] if 'unresolved_interval' in certificate else [])]:
            left, right = x+lo*delta, y if hi == 1. else x+hi*delta
            p.safe_step(left,right-left)
            pairs = []
            for cand in candidates.ev_candidates:
                hit, fraction = cand.ccd(cand.dof(left[p.surface],edges,faces), cand.dof(right[p.surface],edges,faces))
                if hit:
                    edge = list(map(int, edges[cand.edge_id]))
                    pairs.append(dict(edge_global=p.surface[edge].tolist(), vertex_global=int(p.surface[cand.vertex_id]),
                                      native_fraction=float(fraction)))
            limits.append(dict(interval=[lo,hi],**p.step_limits, ev_limits=sorted(pairs,key=lambda pair:pair['native_fraction'])))
        witnesses = sorted(ev_crossing_witnesses(p,x,y),key=lambda w:w['bracket'][0])
        tracker = TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets,p.blade_landmarks)
        for w in witnesses:
            i,j,k = w['vertices_global']
            w['exact_binary64_endpoint_proof'] = verify_witness(x,y,[i,j,k],w['bracket'])
            if not w['exact_binary64_endpoint_proof']['passed']:
                raise ValueError('Interior crossing failed independent rational verification')
            t = sum(w['bracket'])/2
            beta = np.mean([float(item) for item in w['beta']])
            w.update(time_fraction_estimate=t, varying_time_bracket_s=(np.asarray(w['bracket'])*p.material_dt).tolist(),
                collision_lab_mm=(x[k]+t*delta[k]).tolist(),
                original_top_edge=tuple(sorted((i,j))) in tracker.top_edge_map,
                blade_nose_vertex_local=int(k-p.n) if k in tracker.nose_nodes else None)
            if w['original_top_edge']:
                w['material_edge_S_mm'] = tracker.s[[i,j]].tolist()
                w['interpolated_material_offset_mm_estimate'] = float((1-beta)*tracker.s[i]+beta*tracker.s[j]-tracker.origin)
        limit = witnesses[0]['bracket'][0] if witnesses else 1.
        checks = []
        for cand in candidates.ev_candidates:
            ids = p.surface[[*edges[cand.edge_id],cand.vertex_id]].tolist()
            checks.append(dict(vertices_global=ids,**separated_ev_prefix(x,y,ids,limit)))
        ev_only = all(len(getattr(candidates,kind+'_candidates')) == 0 for kind in ('vv','ee','fv','pv'))
        separation = dict(limit=limit,all_swept_candidates_are_ev=ev_only,
                          all_separated=ev_only and all(item['separated'] for item in checks),checks=checks)
        tri = p.mesh.triangles
        f = np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:p.n]-p.mesh.reference_xy)[tri],p.grad)
        d = np.einsum('eai,eaJ->eiJ',delta[:p.n][tri],p.grad)
        j0, j1, c2 = np.linalg.det(f), np.linalg.det(f+d), np.linalg.det(d)
        c1 = f[:,0,0]*d[:,1,1]+d[:,0,0]*f[:,1,1]-f[:,1,0]*d[:,0,1]-d[:,1,0]*f[:,0,1]
        t = np.where(j0 <= j1, 0., 1.)
        positive = c2 > 0
        t[positive] = np.clip(-c1[positive]/(2*c2[positive]),0.,1.)
        minima = j0+t*(c1+t*c2)
        e = int(np.argmin(minima))
        determinant = dict(elements=len(tri), parent_min=float(j0.min()), endpoint_min=float(j1.min()),
            path_min=float(minima[e]), element=e, node_ids=tri[e].tolist(), time_fraction=float(t[e]),
            reference_centroid_xy_mm=p.rest[tri[e]].mean(axis=0).tolist(), min_nonpositive=int((minima<=0).sum()))
        target = report['failed_attempt']['attempted_depth_mm']
        try:
            endpoint = p.acceptance(y,target,0,0)
            endpoint_passed = True
        except LoadStepFailure as exc:
            # A rejected force check is itself an observation on this saved trial.
            _, gradient, _, min_j = p.evaluate(y, p.lag(y), tangent=False)
            norm = float(np.linalg.norm(gradient[p.free]))
            endpoint = dict(depth_mm=target, residual_n=norm if np.isfinite(norm) else None,
                            min_j=float(min_j) if np.isfinite(min_j) else None, error=str(exc))
            endpoint_passed = False
        response = p.material_response(y)
        metric = response['metric']
        np.savez_compressed(output/'reassembled_uncommitted_material.npz',metric=metric)
        fractions = [0.,1.]
        if witnesses:
            fractions += witnesses[0]['bracket']
        samples = [dict(time_fraction=z,intersections=bool(p.ipc.has_intersections(p.cmesh,(y if z==1. else x+z*delta)[p.surface]))) for z in sorted(fractions)]
        result = dict(certificate=certificate,limits=limits,determinant=determinant, cross_witnesses=witnesses,
            exact_pre_crossing_prefix=separation if witnesses else None,
            exact_full_chord_ev_separation=None if witnesses else separation,
            source_status=report['status'], endpoint_checks_passed=endpoint_passed, endpoint_assembly='fresh_native_unbatched',
            broad_phase_counts={kind:len(getattr(candidates,kind+'_candidates')) for kind in ('ev','vv','ee','fv','pv')},
            static_intersection_samples=samples, endpoint=endpoint, dt_s=p.material_dt, epsilon_u_mm=p.friction_smoothing_mm,
            parent_committed_history_identical_to_both_saved_failed_metrics=True,
            uncommitted_trial_metric_max_abs_change=float(abs(metric-p.material_metric).max()),
            uncommitted_trial_metric_min_eigenvalue=float(np.linalg.eigvalsh(metric).min()),
            committed_dissipation_n_mm=p.material_dissipation, max_body_motion_um=1000*float(np.linalg.norm(delta[:p.n],axis=1).max()))
        dump(output/'failed_chord.json',result)
    finally:
        close_performance(p)


def observe_sa(folder, protocol, max_depth):
    """Read-only geometric observation on each SHA-bound original-material surface."""
    from prototype.contact_trace import geometry_metadata
    from prototype.tip_contact import TipContactTracker

    report = read(folder/'result.json')
    audit = read(folder/'audit.json')
    digest = hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()
    if not (audit['passed'] and audit['trajectory_checked'] and audit['source_result_sha256'] == digest):
        raise ValueError('Accepted trajectory and audit do not match')
    config, physical_gap, facets, _ = geometry_metadata(report)
    narrow, wide = protocol['sensitivity_geometric_gaps_mm']
    primary = protocol['primary_geometric_gap_mm']
    if not 0 < narrow < primary < wide or not np.isfinite(wide) or protocol['S0_mm'] != config['drop_s_mm']:
        raise ValueError('Observation protocol and reference origin are incompatible')
    with np.load(folder/'states/0000.npz', allow_pickle=False) as state:
        trackers = {key:TipContactTracker(state['reference_xy'], state['triangles'], int(state['body_nodes']),
            config, gap, facets, report.get('signature',{}).get('blade_landmarks'))
            for key,gap in [('narrow',narrow), ('primary',primary), ('wide',wide)]}
    rows = []
    for i, record in enumerate(report['steps']):
        depth = record['depth_mm']
        if depth > max_depth + 1.e-12:
            break
        proof = audit['rows'][i]
        path = folder/'states'/f'{i:04d}.npz'
        if not (proof['passed'] and proof['trajectory_passed'] and record_digest(record) == proof['accepted_record_sha256']
                and hashlib.sha256(path.read_bytes()).hexdigest() == proof['checkpoint_sha256']):
            raise ValueError('Unbound observation frame: '+str(i))
        with np.load(path, allow_pickle=False) as state:
            x = state['positions']
        row = dict(index=i, depth_mm=depth, checkpoint_sha256=proof['checkpoint_sha256'],
                   accepted_record_sha256=proof['accepted_record_sha256'])
        observations = {}
        for key, tracker in trackers.items():
            item = tracker.observe(x, depth)
            if key == 'primary':
                row.update(pole_lab_mm=item['knife_tip_mm'], pole_gap_mm=item['gap_mm'],
                           nearest_projections=item['nearest_projection_candidates'])
            observations[key] = dict(pole_status=item['status'], pole_offset_mm=item['reference_offset_mm'],
                pole_candidates_mm=[c['reference_offset_mm'] for c in item['candidates']],
                nose_status=item['nose_status'], nose_offset_intervals_mm=item['nose_offset_intervals_mm'])
            if tracker.dhat == physical_gap and 'sliding_amount' in record:
                np.testing.assert_allclose(item['nose_offset_intervals_mm'],record['sliding_amount']['SA_intervals_mm'],rtol=0,atol=1.e-10)
        row['observations'] = observations
        row['native_nose_terms'] = None
        if 'nose_native_candidates' in record:
            row['native_nose_terms'] = []
            tracker = trackers['primary']
            for term in record['nose_native_candidates']:
                i0,i1 = term['edge_nodes']
                u = term['edge_alpha']
                offset = float((1-u)*tracker.s[i0]+u*tracker.s[i1]-tracker.origin)
                members = {key:[i for i,(lo,hi) in enumerate(obs['nose_offset_intervals_mm'])
                                if lo-tracker.tolerance <= offset <= hi+tracker.tolerance] for key,obs in observations.items()}
                row['native_nose_terms'].append(dict(**term,reference_offset_mm=offset,address_in_components=members))
        rows.append(row)
    tracker = trackers['primary']
    return dict(source=folder.name, result_sha256=digest, protocol=protocol, physical_dhat_mm=physical_gap,
                pole_node=tracker.tip_node, reference_tie_tolerance_mm=tracker.tolerance,
                original_top_edges=len(tracker.edges), nose_facets=facets, rows=rows)


def interval_distance_mm(a, b):
    """Hausdorff distance between two sorted unions of closed material intervals."""
    if not len(a) or not len(b):
        return None

    def directed(left, right):
        left, right = np.asarray(left, float), np.asarray(right, float)
        # Distance is piecewise linear, with possible interior maxima at gap midpoints.
        mid = (right[:-1,1]+right[1:,0])/2
        inside = np.any((left[:,0,None]<=mid) & (mid<=left[:,1,None]),axis=0)
        query = np.r_[left.ravel(), mid[inside]][:,None]
        return np.maximum(0.,np.maximum(right[:,0]-query,query-right[:,1])).min(axis=1).max()

    return float(max(directed(a,b),directed(b,a)))


def compare_shallow_reference(report, observations, reference):
    """Pair measured forces and independent primary SA on common accepted depths."""
    if (reference['protocol']!=observations['protocol'] or
            reference['angle']!=report['signature']['config']['inclination_deg']):
        raise ValueError('Reference angle or observation protocol differs')
    if len(report['steps'])!=len(observations['rows']):
        raise ValueError('Incomplete current observation coverage')
    previous={round(row['depth_mm'],12):row for row in reference['rows']}
    rows=[]
    for step,obs in zip(report['steps'],observations['rows']):
        if record_digest(step)!=obs['accepted_record_sha256']:
            raise ValueError('Observation is not bound to this accepted record')
        depth=step['depth_mm']
        old=previous.get(round(depth,12))
        if old is None:continue
        a=obs['observations']['primary']; b=old['observations']['primary']
        left=a['nose_offset_intervals_mm']; right=b['nose_offset_intervals_mm']
        pole=(a['pole_offset_mm']-b['pole_offset_mm'] if a['pole_status']==b['pole_status']=='unique' else None)
        rows.append(dict(depth_mm=depth,current_sha256=obs['checkpoint_sha256'],reference_sha256=old['checkpoint_sha256'],
            Fz_N=step['reaction_fz_n'],reference_Fz_N=old['Fz_N'],Fx_N=step['reaction_x_n'],reference_Fx_N=old['Fx_N'],
            SA_set_distance_mm=interval_distance_mm(left,right),branches=[len(left),len(right)],
            same_existence=bool(left)==bool(right),pole_status=[a['pole_status'],b['pole_status']],unique_pole_difference_mm=pole))
    if not rows or abs(rows[0]['depth_mm'])>1.e-10 or abs(rows[-1]['depth_mm']-report['target_mm'])>1.e-10:
        raise ValueError('Reference does not cover the requested shallow window')
    metrics={}
    for quantity,floor in [('Fz',.5),('Fx',.2)]:
        ref=np.array([r['reference_'+quantity+'_N'] for r in rows])
        error=np.array([r[quantity+'_N'] for r in rows])-ref
        scale=max(floor,float(np.mean(ref**2)**.5))
        rms=float(np.mean(error**2)**.5); maximum=float(np.max(abs(error)))
        metrics[quantity]=dict(rms_N=rms,max_N=maximum,scale_N=scale,rms_fraction=rms/scale,max_fraction=maximum/scale)
    return dict(rows=rows,forces=metrics,reference_result_sha256=reference['result_sha256'],
        reference_dhat_mm=reference['physical_dhat_mm'],current_dhat_mm=observations['physical_dhat_mm'],
        SA_set_max_mm=max((r['SA_set_distance_mm'] for r in rows if r['SA_set_distance_mm'] is not None),default=None),
        identical_depth_schedule=([round(r['depth_mm'],12) for r in observations['rows']]==list(previous)),
        interpretation='Joint contact, preparation and loading-history sensitivity on common saved depths')


def check_preparation(folder, output):
    """Start a distinct two-state history and bind its zero and first child to their own parents."""
    from contextlib import redirect_stdout
    from prototype.ipc_reference import adaptive_load_steps
    from prototype.material_case import audit
    from prototype.material_performance import enable_performance, close_performance
    from run_support import save_npz

    production = read(folder/'result.json')
    original_audit = read(folder/'audit.json')
    if not original_audit['passed'] or original_audit['source_result_sha256'] != hashlib.sha256((folder/'result.json').read_bytes()).hexdigest():
        raise ValueError('Original preparation provenance differs')
    settings = production['settings']
    parameters = {k:v for k,v in settings.items() if k in inspect.signature(make_problem).parameters}
    policy = dict(start='each mesh and angle starts from its own stress-free reference and identity metric',
                  preparation_force_tolerance_n=1.e-6, positive_depth_force_tolerance_n=1.e-4,
                  vertical_clearance_mm=settings['preparation_clearance_mm'],loading_rate_mm_s=settings['loading_rate'],
                  zero_definition='original prescribed geometric zero after preparation', physical_dhat_mm=settings['dhat'])
    study = dict(status='two-state preparation control', settings=dict(parameters,preparation_policy=policy),
                 source_sha256=production['source_sha256'], signature=production['signature'], steps=[],
                 study_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 performance_policy=production['performance_policy'], parent_run_result_sha256=original_audit['source_result_sha256'])
    for name,digest in production['source_sha256'].items():
        if hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Production source mismatch: '+name)
    (output/'states').mkdir(exist_ok=False)
    compare = []
    with study_runtime(settings,output,'preparation'), (output/'preparation.log').open('x',encoding='utf-8') as log, redirect_stdout(log):
        p = make_problem(**parameters)
        review = make_problem(**parameters)
        enable_performance(p,production['performance_policy'])
        try:
            x,old = p.rest.copy(),p.start
            deadline = time.monotonic()+240
            for i,target in enumerate((0.,settings['increment'])):
                p.force_tolerance_n = policy['preparation_force_tolerance_n'] if i == 0 else policy['positive_depth_force_tolerance_n']
                path = folder/'states'/f'{i:04d}.npz'
                proof = original_audit['rows'][i]
                if not proof['passed'] or record_digest(production['steps'][i]) != proof['accepted_record_sha256'] or hashlib.sha256(path.read_bytes()).hexdigest() != proof['checkpoint_sha256']:
                    raise ValueError('Original preparation checkpoint differs')
                with np.load(path,allow_pickle=False) as original:
                    if float(original['depth_mm']) != target:
                        raise ValueError('Original preparation depths differ')
                    if i == 0:
                        np.testing.assert_array_equal(original['previous_positions'],x)
                        np.testing.assert_array_equal(original['previous_material_metric'],p.material_metric)
                    for value,record in adaptive_load_steps(p,x,old,target,deadline,settings['minimum_increment_mm']):
                        if record['depth_mm'] != target:
                            raise ValueError('Unexpected extra preparation substep')
                        new_path = output/'states'/path.name
                        state = dict(positions=value, previous_positions=p.previous, reference_xy=p.rest, triangles=p.mesh.triangles,
                                     body_nodes=p.n, depth_mm=target, **p.checkpoint_data())
                        save_npz(new_path,**state)
                        if float(original['material_dt_s']) != p.material_dt:
                            raise ValueError('Preparation time differs')
                        compare.append(dict(depth_mm=target,source_checkpoint_sha256=proof['checkpoint_sha256'],
                            control_checkpoint_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                            original_positions_sha256=hashlib.sha256(original['positions'].tobytes()).hexdigest(),
                            control_positions_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                            original_metric_sha256=hashlib.sha256(original['material_metric'].tobytes()).hexdigest(),
                            control_metric_sha256=hashlib.sha256(p.material_metric.tobytes()).hexdigest(),
                            parent_positions_sha256=hashlib.sha256(state['previous_positions'].tobytes()).hexdigest(),
                            parent_metric_sha256=hashlib.sha256(state['previous_material_metric'].tobytes()).hexdigest(),
                            max_node_delta_um=1000*float(np.linalg.norm(value-original['positions'],axis=1).max()),
                            max_metric_component_delta=float(abs(p.material_metric-original['material_metric']).max()),
                            delta_Fz_N=record['reaction_fz_n']-production['steps'][i]['reaction_fz_n'],
                            delta_Fx_N=record['reaction_x_n']-production['steps'][i]['reaction_x_n']))
                        study['steps'].append(record)
                        x,old = value,target
            dump(output/'result.json',study)
            checked = audit(review,output,study)
            if not (checked['passed'] and checked['trajectory_checked'] and checked['states']==2):
                raise ValueError('Independent preparation audit failed')
            dump(output/'B6.json',history(output,output,'B6'))
            dump(output/'identity.json',dict(policy=policy,policy_sha256=record_digest(policy),comparison=compare,
                new_result_sha256=checked['source_result_sha256'],audit_sha256=hashlib.sha256((output/'audit.json').read_bytes()).hexdigest()))
        finally:
            close_performance(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement',type=Path,help='Verified historical A/R20 refinement directory')
    parser.add_argument('--output',type=Path,required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--history',type=Path,help='Optional first-principles reconstruction of one accepted case')
    parser.add_argument('--label',default='history')
    modes.add_argument('--replay-failure',type=Path,help='Re-solve only the saved last failed physical increment')
    modes.add_argument('--diagnose-failure',type=Path,help='Identify CCD and determinant restrictions on the saved chord')
    modes.add_argument('--bisect-failure',type=Path,help='One comparison with two physical half steps on that event')
    modes.add_argument('--observe-sa',type=Path,help='Export strict pole and geometric nose sets on accepted frames')
    parser.add_argument('--observation-protocol',type=Path,help='Frozen gap bands and signed reference origin')
    modes.add_argument('--check-preparation',type=Path,help='Bind a common stricter preparation and its first loaded child')
    parser.add_argument('--max-depth',type=float,default=.25)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.check_preparation:
        check_preparation(args.check_preparation,args.output)
        return
    if args.bisect_failure:
        bisect_failed_increment(args.bisect_failure,args.output)
        return
    if args.diagnose_failure:
        diagnose_failed_chord(args.diagnose_failure,args.output)
        return
    if args.observe_sa:
        if args.observation_protocol is None or Path(args.label).name != args.label or not np.isfinite(args.max_depth) or args.max_depth < 0:
            parser.error('SA observation requires a protocol, a filename label and nonnegative finite depth')
        dump(args.output/(args.label+'.json'), observe_sa(args.observe_sa,read(args.observation_protocol),args.max_depth))
        return
    if args.replay_failure:
        replay_failed_increment(args.replay_failure, args.output)
        return
    if args.history:
        if Path(args.label).name!=args.label:parser.error('Use a single filename for --label')
        dump(args.output/(args.label+'.json'),history(args.history,args.output,args.label))
        return
    if args.refinement is None:parser.error('--refinement is required for the benchmark suite')
    start=time.perf_counter()
    dump(args.output/'B1.json',material_points(args.refinement))
    dump(args.output/'B2.json',affine_patch())
    dump(args.output/'B3.json',[manufactured_case(n,g) for g in (GE+GV,GE) for n in (8,16,32,64)])
    cases=[(h,a,.0002,.025,3.,6.6) for a in (.8,1.) for h in (.10,.05,.025)]
    cases.extend([(.025,1.,.0002,.0125,3.,6.6),(.025,1.,.0002,.025,4.,6.6),
                  (.025,1.,.00125,.025,3.,6.6),(.025,1.,.0002,.025,3.,13.2)])
    cases.extend((.0125,a,dh,.0125,3.,6.6) for a in (.8,1.) for dh in (.00125,.0002))
    cases.extend((h,a,.000125,.0125,3.,6.6) for a in (.8,1.) for h in (.0125,.00625))
    dump(args.output/'B4.json',[hertz_case(h,a,dhat=dh,tool_h=th,extent=r,stiffness=k) for h,a,dh,th,r,k in cases])
    dump(args.output/'B5.json',friction_reference())
    print(f'Completed independent intact benchmarks in {time.perf_counter()-start:.2f} seconds',flush=True)


if __name__=='__main__':
    main()
