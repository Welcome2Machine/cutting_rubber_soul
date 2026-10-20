"""Locate whole-P2 Jacobian minima without changing the saved fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prototype.quadratic import quadratic_minimum, shape
from tools import run_augmented_blade as base
from run_support import write_json


def argminimum(v):
    a,b,c,ab,bc,ca = map(float,v)
    rr=2*(a+b-2*ab); ss=2*(a+c-2*ca)
    r=b-a-rr; s=c-a-ss; rs=4*(bc-a-.5*(r+s)-.25*(rr+ss))
    points=[(0.,0.),(1.,0.),(0.,1.)]
    for constant,linear,quad,axis in ((a,r,rr,0),(a,s,ss,1),(c,r-s+rs-2*ss,rr+ss-rs,2)):
        if quad>0:
            t=-linear/(2*quad)
            if 0<t<1: points.append(((t,0),(0,t),(t,1-t))[axis])
    det=4*rr*ss-rs*rs
    if rr>0 and det>0:
        x=(rs*s-2*ss*r)/det; y=(rs*r-2*rr*s)/det
        if x>0 and y>0 and x+y<1:points.append((x,y))
    vals=[a+r*x+s*y+rr*x*x+ss*y*y+rs*x*y for x,y in points]
    k=int(np.argmin(vals));x,y=points[k]
    return float(vals[k]),np.array([1-x-y,x,y])


def element_nodes(p,x,e):
    el=p.enrichment;tri=p.mesh.triangles[e]
    nodes=x[el.elements[e]].copy()
    for k,(a,b) in enumerate(((0,1),(1,2),(2,0))):
        if el.boundary_slots[e,k]:nodes[3+k]=(x[tri[a]]+x[tri[b]])/2
    return nodes


def chain(folder):
    r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    entries=[]
    if 'inherited_prefix' in r:
        entries=[row for row in chain(Path(r['inherited_prefix'])) if row[0]<r['inherited_depth_mm']-1e-12]
    entries.extend((rec['depth_mm'],folder/'states'/f'{i:04d}.npz') for i,rec in enumerate(r['steps']))
    return entries


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    r=json.loads((source/'result.json').read_text(encoding='utf-8'));s=r['settings']
    p=base.make_problem(s['angle'],coarse=s['coarse'],spacing=s['spacing']);el=p.enrichment
    files=[('accepted',source/'states'/f'{len(r["steps"])-1:04d}.npz'),('failed_trial',source/'failed_trial.npz')]
    report={'scope':'Fixed saved fields; trial is NOT equilibrium','states':{},'inputs':{}}
    plotted=[]
    for label,path in files:
        report['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:x=z['positions'];depth=float(z['depth_mm'] if 'depth_mm' in z else z['attempted_depth_mm'])
        u=x[:p.n]-p.mesh.reference_xy
        values=np.linalg.det(el.deformation(u,el.vertex_grad));low=quadratic_minimum(values)
        rows=[]
        for e in np.argsort(low)[:12]:
            minimum,l=argminimum(values[e]);np.testing.assert_allclose(minimum,low[e],atol=1e-12)
            F=np.eye(2)+np.einsum('ai,aJ->iJ',u[el.elements[e]],el.gradients(l[None])[e,0])
            direct=float(F[0,0]*F[1,1]-F[0,1]*F[1,0]);tri=p.mesh.triangles[e]
            nodes=element_nodes(p,x,e);current=shape(l)@nodes
            gauss=np.linalg.det(np.eye(2)+np.einsum('ai,qaJ->qiJ',u[el.elements[e]],el.grad[e]))
            rows.append(dict(element=int(e),min_j=minimum,direct_j=direct,barycentric=l.tolist(),
                reference_sn=(l@p.mesh.local_sn[tri]).tolist(),current_xy=current.tolist(),
                gauss_min_j=float(gauss.min()),corner_j=values[e,:3].tolist(),
                nodes=el.elements[e].tolist(),suppressed_mids=el.boundary_slots[e].tolist(),
                reference_area_mm2=float(el.area[e]),midpoint_bow_mm=np.linalg.norm(nodes[3:]-(nodes[:3]+nodes[[1,2,0]])/2,axis=1).tolist()))
        report['states'][label]={'depth_mm':depth,'worst':rows}
        plotted.append((label,x,int(np.argmin(low))))
    worst=report['states']['accepted']['worst'][0]['element'];hist=[]
    for depth,path in chain(source):
        report['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:u=z['positions'][:p.n]-p.mesh.reference_xy
        F=np.eye(2)+np.einsum('ai,qaJ->qiJ',u[el.elements[worst]],el.vertex_grad[worst])
        G=np.eye(2)+np.einsum('ai,qaJ->qiJ',u[el.elements[worst]],el.grad[worst])
        v,l=argminimum(np.linalg.det(F));hist.append(dict(depth_mm=depth,min_j=v,gauss_min_j=float(np.linalg.det(G).min())))
    report['history']=hist;write_json(out/'geometry.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for ax,(label,x,worst) in zip(axs,plotted):
        center=np.mean(x[p.mesh.triangles[worst]],axis=0)
        near=np.flatnonzero(np.linalg.norm(x[p.mesh.triangles].mean(axis=1)-center,axis=1)<.06)
        for e in near:
            nodes=element_nodes(p,x,e)
            for a,b in ((0,1),(1,2),(2,0)):
                l=np.zeros((31,3));l[:,a]=np.linspace(0,1,31);l[:,b]=1-l[:,a]
                curve=shape(l)@nodes;ax.plot(*curve.T,color='#b2182b' if e==worst else '.6',lw=1)
        node=element_nodes(p,x,worst);ax.scatter(*(shape(el.points)@node).T,s=16,c='#2166ac',label='Material Gauss points')
        row=report['states'][label]['worst'][0];ax.scatter(*row['current_xy'],marker='x',s=80,c='black',label='True min J')
        ax.set(xlabel='Laboratory x (mm)',ylabel='Laboratory y (mm)',title=f'{label}: element {worst}\nmin J={row["min_j"]:.4g}, Gauss={row["gauss_min_j"]:.4g}')
        ax.set_aspect('equal');ax.legend(fontsize=7)
    axs[2].plot([h['depth_mm'] for h in hist],[h['min_j'] for h in hist],label='Whole element')
    axs[2].plot([h['depth_mm'] for h in hist],[h['gauss_min_j'] for h in hist],label='Gauss points')
    axs[2].set(xlabel='Reference-normal q (mm)',ylabel='J (dimensionless)',title='Same material element, accepted history');axs[2].legend()
    fig.savefig(out/'local_map.png',dpi=170);plt.close(fig)
    print(json.dumps({label:{'depth_mm':v['depth_mm'],'worst':v['worst'][0]} for label,v in report['states'].items()}))
    return report


def integration_screen(source,out):
    """Replay own metrics at NEW points on fixed saved displacement fields.

    Local internal forces only, never a new equilibrium or full FE force curve.
    Order-3 reconstruction must reproduce all original selected-point metrics.
    """
    from prototype.ipc_p2 import composite_triangle_rule
    from mechanics.finite_strain import response
    from threadpoolctl import threadpool_limits
    from time import perf_counter
    tick=perf_counter();out.mkdir(parents=True,exist_ok=False)
    r=json.loads((source/'result.json').read_text());s=r['settings'];p=base.make_problem(s['angle'],coarse=s['coarse'],spacing=s['spacing']);el=p.enrichment
    geometry=json.loads((out.parent/'diagnosis/geometry.json').read_text())
    worst=geometry['states']['accepted']['worst'][0]['element'];shared=np.unique(el.elements[worst]);ids=np.flatnonzero(np.any(np.isin(el.elements,shared),axis=1))
    ids=np.unique(np.r_[ids,[v['element'] for v in geometry['states']['accepted']['worst']]])
    paths=chain(source);reports=[];force={};inputs={}
    # Cache just the selected six-DOF fields and original metrics, not full states.
    fields=[];offset=len(p.linear_elements);lookup={int(e):i for i,e in enumerate(p.active_elements)}
    for depth,path in paths:
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            u=z['positions'][:p.n]-p.mesh.reference_xy
            original=z['material_metric'][offset:].reshape(-1,len(el.weights),3,3)[[lookup[int(e)] for e in ids]]
            if not fields:
                initial=z['previous_material_metric'][offset:].reshape(-1,len(el.weights),3,3)[[lookup[int(e)] for e in ids]]
                np.testing.assert_allclose(initial,np.eye(3)[None,None]*np.ones_like(initial),atol=1e-12)
            fields.append((depth,u[el.elements[ids]].copy(),float(z['material_dt_s']),original))
    with threadpool_limits(1):
        for order,level in ((3,0),(5,0),(8,0),(10,0),(3,1),(3,2)):
            label=f'{order}/{level}'
            pts,w=composite_triangle_rule(order,level);grad=el.gradients(pts)[ids];metric=np.tile(np.eye(3),(len(ids)*len(w),1,1));rows=[];err=0.
            for depth,u,dt,expected in fields:
                F=np.eye(2)+np.einsum('eai,eqaJ->eqiJ',u,grad)
                q=response(F.reshape(-1,2,2),metric,dt,p.material,tangent=False,kinematics=p.kinematics)
                metric=q['metric'].copy()
                if order==3 and level==0:err=max(err,float(abs(metric.reshape(expected.shape)-expected).max()))
                local=np.einsum('e,q,eqiJ,eqaJ->eai',el.area[ids]*p.config.width_mm,w,q['P'].reshape(-1,len(w),2,2),grad)
                assembled=np.zeros((p.n,2));np.add.at(assembled,el.elements[ids].ravel(),local.reshape(-1,2))
                energy=float(np.einsum('e,q,eq->',el.area[ids]*p.config.width_mm,w,q['incremental_potential'].reshape(-1,len(w))))
                force[(label,depth)]=assembled;rows.append(dict(q_mm=depth,energy_nmm=energy,min_sample_j=float(np.linalg.det(F).min()),local_force_norm_n=float(np.linalg.norm(assembled))))
            reports.append(dict(order=order,level=level,label=label,rows=rows,original_metric_error=err if label=='3/0' else None))
            print('INTEGRATION',label,rows[-1], 'control_metric_error',err,flush=True)
    for block in reports:
        for row in block['rows']:
            depth=row['q_mm'];row['local_force_change_from_order3_n']=float(np.linalg.norm(force[(block['label'],depth)]-force[('3/0',depth)]))
    if reports[0]['original_metric_error']>1e-10:raise RuntimeError('Original-history replay control failed')
    result=dict(scope='Selected-element fixed-field history integration; NOT equilibrated FE reactions',elements=ids.tolist(),input_sha256=inputs,orders=reports,elapsed_s=perf_counter()-tick)
    write_json(out/'integration.json',result)
    print(json.dumps({'elapsed_s':result['elapsed_s'],'elements':len(ids),'endpoints':[b['rows'][-1] for b in reports]}))
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--integration',action='store_true');q=a.parse_args();(integration_screen if q.integration else run)(q.source,q.output)
