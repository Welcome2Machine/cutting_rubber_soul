"""Cheap swept-distance probe for choosing a CCD query schedule, never acceptance.

Every accepted path still receives full native EV/VV/self-contact CCD. The probe
only decides whether to use the direct query or the complete geometry cover.
"""
import time
import numpy as np


def minimum_swept_gap(v,start,end):
    a=start[v.surface];d=(end-start)[v.surface]
    candidates=v.ipc.Candidates();candidates.build(v.cmesh,a,end[v.surface],v.dhat)
    edges=v.cmesh.edges;minimum=float('inf');witness=None
    pairs={tuple(sorted((c.vertex0_id,c.vertex1_id))) for c in candidates.vv_candidates}
    cross=lambda u,w:u[0]*w[1]-u[1]*w[0]
    for c in candidates.ev_candidates:
        i,j=map(int,edges[c.edge_id]);k=c.vertex_id
        pairs.update((tuple(sorted((i,k))),tuple(sorted((j,k)))))
        e=a[j]-a[i];de=d[j]-d[i];w=a[k]-a[i];dw=d[k]-d[i]
        q0=cross(e,w);q1=cross(e,dw)+cross(de,w);q2=cross(de,dw)
        r0=e@e;r1=2*e@de;r2=de@de
        coeff=[2*q2*r2,3*q2*r1,q1*r1+4*q2*r0-2*q0*r2,2*q1*r0-q0*r1]
        roots=np.r_[np.roots(coeff),np.roots([q2,q1,q0])]
        samples=[0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-9 and 0<z.real<1]
        for s in samples:
            ee=e+s*de;ww=w+s*dw;rr=ee@ee
            if rr<=0:continue
            beta=ww@ee/rr
            if not 0<=beta<=1:continue
            gap=abs(cross(ee,ww))/np.sqrt(rr)
            if gap<minimum:minimum=float(gap);witness=dict(kind='EV',vertices=[i,j,k],s=s,gap_mm=minimum,beta=float(beta))
    for i,j in pairs:
        w=a[i]-a[j];dw=d[i]-d[j];speed=dw@dw
        s=float(np.clip(-w@dw/speed,0,1)) if speed>0 else 0.
        gap=float(np.linalg.norm(w+s*dw))
        if gap<minimum:minimum=gap;witness=dict(kind='VV',vertices=[i,j],s=s,gap_mm=gap)
    return dict(minimum_gap_mm=minimum,minimum_feature=witness,
        ev_candidates=len(candidates.ev_candidates),vv_pairs=len(pairs),acceptance_shortcut=False)



def swept_gap_lower_bound(v,start,end):
    """Bernstein polynomial and VV distance bounds used ONLY for query scheduling."""
    a=start[v.surface];d=(end-start)[v.surface]
    candidates=v.ipc.Candidates();candidates.build(v.cmesh,a,end[v.surface],v.dhat)
    pairs={tuple(sorted((c.vertex0_id,c.vertex1_id))) for c in candidates.vv_candidates}
    minimum=v.dhat
    cross=lambda u,w:u[0]*w[1]-u[1]*w[0]
    bernstein=lambda q:np.array([q[0],q[0]+q[1]/2,sum(q)])
    for c in candidates.ev_candidates:
        i,j=map(int,v.cmesh.edges[c.edge_id]);k=c.vertex_id
        pairs.update((tuple(sorted((i,k))),tuple(sorted((j,k)))))
        e=a[j]-a[i];de=d[j]-d[i];w=a[k]-a[i];dw=d[k]-d[i]
        projection=bernstein([e@w,e@dw+de@w,de@dw])
        rr=bernstein([e@e,2*e@de,de@de])
        if max(projection)<0 or max(rr-projection)<0:continue
        q=bernstein([cross(e,w),cross(e,dw)+cross(de,w),cross(de,dw)])
        lower=max(0.,min(q),-max(q))
        length=max(np.linalg.norm(e),np.linalg.norm(e+de))
        minimum=min(minimum,lower/length if length else 0.)
    for i,j in pairs:
        w=a[i]-a[j];dw=d[i]-d[j];speed=dw@dw
        t=float(np.clip(-w@dw/speed,0,1)) if speed>0 else 0.
        minimum=min(minimum,float(np.linalg.norm(w+t*dw)))
    # Subtract a coordinate-scale roundoff allowance. This is a schedule probe,
    # never an asserted mathematical collision-free certificate.
    allowance=128*np.finfo(float).eps*max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(end[v.surface]))))
    return dict(distance_lower_bound_mm=max(0.,minimum-allowance),roundoff_allowance_mm=allowance,
        ev_candidates=len(candidates.ev_candidates),vv_pairs=len(pairs),acceptance_shortcut=False)

def efficient_chord(v,x,max_queries):
    from prototype.ipc_reference import determinant_step
    from virtual_path_geometry import native_partitioned_chord
    started=time.perf_counter()
    invalid=dict(complete=False,native_queries=0,segments=[],unchanged_straight_path=True)
    if not np.isfinite(x).all() or np.any(x[:v.n,0]<v.config.drop_s_mm):return invalid
    delta=x-v.previous
    f=np.eye(2)+np.einsum('eai,eaJ->eiJ',(v.previous[:v.n]-v.mesh.reference_xy)[v.mesh.triangles],v.grad)
    d=np.einsum('eai,eaJ->eiJ',delta[:v.n][v.mesh.triangles],v.grad)
    if determinant_step(f,d)<1-1e-10:return invalid
    if max_queries<1:return invalid
    probe=swept_gap_lower_bound(v,v.previous,x)
    # A performance switch only: both branches use unchanged native CCD.
    switch=max(100*(getattr(v,'ffm_ccd_tolerance',None) or 0.),1e-4*v.dhat)
    calls=0
    if probe['distance_lower_bound_mm']>switch:
        kwargs={} if v.virtual_narrow_phase_ccd is None else dict(narrow_phase_ccd=v.virtual_narrow_phase_ccd)
        fraction=v.ipc.compute_collision_free_stepsize(v.cmesh,v.previous[v.surface],x[v.surface],**kwargs)
        calls=1
        if fraction>=1-1e-10:
            return dict(complete=True,native_queries=1,segments=[[0.,1.]],unchanged_straight_path=True,
                direct_native_certificate=True,probe=probe,elapsed_s=time.perf_counter()-started)
    result=native_partitioned_chord(v,x,max_queries-calls)
    result.update(native_queries=result['native_queries']+calls,probe=probe,elapsed_s=time.perf_counter()-started)
    return result
