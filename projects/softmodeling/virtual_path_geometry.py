"""Geometry events of an unchanged linear virtual path (no acceptance shortcut)."""
import time
import numpy as np
from numpy.polynomial import Polynomial as P

def real_roots(poly):
    if np.max(np.abs(poly.coef), initial=0)==0:return []
    return sorted(float(z.real) for z in poly.roots() if abs(z.imag)<1e-9 and 0<z.real<1)


def geometry_partitions(v, start, end):
    """All swept EV/VV broad-phase pairs; polynomial feature/activation/extremum roots.

    Rounded polynomial roots and adaptive estimates are NOT rigorous error bounds.
    Dyadic neighborhoods resolve close approaches, including endpoint layers.
    """
    a=start[v.surface];d=(end-start)[v.surface]
    candidates=v.ipc.Candidates();candidates.build(v.cmesh,a,end[v.surface],v.dhat)
    cuts={0.,1.};features=[]
    def dot(u,w):return sum((ui*wi for ui,wi in zip(u,w)),P([0.]))
    def point(k):return [P([a[k,j],d[k,j]]) for j in (0,1)]
    def neighborhoods(s,gap,speed):
        width=max(1e-14,min(.25,gap/max(speed,1e-30)))
        for _ in range(50):
            cuts.add(max(0.,s-width));cuts.add(min(1.,s+width))
            if width>=1:break
            width*=2
    for c in candidates.ev_candidates:
        i,j=map(int,v.cmesh.edges[c.edge_id]);k=c.vertex_id
        e=[q-r for q,r in zip(point(j),point(i))];w=[q-r for q,r in zip(point(k),point(i))]
        cross=e[0]*w[1]-e[1]*w[0];rr=dot(e,e);projection=dot(e,w)
        stationary=real_roots(2*cross.deriv()*rr-cross*rr.deriv())+real_roots(cross)
        for q in (projection,projection-rr,cross*cross-v.dhat**2*rr):
            for s in real_roots(q):cuts.add(s)
        for s in [0.,1.,*stationary]:
            if rr(s)<=0:continue
            beta=projection(s)/rr(s)
            if not 0<=beta<=1:continue
            gap=abs(cross(s))/np.sqrt(rr(s))
            if gap>=v.dhat:continue
            cuts.add(s);speed=max(np.linalg.norm(d[k]-d[i]),np.linalg.norm(d[k]-d[j]))
            neighborhoods(s,gap,speed)
            features.append(dict(kind='EV',vertices=[i,j,k],s=s,gap_mm=float(gap),beta=float(beta)))
    # Include endpoint distances for every swept EV, even when VV broad phase is empty.
    pairs={tuple(sorted((c.vertex0_id,c.vertex1_id))) for c in candidates.vv_candidates}
    for c in candidates.ev_candidates:
        pairs.update(tuple(sorted((int(k),c.vertex_id))) for k in v.cmesh.edges[c.edge_id])
    for i,j in sorted(pairs):
        w=[q-r for q,r in zip(point(i),point(j))];rr=dot(w,w)
        for s in real_roots(rr-v.dhat**2):cuts.add(s)
        for s in [0.,1.,*real_roots(rr.deriv())]:
            gap=np.sqrt(max(0.,rr(s)))
            if gap>=v.dhat:continue
            cuts.add(s);neighborhoods(s,gap,np.linalg.norm(d[i]-d[j]))
            features.append(dict(kind='VV',vertices=[i,j],s=s,gap_mm=float(gap)))
    return sorted(cuts),dict(ev_candidates=len(candidates.ev_candidates),vv_candidates=len(candidates.vv_candidates),
        features=features,partition_method='swept geometry polynomial events plus dyadic gap neighborhoods',
        rigorous_bound=False)



def ev_crossing_witnesses(v, start, end):
    """High precision sign brackets for interior EV crossings of the same chord.

    Only an early rejection witness: never substitutes for full native CCD on an
    accepted path. Decimal recomputation separates crossings from rounded roots.
    """
    from decimal import Decimal, localcontext
    a=start[v.surface];d=(end-start)[v.surface]
    candidates=v.ipc.Candidates();candidates.build(v.cmesh,a,end[v.surface],0.)
    witnesses=[]
    for c in candidates.ev_candidates:
        i,j=map(int,v.cmesh.edges[c.edge_id]);k=c.vertex_id
        e=[P([a[j,q]-a[i,q],d[j,q]-d[i,q]]) for q in (0,1)]
        w=[P([a[k,q]-a[i,q],d[k,q]-d[i,q]]) for q in (0,1)]
        cross=e[0]*w[1]-e[1]*w[0];roots=real_roots(cross)
        for root in roots:
            separation=min([root,1-root]+[abs(root-r) for r in roots if r!=root])
            radius=min(1e-4,separation/8)
            if radius<=1e-14:continue
            lo,hi=root-radius,root+radius
            with localcontext() as context:
                context.prec=60
                def vectors(s):
                    t=Decimal.from_float(s)
                    pts=[[Decimal.from_float(float(a[node,q]))+t*Decimal.from_float(float(d[node,q])) for q in (0,1)] for node in (i,j,k)]
                    ee=[pts[1][q]-pts[0][q] for q in (0,1)];ww=[pts[2][q]-pts[0][q] for q in (0,1)]
                    return ee,ww
                orient=[];betas=[]
                for s in (lo,hi):
                    ee,ww=vectors(s);orient.append(ee[0]*ww[1]-ee[1]*ww[0]);betas.append(sum(x*y for x,y in zip(ee,ww))/sum(x*x for x in ee))
                # Bound all dot products over the bracket with interval arithmetic.
                elo,wlo=vectors(lo);ehi,whi=vectors(hi)
                E=[(min(x,y),max(x,y)) for x,y in zip(elo,ehi)];W=[(min(x,y),max(x,y)) for x,y in zip(wlo,whi)]
                def product(x,y):
                    vals=[xx*yy for xx in x for yy in y];return min(vals),max(vals)
                dot=[product(x,y) for x,y in zip(E,W)];rest=[product(x,(x[0]-y[1],x[1]-y[0])) for x,y in zip(E,W)]
                interior=min(sum(x[0] for x in dot),sum(x[0] for x in rest))>0
                if orient[0]*orient[1]<0 and interior:
                    witnesses.append(dict(kind='EV_crossing',vertices_surface=[i,j,k],vertices_global=v.surface[[i,j,k]].tolist(),
                        bracket=[lo,hi],oriented_area_twice_mm2=[str(z) for z in orient],beta=[str(z) for z in betas],
                        interior_over_bracket=True,decimal_precision=60,acceptance_shortcut=False))
    return witnesses


def native_partitioned_chord(v, x, max_queries):
    from prototype.ipc_reference import determinant_step
    start=time.perf_counter();delta=x-v.previous
    report=dict(complete=False,native_queries=0,segments=[],unchanged_straight_path=True,geometry_seeded=True)
    if not np.isfinite(x).all() or np.any(x[:v.n,0]<v.config.drop_s_mm):return report
    f=np.eye(2)+np.einsum('eai,eaJ->eiJ',(v.previous[:v.n]-v.mesh.reference_xy)[v.mesh.triangles],v.grad)
    d=np.einsum('eai,eaJ->eiJ',delta[:v.n][v.mesh.triangles],v.grad)
    if determinant_step(f,d)<1-1e-10:return report
    witnesses=ev_crossing_witnesses(v,v.previous,x)
    if witnesses:
        report.update(crossing_witnesses=witnesses,elapsed_s=time.perf_counter()-start);return report
    cuts,geometry=geometry_partitions(v,v.previous,x)
    report['geometry_s']=time.perf_counter()-start
    report['minimum_feature']=min(geometry['features'],key=lambda q:q['gap_mm']) if geometry['features'] else None
    # An unresolved gap is not a physical clearance restriction: fail unknown
    # before a native near-zero query can exhaust memory. Never certify it safe.
    closest=report['minimum_feature']
    tolerance=getattr(v,'ffm_ccd_tolerance',None)
    if tolerance is not None and closest and closest['gap_mm'] <= tolerance:
        report.update(failure='gap_below_requested_ccd_resolution',elapsed_s=time.perf_counter()-start)
        return report
    # Visit the closest-approach panels first. All panels remain mandatory.
    pending=list(zip(cuts[:-1],cuts[1:]));closest=report['minimum_feature']
    if closest:pending.sort(key=lambda ab:abs((ab[0]+ab[1])/2-closest['s']),reverse=True)
    kwargs={} if v.virtual_narrow_phase_ccd is None else dict(narrow_phase_ccd=v.virtual_narrow_phase_ccd)
    while pending:
        if report['native_queries']>=max_queries:break
        a,b=pending.pop();first=v.previous+a*delta;last=x if b==1 else v.previous+b*delta
        fraction=v.ipc.compute_collision_free_stepsize(v.cmesh,first[v.surface],last[v.surface],**kwargs)
        report['native_queries']+=1
        if fraction>=1-1e-10:report['segments'].append([a,b]);continue
        mid=(a+b)/2
        if mid==a or mid==b:
            report['failure']='unresolved_native_interval';return report
        pending.extend([(a,mid),(mid,b)])
    report.update(complete=not pending,elapsed_s=time.perf_counter()-start)
    report['segments'].sort()
    return report

