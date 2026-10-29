"""Original straight-path Coulomb power: geometry partitions and two integrations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.integrate import quad

from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from run_support import write_json
from .frozen_scene import load_scene


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


def inspect(case,output,seconds):
    output.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+seconds
    p,base,v,before,after,scene=load_scene(case,'diagnostic')
    if scene['endpoint']['residual_n']>1e-4:raise ValueError('Fixed post endpoint is not equilibrated')
    delta=after-before;trace={};timings={}
    def power(s):
        if time.monotonic()>=deadline:raise TimeoutError('Straight-path integration budget')
        s=float(s)
        if s in trace:return trace[s]['power_n_mm']
        point=before+s*delta;lag=v.lag(point);normal=v.collisions(point)
        value=2*float(v.friction.gradient(lag,v.cmesh,delta[v.surface])@delta[v.surface].ravel())
        coll=[q for kind in ('vv_collisions','ev_collisions') for q in getattr(lag,kind)]
        forces=[float(q.normal_force_magnitude) for q in coll]
        # IPC distance functions use squared distance. Compute geometric gaps explicitly.
        gaps=[]
        for q in coll:
            ids=[i for i in q.vertex_ids(v.cmesh.edges,v.cmesh.faces) if i>=0]
            if len(ids)==2:gaps.append(float(np.linalg.norm(point[v.surface[ids[0]]]-point[v.surface[ids[1]]])))
            else:
                pts=point[v.surface[ids]];a,b,c=pts;edge=c-b
                beta=np.clip((a-b)@edge/(edge@edge),0,1);gaps.append(float(np.linalg.norm(a-b-beta*edge)))
        trace[s]=dict(s=s,power_n_mm=value,contacts=len(coll),normal_load_sum_n=float(sum(forces)),
            max_normal_load_n=max(forces,default=0.),min_gap_mm=min(gaps,default=None),
            signature=[list(q.vertex_ids(v.cmesh.edges,v.cmesh.faces)) for q in coll])
        if len(trace)%100==0:write_json(output/'progress.json',dict(evaluations=len(trace),last_s=s,last_power=value))
        if value< -1e-12:raise ValueError('Negative power')
        return value
    report=dict(schema='softcut-virtual-work-diagnosis/1',scene=scene,quadratures=[],methods={},completed=False,
        stored_release_n_mm=base.stored(before)-v.stored(after),required_fracture_n_mm=p.toughness*float(p.seam_area[v.virtual_removed].sum()))
    try:
        for n in (5,9,17,33,65,129):
            started=time.perf_counter();nodes,weights=np.polynomial.legendre.leggauss(n)
            value=sum(w*power(s) for s,w in zip((nodes+1)/2,weights/2))
            report['quadratures'].append(dict(points=n,work_n_mm=value,elapsed_s=time.perf_counter()-started))
            write_json(output/'partial.json',report)
        cuts,geometry=geometry_partitions(v,before,after);report['geometry']=geometry;report['partition_points']=cuts
        np.savez_compressed(output/'endpoints.npz',before=before,after=after)
        # QUADPACK uses Gauss-Kronrod; many explicit short panels avoid missing narrow spikes.
        total=error=0.;panels=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi-lo<2e-15:continue
            value,err,*details=quad(power,lo,hi,epsabs=1e-9*(hi-lo),epsrel=1e-7,limit=100,full_output=1)
            panels.append(dict(a=lo,b=hi,work=value,error=err,warning=details[1:]));total+=value;error+=err
        report['methods']['adaptive_gk']=dict(work_n_mm=total,empirical_error_n_mm=error,panels=panels,rigorous_bound=False)
        # Independently implement composite Gauss-Legendre on the same geometric panels.
        # Its convergence is checked across orders, not by QUADPACK's embedded estimator.
        for order in (8,16):
            nodes,weights=np.polynomial.legendre.leggauss(order);value=0.
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                value+=(hi-lo)/2*sum(w*power((hi+lo)/2+(hi-lo)/2*z) for z,w in zip(nodes,weights))
            report['methods'][f'panel_gauss{order}']=dict(work_n_mm=value,rigorous_bound=False)
        report['completed']=True
    except TimeoutError as exc:report['error']=str(exc)
    finally:
        report['evaluations']=len(trace);report['timings']=timings
        write_json(output/'power_trace.json',sorted(trace.values(),key=lambda q:q['s']))
        write_json(output/'diagnostic.json',report)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        points=sorted(trace.values(),key=lambda q:q['s']);fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,9))
        for ax,key,label in zip(axes,('power_n_mm','min_gap_mm','normal_load_sum_n'),('Power [N mm]','Minimum gap [mm]','Normal load sum [N]')):
            ax.plot([q['s'] for q in points],[np.nan if q[key] is None else q[key] for q in points],'.-',ms=2);ax.set_ylabel(label);ax.set_yscale('symlog',linthresh=1e-10);ax.grid()
        axes[-1].set_xlabel('s on unchanged virtual straight path');fig.tight_layout();fig.savefig(output/'power.png');plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds',type=float,default=480)
    add_thread_arguments(parser);args=parser.parse_args()
    if not 1<=args.seconds<=540:raise ValueError('Independent work diagnosis <=540 s')
    with numerical_runtime(**thread_arguments(args)) as runtime:
        result=inspect(args.case,args.output,args.seconds);result['thread_runtime']=runtime
    write_json(args.output/'diagnostic.json',result);print(json.dumps(dict(completed=result['completed'],methods=result['methods'])))


if __name__=='__main__':main()
