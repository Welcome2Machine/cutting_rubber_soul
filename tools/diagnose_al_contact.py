"""Saved-field weak/pointwise contact diagnosis and timed original audits."""
import argparse
import contextlib
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits
from run_support import write_json
from tools.run_al_integration import factory
from tools.diagnose_al_failure import restore


def edge_minimum(p,x,e):
    a,b=x[p.body_edges[e]];v=b-a
    ends=p.distance(np.array([a,b]),x)
    slopes=ends[1]@v
    if slopes[0]>=0:t=0.
    elif slopes[1]<=0:t=1.
    else:
        l,r=0.,1.
        for _ in range(45):
            t=(l+r)/2
            if float(p.distance(np.array([a+t*v]),x)[1][0]@v)>0:r=t
            else:l=t
        t=(l+r)/2
    return float(p.distance(np.array([a+t*v]),x)[0][0]),t,ends[0]


def run(source,out,mode='contact',depth=None,threads=1,lazy=False):
    out.mkdir(parents=True,exist_ok=False);r=json.loads((source/'result.json').read_text(encoding='utf-8'));s=r['settings']
    p=factory(s['angle'],coarse=s['coarse'],spacing=s['spacing'],order=s['order'],fast=False);p.ipc.set_num_threads(2)
    index=len(r['steps'])-1 if depth is None else next(i for i,z in enumerate(r['steps']) if abs(z['depth_mm']-depth)<1e-12)
    file=source/'states'/f'{index:04d}.npz'
    with np.load(file) as z:state={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(state['reference_xy'],p.rest)
    result=dict(input_file=str(file.resolve()),input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),mode=mode)
    if mode=='audit_all':
        from tools.run_material_candidate import audit
        from tools import run_augmented_blade as base
        result['frozen_input_sha256']=base.verify_case(source,require_independent=True)
        result['auditor_source_sha256']=base.snapshot(out)
        began=time.perf_counter()
        with threadpool_limits(threads):
            checked=audit(p,source,r,destination=out/'audit.json',deadline=time.monotonic()+600)
        result.update(elapsed_s=time.perf_counter()-began,states=checked['states'],passed=checked['passed'])
        prior=json.loads((source/'audit.json').read_text(encoding='utf-8'))
        for a,b in zip(checked['rows'],prior['rows']):
            assert a['checkpoint_sha256']==b['checkpoint_sha256']
            for key in ('reaction_fz_n','min_j','residual_n'):
                np.testing.assert_allclose(a[key],b[key],rtol=1e-9,atol=1e-8)
        result['prior_audit_agreement']=True
    elif mode=='contact':
        rows=[]
        for label,st in [('accepted',state),('failed_trial',dict(np.load(source/'failed_trial.npz')))]:
            x=restore(p,st);q=p.contact_data(x);g,h,j=p.evaluate(x,tangent=False)[1:]
            edges=[]
            for i,e in enumerate(p.contact_edges):
                minimum,t,ends=edge_minimum(p,x,e)
                if minimum>0.0001:continue
                mask=p.qedges==e;length=float(np.linalg.norm(np.diff(x[p.body_edges[e]],axis=0)))
                edges.append(dict(edge=int(e),nodes=p.body_edges[e].tolist(),minimum_gap_mm=minimum,min_fraction=t,
                    endpoint_gaps_mm=ends.tolist(),mean_gap_mm=float(q['gap_average'][i]),
                    mean_minus_min_mm=float(q['gap_average'][i]-minimum),pressure_mpa=float(p.multipliers[i,0]),
                    current_length_mm=length,reference_length_mm=float(np.linalg.norm(np.diff(p.rest[p.body_edges[e]],axis=0))),
                    gauss_min_mm=float(q['gap'][mask].min()),samples=int(mask.sum())))
            edges.sort(key=lambda e:e['minimum_gap_mm'])
            rows.append(dict(label=label,free_residual_n=float(np.linalg.norm(g[p.free])),min_j=j,worst_edges=edges[:12]))
        result['fields']=rows
    else:
        times={}
        for name in ('material_bulk','material_response','geometry_audit','contact_data','_set_quadrature','_self_terms'):
            original=getattr(p,name)
            def wrap(*a,_fn=original,_name=name,**kw):
                start=time.perf_counter()
                try:return _fn(*a,**kw)
                finally:times.setdefault(_name,[]).append(time.perf_counter()-start)
            setattr(p,name,wrap)
        context=threadpool_limits(threads) if threads else contextlib.nullcontext()
        result['before_threadpools']=threadpool_info();profile=cProfile.Profile();start=time.perf_counter()
        with context:
            result['during_threadpools']=threadpool_info();profile.enable()
            if lazy:
                with np.load(file) as archive:result['audit']=p.audit_state(archive,r['steps'][index])
            else:result['audit']=p.audit_state(state,r['steps'][index])
            profile.disable()
        result['elapsed_s']=time.perf_counter()-start;result['timings']={k:dict(calls=len(v),seconds=sum(v)) for k,v in times.items()}
        stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
        (out/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('before_threadpools','during_threadpools','fields')},ensure_ascii=False))
    if mode=='contact':print(json.dumps([dict(label=z['label'],residual=z['free_residual_n'],min_j=z['min_j'],worst=z['worst_edges'][0]) for z in rows]))
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--mode',choices=('contact','audit','audit_all'),default='contact');a.add_argument('--depth',type=float);a.add_argument('--threads',type=int,default=1)
    a.add_argument('--lazy',action='store_true')
    args=a.parse_args();run(args.source,args.output,args.mode,args.depth,args.threads,args.lazy)
