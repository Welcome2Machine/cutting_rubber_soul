"""Compare complete native CCD covers seeded by swept-path geometry."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve
from numerical_runtime import numerical_runtime,add_thread_arguments,thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from virtual_path_geometry import geometry_partitions, ev_crossing_witnesses
from prototype.ipc_reference import determinant_step


from virtual_path_geometry import native_partitioned_chord

def partitioned_chord(v,x,max_queries=256):
    return native_partitioned_chord(v,x,max_queries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--endpoint',default='production');add_thread_arguments(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p,base,v,before,x,info=load_scene(args.case,args.endpoint)
        v.set_primary_increment(x,x-v.previous);lag=v.lag(x);_,g,h,j=v.evaluate(x,lag,projected=False);v.update_constraint_free(x,g)
        a=v.coupled_friction_tangent(x,lag,h)[v.free][:,v.free];q=spsolve(a,-g[v.free]);d=np.zeros(v.total);d[v.free]=q;d=d.reshape(x.shape)
        result=dict(scene=info,trials=[],runtime=runtime)
        for alpha in (0.,1.,.5,.25,.125):
            trial=x+alpha*d;cuts,geometry=geometry_partitions(v,before,trial)
            v.set_primary_increment(trial,trial-v.previous);_,gt,_,jt=v.evaluate(trial,v.lag(trial),tangent=False)
            item=dict(alpha=alpha,residual_n=float(np.sqrt(2)*np.linalg.norm(gt[v.free])),min_j=float(jt),
                cuts=cuts,features=geometry['features'],local_safe_step=float(v.safe_step(x,alpha*d)))
            result['trials'].append(item);write_json(args.output/'diagnostic.json',result)
            item['cover']=partitioned_chord(v,trial)
            write_json(args.output/'diagnostic.json',result)
    write_json(args.output/'diagnostic.json',result)
    print(json.dumps([dict(alpha=q['alpha'],residual=q['residual_n'],cover={k:v for k,v in q['cover'].items() if k!='segments'}) for q in result['trials']]))


if __name__=='__main__':main()
