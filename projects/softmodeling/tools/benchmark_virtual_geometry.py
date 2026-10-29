"""Fixed-work complete native CCD covers of the same verified cover endpoint."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import numerical_runtime,add_thread_arguments,thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from virtual_path_probe import efficient_chord


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=8);parser.add_argument('--pair',choices=['pre','post'],default='post');add_thread_arguments(parser);args=parser.parse_args()
    if not 1<=args.repeats<=16:raise ValueError('Fixed repeat count 1..16 required')
    args.output.mkdir(parents=True,exist_ok=False)
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p,base,v,before,after,scene=load_scene(args.case,'diagnostic')
        if args.pair=='pre':v,after=base,before
        result=dict(scene=scene,pair=args.pair,runtime=runtime,methods={},repeats=args.repeats)
        for method,queries in [('dyadic',64),('geometry',256),('efficient',256)]:
            v.ffm_chord_partition=method;v.ffm_chord_max_queries=queries
            def check():
                if method=='efficient':
                    v.virtual_chord_certificate=efficient_chord(v,after,queries)
                    return v.virtual_chord_certificate['complete']
                return v.accepted_chord_is_safe(after)
            assert check()
            samples=[]
            for trial in range(3):
                start=time.perf_counter();cpu=time.process_time()
                for _ in range(args.repeats):
                    if not check():raise ValueError('A fixed verified chord lost certification')
                samples.append(dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu))
            result['methods'][method]=dict(samples=samples,median_wall_s=float(np.median([q['wall_s'] for q in samples])),certificate=v.virtual_chord_certificate)
            write_json(args.output/'partial.json',result)
        result['same_chord_ccd_speedup']=result['methods']['dyadic']['median_wall_s']/result['methods']['geometry']['median_wall_s']
    write_json(args.output/'benchmark.json',result);print(json.dumps(dict(speedup=result['same_chord_ccd_speedup'])))


if __name__=='__main__':main()
