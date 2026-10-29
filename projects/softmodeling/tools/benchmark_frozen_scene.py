"""Fixed-work timings on a checked archived virtual state; no loading/solve loop."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse.linalg._dsolve import linsolve

from numerical_runtime import add_thread_arguments, thread_arguments, numerical_runtime
from run_support import write_json
from .frozen_scene import load_scene


def measure(function, count, trials=3):
    function()  # Fixed warm-up excluded; no adaptive amount of timed work.
    samples = []
    for _ in range(trials):
        wall, cpu = time.perf_counter(), time.process_time()
        for _ in range(count):
            function()
        elapsed = time.perf_counter()-wall
        samples.append(dict(wall_s=elapsed, cpu_s=time.process_time()-cpu,
                            calls=count))
    return dict(samples=samples, median_wall_s=float(np.median([s['wall_s'] for s in samples])),
        median_cpu_s=float(np.median([s['cpu_s'] for s in samples])), calls_per_trial=count)


def benchmark(case, output, repeats):
    if not 8 <= repeats <= 512:
        raise ValueError('Explicit fixed work requires 8..512 repeats')
    output.mkdir(parents=True, exist_ok=False)
    started=time.perf_counter()
    p,base,v,before,x,info=load_scene(case, 'production')
    restore_s=time.perf_counter()-started
    v.set_primary_increment(x,x-v.previous)
    lag=v.lag(x)
    _,g,h,j=v.evaluate(x,lag,projected=False)
    v.update_constraint_free(x,g)
    a=v.coupled_friction_tangent(x,lag,h)[v.free][:,v.free].tocsc()
    rhs=-g[v.free]
    lu=splu(a)
    delta=x-before
    u=(x[:v.n]-v.mesh.reference_xy).ravel()
    timings={}
    timings['assembly_fresh_force_and_coupled_jacobian']=measure(
        lambda:v.coupled_friction_tangent(x,v.lag(x),v.evaluate(x,v.lag(x),projected=False)[2]),repeats)
    timings['material_bulk']=measure(lambda:v.material_bulk(u),repeats)
    timings['fresh_friction_contact_set']=measure(lambda:v.lag(x),repeats)
    timings['full_chord_ccd']=measure(lambda:v.accepted_chord_is_safe(x),max(8,repeats//4))
    timings['friction_path_gauss65']=measure(lambda:v.friction_path_work(before,x,65),max(2,repeats//16))
    backend_calls=[]
    original=linsolve._superlu.gssv
    def counted(*args,**kwargs):
        backend_calls.append('SuperLU.gssv')
        return original(*args,**kwargs)
    linsolve._superlu.gssv=counted
    try:
        timings['spsolve_current_matrix']=measure(lambda:spsolve(a,rhs),repeats)
    finally:
        linsolve._superlu.gssv=original
    timings['superlu_factorization']=measure(lambda:splu(a),repeats)
    timings['superlu_triangular_solve']=measure(lambda:lu.solve(rhs),repeats)
    path=output/'fixed_io.npz'
    def io():
        np.savez_compressed(path,positions=x,metric=v.previous_material_metric,history=v.previous_history)
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['positions'],x)
    timings['checkpoint_write_read']=measure(io,max(4,repeats//8))
    for stat in timings.values():
        stat['cpu_equivalents']=stat['median_cpu_s']/stat['median_wall_s']
    np.savez_compressed(output/'numeric_checks.npz',gradient=g,jacobian_data=a.data,
        jacobian_indices=a.indices,jacobian_indptr=a.indptr,direction=spsolve(a,rhs))
    return dict(schema='softcut-fixed-work-benchmark/1',loading_started=False,
        fixed_repeats=repeats,trials=3,restore_and_verify_s=restore_s,scene=info,
        dimensions=dict(nodes=v.n,free_dofs=len(v.free),matrix_nnz=a.nnz),timings=timings,
        scipy_version=scipy.__version__,sparse_backend=dict(
            observed_spsolve_kernels=sorted(set(backend_calls)),observed_calls=len(backend_calls),
            scikit_umfpack_available=not linsolve.noScikit,
            configured_umfpack_preference=bool(linsolve.useUmfpack),
            factor_type=type(lu).__name__,sparse_residual_n=float(np.linalg.norm(a@lu.solve(rhs)-rhs)),
            note='Observed dispatch and measured timings; no assumption that BLAS threads parallelize SuperLU'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=64)
    add_thread_arguments(parser)
    args=parser.parse_args()
    with numerical_runtime(**thread_arguments(args)) as runtime:
        result=benchmark(args.case,args.output,args.repeats)
        result['thread_runtime']=runtime
    write_json(args.output/'benchmark.json',result)
    print(json.dumps(dict(output=str(args.output),timings=result['timings'],runtime=runtime)))


if __name__=='__main__':main()
