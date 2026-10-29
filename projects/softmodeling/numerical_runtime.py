"""Explicit, reversible process thread policy and measured execution identity."""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows integration remains supported.
    resource = None
import time

from threadpoolctl import threadpool_info, threadpool_limits


def add_thread_arguments(parser):
    parser.add_argument('--ipc-threads', type=int, default=2)
    parser.add_argument('--blas-threads', type=int, default=2)
    parser.add_argument('--physical-core-affinity', action='store_true',
                        help='Use one allocated hardware thread per physical core')


def thread_arguments(args):
    return dict(ipc_threads=args.ipc_threads, blas_threads=args.blas_threads,
                physical_core_affinity=args.physical_core_affinity)


def core_topology(cpus):
    result = []
    for cpu in cpus:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            result.append(dict(cpu=cpu, socket=int((base/'physical_package_id').read_text()),
                               core=int((base/'core_id').read_text())))
        except OSError:
            result.append(dict(cpu=cpu, socket=None, core=None))
    return result


@contextmanager
def numerical_runtime(ipc_threads=2, blas_threads=2, physical_core_affinity=False):
    if any(type(n) is not int or not 1 <= n <= 64 for n in (ipc_threads, blas_threads)):
        raise ValueError('Explicit IPC/BLAS counts must be integers in 1..64')
    # Conservative bound also covers a BLAS call nested in a native worker.
    bound = ipc_threads*blas_threads
    budget = int(os.environ.get('SOFTCUT_THREAD_BUDGET', '64'))
    if bound > budget:
        raise ValueError('Possible nested compute threads exceed this process budget')
    from prototype.ipc_reference import toolkit
    ipc = toolkit()
    previous_ipc = ipc.get_num_threads()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    topology = core_topology(affinity)
    selected = affinity
    changed = {}
    if physical_core_affinity:
        if not affinity or any(c['core'] is None for c in topology):
            raise ValueError('Physical CPU topology/affinity unavailable')
        by_core = {}
        for c in topology:
            by_core.setdefault((c['socket'], c['core']), c['cpu'])
        selected = sorted(by_core.values())
    if affinity and bound > len(selected):
        raise ValueError('Thread request exceeds available CPU affinity')
    started, cpu_started = time.monotonic(), time.process_time()
    metadata = dict(requested=dict(ipc_threads=ipc_threads, blas_threads=blas_threads),
        conservative_nested_thread_bound=bound, thread_budget=budget,
        affinity_before=affinity, affinity_effective=selected, allocated_topology=topology,
        one_hardware_thread_per_core=physical_core_affinity,
        host=platform.node(), pid=os.getpid(),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')
               and any(s in k for s in ('JOB_ID','ARRAY','CPUS','NTASK','MEM','NODELIST','PROCID'))})
    try:
        if selected != affinity:
            import psutil
            for thread in psutil.Process().threads():
                try:
                    changed[thread.id] = os.sched_getaffinity(thread.id)
                    os.sched_setaffinity(thread.id, selected)
                except ProcessLookupError:
                    pass
        with threadpool_limits(limits=blas_threads, user_api='blas'):
            ipc.set_num_threads(ipc_threads)
            metadata.update(ipc_threads_effective=ipc.get_num_threads(),
                            threadpools_effective=threadpool_info())
            if metadata['ipc_threads_effective'] != ipc_threads:
                raise RuntimeError('IPC thread setting did not take effect')
            if any(p['num_threads'] != blas_threads for p in metadata['threadpools_effective']
                   if p['user_api'] == 'blas'):
                raise RuntimeError('BLAS thread setting did not take effect')
            yield metadata
    finally:
        metadata.update(wall_s=time.monotonic()-started, process_cpu_s=time.process_time()-cpu_started)
        if resource is not None:
            metadata.update(peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                peak_rss_scope='whole process lifetime; Linux ru_maxrss KiB')
        ipc.set_num_threads(previous_ipc)
        if changed:
            import psutil
            for thread in psutil.Process().threads():
                changed.setdefault(thread.id, affinity)
        for tid, cpus in changed.items():
            try:
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                pass


def runtime_entry(function):
    """Keep positional APIs and legacy 2/2 defaults; explicit CLI overrides win."""
    @wraps(function)
    def wrapped(*args, ipc_threads=2, blas_threads=2, physical_core_affinity=False, **kwargs):
        with numerical_runtime(ipc_threads, blas_threads, physical_core_affinity) as runtime:
            result = function(*args, runtime=runtime, **kwargs)
            if isinstance(result, dict):
                result['thread_runtime'] = runtime
        return result
    return wrapped
