"""Optional shared FP64 PARDISO backend; independent of AL/P2 mechanics.

The public class is also re-exported from al_performance for old callers.
"""
import ctypes
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse.linalg import spsolve


class PardisoDirection:
    """General nonsymmetric FP64 solve, with original-residual guarded fallback."""
    def __init__(self,threads=8,runtime=None):
        if threads<1:raise ValueError('Positive linear thread count required')
        runtime=runtime or os.environ.get('SOFTMODELING_PARDISO_RUNTIME')
        root=Path(runtime) if runtime else Path(__file__).resolve().parents[1]/'.runtime/pardiso'
        self.dll_handle=None
        if root.exists():
            sys.path.insert(0,str(root))
            if os.name=='nt':
                folder=root/'Library/bin';self.dll_handle=os.add_dll_directory(str(folder))
                libraries=sorted(folder.glob('mkl_rt*.dll'))
                if not libraries:raise ImportError('No MKL runtime in '+str(folder))
                os.environ['PYPARDISO_MKL_RT']=str(libraries[-1])
        from pypardiso import PyPardisoSolver
        from pypardiso.pardiso_wrapper import PyPardisoError
        self.library_error=PyPardisoError
        self.solver=PyPardisoSolver(mtype=11);self.threads=int(threads)
        self.set_threads=self.solver.libmkl.MKL_Set_Num_Threads_Local
        self.set_threads.argtypes=[ctypes.c_int];self.set_threads.restype=ctypes.c_int
        self.rows=[]

    def solve(self,matrix,rhs):
        tick=time.perf_counter();old=self.set_threads(self.threads);reason=None
        try:
            A=matrix.tocsr(copy=True);A.sum_duplicates();A.sort_indices()
            value=self.solver.solve(A,rhs)
            residual=float(np.linalg.norm(matrix@value-rhs));limit=max(1e-8*np.linalg.norm(rhs),1e-10)
            if not np.isfinite(value).all() or not np.isfinite(residual) or residual>limit:
                reason=f'PARDISO physical linear residual {residual:g} > {limit:g}'
        except (self.library_error,RuntimeError,ValueError) as exc:
            reason=repr(exc)
        finally:self.set_threads(old)
        if reason:
            value=spsolve(matrix,rhs)
            residual=float(np.linalg.norm(matrix@value-rhs))
        self.rows.append(dict(seconds=time.perf_counter()-tick,residual_n=residual,
            fallback_reason=reason,threads=self.threads))
        return value

    def close(self):
        self.solver.free_memory(everything=True)
