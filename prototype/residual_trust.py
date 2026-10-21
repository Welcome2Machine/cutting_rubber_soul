"""Opt-in residual least-squares globalization of the unchanged IPC equations.

Try the exact nonsymmetric Newton direction, then column-scaled damped LSMR.
The damping changes only the trial direction, not the material or residual.
Every update must reduce the fresh physical residual and have positive model
reduction. A small least-squares gradient is NOT an equilibrium certificate.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsmr

from .ipc_reference import BASELINE, LoadStepFailure


def least_squares_direction(A, r, damp, maxiter=1500):
    scale = 1/np.maximum(np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel()), 1e-12)
    B = A@diags(scale)
    result = lsmr(B, -r, damp=damp, atol=1e-9, btol=1e-9, maxiter=maxiter)
    du = scale*result[0]
    return du, dict(lsmr_stop=int(result[1]), lsmr_iterations=int(result[2]),
                    scaled_gradient_norm=float(np.linalg.norm(B.T@r)),
                    linear_model_residual_n=float(np.linalg.norm(r+A@du)))


def reduction_test(r, trial_r, action, alpha):
    predicted = -float(alpha*(r@action)+.5*alpha**2*(action@action))
    actual = .5*float((r-trial_r)@(r+trial_r))
    ratio = actual/predicted if predicted > 0 else None
    slope = float(r@action)
    accepted = (np.isfinite([predicted, actual, slope]).all() and
                slope < 0 and predicted > 0 and actual > 0 and ratio >= .1)
    return bool(accepted), dict(predicted_reduction_n2=predicted,
        actual_reduction_n2=actual, ratio=ratio, slope=slope)


class ResidualTrustMixin:
    """Explicit candidate, same 500-update ceiling and 40-step progress window."""
    def _trust_record(self, row):
        print("RESIDUAL_TRUST "+json.dumps(row, allow_nan=False), flush=True)
        if getattr(self, "probe_directory", None) is not None:
            with (Path(self.probe_directory)/"directions.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")

    def solve_friction(self, x, depth, deadline):
        if self.nodal_volume is not None or self.enrichment is not None:
            raise ValueError("Residual least-squares candidate qualified only for P1")
        damp, best, progress = .01, np.inf, 0
        direction = np.zeros_like(x)
        try:
            for iteration in range(self.max_iterations+1):
                if time.monotonic() > deadline: raise TimeoutError("Residual trust wall budget")
                self.trial = x.copy()
                lag = self.lag(x)
                _, g, H, minimum_j = self.evaluate(x, lag, projected=False)
                r = g[self.free]; norm = float(np.linalg.norm(r))
                if norm <= BASELINE["force_tolerance_n"]: return x, iteration
                if norm < .9*best: best, progress = norm, iteration
                if self.stagnation_window and iteration-progress >= self.stagnation_window:
                    raise LoadStepFailure(f"Residual trust stagnation: depth={depth:g}, residual={norm:g}, best={best:g}")
                if iteration == self.max_iterations:
                    raise LoadStepFailure(f"Residual trust iteration limit: depth={depth:g}, residual={norm:g}")
                A = self.coupled_friction_tangent(x, lag, H)[self.free][:, self.free].tocsc()
                accepted = False
                for retry in range(9):
                    if time.monotonic() > deadline: raise TimeoutError("Residual trust direction budget")
                    tick = time.monotonic()
                    if retry == 0:
                        du = self.linear_direction(A, -r)
                        detail = dict(kind="newton", linear_relative=float(np.linalg.norm(A@du+r)/norm))
                    else:
                        du, detail = least_squares_direction(A, r, damp)
                        detail.update(kind="lm", damp=damp)
                    action = A@du
                    maximum = float(np.max(abs(du)))
                    if not np.isfinite(du).all() or maximum == 0:
                        raise LoadStepFailure("Nonfinite/zero residual trust direction")
                    direction = np.zeros(self.total); direction[self.free] = du; direction = direction.reshape(x.shape)
                    trial, alpha, geo = self._ptc_trial(x, direction, maximum)
                    linear_s = time.monotonic()-tick
                    # Newton gets bounded line search. LM changes its direction,
                    # rather than repeatedly shortening the same rejected ray.
                    for backtrack in range(4 if retry == 0 else 1):
                        if time.monotonic() > deadline: raise TimeoutError("Residual trust trial budget")
                        if backtrack: alpha *= .5; trial = x+alpha*direction
                        rt = self.evaluate(trial, self.lag(trial), tangent=False)[1][self.free]
                        accept, reduction = reduction_test(r, rt, action, alpha)
                        nt = float(np.linalg.norm(rt))
                        accept = accept and not np.array_equal(trial, x)
                        row = dict(depth_mm=float(depth), iteration=iteration, retry=retry,
                            backtrack=backtrack, norm_n=norm, trial_norm_n=nt, accepted=accept,
                            predicted_norm_n=float(np.linalg.norm(r+alpha*action)),
                            model_error_n=float(np.linalg.norm(rt-r-alpha*action)),
                            geometric_fraction=float(geo), tip_cap_alpha=.5*self.options.tip_radius_mm/maximum,
                            alpha=float(alpha), displacement_infinity_mm=alpha*maximum,
                            min_j=float(minimum_j), linear_s=linear_s, **detail, **reduction)
                        self._trust_record(row)
                        if accept:
                            x = trial; self.trial = x.copy(); accepted = True
                            if retry and reduction["ratio"] > .75: damp = max(1e-7, damp*.5)
                            elif retry and reduction["ratio"] < .25: damp = min(100., damp*2)
                            break
                    if accepted: break
                    if retry: damp = min(100., damp*4)
                if not accepted:
                    raise LoadStepFailure(f"Residual trust no productive direction: depth={depth:g}, residual={norm:g}")
        except Exception:
            if getattr(self, "probe_directory", None) is not None:
                from run_support import save_npz
                save_npz(Path(self.probe_directory)/"failed_direction.npz",
                    positions=x, direction=direction, previous_positions=self.previous,
                    previous_material_metric=self.previous_material_metric,
                    material_dt_s=self.material_dt, depth_mm=depth, **(
                        dict(residual=r) if "r" in locals() else {}))
            raise
