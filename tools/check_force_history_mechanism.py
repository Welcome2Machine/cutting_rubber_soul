"""Screen a positive single-Maxwell force-history hypothesis against image ranges.

This is a reduced hereditary correspondence calculation on an existing elastic
force path, NOT a finite-strain contact solve or calibrated material law. It
keeps loading memory: multiplying the whole force by a decaying modulus is
not equivalent. Source force ranges must be explicitly image estimates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog, nnls

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def maxwell_history(time_s, elastic_force_n, tau_s):
    """Integral exp(-(t-s)/tau) dF(s), exact for piecewise linear F(t).

    F(0) is tared; there is no unrecorded step load at time zero.
    """
    t=np.asarray(time_s,dtype=float);f=np.asarray(elastic_force_n,dtype=float)
    if t.ndim!=1 or f.shape!=t.shape or len(t)<2 or not np.isfinite([t,f]).all():
        raise ValueError('Expected paired finite one-dimensional histories')
    if not np.isfinite(tau_s) or tau_s<=0 or np.any(np.diff(t)<=0):
        raise ValueError('Positive tau and strictly increasing times required')
    h=np.zeros_like(f)
    for i,dt in enumerate(np.diff(t),start=1):
        ratio=dt/tau_s
        h[i]=np.exp(-ratio)*h[i-1]-np.expm1(-ratio)/ratio*(f[i]-f[i-1])
    return h


def positive_branch_interval(base,history,lower,upper):
    """All a>=0 satisfying lower <= base + a*history <= upper."""
    base,h,lo,hi=np.broadcast_arrays(base,history,lower,upper)
    if np.any(lo>hi) or np.any(h<0):raise ValueError('Ordered ranges and positive branch required')
    zero=h==0
    if np.any(zero&((base<lo)|(base>hi))):return None
    nonzero=~zero
    left=max(0.,float(np.max((lo-base)[nonzero]/h[nonzero],initial=-np.inf)))
    right=float(np.min((hi-base)[nonzero]/h[nonzero],initial=np.inf))
    return [left,right] if left<=right else None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    report=json.loads(args.run_result.read_text(encoding='utf-8'))
    provenance=json.loads(args.image_provenance.read_text(encoding='utf-8'))
    # Reuse the existing strict source/type/range check; never substitute V CSV.
    from tools.analyze_prefracture_behavior import validate_image_estimates
    if not provenance.get('reading_csv_sha256') or provenance['reading_csv_sha256']!=sha(args.image_estimates):
        raise ValueError('Image readings require a matching provenance fingerprint')
    if 'screenshot' not in str(provenance.get('source','')).lower():
        raise ValueError('Explicit screenshot provenance required')
    data=validate_image_estimates(pd.read_csv(args.image_estimates)).to_dict('records')
    d=np.array([s['depth_mm'] for s in report['steps'] if s['depth_mm']<=1+1e-10])
    f=np.array([s['reaction_fz_n'] for s in report['steps'] if s['depth_mm']<=1+1e-10])
    if not np.isfinite(args.rate_mm_s) or args.rate_mm_s<=0:raise ValueError('Positive conditional loading rate required')
    if len(d)<2 or d[0]!=0 or np.any(np.diff(d)<=0):
        raise ValueError('Accepted depths must start at zero and be ordered and unique')
    f=f-f[0]
    anchors=np.array([r['depth_mm'] for r in data]);center=np.array([r['force_n'] for r in data])
    lower=np.array([r['lower_n'] for r in data]);upper=np.array([r['upper_n'] for r in data])
    if anchors.min()<d[0] or anchors.max()>d[-1]:raise ValueError('Force references outside accepted path')
    base=np.interp(anchors,d,f)
    records=[]
    for tau in np.geomspace(.005,100,401):
        history=maxwell_history(d/args.rate_mm_s,f,tau)
        h=np.interp(anchors,d,history)
        a=max(0.,float(np.dot(h,center-base)/np.dot(h,h)))
        fixed=base+a*h
        interval=positive_branch_interval(base,h,lower,upper)
        design=np.column_stack((base,h))
        coefficients,_=nnls(design,center)
        # An independent range feasibility solve allows a different equilibrium
        # coefficient. This does not authorize a change to catalogue G.
        feasible=linprog([0,0],A_ub=np.vstack((design,-design)),b_ub=np.r_[upper,-lower],
                        bounds=[(0,None),(0,None)],method='highs')
        fitted=design@coefficients
        records.append(dict(tau_s=float(tau),fixed_equilibrium_a=a,
            fixed_equilibrium_prediction_n=fixed.tolist(),
            fixed_equilibrium_rmse_n=float(np.sqrt(np.mean((fixed-center)**2))),
            fixed_equilibrium_feasible_interval=interval,
            free_equilibrium_b=float(coefficients[0]),free_equilibrium_a=float(coefficients[1]),
            free_equilibrium_prediction_n=fitted.tolist(),
            free_equilibrium_rmse_n=float(np.sqrt(np.mean((fitted-center)**2))),
            free_equilibrium_ranges_feasible=bool(feasible.success)))
    best_fixed=min(records,key=lambda r:r['fixed_equilibrium_rmse_n'])
    best_free=min(records,key=lambda r:r['free_equilibrium_rmse_n'])
    # The recurrence is exact for the declared linear segments, but a tau below
    # the stored time spacing cannot resolve the real within-step elastic path.
    minimum_resolved_tau=5*float(np.max(np.diff(d)))/args.rate_mm_s
    resolved=[r for r in records if r['tau_s']>=minimum_resolved_tau]
    if not resolved:raise ValueError('No tau in the declared grid resolves the stored loading intervals')
    feasible_resolved=[r for r in resolved if r['fixed_equilibrium_feasible_interval'] is not None]
    selected=None
    if feasible_resolved:
        candidate=min(feasible_resolved,key=lambda r:r['fixed_equilibrium_feasible_interval'][0])
        a=sum(candidate['fixed_equilibrium_feasible_interval'])/2
        tau=candidate['tau_s']
        history=maxwell_history(d/args.rate_mm_s,f,tau)
        prediction=base+a*np.interp(anchors,d,history)
        selected=dict(selection='Smallest added stiffness among resolved grid points entering every visual range',
            b=1.,a=a,tau_s=tau,relaxation_length_mm=tau*args.rate_mm_s,
            candidate_instantaneous_G_MPa=report['config']['shear_modulus_n_mm2']*(1+a),
            prediction_n=prediction.tolist(),half_depth_ratio=(float(np.interp(.5,anchors,prediction)/np.interp(1,anchors,prediction))
                if anchors.min()<=.5 and anchors.max()>=1 else None),
            rate_counterfactuals=[])
        for rate in (.1,.2,.4):
            predicted=base+a*np.interp(anchors,d,maxwell_history(d/rate,f,tau))
            selected['rate_counterfactuals'].append(dict(rate_mm_s=rate,prediction_n=predicted.tolist()))
    scale_interval=positive_branch_interval(np.zeros_like(base),base,lower,upper)
    result=dict(scope=__doc__,status='screened_not_FE',force_source='user image estimates only',
        source_result_sha256=sha(args.run_result),image_estimates_sha256=sha(args.image_estimates),
        image_provenance_sha256=sha(args.image_provenance),script_sha256=sha(__file__),
        source_provenance=provenance,rate_mm_s=args.rate_mm_s,rate_status='conditional assumption, not a fitted rate',
        tau_grid_s=[.005,100],tau_count=len(records),
        minimum_resolved_tau_s=minimum_resolved_tau,resolution_rule='Engineering screen tau >= 5 max stored dt; not a proven error bound',
        selected_resolved_fixed_equilibrium_candidate=selected,
        best_resolved_free_equilibrium=min(resolved,key=lambda r:r['free_equilibrium_rmse_n']),
        formula='F = b F_el + a integral exp(-(t-s)/tau) dF_el(s); a,b >= 0',
        equilibrium_G_MPa=report['config']['shear_modulus_n_mm2'],
        anchors=[dict(depth_mm=float(x),base_force_n=float(y),image_center_n=float(c),
                      image_lower_n=float(l),image_upper_n=float(u),required_scale=[float(l/y),float(u/y)])
                 for x,y,c,l,u in zip(anchors,base,center,lower,upper)],
        constant_scale_feasible_interval=scale_interval,
        fixed_equilibrium_feasible_tau_count=sum(r['fixed_equilibrium_feasible_interval'] is not None for r in records),
        free_equilibrium_feasible_tau_count=sum(r['free_equilibrium_ranges_feasible'] for r in records),
        best_fixed_equilibrium=best_fixed,best_free_equilibrium=best_free,grid=records,
        limitations=['Frozen elastic kinematics and contact footprint; no redistribution of friction/pressure.',
                    'Single-rate image ranges do not identify a finite-strain material or independent fracture-free window.',
                    'A failed screen only rejects this reduced correspondence hypothesis on the specified grid.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('constant_scale_feasible_interval','fixed_equilibrium_feasible_tau_count',
            'free_equilibrium_feasible_tau_count','selected_resolved_fixed_equilibrium_candidate','best_resolved_free_equilibrium')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-result',type=Path,required=True)
    parser.add_argument('--image-estimates',type=Path,required=True)
    parser.add_argument('--image-provenance',type=Path,required=True)
    parser.add_argument('--rate-mm-s',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
