"""Absolute-time sampled force consistency; never a continuous error bound."""
import itertools
import numpy as np


def review(audits,absolute_tolerance_n,*,raw_cases=None):
    if len(audits)!=3 or not 0<absolute_tolerance_n<=1e-5:
        raise ValueError('Three levels and explicitly authorized force tolerance required')
    if raw_cases is None and any('input_identity' in a for a in audits):
        raise ValueError('Bound raw schedules required for audited physical comparisons')
    if raw_cases is None:
        maps=[{round(s['time_s'],12):s for s in a['states']} for a in audits]
    else:
        from .coupled_audit_contract import registered_offsets,common_offset_indices
        offsets=[registered_offsets(raw,audit) for raw,audit in zip(raw_cases,audits)]
        associations=[dict(common_offset_indices(o,offsets[0],require_all=False)) for o in offsets]
        common=set.intersection(*(set(m) for m in associations))
        maps=[]
        for a,m in zip(audits,associations):
            mapped={raw_cases[0]['initial_time_s']+float(offsets[0][i]):a['states'][m[i]] for i in common}
            # Retain every extra point for peak evidence without inventing a
            # shared time key or moving its actual physical timestamp.
            used={m[i] for i in common}
            mapped.update({('extra',j):s for j,s in enumerate(a['states']) if j not in used})
            maps.append(mapped)
    if any(len(m)!=len(a['states']) for m,a in zip(maps,audits)):
        raise ValueError('Duplicate physical comparison time')
    times=sorted(k for k in set(maps[0]).intersection(*maps[1:]) if not isinstance(k,tuple))
    if not times:raise ValueError('No common saved absolute times; interpolation forbidden')
    def peak(rows):
        if not rows:return None
        s=max(rows,key=lambda r:abs(r['independent_fz_n']))
        return dict(time_s=s['time_s'],fz_n=s['independent_fz_n'])
    peaks=[dict(all_saved_peak=peak(a['states']),common_peak=peak([m[t] for t in times]),
                extra_samples_peak=peak([s for t,s in m.items() if t not in times]),extra_count=len(m)-len(times))
           for m,a in zip(maps,audits)]
    pairs=[]
    for i,j in itertools.combinations(range(3),2):
        d=np.array([maps[j][t]['independent_fz_n']-maps[i][t]['independent_fz_n'] for t in times]);k=int(np.argmax(abs(d)))
        pairs.append(dict(levels=[i,j],maximum_absolute_difference_n=float(abs(d[k])),signed_difference_n=float(d[k]),time_s=times[k]))
    return dict(common_times_s=times,common_count=len(times),peaks=peaks,pairs=pairs,
        absolute_tolerance_n=absolute_tolerance_n,all_three_sampled_force_consistent=all(p['maximum_absolute_difference_n']<=absolute_tolerance_n for p in pairs),
        scope='Saved absolute-time samples only; no smoothing, event alignment, interpolation or continuous-time error bound')


def mechanism_gate(comparison,audits,absolute_tolerance_n,*,cases=None):
    result=review(audits,absolute_tolerance_n,raw_cases=None if cases is None else [q for _,q,_ in cases])
    retained={k:v for k,v in comparison['gates'].items() if k!='force_difference_decreases'}
    result['retained_gates']=retained
    result['strict_force_difference_decreases']=comparison['gates']['force_difference_decreases']
    result['strict_passed']=comparison['passed']
    result['mechanism_force_consistency_passed']=bool(all(a['passed'] for a in audits) and all(retained.values()) and result['all_three_sampled_force_consistent'])
    return result
