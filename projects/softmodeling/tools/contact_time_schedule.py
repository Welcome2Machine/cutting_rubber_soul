"""Prescribed absolute-time partitions: no hidden physical-step cutbacks."""
import numpy as np


def requested_schedule(limits,dt):
    if dt not in limits['dt_s']:raise ValueError('Unregistered nominal step')
    window=float(limits['window_s'])
    if 'schedules' not in limits:
        n=int(round(window/dt))
        if n<1 or abs(n*dt-window)>1e-12:raise ValueError('Window must contain whole fixed steps')
        return np.arange(1,n+1)*dt,None
    offsets=np.asarray(limits['schedules'][str(dt)],float)
    provenance=limits.get('schedule_provenance')
    if not provenance or offsets.ndim!=1 or not len(offsets) or not np.all(np.isfinite(offsets)):
        raise ValueError('Explicit finite partition and provenance required')
    h=np.diff(np.r_[0.,offsets]);bound=np.full(len(h),dt)
    prefix=limits.get('common_prefix')
    if prefix is not None:
        if set(prefix)!={'through_s','dt_s'} or not 0<prefix['through_s']<window or prefix['dt_s']<=0:raise ValueError('Invalid shared prefix')
        n=int(round(prefix['through_s']/prefix['dt_s']))
        if abs(n*prefix['dt_s']-prefix['through_s'])>1e-12:raise ValueError('Nonintegral shared prefix')
        np.testing.assert_allclose(offsets[:n],np.arange(1,n+1)*prefix['dt_s'],atol=1e-15,rtol=0)
        bound[:n]=prefix['dt_s']
    if np.any(h<=0) or np.any(h>bound+1e-12) or abs(offsets[-1]-window)>1e-12:
        raise ValueError('Invalid partition order, step maximum or endpoint')
    return offsets,dict(kind='preregistered_partition',provenance=provenance,
        end_offsets_s=offsets.tolist(),nominal_maximum_dt_s=dt,common_prefix=prefix,nominal_step_scope='post-prefix segment' if prefix else 'whole window',
        minimum_dt_s=float(h.min()),maximum_dt_s=float(h.max()),hidden_cutbacks=False)
