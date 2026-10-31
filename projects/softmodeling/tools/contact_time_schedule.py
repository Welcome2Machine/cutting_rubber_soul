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
    h=np.diff(np.r_[0.,offsets])
    if np.any(h<=0) or np.any(h>dt+1e-12) or abs(offsets[-1]-window)>1e-12:
        raise ValueError('Invalid partition order, step maximum or endpoint')
    return offsets,dict(kind='preregistered_partition',provenance=provenance,
        end_offsets_s=offsets.tolist(),nominal_maximum_dt_s=dt,
        minimum_dt_s=float(h.min()),maximum_dt_s=float(h.max()),hidden_cutbacks=False)
