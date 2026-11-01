"""Bounded physical-step bisection retaining every registered outer clock point."""
from collections import deque
import numpy as np

ALLOWED_FAILURES={'physical_energy_resolution','physical_nonlinear_failure'}

class Partition:
    def __init__(self,offsets,control):
        legacy={'maximum_bisections':12,'recovery':'original_physical_gates_only','maximum_step_growth':False}
        regulated=dict(legacy,maximum_step_growth='accepted_error_ratio_at_most_quarter')
        if control not in (legacy,regulated):
            raise ValueError('Unregistered bounded physical partition')
        self.control=control;self.outer=list(map(float,offsets));self.pending=deque((a,b,0,i) for i,(a,b) in enumerate(zip([0.,*self.outer[:-1]],self.outer)));self.trace=[];self.accepted=[]
    def decision(self,status,kind=None,resolution_ratio=None):
        a,b,level,index=self.pending.popleft();record=dict(left_s=a,right_s=b,level=level,outer_index=index,status=status)
        if status=='accepted':self.accepted.append(b);recover=False
        elif status=='rejected':
            record['failure_kind']=kind;recover=kind in ALLOWED_FAILURES and level<12
            if recover:
                mid=(a+b)/2
                self.pending.appendleft((mid,b,level+1,index));self.pending.appendleft((a,mid,level+1,index))
            else:self.pending.appendleft((a,b,level,index))
        else:raise ValueError('Unknown partition decision')
        record['bisected']=recover
        if self.control['maximum_step_growth']:
            if status=='accepted':
                if resolution_ratio is None or not np.isfinite(resolution_ratio) or not 0<=resolution_ratio<=1:raise ValueError('Accepted event resolution evidence required')
                record['resolution_ratio']=float(resolution_ratio)
                bound=(b-a)*(2 if resolution_ratio<=.25 else 1)
                # Preserve all registered outer endpoints; only split upcoming
                # intervals, never merge or skip an endpoint. Low error permits
                # at most a factor-two recovery, not a jump to the original cap.
                while self.pending and self.pending[0][1]-self.pending[0][0]>bound*(1+1e-10):
                    x,y,j,k=self.pending.popleft()
                    if j>=12:raise ValueError('Event recovery exceeds registered subdivision depth')
                    mid=(x+y)/2;self.pending.appendleft((mid,y,j+1,k));self.pending.appendleft((x,mid,j+1,k))
            elif resolution_ratio is not None:raise ValueError('Rejected step cannot supply an accepted recovery ratio')
        self.trace.append(record);return recover
    def saved(self):
        return dict(kind='bounded_dyadic_partition',provenance='Registered common outer grid; physical-energy/nonlinear rejections alone may bisect; numerical precision and budget never alter dt',control=self.control,outer_end_offsets_s=self.outer,end_offsets_s=self.accepted,trace=self.trace,hidden_cutbacks=False)


def verify(schedule,limits,dt):
    from .contact_time_schedule import requested_schedule
    outer,_=requested_schedule(limits,dt);p=Partition(outer,limits['adaptive_dyadic'])
    if schedule['outer_end_offsets_s']!=p.outer or schedule['control']!=p.control:raise ValueError('Changed outer grid/control')
    for row in schedule['trace']:
        if not p.pending:raise ValueError('Extra partition decision')
        if tuple(row[k] for k in ('left_s','right_s','level','outer_index'))!=p.pending[0]:raise ValueError('Unreproducible physical partition')
        p.decision(row['status'],row.get('failure_kind'),row.get('resolution_ratio'))
        if p.trace[-1]!=row:raise ValueError('Forbidden physical dt recovery')
        if row['status']=='rejected' and not row['bisected'] and row is not schedule['trace'][-1]:raise ValueError('Continued after unresolved numerical decision')
    if schedule['end_offsets_s']!=p.accepted:raise ValueError('Accepted partition differs from trace')
    return np.asarray(p.accepted),not p.pending
