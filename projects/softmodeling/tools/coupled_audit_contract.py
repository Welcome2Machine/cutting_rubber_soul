"""Independent clock and immutable comparison contract for coupled windows."""
import hashlib,json
from pathlib import Path
import numpy as np
from .frozen_scene import digest


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def close(a,b,label):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a-b)>1e-12+1e-10*abs(b):
        raise ValueError(f'{label}: {a:.17g} != {b:.17g}')


class TimeChain:
    def __init__(self,raw,parent,rate,start_depth,plan=None):
        self.raw=raw;self.rate=float(rate);self.depth=float(parent['depth_mm'])
        self.t0=(self.depth-start_depth)/rate;self.time=self.t0;self.count=0
        self.window=float(raw['window_s']);self.h=float(raw['dt_requested_s'])
        if min(self.rate,self.window,self.h)<=0:raise ValueError('Positive clock data required')
        close(raw['initial_depth_mm'],self.depth,'Parent depth');close(raw['initial_time_s'],self.t0,'Parent physical time')
        if plan is not None:
            close(plan['window_s'],self.window,'Requested window')
            if not any(abs(self.h-z)<1e-15 for z in plan['dt_s']):raise ValueError('Unregistered requested dt')
        schedule=raw.get('time_schedule')
        if schedule is None:
            n=int(round(self.window/self.h));close(n*self.h,self.window,'Fixed window/count')
            self.offsets=np.arange(1,n+1)*self.h;self.kind='fixed'
        else:
            if schedule['kind']!='preregistered_partition' or not schedule.get('provenance'):raise ValueError('Unspecified variable schedule')
            self.offsets=np.asarray(schedule['end_offsets_s'],float);self.kind=schedule['kind']
            if plan is None or 'schedules' not in plan:raise ValueError('Partition requires independent input plan')
            np.testing.assert_array_equal(self.offsets,np.asarray(plan['schedules'][str(self.h)],float))
            if not len(self.offsets) or np.any(np.diff(np.r_[0.,self.offsets])<=0):raise ValueError('Nonincreasing partition')
            close(self.offsets[-1],self.window,'Partition endpoint')
        self.requested_count=len(self.offsets)

    def advance(self,state,row):
        if self.count>=self.requested_count:raise ValueError('Too many states for requested window')
        h=float(state['material_dt_s']);depth=float(state['depth_mm'])
        if h<=0:raise ValueError('Nonpositive physical dt')
        close((depth-self.depth)/self.rate,h,'NPZ depth/speed vs material dt')
        close(row['dt_s'],h,'Production dt vs NPZ')
        close(row['depth_mm'],depth,'Production depth vs NPZ')
        time=self.time+h
        close(row['time_s']-self.time,h,'Production adjacent time vs NPZ')
        close(time,self.t0+self.offsets[self.count],'Requested schedule vs actual clock')
        self.depth=depth;self.time=time;self.count+=1
        return dict(time_s=time,depth_mm=depth,dt_s=h)

    def finish(self):
        complete=self.count==self.requested_count and abs(self.time-self.t0-self.window)<=1e-12
        if complete!=(self.raw['status']=='completed_window'):raise ValueError('Completion label disagrees with reconstructed physical window')
        if complete:close(self.depth,self.raw['initial_depth_mm']+self.rate*self.window,'Window target depth')
        return dict(initial_time_s=self.t0,initial_depth_mm=self.raw['initial_depth_mm'],actual_elapsed_s=self.time-self.t0,
            terminal_time_s=self.time,terminal_depth_mm=self.depth,requested_window_s=self.window,
            actual_state_count=self.count,requested_state_count=self.requested_count,schedule_kind=self.kind,
            schedule_sha256=canonical(self.offsets.tolist()),complete_requested_window=complete)


def input_identity(raw):
    return dict(parent_state_sha256=raw['parent']['state_sha256'],parent_result_sha256=raw['parent']['result_sha256'],
        config_sha256=canonical(raw['config']),material_layer=raw['material_layer'],
        source_hashes=raw['source_hashes'],preregistration_sha256=raw.get('preregistration_sha256'),initial_depth_mm=raw['initial_depth_mm'],initial_time_s=raw['initial_time_s'],
        window_s=raw['window_s'],contact_model=raw['contact_model'],
        solve_tolerance_n=raw.get('solve_tolerance_n',1e-4),physical_acceptance_n=raw.get('physical_acceptance_n',1e-4),
        conditional_comparison=raw['conditional_comparison'],production_continuation=raw['production_continuation'])


def bind_case(path,raw,audit):
    path=Path(path)
    if audit.get('schema')!='softcut-coupled-audit/2' or not audit['passed']:raise ValueError('New independent clock audit required')
    if json.loads((path/'result.json').read_text())!=raw:raise ValueError('Parsed result differs from current file')
    if digest(path/'result.json')!=audit['result_sha256']:raise ValueError('Stale audit/result binding')
    if audit['input_identity']!=input_identity(raw):raise ValueError('Audit input identity mismatch')
    if raw.get('preregistration_sha256') and digest(path/'preregistration.json')!=raw['preregistration_sha256']:raise ValueError('Preregistration changed')
    if digest(path/'parent_state.npz')!=raw['parent']['state_sha256']:raise ValueError('Parent state changed')
    if not audit['time_chain']['complete_requested_window']:raise ValueError('Incomplete independently reconstructed window')
    if len(raw['steps'])!=len(audit['states']):raise ValueError('Audit state count mismatch')
    for row,check in zip(raw['steps'],audit['states']):
        file=path/row['state_file']
        if file.resolve().parent!=path.resolve():raise ValueError('State escaped case directory')
        if row['state_file']!=check['state_file'] or digest(file)!=check['state_sha256'] or row['state_sha256']!=check['state_sha256']:raise ValueError('State/audit binding mismatch')
    return audit['input_identity']


def common_inputs(cases,allowed_variations=('dt_requested_s','time_schedule')):
    allowed=set(allowed_variations)
    if not allowed<= {'dt_requested_s','time_schedule','solve_tolerance_n'}:raise ValueError('Unapproved comparison variation')
    reference=None
    for path,raw,audit in cases:
        identity=bind_case(path,raw,audit).copy()
        if 'solve_tolerance_n' in allowed:identity.pop('solve_tolerance_n')
        if 'dt_requested_s' not in allowed:identity['dt_requested_s']=raw['dt_requested_s']
        if 'time_schedule' not in allowed:identity['time_schedule']=raw.get('time_schedule')
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('Different parent/model/configuration/window in comparison')
    return reference
