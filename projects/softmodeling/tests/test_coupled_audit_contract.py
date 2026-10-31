"""Clock and identity counterexamples are independent of solver success labels."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from tools.coupled_audit_contract import TimeChain,input_identity,bind_case,common_inputs
from tools.frozen_scene import digest


def clock_raw():
    return dict(initial_depth_mm=.355,initial_time_s=1.875,window_s=.001,
                dt_requested_s=.0005,status='completed_window')


def advance(clock,time=1.8755,depth=.3551,dt=.0005):
    return clock.advance(dict(material_dt_s=dt,depth_mm=depth),dict(dt_s=dt,time_s=time,depth_mm=depth))


def test_independent_clock_rejects_single_step_claimed_complete():
    q=clock_raw();c=TimeChain(q,{'depth_mm':.355},.2,-.02)
    advance(c)
    with pytest.raises(ValueError,match='Completion label'):c.finish()


@pytest.mark.parametrize('field,value',[('dt_s',.001),('time_s',1.876),('depth_mm',.3552)])
def test_clock_rejects_producer_fields_not_in_npz(field,value):
    q=clock_raw();c=TimeChain(q,{'depth_mm':.355},.2,-.02)
    row=dict(dt_s=.0005,time_s=1.8755,depth_mm=.3551);row[field]=value
    with pytest.raises(ValueError):c.advance(dict(material_dt_s=.0005,depth_mm=.3551),row)


def test_clock_reconstructs_complete_window_and_rejects_wrong_parent():
    q=clock_raw();c=TimeChain(q,{'depth_mm':.355},.2,-.02)
    advance(c);advance(c,1.876,.3552)
    assert c.finish()['complete_requested_window']
    q['initial_time_s']+=.01
    with pytest.raises(ValueError,match='Parent physical time'):TimeChain(q,{'depth_mm':.355},.2,-.02)


def test_variable_partition_has_independent_plan_and_actual_schedule():
    q=clock_raw();offsets=[.00025,.0005,.001]
    q['time_schedule']=dict(kind='preregistered_partition',provenance='fixed absolute refinement interval',end_offsets_s=offsets)
    plan=dict(window_s=.001,dt_s=[.0005],schedules={'0.0005':offsets})
    c=TimeChain(q,{'depth_mm':.355},.2,-.02,plan)
    for a,b in zip([0.,*offsets[:-1]],offsets):advance(c,1.875+b,.355+.2*b,b-a)
    assert c.finish()['actual_state_count']==3
    with pytest.raises(ValueError,match='independent input plan'):TimeChain(q,{'depth_mm':.355},.2,-.02)


def case(tmp_path,name='a',parent='same',config=None):
    path=tmp_path/name;path.mkdir()
    (path/'parent_state.npz').write_bytes(parent.encode());(path/'state.npz').write_bytes(b'accepted state')
    q=dict(parent=dict(state_sha256=digest(path/'parent_state.npz'),result_sha256='parent_result'),
           config=config or {'mu':.35},material_layer='maxwell',source_hashes={'equation':'one'},
           initial_depth_mm=.355,initial_time_s=1.875,window_s=.05,dt_requested_s=.0005,
           contact_model='IPC',conditional_comparison=True,production_continuation=False,
           steps=[dict(state_file='state.npz',state_sha256=digest(path/'state.npz'))])
    (path/'result.json').write_text(json.dumps(q))
    a=dict(schema='softcut-coupled-audit/2',passed=True,result_sha256=digest(path/'result.json'),
           input_identity=input_identity(q),time_chain={'complete_requested_window':True},states=copy.deepcopy(q['steps']))
    return path,q,a


def test_stale_audit_is_rejected(tmp_path):
    path,q,a=case(tmp_path);bind_case(path,q,a)
    (path/'result.json').write_text(json.dumps(q)+'\n')
    with pytest.raises(ValueError,match='Stale'):bind_case(path,q,a)


@pytest.mark.parametrize('other',[{'parent':'different'},{'config':{'mu':.2}}])
def test_different_parent_or_configuration_cannot_enter_matrix(tmp_path,other):
    first=case(tmp_path);second=case(tmp_path,'b',**other)
    with pytest.raises(ValueError,match='Different parent/model'):common_inputs([first,second])


def test_saved_state_replacement_invalidates_comparison(tmp_path):
    path,q,a=case(tmp_path);(path/'state.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='State/audit'):bind_case(path,q,a)


def test_partition_driver_never_silently_changes_or_truncates_requested_window():
    from tools.contact_time_schedule import requested_schedule
    plan=dict(window_s=.001,dt_s=[.0005],schedules={'0.0005':[.00025,.0005,.001]},schedule_provenance='diagnosed absolute band')
    ends,schedule=requested_schedule(plan,.0005)
    assert ends.tolist()==[.00025,.0005,.001] and schedule['hidden_cutbacks'] is False
    for bad in ([.0005,.0004,.001],[.0005,.0009],[.0008,.001]):
        plan['schedules']['0.0005']=bad
        with pytest.raises(ValueError):requested_schedule(plan,.0005)


def test_real_accepted_step_cannot_masquerade_as_full_window(tmp_path):
    import time
    from test_precrack_entry import cfg
    from coupled_contact import problem
    from tools.run_fracture_route import snapshot,observables
    from tools.run_coupled_contact import sources
    from tools.audit_coupled_contact import inspect
    c=cfg(.2);p=problem(c,'maxwell');old=p.rest.copy();released=p.released.copy()
    # Build a complete parent identity before the actual qualified FE step.
    parent=dict(depth_mm=p.start,positions=old,velocity=p.velocity.copy(),material_metric=p.material_metric.copy(),
                history=p.history.copy(),active_faces=p.active_faces.copy())
    np.savez_compressed(tmp_path/'parent_state.npz',**parent)
    x,row=p.step(old,p.start,p.start+.0001,time.monotonic()+20)
    row.update(time_s=.0005,dt_s=p.material_dt);observables(p,x,row)
    np.savez_compressed(tmp_path/'state.npz',**snapshot(p,x,released,p.start+.0001))
    row.update(state_file='state.npz',state_sha256=digest(tmp_path/'state.npz'))
    plan={'coupled_window':dict(window_s=.001,dt_s=[.0005])}
    (tmp_path/'preregistration.json').write_text(json.dumps(plan))
    raw=dict(config=c,material_layer='maxwell',source_hashes=sources(),parent=dict(state_sha256=digest(tmp_path/'parent_state.npz'),result_sha256='fixture'),
             preregistration_sha256=digest(tmp_path/'preregistration.json'),initial_depth_mm=p.start,initial_time_s=0.,window_s=.001,
             dt_requested_s=.0005,contact_model='IPC',conditional_comparison=True,production_continuation=False,status='completed_window',steps=[row])
    (tmp_path/'result.json').write_text(json.dumps(raw))
    proof=inspect(tmp_path,tmp_path/'audit_one.json',ipc_threads=2,blas_threads=1)
    assert not proof['passed'] and 'Completion label' in proof['error']
    assert len(proof['states'])==1, 'The actual FE step passed mechanics; the full window did not'
    raw['steps'][0]['time_s']=.001
    (tmp_path/'result.json').write_text(json.dumps(raw))
    proof=inspect(tmp_path,tmp_path/'audit_clock.json',ipc_threads=2,blas_threads=1)
    assert not proof['passed'] and 'Production adjacent time' in proof['error']
