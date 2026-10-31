from types import SimpleNamespace
import numpy as np
from tools import report_coupled_growth as report,report_coupled_contact as contact


def test_failure_window_requires_every_point_and_does_not_invent_activation(tmp_path,monkeypatch):
    p=SimpleNamespace(toughness=.006,strength=.2,precracked_faces=np.array([False,True]),rebuild_contact=lambda:None)
    monkeypatch.setattr(report,'problem',lambda config:p)
    monkeypatch.setattr(report,'face_contacts',lambda *args:[])
    base=dict(passed=False,gates=dict(force=True,mass_velocity=True,force_difference_decreases=True,
        activation_replayed=False,activation_brackets_overlap=False))
    monkeypatch.setattr(contact,'compare',lambda *args,**kwargs:base)
    cases=[]
    for level in range(3):
        path=tmp_path/str(level);path.mkdir()
        np.savez(path/'parent_state.npz',positions=np.zeros((3,2)),released=[True,True,True],active_faces=[True,True],history=[[.01,.01],[.06,.06]])
        states=[]
        for i in range(3):
            name=f'{i}.npz';np.savez(path/name,positions=np.zeros((3,2)),active_faces=[True,True],history=[[.04,.04] if i==0 else [.06,.07],[.06,.06]],released=[True,True,True])
            states.append(dict(state_file=name,time_s=(i+1)*.1,depth_mm=(i+1)*.02,dt_s=.1,
                independent_fz_n=1.,activation_chain=dict(complete=True,events=[])))
        cases.append((path,dict(config={},initial_time_s=0.,initial_depth_mm=0.,dt_requested_s=.1,window_s=.3),dict(passed=True,states=states)))
    result=report.compare_failure_window(cases,dict(mechanism_force_absolute_n=1e-5),0)
    assert result['passed'] and result['force_sampling']['mechanism_force_consistency_passed']
    assert result['activation_window_only_gates']['activation_replayed'] is False
    # Maximum point history alone must never establish a full failed face.
    for i in (1,2):np.savez(cases[2][0]/f'{i}.npz',positions=np.zeros((3,2)),active_faces=[True,True],history=[[.04,.07],[.06,.06]],released=[True,True,True])
    result=report.compare_failure_window(cases,dict(mechanism_force_absolute_n=1e-5),0)
    assert not result['passed'] and not result['gates']['full_failure_brackets_overlap']
