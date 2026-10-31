import json,time
import numpy as np
import pytest
from test_coupled_cohesive import config
from tools import run_coupled_growth as growth
from tools import run_coupled_growth_window as window
from tools import run_fracture_route as route
from tools.face_witness import accepted_fields,array_hash


def control():return dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-7))


def test_conditional_growth_preserves_committed_arrays_and_rejects_invalid_topology():
    c=control();p=growth.problem(c);released=p.released.copy()
    x,row=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    s=route.snapshot(p,x,released,p.start+1e-4);q=growth.problem(c)
    y=window.restore_parent(q,s)
    assert array_hash(accepted_fields(p,x))==array_hash(accepted_fields(q,y))
    s['released']=~s['released']
    with pytest.raises(AssertionError):window.restore_parent(growth.problem(c),s)


def test_conditional_growth_cannot_silently_change_parent_layer_or_root_control():
    with pytest.raises(ValueError,match='cannot change'):window.candidate(control(),'frozen_metric',1e-7)
    with pytest.raises(ValueError,match='cannot change'):window.candidate(control(),'maxwell',1e-9)


def test_conditional_window_audit_binds_real_parent_clock_and_complete_window(tmp_path,monkeypatch):
    c=control();parent=tmp_path/'parent';proof=tmp_path/'parent.audit.json'
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    r=route.run(c,parent,30,stop_depth=.001,ipc_threads=2,blas_threads=1)
    a=growth.audit(parent,proof,30,ipc_threads=2,blas_threads=1);assert a['passed']
    plan=tmp_path/'plan.json';index=len(r['steps'])-1
    plan.write_text(json.dumps({'coupled_window':dict(parent_state_index=index,dt_s=[.0005],window_s=.001,solve_tolerance_n=1e-7)}))
    out=tmp_path/'window'
    q=window.run(parent,proof,index,plan,out,.0005,30,ipc_threads=2,blas_threads=1)
    assert q['status']=='completed_window' and q['conditional_comparison']
    a=window.audit(out,tmp_path/'window.audit.json',30,ipc_threads=2,blas_threads=1)
    assert a['passed'],a.get('error')
    assert a['time_chain']['complete_requested_window'] and not a['entry_present_at_parent']
    q['steps'][0]['time_s']+=.001
    (out/'result.json').write_text(json.dumps(q))
    a=window.audit(out,tmp_path/'wrong_clock.audit.json',30,ipc_threads=2,blas_threads=1)
    assert not a['passed'] and 'Production adjacent time' in a['error']
