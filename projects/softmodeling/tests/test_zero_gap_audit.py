import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pytest
from tools.run_zero_gap_cut import problem, snapshot, hashes
from tools.audit_zero_gap_cut import inspect


@pytest.fixture
def saved(tmp_path):
    cfg=json.loads((Path(__file__).parents[1]/'configs/zero_gap_dynamic_control.json').read_text())
    p=problem(cfg)
    old=p.released.copy()
    x,rec=p.step(p.rest.copy(),p.start,-.015,time.monotonic()+20)
    state=snapshot(p,x,old,-.015)
    path=tmp_path/'state_00000.npz'
    np.savez_compressed(path,**state)
    rec.update(state_file=path.name,state_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(schema='softcut-zero-gap-result/1',config=cfg,source_hashes=hashes(),steps=[rec])
    (tmp_path/'result.json').write_text(json.dumps(result))
    return tmp_path,state,result


def test_saved_audit_checks_initial_history_mass_and_hash(saved):
    directory,state,result=saved
    report=inspect(directory,20)
    assert report['passed'] and report['producer_source_matches_current']
    assert not report['loading_started']
    path=directory/result['steps'][0]['state_file']
    state['mass']=state['mass']*2
    np.savez_compressed(path,**state)
    assert not inspect(directory,20)['passed']
    result['steps'][0]['state_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/'result.json').write_text(json.dumps(result))
    # Updating a file hash cannot legitimize a change to the physical mass.
    assert not inspect(directory,20)['passed']


def test_saved_endpoint_audit_does_not_promote_unsafe_temporal_motion(saved,monkeypatch):
    directory,_,_=saved
    monkeypatch.setattr('tools.audit_zero_gap_cut.audit',lambda *a:dict(
        passed=True,accepted_chord_ccd_safe=False))
    report=inspect(directory,20)
    assert report['endpoint_history_passed']
    assert not report['accepted_chord_ccd_verified']
    assert not report['passed']


@pytest.mark.parametrize('seconds',[0,601,float('nan')])
@pytest.mark.parametrize('entry',['path','finite'])
def test_independent_audit_enforces_six_hundred_second_ceiling(tmp_path,seconds,entry):
    from tools.audit_finite_increment import inspect as finite
    with pytest.raises(ValueError,match='1..600'):
        (inspect if entry=='path' else finite)(tmp_path,seconds)


def test_contact_object_trace_distinguishes_original_surface_from_seam():
    from tools.audit_zero_gap_cut import blade_contacts
    cfg=json.loads((Path(__file__).parents[1]/'configs/zero_gap_dynamic_control.json').read_text())
    p=problem(cfg);x=p.rest.copy()
    assert blade_contacts(p,x)==[]
    x[p.n:]=p.prescribed_blade_positions(.003)
    records=blade_contacts(p,x)
    assert records and all(q['body_reference_group']=='original_top_surface' for q in records)
    assert all(all(i<p.n for i in q['body_nodes']) and all(i>=p.n for i in q['blade_nodes']) for q in records)
