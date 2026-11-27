import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from tools.audit_virtual_chord import separated_ev_prefix, verify_witness


def test_exact_rational_crossing_and_outside_edge_rejection():
    start=np.array([[0.,0.],[1.,0.],[.5,1.]])
    end=start.copy();end[2,1]=-1
    assert verify_witness(start,end,[0,1,2],[.4,.6])['passed']
    start[2,0]=2;end[2,0]=2
    assert not verify_witness(start,end,[0,1,2],[.4,.6])['passed']


def test_near_miss_does_not_become_a_crossing():
    start=np.array([[0.,0.],[1.,0.],[.5,1e-12]])
    end=start.copy();end[2,0]=.6
    assert not verify_witness(start,end,[0,1,2],[.4,.6])['passed']


def test_prefix_stops_before_contact_and_accepts_collinear_extension():
    x=np.array([[0.,0.],[1.,0.],[.5,1.]])
    y=x.copy();y[2,1]=-1.
    assert separated_ev_prefix(x,y,[0,1,2],.49)['separated']
    assert not separated_ev_prefix(x,y,[0,1,2],.5)['separated']
    x[2]=[2.,0.];y[2]=[3.,0.]
    assert separated_ev_prefix(x,y,[0,1,2],1.)['separated']


def test_timeout_diagnostic_requires_an_attempt_and_matching_audit(tmp_path, monkeypatch):
    from tools import verify_intact_numerics as study

    report = dict(status='budget_exhausted', steps=[{}, {}], source_sha256={},
                  settings={'diagnostics': False}, failed_attempt={'attempted_depth_mm': .0625})
    result_path = tmp_path/'result.json'
    audit_path = tmp_path/'audit.json'
    result_path.write_text(json.dumps(report))
    audit = dict(passed=True, trajectory_checked=True, states=2, qualified_prefix_states=2,
                 source_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    audit_path.write_text(json.dumps(audit))
    problem = SimpleNamespace()
    monkeypatch.setattr(study, 'make_problem', lambda: problem)
    monkeypatch.setattr(study, 'restore_failed_pair', lambda model, folder, saved, proof: (model, saved, proof))
    assert study.failed_increment(tmp_path, use_production_backend=False) == (problem, report, audit)
    audit_path.write_text(json.dumps(dict(audit, source_result_sha256='wrong')))
    with pytest.raises(ValueError, match='Audit and failed result differ'):
        study.failed_increment(tmp_path, use_production_backend=False)
    result_path.write_text(json.dumps(dict(report, failed_attempt=None)))
    with pytest.raises(ValueError, match='independently audited parent'):
        study.failed_increment(tmp_path, use_production_backend=False)
