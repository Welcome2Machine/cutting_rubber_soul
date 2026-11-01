"""Error-budget stops require observed, resolvable and reversible probes."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import eye
from nonlinear_accuracy import inspect,try_stop,validate_policy


def policy():return dict(version='resolved_probe_budget_v1',time_error_fraction=.05,nonlinear_fraction=.1,force_change_n=1e-6,equation_work_rate_n_mm_s=.1,impulse_horizon_s=.001,resolved_prediction_rtol=.05,condition_limit=1e10)


class LinearControl:
    n=1;total=4;free=np.array([1]);mass=np.array([1.]);mass_dof=np.ones(4);material_dt=1e-4;loading_rate=.2
    config=SimpleNamespace(drop_s_mm=0.);seam=np.array([],dtype=int);strength=.2;toughness=.006
    def __init__(self,wrong_tangent=False,unsafe=False):
        self.nonlinear_error_policy=policy();self.previous=np.zeros((2,2));self.previous_velocity=np.zeros((2,2));self.velocity=np.zeros((2,2));self.history=np.zeros((1,1));self.material_metric=np.eye(3)[None];self.previous_history=self.history.copy();self.previous_material_metric=self.material_metric.copy();self.active_faces=np.array([True]);self.active_contact=np.array([False]);self.released=np.array([],bool);self.wrong=wrong_tangent;self.unsafe=unsafe;self.inc=self.previous.copy();self.x=self.previous.copy();self.cmesh=SimpleNamespace(edges=[],faces=[])
    def step_increment(self,x):return self.inc.copy()
    def set_primary_increment(self,x,du):self.x=x.copy();self.inc=du.copy()
    def endpoint_velocity(self,x):return self.inc/self.material_dt
    def lag(self,x):return None
    def collisions(self,x):return SimpleNamespace(ev_collisions=[],vv_collisions=[])
    def safe_step(self,x,d):return 0. if self.unsafe else 1.
    def accepted_chord_is_safe(self,x):return not self.unsafe
    def cohesive_response(self,x):return dict(history=self.history.copy())
    def material_response(self,x):return dict(metric=self.material_metric.copy())
    def evaluate(self,x,lag,tangent=True,projected=False):
        g=np.zeros(4);g[1]=x[0,1];return 0.,g,(2 if self.wrong else 1)*eye(4,format='csc'),1.
    def coupled_friction_tangent(self,x,lag,H):return H


def control(**kw):
    p=LinearControl(**kw);x=p.previous.copy();x[0,1]=2e-10;p.set_primary_increment(x,x.copy());return p,x


def test_actual_probes_are_replayable_and_do_not_commit_history():
    p,x=control();du=p.step_increment(x);a=inspect(p,x)
    assert a['passed'],a.get('reason')
    assert [q['amplifier'] for q in a['probes']]==[1,16,256]
    assert max(q['relative_prediction_error'] for q in a['probes'])<1e-12
    np.testing.assert_array_equal(p.step_increment(x),du)
    np.testing.assert_array_equal(p.history,[[0.]])
    assert inspect(p,x)==a


@pytest.mark.parametrize('mode',['tangent','trajectory','allocation','friction','physical'])
def test_bad_reliability_or_budget_never_certifies(mode):
    p,x=control(wrong_tangent=mode=='tangent',unsafe=mode=='trajectory')
    if mode=='allocation':p.nonlinear_error_policy['impulse_horizon_s']=1e9
    if mode=='friction':p._friction_homotopy_factor=10
    if mode=='physical':x[0,1]=.001;p.set_primary_increment(x,x.copy())
    before=p.step_increment(x);a=inspect(p,x)
    assert not a['passed'] and a['reason']
    np.testing.assert_array_equal(p.step_increment(x),before)


def test_optional_policy_does_not_change_legacy_solve():
    p,x=control();del p.nonlinear_error_policy
    assert not try_stop(p,x,float('inf'))
    for key,value in [('nonlinear_fraction',.11),('time_error_fraction',.06),('force_change_n',1e-5),('condition_limit',1e12)]:
        q=policy();q[key]=value
        with pytest.raises(ValueError):validate_policy(q)


def test_probe_search_requires_two_consecutive_resolved_amplitudes():
    p,x=control();original=p.evaluate
    def noisy(x,lag,tangent=True,projected=False):
        e,g,H,j=original(x,lag,tangent,projected)
        if x[0,1]<0:g[1]+=2e-10
        return e,g,H,j
    p.evaluate=noisy
    a=inspect(p,x)
    assert a['passed'],a.get('reason')
    assert a['probes'][1]['relative_prediction_error']>.05
    assert a['resolved_amplifiers']==[256.,4096.]
    assert a['probes'][1]['amplifier']==16


def test_plateau_only_requests_original_equation_certificate(monkeypatch):
    import nonlinear_accuracy
    from prototype.force_globalization import accuracy_plateau
    from prototype.ipc_reference import LoadStepFailure
    p,x=control();h=[];calls=[]
    monkeypatch.setattr(nonlinear_accuracy,'try_stop',lambda *a:calls.append(a) or False)
    for i in range(8):assert not accuracy_plateau(p,x,1e-8,1e-9,h,1e10)
    assert not calls
    assert not accuracy_plateau(p,x,1e-8,1e-9,h,1e10) and len(calls)==1
    p._friction_homotopy_factor=10
    with pytest.raises(LoadStepFailure,match='original-equation'):accuracy_plateau(p,x,1e-8,1e-9,h,1e10)
    assert len(calls)==1


def test_policy_parent_rejects_physical_source_change_and_missing_declaration(tmp_path):
    import json
    from tools.branch_coupled_window import numerical_policy_parent
    from tools.frozen_scene import digest
    (tmp_path/'result.json').write_text('{}');audit=tmp_path/'audit.json';audit.write_text('{}')
    raw=dict(source_hashes={'nonlinear_accuracy.py':'old','coupled_contact.py':'same'})
    proof=dict(passed=True,source_hashes=raw['source_hashes']);current={**raw['source_hashes'],'nonlinear_accuracy.py':'new'}
    q=dict(parent_result_sha256=digest(tmp_path/'result.json'),parent_audit_sha256=digest(audit),parent_source_hashes=raw['source_hashes'],child_source_hashes=current,changed_files=['nonlinear_accuracy.py'],production_continuation=False,reviewed_numerical_strategy_diff='Bounded probe selection only')
    manifest=tmp_path/'policy.json';manifest.write_text(json.dumps(q))
    assert numerical_policy_parent(raw,proof,tmp_path,audit,manifest,current)['production_continuation'] is False
    with pytest.raises(ValueError,match='Physical-equation'):numerical_policy_parent(raw,proof,tmp_path,audit,manifest,{**current,'coupled_contact.py':'changed'})
    q['parent_result_sha256']='wrong';manifest.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='Unbound'):numerical_policy_parent(raw,proof,tmp_path,audit,manifest,current)
