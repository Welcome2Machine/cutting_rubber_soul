import json
import pytest
from test_coupled_growth_window import control
from tools import run_coupled_growth as growth,run_fracture_route as route,continue_coupled_growth as cont,audit_growth_continuation as suffix


def test_bound_prefix_reuse_matches_full_audit_and_rejects_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    c=control();c['coupled_control']['depth_step_caps']=[dict(through_mm=2.,max_dt_s=.005)];rt=dict(ipc_threads=2,blas_threads=1)
    p=tmp_path/'parent';route.run(c,p,30,stop_depth=.0012,**rt);pa=tmp_path/'parent.audit.json';assert growth.audit(p,pa,30,**rt)['passed']
    child=tmp_path/'child';cont.run(p,pa,child,.0022,30,**rt)
    full=growth.audit(child,tmp_path/'full.json',30,**rt);fresh=suffix.audit(child,p,pa,tmp_path/'reuse.json',30,**rt)
    assert full['passed'] and fresh['passed'],fresh.get('error')
    assert full['states']==fresh['states']
    for key in ('new_fracture_dissipation_n_mm','absolute_numerical_work_n_mm','terminal_time_s'):assert full[key]==fresh[key]
    raw=json.loads((child/'result.json').read_text())
    import copy
    stopped=copy.deepcopy(raw);stopped.update(status='stopped',failure_kind='physical_solver_accuracy_incomplete',requested_stop_depth_mm=.0032)
    stage=stopped['continuation_stages'][-1];stage.update(status='stopped',requested_stop_depth_mm=.0032)
    trial=dict(from_mm=raw['steps'][-1]['depth_mm'],target_mm=.0032,dt_s=.005,failure_kind=stopped['failure_kind'])
    stopped['attempts'].append(trial);stage['attempt_range'][1]+=1
    model=growth.problem(c)
    with pytest.raises(ValueError,match='not resumable'):cont.controller_increment(model,stopped)
    assert cont.controller_history(model,stopped,terminal_accuracy_audit=True)[0]==cont.controller_increment(model,raw)
    trial['target_mm']+=.0001
    with pytest.raises(ValueError,match='trial schedule'):cont.controller_history(model,stopped,terminal_accuracy_audit=True)
    (child/raw['steps'][0]['state_file']).write_bytes(b'tampered')
    bad=suffix.audit(child,p,pa,tmp_path/'invalid.json',30,**rt);assert not bad['passed'] and 'Inherited state bytes' in bad['error']
