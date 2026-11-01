import json
import numpy as np
import pytest
from test_coupled_growth_window import control
from tools import run_coupled_growth as growth,run_fracture_route as route
from tools import continue_coupled_growth as continuation


def test_split_path_preserves_all_histories_controller_and_predictor(tmp_path,monkeypatch):
    c=control();c['coupled_control']['depth_step_caps']=[dict(through_mm=.001,max_dt_s=.005),dict(through_mm=2.,max_dt_s=.005)]
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    runtime=dict(ipc_threads=2,blas_threads=1)
    full=route.run(c,tmp_path/'full',30,stop_depth=.003,**runtime)
    parent=route.run(c,tmp_path/'parent',30,stop_depth=.001,**runtime)
    proof=tmp_path/'parent.audit.json';assert growth.audit(tmp_path/'parent',proof,30,**runtime)['passed']
    result=continuation.run(tmp_path/'parent',proof,tmp_path/'suffix',.003,30,**runtime)
    assert result['status']=='completed_pilot'
    audit=growth.audit(tmp_path/'suffix',tmp_path/'suffix.audit.json',30,**runtime)
    assert audit['passed'],audit.get('error')
    assert len(result['steps'])==len(full['steps'])
    for a,b in zip(full['steps'],result['steps']):
        assert a['depth_mm']==b['depth_mm']
        with np.load(tmp_path/'full'/a['state_file']) as x,np.load(tmp_path/'suffix'/b['state_file']) as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    bad=json.loads((tmp_path/'parent/result.json').read_text());bad['config']['base']['mu']=.1
    (tmp_path/'parent/result.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):continuation.run(tmp_path/'parent',proof,tmp_path/'invalid',.003,30,**runtime)


def test_two_successive_off_cap_stages_match_same_uninterrupted_segment_schedule(tmp_path,monkeypatch):
    c=control();c['coupled_control']['depth_step_caps']=[dict(through_mm=2.,max_dt_s=.005)]
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    rt=dict(ipc_threads=2,blas_threads=1);targets=[.0012,.0022,.0032]
    out=tmp_path/'stage0';q=route.run(c,out,30,stop_depth=targets[0],**rt)
    for k,target in enumerate(targets[1:],1):
        proof=tmp_path/f'stage{k-1}.audit.json';a=growth.audit(out,proof,30,**rt);assert a['passed'],a.get('error')
        new=tmp_path/f'stage{k}';q=continuation.run(out,proof,new,target,30,**rt);out=new
        assert q['status']=='completed_pilot',q.get('error')
    assert [s['requested_stop_depth_mm'] for s in q['continuation_stages']]==targets
    assert growth.audit(out,tmp_path/'stage2.audit.json',30,**rt)['passed']
    continuation.controller_increment(growth.problem(c),q)
    # A live object follows the exact SAME stage truncations without restart.
    # It is deliberately not compared with a path lacking those truncations.
    def segmented_factory(config):
        p=growth.problem(config);next_depth=p.next_loading_depth
        def segmented(old,increment,target):
            boundary=next(t for t in targets if old<t-1e-12)
            return next_depth(old,increment,min(target,boundary))
        p.next_loading_depth=segmented;return p
    monkeypatch.setattr(route,'problem',segmented_factory)
    ref=route.run(c,tmp_path/'reference',30,stop_depth=targets[-1],**rt)
    assert len(ref['steps'])==len(q['steps'])
    for a,b in zip(ref['steps'],q['steps']):
        assert a['depth_mm']==b['depth_mm'] and a['face_events']==b['face_events']
        with np.load(tmp_path/'reference'/a['state_file']) as x,np.load(out/b['state_file']) as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    import copy
    for mutation in ('target','range','controller','parent'):
        bad=copy.deepcopy(q);s=bad['continuation_stages'][1]
        if mutation=='target':s['requested_stop_depth_mm']+=.0001
        elif mutation=='range':s['state_range'][0]+=1
        elif mutation=='controller':s['initial_controller_increment_mm']*=2
        else:s['parent_state']['state_sha256']='bad'
        with pytest.raises(ValueError):continuation.controller_increment(growth.problem(c),bad)
