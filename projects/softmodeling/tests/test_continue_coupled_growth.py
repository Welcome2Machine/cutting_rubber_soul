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
