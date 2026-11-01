import json
import numpy as np
from test_coupled_growth_window import control
from tools import run_coupled_growth as growth,run_fracture_route as route,run_coupled_growth_window as window,branch_coupled_window as branch


def test_audited_window_branch_restores_shared_history_and_predictors(tmp_path,monkeypatch):
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    rt=dict(ipc_threads=2,blas_threads=1);parent=tmp_path/'path';raw=route.run(control(),parent,30,stop_depth=.001,**rt)
    proof=tmp_path/'path.audit.json';assert growth.audit(parent,proof,30,**rt)['passed']
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(coupled_window=dict(parent_state_index=len(raw['steps'])-1,dt_s=[.0005],window_s=.002,solve_tolerance_n=1e-7))))
    win=tmp_path/'window';q=window.run(parent,proof,len(raw['steps'])-1,plan,win,.0005,30,**rt)
    wa=tmp_path/'window.audit.json';assert window.audit(win,wa,30,**rt)['passed']
    bp=tmp_path/'branchplan.json';bp.write_text(json.dumps(dict(coupled_window=dict(parent_state_index=1,dt_s=[.0005,.00025],window_s=.001,solve_tolerance_n=1e-7))))
    for name,h in [('restore',.0005),('hold',.00025)]:
        case=tmp_path/name;z=branch.run(win,wa,1,bp,case,h,30,**rt);assert z['status']=='completed_window'
        assert z['parent']['state_sha256']==q['steps'][1]['state_sha256'] and len(z['branch_control']['predictor_states'])==2
        audit=branch.audit(case,tmp_path/(name+'.audit.json'),30,**rt);assert audit['passed'],audit.get('error')
    with np.load(tmp_path/'restore/parent_state.npz') as a,np.load(tmp_path/'hold/parent_state.npz') as b:
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
