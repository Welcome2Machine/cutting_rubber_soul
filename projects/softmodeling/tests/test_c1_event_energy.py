import pytest
from tools.report_c1_event import event_energy_scale


def test_reservoir_scale_does_not_pay_release_and_dissipation_twice():
    ledger=dict(positive_external_work_n_mm=.0001,independent_bulk_delta_n_mm=-.8,independent_cohesive_delta_n_mm=.1,independent_barrier_delta_n_mm=-.01,independent_fracture_n_mm=.6,independent_friction_n_mm=.2)
    a=event_energy_scale(.03,ledger)
    assert a['value_n_mm']==pytest.approx(.8401)
    ledger['independent_friction_n_mm']=200
    assert event_energy_scale(.03,ledger)==a
    assert a['components']['cohesive_net_recoverable_release_n_mm']==0


def test_zero_or_invalid_event_reference_cannot_certify_accuracy():
    d=dict(positive_external_work_n_mm=0.,independent_bulk_delta_n_mm=.1,independent_cohesive_delta_n_mm=0.,independent_barrier_delta_n_mm=0.)
    with pytest.raises(ValueError):event_energy_scale(0.,d)
    with pytest.raises(ValueError):event_energy_scale(-1.,d)


def test_closed_event_report_binds_parent_audit_states_and_raw_energies(tmp_path,monkeypatch):
    import json
    from test_coupled_growth_window import control
    from tools import run_coupled_growth as growth,run_fracture_route as route,branch_coupled_window as branch
    from tools.report_c1_event import run
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    rt=dict(ipc_threads=2,blas_threads=1);c=control();c['initial_slit_halfwidth_mm']=0.
    parent=tmp_path/'path';raw=route.run(c,parent,30,stop_depth=.001,**rt)
    proof=tmp_path/'path.audit.json';assert growth.audit(parent,proof,30,**rt)['passed']
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(coupled_window=dict(parent_state_index=len(raw['steps'])-1,dt_s=[.0001],window_s=.0003,solve_tolerance_n=1e-7))))
    case=tmp_path/'window';branch.run(parent,proof,len(raw['steps'])-1,plan,case,.0001,30,**rt)
    audit=tmp_path/'window.audit.json';assert branch.audit(case,audit,30,**rt)['passed']
    report=run(case,audit,tmp_path/'report',seconds=30,**rt)
    assert report['accepted_states']==3 and report['duration_s']==pytest.approx(.0003)
    assert not report['event_complete'] and not report['time_converged']
    assert report['event_energy_reference']['components']['positive_external_work_n_mm']>0
    assert (tmp_path/'report/geometry_parent.png').exists()
    p=case/'state_00000.npz';p.write_bytes(b'changed state')
    with pytest.raises(ValueError,match='state binding'):run(case,audit,tmp_path/'bad',seconds=30,**rt)


def test_original_node_contact_uses_nested_audit_and_excludes_physical_ends():
    from types import SimpleNamespace
    import numpy as np
    from tools.report_c1_event import original_node_contacts
    p=SimpleNamespace(rest=np.array([[15.,4.5],[15.,4.8],[15.,5.]]),config=SimpleNamespace(height_mm=5.))
    contacts=[dict(contact=dict(body_nodes=nodes)) for nodes in ([0],[1],[2],[0,1])]
    assert original_node_contacts(p,contacts,.5)==[contacts[1]]


def test_common_event_comparison_binds_reference_and_keeps_missing_events_failed(tmp_path,monkeypatch):
    import json
    from test_coupled_growth_window import control
    from tools import run_coupled_growth as growth,run_fracture_route as route,branch_coupled_window as branch
    from tools.report_c1_event import run as report
    from tools.compare_c1_event import run as compare
    from tools.frozen_scene import digest
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    rt=dict(ipc_threads=2,blas_threads=1);c=control();c['initial_slit_halfwidth_mm']=0.
    parent=tmp_path/'path';raw=route.run(c,parent,30,stop_depth=.001,**rt);idx=len(raw['steps'])-1
    proof=tmp_path/'path.audit.json';assert growth.audit(parent,proof,30,**rt)['passed']
    pre=tmp_path/'pilot-plan.json';limits=dict(parent_state_index=idx,dt_s=[.0001],window_s=.0002,solve_tolerance_n=1e-7)
    pre.write_text(json.dumps(dict(coupled_window=limits)));pilot=tmp_path/'pilot';branch.run(parent,proof,idx,pre,pilot,.0001,30,**rt)
    pa=tmp_path/'pilot.audit.json';assert branch.audit(pilot,pa,30,**rt)['passed'];rr=report(pilot,pa,tmp_path/'pilot_report',seconds=30,**rt)
    root=tmp_path/'matrix';root.mkdir();pre=root/'preregistration.json'
    limits.update(dt_s=[.0001,.00005,.000025],event_energy_reference=dict(report_file=str(tmp_path/'pilot_report/summary.json'),report_sha256=digest(tmp_path/'pilot_report/summary.json'),value_n_mm=rr['event_energy_reference']['value_n_mm']),convergence_targets=dict(force_relative=.05,velocity_mass_relative=.05,event_bracket_mm=.0001,absolute_numerical_work_fraction=.01,impulse_relative=.05,fracture_relative=.05,mechanism_force_absolute_n=1e-5))
    pre.write_text(json.dumps(dict(coupled_window=limits)))
    for name,dt in zip(('coarse','middle','fine'),limits['dt_s']):
        branch.run(parent,proof,idx,pre,root/name,dt,30,**rt);a=root/(name+'.audit.json');assert branch.audit(root/name,a,30,**rt)['passed'];report(root/name,a,root/(name+'_event'),seconds=30,**rt)
    q=compare(root)
    assert not q['strict_time_comparison_passed'] and not q['gates']['new_failure_observed']
    assert len(q['energy'])==3
    assert len({row['reference_scale_n_mm'] for row in q['energy']})==1
    (tmp_path/'pilot_report/summary.json').write_text('{}')
    with pytest.raises(ValueError,match='Reference pilot report changed'):compare(root)
