"""Endpoint event localization tests; no archived run is a test dependency."""
import numpy as np
import pytest
from test_face_cut import config
from tools.run_fracture_route import problem
from tools.face_events import endpoint_forces, outgoing_force, check_connected_endpoint
from prototype.zero_gap_cut import EnergyResolutionRequired


def test_endpoint_force_is_history_pure_and_retains_native_velocity_friction():
    p=problem(config(.5,'endpoint_friction'));x=p.rest.copy()
    p.material_dt=.01;p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*.01)
    p.set_primary_increment(x,np.zeros_like(x))
    saved={k:getattr(p,k).copy() for k in ('history','previous_history','material_metric','previous_material_metric','velocity','previous_velocity','released','active_faces')}
    fields=endpoint_forces(p,x)
    assert set(fields)=={'bulk','cohesive','normal','friction'}
    assert all(v.shape==(p.total,) for v in fields.values())
    for k,v in saved.items():np.testing.assert_array_equal(getattr(p,k),v)
    assert check_connected_endpoint(p,x)['maximum_connected_endpoint_excess_half_n']==0


def test_average_force_below_strength_cannot_hide_endpoint_crossing(monkeypatch):
    import tools.face_events as event
    p=problem(config());x=p.rest.copy();j=48;limit=outgoing_force(p)[j]
    # A completed force solve can have an admissible mean force while its
    # instantaneous relaxed endpoint has crossed strength. No successor dt
    # reduction can remove a crossing already committed to the initial state.
    mean=np.zeros(p.total);mean[2*p.seam[j]]=-limit+.004
    monkeypatch.setattr(p,'evaluate',lambda *a,**k:(0.,mean,None,1.))
    instantaneous=np.zeros(p.total);instantaneous[2*p.seam[j]]=-limit-.003
    monkeypatch.setattr(event,'endpoint_forces',lambda *a:{'bulk':instantaneous})
    for dt in (.025,1e-6,5e-11):
        p.material_dt=dt
        with pytest.raises(EnergyResolutionRequired,match='current physical step'):
            check_connected_endpoint(p,x)
        assert not p.released.any() and not p.history.any()
    instantaneous[2*p.seam[j]]=-limit-.5*p.force_tolerance_n
    assert check_connected_endpoint(p,x)['maximum_connected_endpoint_excess_half_n']<=p.force_tolerance_n


def test_endpoint_route_rejects_crossing_before_history_commit(monkeypatch):
    from tools.run_endpoint_face_route import problem as endpoint_problem
    import tools.run_endpoint_face_route as route
    p=endpoint_problem(config());before={k:getattr(p,k).copy() for k in ('history','material_metric','velocity','released','active_faces')}
    def crossed(*args):raise EnergyResolutionRequired('controlled endpoint crossing')
    monkeypatch.setattr(route,'check_connected_endpoint',crossed)
    import time
    with pytest.raises(EnergyResolutionRequired):p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    for k,v in before.items():np.testing.assert_array_equal(getattr(p,k),v)


def test_numpy_depth_cutback_report_is_serializable_without_summing_state_energy(tmp_path):
    from tools.run_endpoint_face_route import finish
    # Exercise the exact original failure: bool(rows) and numpy depth >=2.
    energy={'external_work_n_mm':1.,'kinetic_energy_n_mm':.2,'mechanical_reservoir_n_mm':3.}
    row=dict(depth_mm=np.float64(.1),time_s=1.,dt_s=.01,residual_n=1e-5,released_nodes=0,energy=energy,surface_connected_failed_length_mm=0.,blade_entered_surface_channel=False)
    for k in ['reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n','instantaneous_fz_n','instantaneous_normal_fz_n','instantaneous_friction_fz_n','maximum_speed_mm_s','mass_velocity_norm_sqrt_nmm','blade_normal_impulse_n_s','blade_friction_impulse_n_s']:row[k]=0.
    raw=dict(steps=[row,row.copy()],config={'precrack_mm':0.})
    finish(tmp_path,raw)
    import json
    saved=json.loads((tmp_path/'result.json').read_text())
    assert saved['summary']['completed_2mm'] is False
    assert saved['summary']['terminal_kinetic_energy_n_mm']==.2
    assert 'kinetic_energy_n_mm' not in saved['summary']['ledger_n_mm']
    assert saved['summary']['ledger_n_mm']['external_work_n_mm']==2.


def test_precrack_tip_strength_uses_only_the_remaining_intact_face():
    p=problem(config(.5));tip=int(np.flatnonzero(p.rest[p.seam,1]==4.5)[0])
    # Uniform traction on the remaining 0.1 mm x 20 mm P1 face supplies
    # half its resultant to the tip. The precracked neighbour supplies zero.
    assert outgoing_force(p)[tip]==pytest.approx(.2*.1*20/2)
    assert outgoing_force(p)[tip-1]==pytest.approx(.2*.1*20)
    assert p.seam_area[tip]*p.strength==pytest.approx(.2*.1*20)
    assert p.rule['face_area_mm2'][tip-1]==pytest.approx(2.)
