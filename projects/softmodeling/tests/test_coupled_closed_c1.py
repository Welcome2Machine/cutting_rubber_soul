import json
from pathlib import Path
import numpy as np
import pytest
from coupled_cohesive import CoupledCohesiveProblem
from tools.run_coupled_growth import problem


def test_closed_c1_rebuilds_real_reference_mass_area_and_static_history():
    cfgs=Path(__file__).resolve().parents[1]/'configs'
    c=json.loads((cfgs/'coupled_closed_c1_growth_precision.json').read_text());wide=json.loads((cfgs/'coupled_wide_notch_growth_precision.json').read_text())
    assert c['base']==wide['base'] and c['precrack_mm']==.5 and c['pure_precrack'] is False
    p=problem(c);q=problem(wide)
    assert type(p) is CoupledCohesiveProblem and p.material_layer=='maxwell' and p.kinematic_theta==1
    assert p.initial_slit_halfwidth_mm==0 and p.released[-1] and p.rest[p.seam[-1],0]==15.
    np.testing.assert_array_equal(p.velocity,0*p.velocity)
    assert 2*p.mass.sum()==pytest.approx(3e-6) and p.mass.sum()>q.mass.sum()
    assert p.initial_fracture_energy_n_mm==pytest.approx(.06)
    assert p.rule['face_area_mm2'][p.precracked_faces].sum()==pytest.approx(10.)
    assert p.initial_fracture_energy_n_mm<q.initial_fracture_energy_n_mm
    assert np.all(p.history[p.precracked_faces]>=2*p.toughness/p.strength)
    assert not np.any(p.history[~p.precracked_faces]) and p.material==q.material
    np.testing.assert_allclose(p.deformation(p.rest),np.broadcast_to(np.eye(2),(len(p.mesh.triangles),2,2)),atol=1e-12)
