"""Native contact translation invariance and independent near-gap work checks."""
import time
import numpy as np
from test_virtual_ccd_precision import buffer_near_miss
from virtual_path_geometry import geometry_partitions
from virtual_friction_work import contact_power_function, virtual_work
from tools.virtual_work_audit import audit_work


def scene():
    v,start,end=buffer_near_miss()
    v.dhat=.0025;v.ffm_ccd_tolerance=1e-9
    v.config.friction_coefficient=.35
    v.barrier=v.ipc.BarrierPotential(v.dhat,132.,True)
    v.friction=v.ipc.FrictionPotential(1e-6)
    return v,start,end


def test_native_power_unchanged_away_from_unresolved_coordinate_scale():
    v,start,end=scene();_,geometry=geometry_partitions(v,start,end)
    power,meta=contact_power_function(v,start,end,geometry);delta=end-start
    for s in (0.,.1,.25,.75,1.):
        direct=2*float(v.friction.gradient(v.lag(start+s*delta),v.cmesh,delta)@delta.ravel())
        assert direct>0
        np.testing.assert_allclose(power(s),direct,rtol=1e-8,atol=1e-12)
    assert meta['unchanged_mathematical_path']


def test_near_gap_work_matches_distinct_extended_precision_legendre_audit():
    v,start,end=scene();q=virtual_work(v,start,end,time.monotonic()+10)
    check=audit_work(v,start,end,time.monotonic()+10)
    assert q['converged'] and check['converged']
    assert abs(q['work_n_mm']-check['work_n_mm'])<=q['empirical_uncertainty_n_mm']+check['empirical_uncertainty_n_mm']+1e-7
    assert q['coordinate_evaluation']['interpolation_dtype']=='float64'
    assert check['coordinate_evaluation']['mantissa_bits']>=53


def test_contact_rebasing_never_mutates_endpoints_or_anchor():
    v,start,end=scene();copies=[x.copy() for x in (start,end,v.previous,v.rest)]
    q=virtual_work(v,start,end,time.monotonic()+10)
    assert q['converged']
    for x,original in zip((start,end,v.previous,v.rest),copies):
        np.testing.assert_array_equal(x,original)
