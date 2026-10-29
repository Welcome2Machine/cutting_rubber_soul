import numpy as np
from test_virtual_ccd_precision import buffer_near_miss,near_miss
from virtual_path_geometry import geometry_partitions
from virtual_path_probe import minimum_swept_gap,efficient_chord


def test_probe_matches_independent_polynomial_events_on_near_miss():
    v,start,end=buffer_near_miss();v.dhat=.0025;v.ffm_ccd_tolerance=1e-9
    probe=minimum_swept_gap(v,start,end);_,geometry=geometry_partitions(v,start,end)
    exact=min(q['gap_mm'] for q in geometry['features'])
    assert abs(probe['minimum_gap_mm']-exact)<1e-14
    assert efficient_chord(v,end,256)['complete']


def test_fast_schedule_still_rejects_crossing_and_missing_query_budget():
    v,start,end=near_miss();v.dhat=.0025;v.ffm_ccd_tolerance=1e-9
    assert not efficient_chord(v,end,0)['complete']
    end=start.copy();end[0]=.5*(start[1]+start[2])-[.01,0.]
    v.mesh.triangles=np.empty((0,3),int);v.grad=np.empty((0,3,2))
    assert minimum_swept_gap(v,start,end)['minimum_gap_mm']<1e-12
    assert not efficient_chord(v,end,256)['complete']


def test_simple_path_uses_one_complete_native_query():
    v,start,_=near_miss();v.dhat=.0025;v.ffm_ccd_tolerance=1e-9
    q=efficient_chord(v,start,256)
    assert q['complete'] and q['native_queries']==1 and q['segments']==[[0.,1.]]


def test_bernstein_probe_never_replaces_the_native_certificate():
    from virtual_path_probe import swept_gap_lower_bound
    v,start,end=buffer_near_miss();v.dhat=.0025
    bound=swept_gap_lower_bound(v,start,end)
    assert bound['distance_lower_bound_mm']<=minimum_swept_gap(v,start,end)['minimum_gap_mm']
    assert not bound['acceptance_shortcut']
