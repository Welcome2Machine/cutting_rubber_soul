"""Geometry seeds partition queries; only native certificates can accept."""
import numpy as np
import pytest
from test_virtual_ccd_precision import near_miss,buffer_near_miss
from virtual_path_geometry import geometry_partitions,ev_crossing_witnesses,native_partitioned_chord


def test_near_miss_partition_certifies_every_segment():
    v,start,end=buffer_near_miss();v.dhat=.0025;v.ffm_ccd_tolerance=1e-9
    assert not ev_crossing_witnesses(v,start,end)
    cuts,_=geometry_partitions(v,start,end)
    assert cuts[0]==0 and cuts[-1]==1 and all(b>a for a,b in zip(cuts[:-1],cuts[1:]))
    cert=native_partitioned_chord(v,end,256)
    assert cert['complete']
    assert cert['segments'][0][0]==0 and cert['segments'][-1][1]==1
    assert all(a[1]==b[0] for a,b in zip(cert['segments'][:-1],cert['segments'][1:]))
    for a,b in cert['segments']:
        assert v.ipc.compute_collision_free_stepsize(v.cmesh,start+a*(end-start),start+b*(end-start),narrow_phase_ccd=v.virtual_narrow_phase_ccd)>=1-1e-10


def test_real_crossing_is_witnessed_without_acceptance_shortcut():
    v,start,_=near_miss();v.dhat=.0025
    v.mesh.triangles=np.empty((0,3),int);v.grad=np.empty((0,3,2))
    end=start.copy();end[0]=.5*(start[1]+start[2])-[.01,0.]
    witnesses=ev_crossing_witnesses(v,start,end)
    assert witnesses and all(w['interior_over_bracket'] for w in witnesses)
    cert=native_partitioned_chord(v,end,256)
    assert not cert['complete'] and cert['native_queries']==0


def test_work_limit_and_unresolved_precision_never_certify():
    v,start,end=buffer_near_miss();v.dhat=.0025
    assert not native_partitioned_chord(v,end,1)['complete']
    v.ffm_ccd_tolerance=1.
    cert=native_partitioned_chord(v,end,256)
    assert not cert['complete'] and cert['failure']=='gap_below_requested_ccd_resolution'
