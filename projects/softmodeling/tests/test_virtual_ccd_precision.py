"""Native CCD near-miss regression and unchanged unilateral/J constraints."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.ipc_reference import toolkit
from tools.run_zero_gap_cut import problem
from test_finite_increment import config


def near_miss(start=None, end=None):
    # Minimal three-vertex witness from the server's virtual failure. The point
    # (0) approaches the opposite edge (1,2) without ever becoming collinear.
    if start is None:
        start = [[15., 3.715043108781945], [15.1029213229608, 3.8368949894176096],
                 [15., 3.718068930652552]]
        end = [[15.04047744071824, 3.757255779154387],
               [15.113619320202934, 3.867404703914534], [15., 3.7014800992549253]]
    start,end = np.array(start),np.array(end)
    v = object.__new__(FrozenVirtualCrack)
    v.ipc = toolkit();v.ipc.set_num_threads(1)
    v.n = 3;v.rest = start.copy();v.previous = start.copy();v.surface = np.arange(3)
    v.mesh = SimpleNamespace(reference_xy=start.copy(), triangles=np.array([[0, 1, 2]]))
    inverse = np.linalg.inv(np.column_stack((start[1]-start[0], start[2]-start[0])))
    v.grad = np.array([np.vstack((-inverse.sum(axis=0), inverse))])
    v.config = SimpleNamespace(drop_s_mm=15.)
    v.cmesh = v.ipc.CollisionMesh(start, np.array([[0, 1], [1, 2]]))
    v.cmesh.can_collide = v.ipc.make_static_obstacle_filter(3)
    v.virtual_narrow_phase_ccd = v.ipc.TightInclusionCCD(tolerance=1e-9)
    return v, start, end


def test_tight_ccd_resolves_certified_near_miss_without_dropping_self_contact():
    v, start, end = near_miss()
    q0 = [[Fraction(x) for x in row] for row in start]
    q1 = [[Fraction(x) for x in row] for row in end]
    sub = lambda a,b: [x-y for x,y in zip(a,b)]
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    a,b = sub(q0[1],q0[0]),sub(q0[2],q0[0])
    da = sub(sub(q1[1],q0[1]),sub(q1[0],q0[0]))
    db = sub(sub(q1[2],q0[2]),sub(q1[0],q0[0]))
    c2,c1,c0 = cross(da,db),cross(da,b)+cross(a,db),cross(a,b)
    assert c0 > 0 and c2 > 0 and c1*c1-4*c2*c0 < 0
    # Exact rational arithmetic on the binary64 inputs proves the signed area
    # stays positive. Default inclusion bounds remain overly conservative.
    assert v.ipc.compute_collision_free_stepsize(v.cmesh,start,end) < 1
    assert v.safe_step(start,end-start) == 1
    v.virtual_narrow_phase_ccd = None
    assert v.safe_step(start,end-start) < 1


def test_tight_ccd_still_rejects_real_crossing_and_inversion():
    v, start, _ = near_miss()
    end = start.copy()
    end[0] = .5*(start[1]+start[2]) - [.01,0.]
    fraction = v.ipc.compute_collision_free_stepsize(v.cmesh,start,end,
        narrow_phase_ccd=v.virtual_narrow_phase_ccd)
    assert 0 < fraction < 1
    assert v.safe_step(start,end-start) < 1


def test_tight_virtual_precision_retains_exact_halfspace():
    v, start, end = near_miss()
    end[0,0] = 14.99
    assert v.safe_step(start,end-start) == 0


@pytest.mark.parametrize('tolerance', [0., -1e-9, 1e-5, float('nan')])
def test_virtual_precision_cannot_weaken_native_error_bound(tolerance):
    with pytest.raises(ValueError, match='only tighten'):
        problem(dict(config(),ffm_ccd_tolerance=tolerance))


def test_virtual_precision_does_not_change_physical_loading_ccd():
    p = problem(dict(config(),ffm_ccd_tolerance=1e-9))
    v = FrozenVirtualCrack(p,p.rest,p.material_metric,p.history,p.released,
                          np.zeros_like(p.released),p.ffm_epsilon)
    assert not hasattr(p,'virtual_narrow_phase_ccd')
    assert v.virtual_narrow_phase_ccd.tolerance == 1e-9
    assert v.virtual_narrow_phase_ccd.conservative_rescaling == .8


def buffer_near_miss():
    return near_miss([[15.,3.715052882775612], [15.102925435649878,3.836903048806188],
                     [15.,3.71807881412435]],
                    [[15.040477467774148,3.75725579433116], [15.113619334493869,3.8674047692065208],
                     [15.,3.7014800389597258]])


def test_native_subdivision_covers_entire_same_chord_without_changing_scaling():
    v,start,end = buffer_near_miss()
    assert v.ipc.compute_collision_free_stepsize(v.cmesh,start,end,
        narrow_phase_ccd=v.virtual_narrow_phase_ccd) < 1
    v.ffm_chord_max_queries = 64
    assert v.accepted_chord_is_safe(end)
    cert = v.virtual_chord_certificate
    segments = cert['segments']
    assert cert['complete'] and 1 < cert['native_queries'] <= 64
    assert segments[0][0] == 0 and segments[-1][1] == 1
    assert all(a[1] == b[0] for a,b in zip(segments[:-1],segments[1:]))
    assert v.virtual_narrow_phase_ccd.conservative_rescaling == .8
    for a,b in segments:
        assert v.ipc.compute_collision_free_stepsize(v.cmesh,start+a*(end-start),start+b*(end-start),
            narrow_phase_ccd=v.virtual_narrow_phase_ccd) >= 1-1e-10


def test_incomplete_cover_cannot_accept_a_native_query_budget_exhaustion():
    v,start,end = buffer_near_miss()
    v.ffm_chord_max_queries = 1
    assert not v.accepted_chord_is_safe(end)
    assert not v.virtual_chord_certificate['complete']
    assert v.virtual_chord_certificate['native_queries'] == 1


def test_subdivisions_cannot_step_across_real_collision_even_without_bulk_gate():
    v,start,_ = near_miss()
    # Isolate surface CCD: remove the bulk triangle, leaving the native EV pair.
    v.mesh.triangles = np.empty((0,3),int);v.grad = np.empty((0,3,2))
    end = start.copy();end[0] = .5*(start[1]+start[2])-[.01,0.]
    v.ffm_chord_max_queries = 64
    assert not v.accepted_chord_is_safe(end)
    assert not v.virtual_chord_certificate['complete']


def test_subdivision_keeps_whole_chord_j_and_halfspace_gates():
    v,start,_ = near_miss();v.ffm_chord_max_queries = 64
    end = start.copy();end[0] = .5*(start[1]+start[2])-[.01,0.]
    assert not v.accepted_chord_is_safe(end)
    assert v.virtual_chord_certificate['native_queries'] == 0
    end=start.copy();end[0,0] -= .01
    assert not v.accepted_chord_is_safe(end)
    assert v.virtual_chord_certificate['native_queries'] == 0


@pytest.mark.parametrize('queries', [False,-1,257,1.5])
def test_virtual_chord_work_bound_is_explicit(queries):
    with pytest.raises(ValueError, match='0..256'):
        problem(dict(config(),ffm_chord_max_queries=queries))
