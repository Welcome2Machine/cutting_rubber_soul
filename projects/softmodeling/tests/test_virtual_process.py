"""Energy-accounting and native contact decomposition for independent processes."""
import numpy as np
import pytest
from tools.controlled_virtual_release import auxiliary_work
from tools.diagnose_virtual_work import contact_partition
from test_virtual_work_coordinates import scene


@pytest.mark.parametrize('intervals',[8,16,32])
def test_auxiliary_extraction_cannot_be_omitted_or_counted_as_free_fracture_energy(intervals):
    # Two mirrored unit linear springs: prescribed load decreases 1 -> 0.
    # Quasistatic displacement follows load; storage is x^2 in the full domain.
    # All released energy goes to the unloading apparatus, none pays Gc for free.
    points=np.linspace(1.,0.,intervals+1);work=0.
    for a,b in zip(points[:-1],points[1:]):
        work+=auxiliary_work(np.array([a]),np.array([b]),np.array([a]),np.array([b]))
    delta=points[-1]**2-points[0]**2
    assert work==-1 and work-delta==0
    assert work-delta-.012 < 0
    assert -delta > 0  # omitting the auxiliary apparatus invents available energy


def test_contact_classification_sums_native_stencils_without_altering_power():
    v,start,end=scene();delta=end-start
    for s in (.1,.4745,.9):
        lag=v.lag(start+s*delta)
        for body_nodes in (0,len(v.surface)//2,len(v.surface)):
            v.n=body_nodes
            q=contact_partition(v,lag,delta[v.surface])
            full=2*float(v.friction.gradient(lag,v.cmesh,delta[v.surface])@delta[v.surface].ravel())
            assert sum(q['power_by_contact_n_mm'].values())==pytest.approx(full,rel=1e-12)
            assert all(c['power_n_mm']>=0 for c in q['contacts'])
        assert q['power_by_contact_n_mm']['blade']==0


def test_archived_equation_guard_allows_step_control_but_rejects_force_changes(tmp_path,monkeypatch):
    import hashlib
    from tools import frozen_scene
    zero='class ZeroGapCutProblem:\n    def step(self): return 1\n    def solve_active(self): return 2\n    def force(self): return 3\n'
    factory='def problem(config): return 0.14\n'
    sources={'prototype/zero_gap_cut.py':zero,'tools/run_zero_gap_cut.py':factory}
    for name,text in sources.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
    raw=dict(config={},source_hashes={k:digest(v) for k,v in sources.items()},execution_identity=dict(git_commit='0'*40))
    def current():return {k:digest((tmp_path/k).read_text()) for k in sources}
    monkeypatch.setattr(frozen_scene,'hashes',current)
    monkeypatch.setattr(frozen_scene.subprocess,'check_output',lambda args,**kw:sources[args[-1].split('projects/softmodeling/')[1]].encode())
    path=tmp_path/'prototype/zero_gap_cut.py'
    path.write_text(zero.replace('def step(self): return 1','def step(self): return 10'))
    q=frozen_scene.check_archived_equations(raw,tmp_path)
    assert not q['exact_producer_source'] and not q['physical_continuation_permitted']
    path.write_text(path.read_text().replace('def force(self): return 3','def force(self): return 30'))
    with pytest.raises(ValueError,match='physical zero-gap equation changed'):
        frozen_scene.check_archived_equations(raw,tmp_path)
