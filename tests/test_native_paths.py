"""Physics/identity tests for native paths, independent of the P2 implementation."""
import contextlib
import io
import time
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tools import run_native_path_study as native
from tools import run_angle_path_study as high
from tools.run_material_candidate import resume_compatibility


class NativePathsTests(unittest.TestCase):
    def setUp(self):
        limits=threadpool_limits(2);limits.__enter__()
        self.addCleanup(limits.__exit__,None,None,None)

    def test_shared_motion_trace_and_no_enriched_history(self):
        for angle in (33,57):
            b=native.make_problem(angle,'normal',coarse=True)
            c=native.make_problem(angle,'oblique',coarse=True)
            p=high.make_problem(angle,'oblique',coarse=True)
            self.assertIsNone(c.enrichment)
            self.assertEqual(len(c.material_metric),len(c.mesh.triangles))
            self.assertEqual(c.n,p.base_nodes)
            np.testing.assert_array_equal(b.rest,c.rest)
            np.testing.assert_array_equal(c.rest[c.surface],p.rest[p.surface])
            np.testing.assert_array_equal(c.cmesh.edges,p.cmesh.edges)
            for q in (c.start,0.,.04):
                np.testing.assert_array_equal(c.prescribed_blade_positions(q),p.prescribed_blade_positions(q))
            delta=c.prescribed_blade_positions(.04)[0]-c.blade_at_zero[0]
            np.testing.assert_allclose(delta,[0.,-.04/np.cos(np.deg2rad(angle))],atol=2e-14)
            # Cross-mode and cross-discretization restarts must remain forbidden.
            r=dict(signature=native.signature(c),source_sha256={})
            for other in (native.signature(b),high.signature(p)):
                with self.assertRaises(ValueError):resume_compatibility(r,dict(r,signature=other))

    def test_affine_field_matches_P2_integration_with_same_metric(self):
        a=native.make_problem(33,'oblique',coarse=True)
        b=high.make_problem(33,'oblique',coarse=True)
        for p in (a,b):
            p.material_dt=.05
            x=p.rest.copy();u=np.column_stack((.002*p.mesh.local_sn[:,1],-.003*p.mesh.local_sn[:,1]))
            x[:p.n]+=u
            r=p.material_response(x,tangent=True)
            if p is a:expected=r
            else:
                # Uniform gradient: all quadrature material points must agree.
                for key in ('P','tangent','metric','energy','J'):
                    np.testing.assert_allclose(r[key],np.broadcast_to(expected[key][0],r[key].shape),rtol=2e-10,atol=2e-10)

    def test_real_increment_audit_and_power_with_material_history(self):
        p=native.make_problem(57,'oblique',coarse=True)
        x=p.rest.copy();old=p.start
        with contextlib.redirect_stdout(io.StringIO()):
            for q in (0.,.005):
                previous=p.material_metric.copy()
                x,row=p.step(x,old,q,time.monotonic()+45);old=q
                np.testing.assert_array_equal(p.previous_material_metric,previous)
                state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                    previous_positions=p.previous,depth_mm=q,**p.checkpoint_data())
                self.assertTrue(p.audit_state(state,row)['passed'])
        lag=p.lag(x);direction=np.zeros_like(x);direction[p.n:]=p.loading_direction
        h=1e-6
        derivative=(p.evaluate(x+h*direction,lag,tangent=False)[0]-p.evaluate(x-h*direction,lag,tangent=False)[0])/(2*h)
        self.assertAlmostEqual(derivative,row['motion']['generalized_resistance_n'],delta=1e-4)
        self.assertAlmostEqual(row['reaction_fz_n'],row['motion']['normal_reaction_n']*np.cos(np.deg2rad(57))+
            row['motion']['tangent_resistance_n']*np.sin(np.deg2rad(57)),places=10)

    def test_batched_material_matches_history_tangent_and_actual_path(self):
        a=native.make_problem(33,'oblique',coarse=True)
        b=native.make_problem(33,'oblique',coarse=True,material_batch_size=257)
        xa=a.rest.copy();xb=b.rest.copy();old=a.start
        with contextlib.redirect_stdout(io.StringIO()):
            for q in (0.,.005,.01):
                xa,ra=a.step(xa,old,q,time.monotonic()+45)
                xb,rb=b.step(xb,old,q,time.monotonic()+45);old=q
                np.testing.assert_allclose(xa,xb,atol=1e-7,rtol=0)
                np.testing.assert_allclose(a.material_metric,b.material_metric,atol=1e-8,rtol=1e-8)
                self.assertAlmostEqual(ra['reaction_fz_n'],rb['reaction_fz_n'],delta=1e-6)
        original=a.material_response(xa,tangent=True)
        b.previous_material_metric=a.previous_material_metric.copy();b.material_dt=a.material_dt
        chunked=b.material_response(xa,tangent=True)
        for key in original:
            np.testing.assert_allclose(original[key],chunked[key],rtol=1e-8,atol=1e-8,err_msg=key)
        # A changed history must invalidate all previous material responses.
        b.previous_material_metric=np.tile(np.diag([1.02,1/1.02,1.]),(len(b.mesh.triangles),1,1))
        self.assertGreater(np.max(abs(b.material_response(xa)['P']-original['P'])),1e-5)

    def test_incomplete_paths_cannot_trigger_deep_jobs(self):
        jobs=[dict(case=name,stage='shallow',mechanics_completed=False) for name,_,_ in native.CASES]
        gate=native.depth_gate(Path('unused'),{},dict(jobs=jobs))
        self.assertFalse(gate['passed'])
        self.assertEqual(len(gate['checks']),4)
        self.assertFalse(any(c['passed'] for c in gate['checks']))

    def test_pair_rejects_different_material_time_schedules(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'b',Path(tmp)/'c'
            for p,mode in ((a,'normal'),(b,'oblique')):
                p.mkdir();(p/'states').mkdir()
                sig={k:0 for k in ('config','material','loading_rate','mesh_sha256',
                    'contact_formulation','dhat','discretization','material_batch_size')}
                records=[dict(depth_mm=q,material={'dt_s':.05}) for q in (0.,.005)]
                (p/'result.json').write_text(json.dumps(dict(signature=sig,settings={'mode':mode},steps=records)))
                np.savez(p/'states/0000.npz',reference_xy=np.zeros((3,2)),positions=np.zeros((3,2)),material_metric=np.eye(3)[None])
            self.assertTrue(native.pair_check(a,b)['passed'])
            r=json.loads((b/'result.json').read_text());r['steps'].insert(1,dict(depth_mm=.0025,material={'dt_s':.025}))
            (b/'result.json').write_text(json.dumps(r))
            self.assertFalse(native.pair_check(a,b)['passed'])


if __name__=='__main__':unittest.main()
