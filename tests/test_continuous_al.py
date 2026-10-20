"""Contact-space patch, mixed derivative, exact history and fresh audit tests."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from tools.run_al_integration import factory,run_case
from mechanics.finite_strain import response
from prototype.augmented_blade import blocks


class ContinuousContactTests(unittest.TestCase):
    def setUp(self):
        c=threadpool_limits(1);c.__enter__();self.addCleanup(c.__exit__,None,None,None)

    def problem(self,fast=False):
        p=factory(57,coarse=True,order=3,fast=fast,contact_basis='P1');p.ipc.set_num_threads(2)
        if fast:self.addCleanup(p.close_compute)
        return p

    def test_planar_patch_removes_alternating_gap_blind_mode(self):
        # P1 material gap on four straight edges. P0 moments annihilate an
        # alternating endpoint field; consistent P1 moments do not.
        p=self.problem();s,w=np.polynomial.legendre.leggauss(3)
        # Derive topology from actual ordered boundary edges, independent of IDs.
        e=int(np.argmin(np.linalg.norm(p.rest[p.body_edges].mean(axis=1)-p.blade_at_zero[0],axis=1)))
        p._set_quadrature(np.full(3,e),(s+1)/2,w/2*np.linalg.norm(np.diff(p.rest[p.body_edges[e]],axis=0))*p.config.width_mm)
        N=p.edgeQ.toarray();v=np.array([1.,-1.]);gap=N@v
        self.assertLess(abs(p.qweights@gap),1e-14)
        self.assertGreater(np.linalg.norm(p.average@gap),.4)
        mass=N.T@np.diag(p.qweights)@N
        self.assertGreater(np.linalg.eigvalsh(mass).min(),0)
        np.testing.assert_allclose(p.edgeQ@np.ones(2),1,atol=1e-14)

    def test_constant_traction_cone_action_reaction_and_mixed_tangent(self):
        p=self.problem();old=factory(57,coarse=True,order=3,fast=False)
        x=p.rest.copy();x[p.n:]=p.prescribed_blade_positions(.001)
        p.material_dt=.06;p.prepare_quadrature(x);old._set_quadrature(p.qedges,p.qfraction,p.qweights)
        p.multipliers=np.tile([.05,.012],(len(p.contact_edges),1))
        old.multipliers=np.tile([.05,.012],(len(old.contact_edges),1))
        q=p.contact_data(x);q0=old.contact_data(x)
        np.testing.assert_allclose(q['force'],q0['force'],atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(q['force'].reshape(-1,2).sum(axis=0),0,atol=1e-13)
        self.assertAlmostEqual(p.edge_area.sum(),old.edge_area.sum(),places=12)
        rng=np.random.default_rng(18);lam=rng.uniform(.01,.5,(len(p.contact_edges),2));lam[:,1]=(lam[:,1]-.25)*lam[:,0]*.35
        interp=p.edgeQ@lam;self.assertTrue(np.all(abs(interp[:,1])<=.35*interp[:,0]))
        # Full contact mixed derivative, including normal/tangent rotation and
        # cross blocks; no bulk kernel is necessary for this contact-space test.
        d=rng.normal(size=x.shape);d/=np.max(abs(d));dl=rng.normal(size=p.multipliers.shape)*.01
        before=p.multipliers.copy();h=1e-8
        def f(xx,ll):
            p.multipliers=ll;z=p.contact_data(xx)
            return np.r_[z['force'],z['constraint'].ravel()]
        fd=(f(x+h*d,before+h*dl)-f(x-h*d,before-h*dl))/(2*h);p.multipliers=before
        H=p.D.T@blocks(q['jacobian'])@p.D
        exact=np.r_[H@d.ravel()+q['force_lambda']@dl.ravel(),q['constraint_u']@d.ravel()+q['constraint_lambda']@dl.ravel()]
        self.assertLess(np.linalg.norm(fd-exact)/max(1.,np.linalg.norm(exact)),3e-6)

    def test_actual_step_original_audit_and_corrupt_history_rejection(self):
        p=self.problem(True);x=p.rest.copy();old=p.start
        with contextlib.redirect_stdout(io.StringIO()):
            for q in (0.,.005):x,record=p.step(x,old,q,time.monotonic()+60);old=q
        state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                   previous_positions=p.previous.copy(),depth_mm=old,**p.checkpoint_data())
        original=self.problem()
        with patch('prototype.ipc_viscoelastic_p2.response',wraps=response) as kernel:
            a=original.audit_state(state,record)
        self.assertEqual(kernel.call_count,1);self.assertTrue(a['passed'])
        self.assertFalse(hasattr(original,'_fresh_material_cache'))
        with patch.object(original,'_audit_response',side_effect=lambda F,**kw:kw['compute']()):
            b=original.audit_state(state,record)
        self.assertEqual(a,b)
        bad=dict(state,material_metric=state['material_metric'].copy());bad['material_metric'][0,0,0]+=.001
        with self.assertRaisesRegex(RuntimeError,'history'):original.audit_state(bad,record)
        bad=dict(record,motion=dict(record['motion'],tangent_resistance_n=123.))
        with self.assertRaises(AssertionError):original.audit_state(state,bad)
        bad=dict(state,al_multiplier_basis=np.array('P0'))
        with self.assertRaisesRegex(ValueError,'basis'):original.audit_state(bad,record)
        self.assertFalse(hasattr(original,'_fresh_material_cache'))
        saved=(p.material_metric.copy(),p.multipliers.copy(),p.contact_edges.copy())
        with self.assertRaises(TimeoutError):p.step(x,old,.01,time.monotonic()-1)
        np.testing.assert_array_equal(p.material_metric,saved[0]);np.testing.assert_array_equal(p.multipliers,saved[1]);np.testing.assert_array_equal(p.contact_edges,saved[2])

    def test_audit_cache_invalidates_exact_inputs(self):
        p=self.problem();p.material_dt=.1;F=np.eye(2)[None];calls=[]
        def compute():calls.append(1);return len(calls)
        p._fresh_material_cache=None
        def f():return p._audit_response(F,tangent=False,kinematics=p.kinematics,compute=compute)
        self.assertEqual(f(),1);self.assertEqual(f(),1)
        F[0,0,0]+=1e-10;self.assertEqual(f(),2)
        p.previous_material_metric[0,0,0]+=1e-10;self.assertEqual(f(),3)
        p.material_dt+=1e-10;self.assertEqual(f(),4)
        del p._fresh_material_cache;self.assertEqual(f(),5);self.assertEqual(f(),6)

    def test_restart_uses_only_own_contact_basis_and_exact_history(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            def args(name,depth,resume=None,basis='P1'):
                return argparse.Namespace(output=root/name,angle=57.,coarse=True,spacing=.000625,order=3,refinement=0,
                    depth=depth,increment=.005,seconds=120.,resume=resume,phase='case',blas_threads=1,contact_basis=basis)
            a=run_case(args('a',.005));self.assertTrue(a['completed'],a.get('error'))
            b=run_case(args('b',.01,root/'a'));c=run_case(args('c',.01));self.assertTrue(b['completed']);self.assertTrue(c['completed'])
            for key in ('positions','material_metric','previous_material_metric','al_multipliers'):
                with np.load(root/'b/states/0002.npz') as x,np.load(root/'c/states/0002.npz') as y:np.testing.assert_allclose(x[key],y[key],rtol=1e-11,atol=1e-11)
            with self.assertRaisesRegex(ValueError,'independent history'):run_case(args('wrong',.01,root/'a',basis='P0'))


if __name__=='__main__':unittest.main()
