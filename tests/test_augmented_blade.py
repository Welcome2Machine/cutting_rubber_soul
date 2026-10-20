"""Independent mechanics checks for the explicit AL blade candidate."""
import contextlib
import io
import time
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.augmented_blade import (augmented_traction,convex_edge_gap_bounds,
                                       lipschitz_edge_gap_bounds,AugmentedBladeProblem)


def problem(angle=57,coarse=True,spacing=.00125):
    return AugmentedBladeProblem(specimen_angle=angle,path_mode='oblique',normal_speed=.1,
        maxwell_shear=1.24,relaxation_s=1.7,bulk_modulus=23.34333333333331,
        case='oblique',mu=.35,shear_modulus=.14,ogden_alpha=2.,dhat=.0025,
        local_spacing=.25 if coarse else .05,surface_spacing=.005,
        nose_facets=64,tip_radius=.02,fine_halfwidth=1. if coarse else 3.,
        max_iterations=500,predictor='affine',hessian_mode='regularized',element='p1',
        kinematics='plane_strain',enrichment_radius=.5,quadrature_order=3,
        contact_spacing=spacing)


class AugmentedBladeTests(unittest.TestCase):
    def setUp(self):
        c=threadpool_limits(2);c.__enter__();self.addCleanup(c.__exit__,None,None,None)

    def test_curved_fresh_tangent_slip_stick_and_separation(self):
        radius=.02
        for lam,rel in (([.4,0.],[.00001,0.]),([.4,.2],[.003,0.]),([0.,0.],[.001,0.])):
            x=np.array([.003,-.0202 if lam[0]==0 else -.0198]);previous=x-np.array(rel)
            def evaluate(z):
                length=np.linalg.norm(z);n=z/length;dn=(np.eye(2)-np.outer(n,n))/length
                return augmented_traction(np.array([length-radius]),n[None],dn[None],
                    (z-previous)[None],np.array([lam]),250.,.35)
            q=evaluate(x);h=1e-8
            fd=np.column_stack([(evaluate(x+h*np.eye(2)[j])['gradient'][0]-
                evaluate(x-h*np.eye(2)[j])['gradient'][0])/(2*h) for j in range(2)])
            np.testing.assert_allclose(q['jacobian'][0],fd,rtol=2e-6,atol=1e-5)

    def test_exact_spring_contact_stick_and_sliding(self):
        for target_x,expected_x,expected_shear in ((.02,0.,-.2),(.1,.065,-.35)):
            x=np.array([.001,-.01]);lam=np.zeros((1,2));K=10.
            def field(z):
                return augmented_traction(np.array([z[1]]),np.array([[0.,1.]]),
                    np.zeros((1,2,2)),z[None],lam,250.,.35)
            for _ in range(40):
                for _ in range(12):
                    q=field(x);r=K*(x-[target_x,-.1])+q['gradient'][0]
                    if np.linalg.norm(r)<1e-12:break
                    x-=np.linalg.solve(K*np.eye(2)+q['jacobian'][0],r)
                lam[:]=[q['pressure'][0],q['shear'][0]]
                if np.linalg.norm(np.r_[field(x)['pressure'],field(x)['shear']]-lam.ravel())<1e-11:break
            np.testing.assert_allclose(x,[expected_x,0.],atol=1e-10)
            np.testing.assert_allclose(lam,[[1.,expected_shear]],atol=1e-9)
            self.assertGreaterEqual(float(lam[0,1]*(-x[0])),-1e-12)

    def test_whole_edge_detects_penetration_missed_by_vertices(self):
        def circle(x):
            d=np.linalg.norm(x,axis=1);n=x/d[:,None]
            return d-1,n,(np.eye(2)-n[:,:,None]*n[:,None,:])/d[:,None,None]
        a=np.array([-1.1,.9]);b=np.array([1.1,.9])
        self.assertTrue(np.all(circle(np.array([a,b]))[0]>0))
        lo,up,_=convex_edge_gap_bounds(circle,a,b)
        self.assertLessEqual(lo,-.1+1e-12);self.assertGreaterEqual(up,-.1-1e-12)
        self.assertLess(up-lo,1e-7)

    def test_traction_transforms_under_rigid_rotation(self):
        phi=.7;Q=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        n=np.array([[.6,-.8]]);dn=(np.eye(2)-n[:,:,None]*n[:,None,:])/.02
        rel=np.array([[.003,.001]]);lam=np.array([[.4,.1]])
        a=augmented_traction(np.array([-.0001]),n,dn,rel,lam,250,.35)
        b=augmented_traction(np.array([-.0001]),n@Q.T,Q@dn@Q.T,rel@Q.T,lam,250,.35)
        np.testing.assert_allclose(a['gradient']@Q.T,b['gradient'],atol=1e-12)
        np.testing.assert_allclose(Q@a['jacobian']@Q.T,b['jacobian'],atol=1e-10)

    def test_gap_bounds_include_monotone_endpoints_and_exact_circle_minimum(self):
        def circle(x):
            d=np.linalg.norm(x,axis=1)
            return d-1,x/d[:,None],None
        rng=np.random.default_rng(41)
        pairs=[(np.array([1.,.1]),np.array([2.,.1])),
               (np.array([.3,.1]),np.array([.7,.1])),
               (np.array([1.,.1]),np.array([1.,.1]))]
        pairs += [(rng.uniform(-2,2,2),rng.uniform(-2,2,2)) for _ in range(50)]
        for a,b in pairs:
            v=b-a;vv=float(v@v)
            t=float(np.clip(-a@v/vv,0,1)) if vv else 0.
            exact=np.linalg.norm(a+t*v)-1
            for aa,bb in ((a,b),(b,a)):
                lo,up,_=convex_edge_gap_bounds(circle,aa,bb)
                self.assertLessEqual(lo,exact+2e-13)
                self.assertGreaterEqual(up,exact-2e-13)
                self.assertLessEqual(lo,up)
                self.assertLess(up-lo,1e-7)
                # Even a capped search must remain conservative.
                capped=convex_edge_gap_bounds(circle,aa,bb,maximum=1)
                self.assertLessEqual(capped[0],exact+2e-13)
                self.assertGreaterEqual(capped[1],exact-2e-13)

    def test_independent_distance_gate_and_invalid_distance_rejected(self):
        def circle(x):
            d=np.linalg.norm(x,axis=1)
            return d-1,x/d[:,None],None
        for y,expected in ((.9,False),(.9999,True),(1.01,True)):
            lo,up,_=lipschitz_edge_gap_bounds(circle,np.array([-.2,y]),
                                            np.array([.2,y]),threshold=-.0002)
            self.assertLessEqual(lo,y-1+1e-12)
            self.assertGreaterEqual(up,y-1-1e-12)
            self.assertEqual(lo>=-.0002,expected)
        def invalid(x):return np.full(len(x),np.nan),np.ones_like(x),None
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            convex_edge_gap_bounds(invalid,np.array([0.,1.]),np.array([1.,1.]))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            lipschitz_edge_gap_bounds(invalid,np.array([0.,1.]),np.array([1.,1.]),threshold=-.0002)

    def test_precision_comparison_requires_full_range_and_matching_physics(self):
        from tools.run_augmented_blade import compare
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'base',Path(tmp)/'control'
            angle=np.deg2rad(57);ref=np.array([[np.sin(angle),np.cos(angle)]])
            def write(folder,depths,force=1.,mu=.35):
                (folder/'states').mkdir(parents=True,exist_ok=True)
                rows=[]
                for i,q in enumerate(depths):
                    np.savez(folder/'states'/f'{i:04d}.npz',reference_xy=ref,positions=ref,
                             base_body_nodes=1)
                    rows.append(dict(depth_mm=q,reaction_fz_n=force,motion=dict(
                        normal_reaction_n=force,tangent_resistance_n=force,generalized_resistance_n=force)))
                r=dict(settings=dict(angle=57),steps=rows,signature=dict(config=dict(height_mm=1.,mu=mu),
                    material={},loading_rate=.1,contact_formulation='AL',motion={},mesh_sha256='same'))
                (folder/'result.json').write_text(json.dumps(r),encoding='utf-8')
            write(a,[0.,.05,.1]);write(b,[0.,.025,.05,.075,.1],force=1.01)
            self.assertTrue(compare(a,b,.1)['passed'])
            write(b,[0.,.05])
            self.assertFalse(compare(a,b,.1)['passed'])
            write(a,[.1]);write(b,[.1])
            self.assertFalse(compare(a,b,.1)['passed'])
            write(a,[0.,.05,.1]);write(b,[0.,.05,.1],mu=.3)
            with self.assertRaisesRegex(ValueError,'physical mismatch'):compare(a,b,.1)

    def test_real_geometry_material_layout_and_action_reaction(self):
        p=problem();p.material_dt=.05;p.previous_material_metric=p.material_metric.copy()
        x=p.rest.copy();x[p.n:]=p.prescribed_blade_positions(.001);p.prepare_quadrature(x)
        p.multipliers[:,0]=.5
        q=p.contact_data(x)
        self.assertGreater(np.linalg.norm(q['force']),.01)
        np.testing.assert_allclose(q['force'].reshape(-1,2).sum(axis=0),0.,atol=1e-13)
        before=p.material_metric.copy();p.evaluate(x,tangent=False)
        np.testing.assert_array_equal(p.material_metric,before)
        self.assertGreater(len(p.active_elements),0)
        self.assertEqual(p.contact_formulation,'weak-P0-AL-rigid-blade-v2')
        # Every boundary edge can trigger the independent gap audit.
        x[p.n:]=p.prescribed_blade_positions(.1)
        with self.assertRaisesRegex(RuntimeError,'penetration|outside quadrature'):
            p.geometry_audit(x)

    def test_full_assembled_directional_derivative_and_no_history_commit(self):
        p=problem();p.material_dt=.06;p.previous_material_metric=p.material_metric.copy()
        x=p.rest.copy();x[p.n:]=p.prescribed_blade_positions(.001)
        p.prepare_quadrature(x);p.multipliers[:,0]=.05
        _,g,H,_=p.evaluate(x)
        rng=np.random.default_rng(91);d=np.zeros(p.total);d[p.free]=rng.normal(size=len(p.free))
        d=d.reshape(-1,2);d/=np.max(abs(d));h=2e-8
        fp=p.evaluate(x+h*d,tangent=False)[1];fm=p.evaluate(x-h*d,tangent=False)[1]
        fd=(fp-fm)/(2*h);exact=H@d.ravel()
        self.assertLess(np.linalg.norm(fd-exact)/max(1.,np.linalg.norm(exact)),3e-6)
        np.testing.assert_array_equal(p.material_metric,p.previous_material_metric)
        # Full mixed residual: geometry, pressure and Coulomb cross derivatives.
        q=p.contact_data(x);dl=rng.normal(size=p.multipliers.shape)*.01;before=p.multipliers.copy()
        def residual(z,m):
            p.multipliers=m
            return np.r_[p.evaluate(z,tangent=False)[1],p.contact_data(z)['constraint'].ravel()]
        fd=(residual(x+h*d,before+h*dl)-residual(x-h*d,before-h*dl))/(2*h)
        p.multipliers=before
        exact=np.r_[H@d.ravel()+q['force_lambda']@dl.ravel(),
                     q['constraint_u']@d.ravel()+q['constraint_lambda']@dl.ravel()]
        self.assertLess(np.linalg.norm(fd-exact)/max(1.,np.linalg.norm(exact)),3e-6)
        with self.assertRaises(AssertionError):
            p._set_quadrature(p.qedges,p.qfraction,1.1*p.qweights)

    def test_actual_path_fresh_dual_history_and_rejected_step(self):
        p=problem();x=p.rest.copy();old=p.start
        with contextlib.redirect_stdout(io.StringIO()):
            for q in (0.,.005):
                x,record=p.step(x,old,q,time.monotonic()+40);old=q
            state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                previous_positions=p.previous.copy(),depth_mm=old,**p.checkpoint_data())
            fresh=p.audit_state(state,record)
            self.assertTrue(fresh['passed'])
            self.assertLessEqual(fresh['geometry_lipschitz']['penetration_bound_mm'],.0002)
            self.assertLess(record['al_geometry']['penetration_bound_mm'],.0002)
            self.assertGreaterEqual(record['friction_work_n_mm'],-1e-10)
            metric=p.material_metric.copy();history=[(d,z.copy()) for d,z in p._accepted_predictor_history]
            with self.assertRaises(TimeoutError):p.step(x,old,.01,time.monotonic()-1)
            np.testing.assert_array_equal(metric,p.material_metric)
            for a,b in zip(history,p._accepted_predictor_history):
                self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_exact_prefix_resume_matches_uninterrupted_loading(self):
        import argparse
        from tools import run_augmented_blade as runner
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            def run(name,depth,resume=None):
                return runner.run(argparse.Namespace(output=root/name,angle=57.,coarse=True,
                    spacing=.000625,depth=depth,increment=.005,seconds=90.,resume=resume,phase='run'))
            direct=run('direct',.01);prefix=run('prefix',.005)
            resumed=run('resumed',.01,root/'prefix')
            self.assertTrue(direct['completed']);self.assertTrue(prefix['completed']);self.assertTrue(resumed['completed'])
            runner.verify_case(root/'resumed',require_independent=True)
            with np.load(root/'direct/states/0002.npz') as a,np.load(root/'resumed/states/0002.npz') as b:
                for key in ('positions','previous_positions','material_metric','previous_material_metric',
                            'al_multipliers','material_dt_s','cumulative_viscous_dissipation_n_mm'):
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12,err_msg=key)
            p=runner.make_problem(33,coarse=True,spacing=.000625)
            bad=dict(signature=dict(runner.signature(p)),source_sha256=resumed['source_sha256'])
            with self.assertRaisesRegex(ValueError,'exact physical'):
                runner.restore_prefix(p,root/'prefix',root/'resumed',bad)

    def test_subedge_projection_preserves_constant_traction_and_resolves_moments(self):
        from tools.run_al_depth import split_problem
        a=problem(spacing=.000625);b=split_problem(angle=57,coarse=True,spacing=.000625)
        x=a.rest.copy();x[a.n:]=a.prescribed_blade_positions(.001)
        a.prepare_quadrature(x);b._set_quadrature(a.qedges,a.qfraction,a.qweights)
        a.multipliers=np.tile([.8,.2],(len(a.contact_edges),1));b.multipliers=np.tile([.8,.2],(len(b.contact_edges),1))
        self.assertGreater(len(b.contact_edges),len(a.contact_edges))
        np.testing.assert_array_equal(a.Q.toarray(),b.Q.toarray())
        np.testing.assert_allclose(a.contact_data(x)['force'],b.contact_data(x)['force'],rtol=1e-13,atol=1e-13)
        # Subcell grouping must preserve reference area and first moments.
        for i,e in enumerate(a.contact_edges):
            ids=np.flatnonzero(b.contact_edges//2==e)
            self.assertAlmostEqual(float(b.edge_area[ids].sum()),a.edge_area[i],places=12)
            moments=np.asarray(b.average@b.qfraction).ravel()
            self.assertAlmostEqual(float(b.edge_area[ids]@moments[ids]),a.edge_area[i]*.5,places=12)
        # New constraints change the space; they are not claimed identical.
        self.assertNotEqual(b.contact_formulation,a.contact_formulation)

    def test_conditional_probe_audits_fixed_history_and_rejects_timeout(self):
        import argparse
        from tools import run_augmented_blade as runner
        from tools.run_al_depth import probe,observe
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'parent'
            r=runner.run(argparse.Namespace(output=source,angle=57.,coarse=True,
                spacing=.000625,depth=.005,increment=.005,seconds=60.,resume=None,phase='run'))
            self.assertTrue(r['completed'])
            accepted=probe(root/'zero',source,0.,seconds=10.)
            self.assertTrue(accepted['completed'],accepted.get('error'))
            runner.verify_case(root/'zero',require_independent=True)
            with np.load(source/'states/0000.npz') as a,np.load(root/'zero/states/0000.npz') as b:
                for key in ('previous_positions','previous_material_metric','material_dt_s'):
                    np.testing.assert_array_equal(a[key],b[key])
            failed=probe(root/'timed',source,.005,seconds=1e-12)
            self.assertFalse(failed['completed']);self.assertFalse(failed['steps'])
            self.assertNotIn('force_fractions',failed)
            obs=observe(source)
            self.assertEqual(len(obs),2)
            self.assertGreaterEqual(obs[-1]['active_original_edges'],1)


if __name__=='__main__':unittest.main()
