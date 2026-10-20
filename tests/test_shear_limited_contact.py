"""Current-area pressure, complete fresh tangent and native-limit checks."""
import contextlib
import io
import time
import unittest
import csv
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.shear_limited_contact import ShearLimitedContact, limited_mu


class Candidate(ShearLimitedContact, ViscoelasticProblem):
    pass


def patch_problem(spacing=.2):
    p=Candidate(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
        shear_modulus=.14,bulk_modulus=23.34333333333331,
        case='patch',mu=.35,local_spacing=spacing,surface_spacing=spacing,dhat=.0025,
        max_iterations=500,predictor='affine')
    p.enable_shear_limit(1.)
    return p


def sliding_field(p):
    x=p.rest.copy();x[:p.n,1]*=.97
    x[p.n:,1]-=x[p.n:,1].min()-.9713
    x[p.n:,0]+=.0174
    p.previous=x.copy();p.previous[:p.n,0]-=.0004
    return x


class ShearLimitTests(unittest.TestCase):
    def test_queue_review_never_promotes_an_incomplete_case(self):
        from tools.check_shear_limit import review_queue
        from tools.run_material_candidate import load_force_csv
        # Real format reader, synthetic accepted-state fixture; no FE claim.
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);candidate=out/'candidate';base=out/'base'
            for folder in (candidate,base):(folder/'states').mkdir(parents=True)
            signature=dict(config=dict(inclination_deg=57),material=dict(shared=True))
            steps=[dict(depth_mm=.25,reaction_fz_n=4.6,min_j=.8,tip_contact=dict(status='absent'),interface=dict())]
            br=dict(signature=signature,steps=steps)
            cr=dict(signature=dict(signature,interface_law=dict(name='nodal-current-area-smooth-shear-limit',
                strength_mpa=1.,smooth_min_power=4,area='current rubber boundary dual area')),steps=steps)
            for folder,record in ((candidate,cr),(base,br)):
                (folder/'result.json').write_text(json.dumps(record),encoding='utf-8')
                np.savez(folder/'states/0000.npz',reference_xy=np.array([[.5,1.]]),positions=np.array([[.5,.9]]))
            image=out/'image.csv';image.write_text('depth_mm,force_n,lower_n,upper_n\n0.25,4,3,5\n',encoding='utf-8')
            manifest=dict(experiment=str(root/'docs/1014n57a1__Fz__curve_0.csv'),image_estimates=str(image),shear_limit_mpa=1.)
            state=dict(jobs=[dict(angle=57,output=str(candidate),baseline=str(base),mechanics_completed=False)])
            fake=SimpleNamespace(mesh=SimpleNamespace(local_sn=np.array([[.5,1.]])),config=SimpleNamespace(height_mm=1.,drop_s_mm=.5),n=1)
            with patch('tools.check_shear_limit.make_problem',return_value=fake):r=review_queue(out,manifest,state)
            self.assertEqual(r['status'],'partial_or_unavailable')
            self.assertFalse(r['force_screen_passed'])
            self.assertFalse(r['experimental_validation'])
            self.assertEqual(r['rows'][0]['tip_status'],'absent')

    def test_cap_limits_derivative_and_units(self):
        pressure=np.r_[0.,np.geomspace(1e-5,1e5,60)];mu=.35;strength=1.
        m,e=limited_mu(pressure,mu,strength)
        self.assertEqual(m[0],mu);self.assertEqual(e[0],0.)
        self.assertTrue(np.all(m*pressure<=strength+1e-14))
        self.assertAlmostEqual((m*pressure)[-1],strength,places=12)
        h=1e-5;plus=limited_mu(pressure[1:]*np.exp(h),mu,strength)[0]
        minus=limited_mu(pressure[1:]*np.exp(-h),mu,strength)[0]
        np.testing.assert_allclose((plus-minus)/(2*h),-m[1:]*e[1:],rtol=2e-5,atol=1e-10)

    def test_infinite_limit_recovers_native_and_normal_contact_is_unchanged(self):
        p=patch_problem();x=sliding_field(p)
        p.shear_limit_mpa=1e12
        limited=p.lag(x);native=IPCProblem.lag(p,x)
        delta=(x-p.previous)[p.surface]
        np.testing.assert_allclose(p.friction.gradient(limited,p.cmesh,delta),
            p.friction.gradient(native,p.cmesh,delta),rtol=1e-13,atol=1e-13)
        normal=p.barrier.gradient(p.collisions(x),p.cmesh,x[p.surface])
        p.shear_limit_mpa=1.
        p.lag(x)
        np.testing.assert_array_equal(normal,p.barrier.gradient(p.collisions(x),p.cmesh,x[p.surface]))

    def test_pressure_area_derivative_and_rotation_covariance(self):
        p=patch_problem();x=sliding_field(p)
        field=p.interface_field(x,derivative=True)
        rng=np.random.default_rng(42);v=rng.normal(size=x.shape)*.1;h=2e-7
        plus=p.interface_field(x+h*v);minus=p.interface_field(x-h*v)
        actual=(plus['mu']-minus['mu'])/(2*h)
        expected=field['mu_jacobian']@v[p.surface].ravel()
        np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=1e-7)
        angle=.51;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotated=p.interface_field(x@Q.T)
        np.testing.assert_allclose(rotated['mu'],field['mu'],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(rotated['current_area_mm2'],field['current_area_mm2'],atol=1e-12)

    def test_full_fresh_friction_jacobian_includes_pressure_and_interpolation(self):
        p=patch_problem();x=sliding_field(p);lag=p.lag(x)
        local=p.friction.hessian(lag,p.cmesh,(x-p.previous)[p.surface],p.ipc.PSDProjectionMethod.NONE).tocoo()
        H=coo_matrix((local.data,(p.surface_dofs[local.row],p.surface_dofs[local.col])),shape=(p.total,p.total)).tocsc()
        J=p.coupled_friction_tangent(x,lag,H)
        def gradient(value):
            g=np.zeros(p.total)
            g[p.surface_dofs]=p.friction.gradient(p.lag(value),p.cmesh,(value-p.previous)[p.surface])
            return g
        rng=np.random.default_rng(821)
        for h in (4e-7,1e-7):
            v=rng.normal(size=x.shape)*.05
            actual=(gradient(x+h*v)-gradient(x-h*v))/(2*h)
            expected=J@v.ravel()
            self.assertLess(np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(actual)),3e-5)
        self.assertGreaterEqual(float(gradient(x)@(x-p.previous).ravel()),0.)

    def test_uniform_flat_contact_traction_is_mesh_consistent(self):
        # Interior nodes of a flat contact: dual current areas give the same
        # pressure despite tangential boundary subdivision. Endpoints excluded.
        means=[]
        for spacing in (.2,.1):
            p=patch_problem(spacing);x=sliding_field(p);f=p.interface_field(x)
            ids=p.surface[:p.interface_body_count]
            inside=(abs(p.rest[ids,1]-1)<1e-10)&(p.rest[ids,0]>.21)&(p.rest[ids,0]<.79)
            self.assertTrue(inside.any())
            means.append(float(np.mean(f['pressure_mpa'][inside])))
        np.testing.assert_allclose(means[0],means[1],rtol=1e-10,atol=1e-10)

    def test_actual_patch_loading_accepts_original_force_and_material_history(self):
        p=patch_problem();x=p.rest.copy();old=p.start
        with threadpool_limits(2),contextlib.redirect_stdout(io.StringIO()):
            for d in (0.,.01,.02):
                x,r=p.step(x,old,d,time.monotonic()+30)
                self.assertLessEqual(r['residual_n'],1e-4)
                self.assertGreater(r['min_j'],0)
                self.assertGreaterEqual(r['interface']['friction_increment_work_n_mm'],-1e-10)
                old=d
        from prototype.shear_limited_contact import export_interface_history
        from run_support import save_npz
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'states').mkdir()
            save_npz(out/'states/0000.npz',positions=x,previous_positions=p.previous)
            export_interface_history(p,out,dict(steps=[r]))
            with (out/'interface_motion.csv').open(newline='',encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(float(row['depth_mm']),.02)
                self.assertGreater(float(row['current_dual_area_mm2']),0)
                if row['blade_contact_supported']=='0':
                    self.assertEqual(row['relative_tangent_increment_mm'],'')


if __name__=='__main__':unittest.main()
