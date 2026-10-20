"""Qualification of the explicit analytic-blade candidate, not its defaults."""
import unittest
import numpy as np
from prototype.curve_contact import CurveBladeProblem,physical_barrier,point_contact
from prototype.initiation import rounded_blade_distance,_blade_boundary_vertices
from prototype.model_config import local_to_lab,_rotation_vectors


class CurveContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=CurveBladeProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
            bulk_modulus=23.34333333333331,shear_modulus=.14,local_spacing=.2,
            surface_spacing=.02,mu=.35,tip_radius=.02,dhat=.0025,inclination=60)
        cls.p.material_dt=.05

    def point(self,xy,old,mu=.35):
        p=self.p
        pen,n,dn=rounded_blade_distance(np.atleast_2d(xy),p.config,p.options,p.start,with_tangent=True)
        return point_contact(-pen,n,dn,np.atleast_2d(xy-old),np.ones(len(n)),p.dhat,
                             p.contact_stiffness,mu,1e-5)

    def center(self):
        p=self.p;_,en=_rotation_vectors(p.config.inclination_deg)
        return local_to_lab(np.array([[p.config.drop_s_mm,p.config.height_mm]]),p.config)[0]+p.options.tip_radius_mm*en-[0,p.start]

    def test_native_physical_barrier_scaling_and_derivatives(self):
        p=self.p;d=np.array([.0002,.001,.002,.003]);D=p.dhat**2
        b,bp,bpp=physical_barrier(d,p.dhat,p.contact_stiffness);factor=p.contact_stiffness/p.dhat**3
        expected=np.array([p.ipc.barrier(x*x,D) for x in d])*factor
        np.testing.assert_allclose(b,expected,rtol=1e-14)
        h=1e-9;plus=physical_barrier(d+h,p.dhat,p.contact_stiffness);minus=physical_barrier(d-h,p.dhat,p.contact_stiffness)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*h),bp,rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*h),bpp,rtol=1e-8,atol=1e-5)
        with self.assertRaises(FloatingPointError):physical_barrier(np.array([0.]),p.dhat,1.)

    def test_fresh_friction_tangent_stick_slip_and_normal_coupling(self):
        c=self.center();r=self.p.options.tip_radius_mm;h=1e-9
        for theta in (-2.4,-1.57,-.7):
            n=np.array([np.cos(theta),np.sin(theta)]);t=np.array([-n[1],n[0]])
            x=c+(r+.001)*n
            for slip in (-2e-5,-3e-6,0,3e-6,2e-5):
                old=x-slip*t-.0001*n;q=self.point(x,old)
                numerical=np.column_stack([(self.point(x+h*e,old)['gradient'][0]-self.point(x-h*e,old)['gradient'][0])/(2*h) for e in np.eye(2)])
                np.testing.assert_allclose(q['jacobian'][0],numerical,rtol=1e-4,atol=.1)
                self.assertGreaterEqual(q['friction_work'],0.)
                self.assertGreaterEqual(np.linalg.eigvalsh(q['metric'][0]).min(),-1e-8)

    def test_removed_polygon_nose_switch_is_smooth(self):
        p=self.p;c=self.center();a=np.deg2rad(p.config.blade_tip_angle_deg/2)
        theta=np.linspace(-np.pi+a,-a,65)[25];r=p.options.tip_radius_mm+.001
        x=c+r*np.array([np.cos(theta),np.sin(theta)]);old=x-[.0001,.0002]
        q=self.point(x,old)
        for sign in (-1,1):
            y=c+r*np.array([np.cos(theta+sign*1e-7),np.sin(theta+sign*1e-7)])
            other=self.point(y,old)
            np.testing.assert_allclose(other['jacobian'],q['jacobian'],rtol=3e-5,atol=.1)
            np.testing.assert_allclose(other['gradient'],q['gradient'],rtol=3e-5,atol=.001)

    def test_circle_to_bevel_force_continuity_not_c2_claim(self):
        p=self.p;c=self.center();a=np.deg2rad(p.config.blade_tip_angle_deg/2)
        n=np.array([np.cos(-a),np.sin(-a)]);t=np.array([-n[1],n[0]])
        arc=c+p.options.tip_radius_mm*n;old=arc+.001*n-[.0001,.0002]
        left=c+(p.options.tip_radius_mm+.001)*np.array([np.cos(-a-1e-8),np.sin(-a-1e-8)])
        right=arc+.001*n+1e-9*t
        np.testing.assert_allclose(self.point(left,old)['gradient'],self.point(right,old)['gradient'],rtol=1e-5,atol=1e-4)

    def test_reference_quadrature_conservation_and_zero_prestress(self):
        p=self.p
        perimeter=np.linalg.norm(np.diff(p.rest[p.body_edges],axis=1)[:,0],axis=1).sum()
        self.assertAlmostEqual(p.weights.sum(),perimeter,places=10)
        np.testing.assert_allclose(p.D@np.tile([.73,-.4],len(p.rest)),0,atol=1e-15)
        self.assertEqual(p.acceptance(p.rest,p.start,0,0)['residual_n'],0.)

    def test_native_material_self_contact_is_retained(self):
        p=self.p;x=p.rest.copy();edge=p.body_edges[0]
        point=next(v for v in p.body_surface if v not in edge and
                   all(v not in e for e in p.body_edges if any(k in e for k in edge)))
        a,b=x[edge];t=b-a;n=np.array([-t[1],t[0]])/np.linalg.norm(t)
        x[point]=(a+b)/2+.001*n
        c=p._self_collisions(x)
        self.assertGreater(p.barrier(c,p.self_mesh,x[p.body_surface]),0.)

    def test_circumscribed_envelope_and_whole_edge_ccd(self):
        p=self.p;c=self.center();a=np.deg2rad(p.config.blade_tip_angle_deg/2);r=p.options.tip_radius_mm
        theta=np.linspace(-np.pi+a,-a,1025);arc=c+r*np.column_stack((np.cos(theta),np.sin(theta)))
        v=p.guard_rest;edges=np.roll(v,-1,axis=0)-v
        # Polygon is counterclockwise: exact arc lies on/in every supporting line.
        cross=edges[None,:,0]*(arc[:,None,1]-v[None,:,1])-edges[None,:,1]*(arc[:,None,0]-v[None,:,0])
        self.assertGreaterEqual(cross.min(),-1e-12)
        # A long material edge can cross the circular nose between sample points.
        body=np.array([c+[-.04,-.021],c+[.04,-.021]])
        end=body+[0,.004];vertices=np.vstack([body,v]);target=np.vstack([end,v])
        boundary=np.column_stack((np.arange(len(v)),np.roll(np.arange(len(v)),-1)))+2
        mesh=p.ipc.CollisionMesh(vertices,np.vstack([[[0,1]],boundary]));mesh.can_collide=p.ipc.make_static_obstacle_filter(2)
        self.assertFalse(p.ipc.has_intersections(mesh,vertices))
        self.assertTrue(p.ipc.has_intersections(mesh,target))
        safe=p.ipc.compute_collision_free_stepsize(mesh,vertices,target)
        self.assertGreater(safe,0);self.assertLess(safe,1)
        self.assertFalse(p.ipc.has_intersections(mesh,vertices+safe*(target-vertices)))

        # Collision-free endpoints do not imply a collision-free motion path.
        body=np.array([c+[-.03,-.01],c+[-.03,-.009]])
        vertices=np.vstack([body,v]);target=np.vstack([body+[.06,0],v])
        mesh=p.ipc.CollisionMesh(vertices,np.vstack([[[0,1]],boundary]));mesh.can_collide=p.ipc.make_static_obstacle_filter(2)
        self.assertFalse(p.ipc.has_intersections(mesh,vertices))
        self.assertFalse(p.ipc.has_intersections(mesh,target))
        safe=p.ipc.compute_collision_free_stepsize(mesh,vertices,target)
        self.assertGreater(safe,0);self.assertLess(safe,1)
        self.assertFalse(p.ipc.has_intersections(mesh,vertices+safe*(target-vertices)))

if __name__=='__main__':unittest.main()
