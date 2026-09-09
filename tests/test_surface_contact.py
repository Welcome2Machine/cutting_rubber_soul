import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from prototype.model_config import FEMConfig
from prototype.initiation import InitiationOptions, IntactProblem
from prototype.contact_study import mesh_metrics
from prototype.surface_contact import SurfaceContact


class SurfaceContactTests(unittest.TestCase):
    def flat_boundary(self):
        p=SimpleNamespace(config=FEMConfig(width_mm=2.),options=InitiationOptions(contact_discretization='integral'),
            contact_nodes=np.arange(3),edges=np.array([[0,1],[1,2]]),weights=np.array([1.,2.,1.]),
            cdofs=np.arange(6).reshape(-1,2),mesh=SimpleNamespace(dof_count=6,reference_xy=np.array([[0.,0.],[1.,0.],[2.,0.]])))
        return p,SurfaceContact(p)

    @staticmethod
    def plane(points,config,options,depth,**kwargs):
        return -points[:,1]-depth,np.tile([0.,1.],(len(points),1)),np.zeros((len(points),2,2))

    def test_flat_patch_force_and_affine_gap(self):
        p,s=self.flat_boundary()
        u=np.zeros(6);lam=np.tile([3.,-.5],(3,1))
        with patch('prototype.initiation.rounded_blade_distance',self.plane):
            q=s.evaluate(u,u,0.,0.,lam,with_tangent=True,formulation=None,roundoff_guard=False)
        np.testing.assert_allclose(q['force'].sum(axis=0),[-2.,12.],atol=1e-13)
        np.testing.assert_allclose(q['gap'],0.,atol=1e-14)
        np.testing.assert_allclose(q['pressure'],lam[:,0],atol=1e-14)
        # A linear normal displacement integrates against P1 test functions;
        # endpoint averages must differ from collocation at endpoints.
        u[1::2]=[0.,1.,2.]
        with patch('prototype.initiation.rounded_blade_distance',self.plane):
            q=s.evaluate(u,u,0.,0.,lam,with_tangent=False,formulation=None,roundoff_guard=False)
        np.testing.assert_allclose(q['gap'],[-1/3,-1.,-5/3],atol=1e-13)

    def test_friction_work_and_rigid_translation(self):
        p,s=self.flat_boundary();old=np.zeros(6);u=np.tile([.01,0.],3);lam=np.tile([3.,-1.05],(3,1))
        with patch('prototype.initiation.rounded_blade_distance',self.plane):
            q=s.evaluate(u,old,0.,0.,lam,with_tangent=False,formulation=None,roundoff_guard=False)
            moved=s.evaluate(u+np.tile([0.,-.1],3),old,.1,0.,lam,with_tangent=False,formulation=None,roundoff_guard=False)
        self.assertAlmostEqual(q['dissipation'].sum(),.042)
        self.assertAlmostEqual(-np.sum(q['force']*u.reshape(-1,2)),q['dissipation'].sum())
        np.testing.assert_allclose(q['gap'],moved['gap'],atol=1e-14)
        np.testing.assert_allclose(q['slip'],moved['slip'],atol=1e-14)

    def test_reduced_multiplier_force_and_work_use_all_displacement_nodes(self):
        p,_=self.flat_boundary()
        p.options=InitiationOptions(contact_discretization='macro',volumetric_discretization='macro_mixed')
        p.mesh.macro_boundary_edges=np.array([[0,2]])
        p.contact_nodes=np.array([0,2]);p.cdofs=np.array([[0,1],[4,5]]);p.weights=np.array([2.,2.])
        s=SurfaceContact(p);old=np.zeros(6);u=np.array([.01,0.,.03,0.,.02,0.]);lam=np.tile([3.,-1.05],(2,1))
        with patch('prototype.initiation.rounded_blade_distance',self.plane):
            q=s.evaluate(u,old,0.,0.,lam,with_tangent=True,formulation=None,roundoff_guard=False)
        np.testing.assert_allclose(q['force'].sum(axis=0),[-4.2,12.],atol=1e-12)
        self.assertGreater(np.linalg.norm(q['global_force'][2:4]),0.)
        self.assertAlmostEqual(-q['global_force']@u,q['dissipation'].sum(),places=12)

    def test_curved_contact_full_assembly_directional_derivatives(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.02,contact_discretization='integral'))
        self.check_curved_derivatives(p)

    def test_macro_curved_contact_directional_derivatives(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.02,mesh_grading='graded',contact_discretization='macro',volumetric_discretization='macro_mixed'))
        self.assertGreater(len(p.surface_contact.displacement_nodes),len(p.contact_nodes))
        self.check_curved_derivatives(p)

    def test_quadratic_curved_contact_directional_derivatives(self):
        from prototype.quadratic import QuadraticProblem
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.02,mesh_grading='graded',contact_discretization='macro',volumetric_discretization='taylor_hood'))
        self.check_curved_derivatives(p)

    def test_quadratic_nodal_contact_directional_derivatives(self):
        from prototype.quadratic import QuadraticProblem
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.02,mesh_grading='graded',contact_discretization='quadratic_nodes',volumetric_discretization='taylor_hood'))
        self.check_curved_derivatives(p)

    def test_quadratic_integral_contact_directional_derivatives(self):
        from prototype.quadratic import QuadraticProblem
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.02,mesh_grading='graded',contact_discretization='quadratic_integral',volumetric_discretization='taylor_hood'))
        self.check_curved_derivatives(p)

    def check_curved_derivatives(self,p):
        rng=np.random.default_rng(9);u=np.zeros(p.mesh.dof_count);previous=u.copy()
        lam=p.initial_contact_state();lam[:,0]=.3;lam[:,1]=.02
        du=rng.normal(size=len(u));du*=.01/np.linalg.norm(du)
        dl=rng.normal(size=lam.shape)*.02
        for formulation in ('alart_curnier','jean_moreau'):
            def evaluate(v,l,tangent):
                r,k,q,_=p.assemble(v,previous,.0173,0,.1,l[:,0],l[:,1],mixed=True,
                    with_tangent=tangent,formulation=formulation,roundoff_guard=False)
                return r,k,q,np.column_stack((q['pressure'],q['shear']))-l
            r,k,q,f=evaluate(u,lam,True)
            h=1e-6
            a=evaluate(u+h*du,lam+h*dl,False);b=evaluate(u-h*du,lam-h*dl,False)
            predicted=k@du+q['global_B']@dl.ravel()
            if 'pressure_mass_matrix' in q:
                H=q['pressure_coupling'];predicted+=H.T@p.mass_factor.solve(H@du)
            np.testing.assert_allclose((a[0]-b[0])/(2*h),predicted,rtol=2e-5,atol=2e-5)
            predicted=q['global_C']@du+q['global_D']@dl.ravel()
            np.testing.assert_allclose(((a[3]-b[3])/(2*h)).ravel(),predicted,rtol=2e-5,atol=2e-5)

    def test_graded_refinement_preserves_field_area_and_conformity(self):
        p=IntactProblem(FEMConfig(),InitiationOptions(mesh_grading='graded',contact_discretization='integral'))
        self.assertGreater(mesh_metrics(p)['min_angle_deg'],20.)
        gradient=np.array([[.02,.01],[-.01,.03]])
        u=(p.mesh.reference_xy@gradient).ravel()
        state=p.initial_contact_state();state[:,0]=2.;state[:,1]=.3
        resultant=p.weights@state
        for _ in range(2):
            centers=p.mesh.local_sn[p.edges].mean(axis=1)
            edge=np.linalg.norm(centers-[p.config.drop_s_mm,p.config.height_mm],axis=1).argmin()
            u,state=p.refine_contact(u,state,[edge])
            np.testing.assert_allclose(u.reshape(-1,2),p.mesh.reference_xy@gradient,atol=1e-13)
            np.testing.assert_allclose(p.weights@state,resultant,atol=1e-12)
            self.assertAlmostEqual(p.mesh.bulk_data[0].sum(),150.)
            self.assertGreater(mesh_metrics(p)['min_angle_deg'],10.)
            tri=p.mesh.triangles
            edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
            unique,count=np.unique(edges,axis=0,return_counts=True)
            self.assertTrue(np.all(count<=2))
            boundary=unique[count==1]
            boundary=boundary[~np.all(np.isclose(p.mesh.local_sn[boundary,1],0),axis=1)]
            self.assertEqual(set(map(tuple,boundary)),set(map(tuple,np.sort(p.edges,axis=1))))



if __name__=='__main__':unittest.main()
