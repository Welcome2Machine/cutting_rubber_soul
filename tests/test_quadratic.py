import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prototype.quadratic import QuadraticProblem,quadratic_minimum
from prototype.initiation import InitiationOptions
from prototype.model_config import FEMConfig


def problem(contact='macro',volume='taylor_hood'):
    return QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.1,
        mesh_grading='graded',contact_discretization=contact,volumetric_discretization=volume))


class QuadraticTests(unittest.TestCase):
    def test_exact_element_minimum_includes_interior_and_edge_extrema(self):
        xy=np.array([[0,0],[1,0],[0,1],[.5,0],[.5,.5],[0,.5]])
        values=np.array([(xy[:,0]-.2)**2+(xy[:,1]-.3)**2-.01,
            (xy[:,0]-.4)**2+(xy[:,1]+.1)**2,
            1-xy[:,0]-xy[:,1]])
        np.testing.assert_allclose(quadratic_minimum(values),[-.01,.01,0],atol=1e-15)

    def test_energy_residual_and_pressure_schur_tangent(self):
        p=problem();rng=np.random.default_rng(47)
        u=rng.normal(size=p.mesh.dof_count)*1e-4;v=rng.normal(size=len(u));v/=np.linalg.norm(v)
        lam=p.initial_contact_state();eps=1e-6
        def evaluate(x):return p.assemble(x,u,-2.,-2.,.1,lam[:,0],lam[:,1])
        r,k,q,d=evaluate(u);rp,_,_,dp=evaluate(u+eps*v);rm,_,_,dm=evaluate(u-eps*v)
        H=q['pressure_coupling'];action=k@v+H.T@p.mass_factor.solve(H@v)
        np.testing.assert_allclose(action,(rp-rm)/(2*eps),rtol=2e-6,atol=2e-7)
        self.assertAlmostEqual(r@v,(dp['bulk_energy_n_mm']-dm['bulk_energy_n_mm'])/(2*eps),places=7)

    def test_full_integration_energy_and_tangent_use_actual_material_response(self):
        from prototype.material import response
        p=problem('quadratic_integral','quadratic_full');rng=np.random.default_rng(14)
        u=rng.normal(size=p.mesh.dof_count)*1e-4;v=rng.normal(size=len(u));v/=np.linalg.norm(v)
        lam=p.initial_contact_state();eps=1e-6
        r,k,q,d=p.assemble(u,u,-2.,-2.,.1,lam[:,0],lam[:,1])
        a=p.assemble(u+eps*v,u,-2.,-2.,.1,lam[:,0],lam[:,1],with_tangent=False)
        b=p.assemble(u-eps*v,u,-2.,-2.,.1,lam[:,0],lam[:,1],with_tangent=False)
        np.testing.assert_allclose(k@v,(a[0]-b[0])/(2*eps),rtol=2e-6,atol=2e-7)
        self.assertAlmostEqual(r@v,(a[3]['bulk_energy_n_mm']-b[3]['bulk_energy_n_mm'])/(2*eps),places=7)
        self.assertNotIn('pressure_coupling',q)
        fields=p.fields(u);raw=response(fields['deformation_gradient_xzy'][:,:2,:2],p.config,tangent=False)
        np.testing.assert_array_equal(fields['cauchy_stress_xzy_mpa'],raw['cauchy'])

    def test_quadratic_history_transfer_and_checkpoint(self):
        from prototype.contact_study import save_state,load_failed_problem
        from run_support import write_json
        p=problem()
        def field(x):return np.column_stack((.002*x[:,0]**2+.001*x[:,0]*x[:,1],-.003*x[:,1]**2+.002*x[:,0]))
        u=field(p.mesh.reference_xy).ravel();state=p.initial_contact_state();state[:]=[2.,.3]
        resultant=p.weights@state;n=p.mesh.pressure_node_count
        centers=p.mesh.local_sn[p.edges].mean(axis=1)
        index=np.linalg.norm(centers-[p.config.drop_s_mm,p.config.height_mm],axis=1).argmin()
        u,state=p.refine_contact(u,state,[index])
        self.assertGreater(p.mesh.pressure_node_count,n)
        np.testing.assert_allclose(u.reshape(-1,2),field(p.mesh.reference_xy),atol=1e-13)
        np.testing.assert_allclose(p.weights@state,resultant,atol=1e-11)
        p._accepted_velocity=u/.1;p._velocity_depth=.1;p._velocity_loading=True
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);save_state(out/'last_state.npz',p,u,state,.1)
            write_json(out/'result.json',dict(config=p.config.to_dict(),options=vars(p.options)))
            restored,v,w,depth=load_failed_problem(out)
            self.assertIsInstance(restored,QuadraticProblem)
            np.testing.assert_array_equal(u,v);np.testing.assert_array_equal(state,w)
            np.testing.assert_allclose(p.fields(u)['J'],restored.fields(v)['J'])
            np.testing.assert_array_equal(p._accepted_velocity,restored._accepted_velocity)
            self.assertEqual(restored._velocity_depth,.1)
            self.assertTrue(restored._velocity_loading)
            from prototype.loading import predictor_checkpoint,restore_predictor
            self.assertEqual(predictor_checkpoint(p,.09),{})
            invalid=predictor_checkpoint(p,.1);invalid['predictor_depth_mm']=.09
            with self.assertRaisesRegex(ValueError,'predictor checkpoint'):
                restore_predictor(restored,invalid,.1)

    def test_nodal_contact_weights_work_and_refinement(self):
        p=problem('quadratic_nodes');u=np.zeros(p.mesh.dof_count);lam=p.initial_contact_state();lam[:]=[3.,1.05]
        def plane(x,c,o,depth,**kwargs):
            return x[:,1]*0,np.tile([0.,-1.],(len(x),1)),np.zeros((len(x),2,2))
        moved=np.tile([.01,0.],p.mesh.node_count)
        with patch('prototype.initiation.rounded_blade_distance',plane):
            q=p.surface_contact.evaluate(moved,u,0.,0.,lam,with_tangent=True,formulation=None,roundoff_guard=False)
        self.assertGreater(p.weights.min(),0.)
        self.assertAlmostEqual(p.weights.sum(),p.config.width_mm*(p.config.length_mm+2*p.config.height_mm))
        self.assertAlmostEqual(-q['global_force']@moved,q['dissipation'].sum())
        resultant=p.weights@lam
        u,lam=p.refine_contact(u,lam,[len(p.edges)//2])
        np.testing.assert_allclose(p.weights@lam,resultant,atol=1e-11)
        np.testing.assert_allclose(lam,np.tile([3.,1.05],(len(lam),1)))

    def test_curve_audit_detects_penetration_between_admissible_nodes(self):
        p=problem('quadratic_nodes');u=np.zeros((p.mesh.node_count,2));edge=0
        nodes=np.r_[p.edges[edge],p.mesh.boundary_midpoints[edge]]
        t=np.array([0.,1.,.5]);gap=.00010001-.005*(t-.23)**2
        self.assertLess(gap.max(),0.)
        u[nodes,1]=1+gap-p.mesh.reference_xy[nodes,1]
        def plane(x,c,o,depth,**kwargs):return x[:,1]-1,np.tile([0.,-1.],(len(x),1))
        with patch('prototype.initiation.rounded_blade_distance',plane):
            maximum,edges=p.edge_penetration(u.ravel(),0.,with_edges=True)
        self.assertIn(edge,edges)
        self.assertGreater(maximum,p.config.contact_penetration_tolerance_mm)

    def test_finite_compression_contact(self):
        self.check_finite_compression('macro')

    def test_finite_compression_nodal_contact(self):
        self.check_finite_compression('quadratic_nodes')

    def test_finite_compression_integral_contact(self):
        self.check_finite_compression('quadratic_integral')

    def test_finite_compression_full_integration_contact(self):
        self.check_finite_compression('quadratic_integral','quadratic_full',branch_probe=True)

    def test_finite_compression_full_integration_nodal_contact(self):
        self.check_finite_compression('quadratic_nodes','quadratic_full')

    def test_bulk_refinement_preserves_displacement_and_reduces_projection_l2_error(self):
        p=problem('quadratic_integral');x=p.mesh.reference_xy
        u=np.column_stack((.002*x[:,0]**2+.001*x[:,0]*x[:,1],.003*x[:,1]**2)).ravel()
        lam=p.initial_contact_state();lam[:]=[2.,.3]
        q=p.bulk_response(u);error=p.config.bulk_modulus_n_mm2*(q['J']-1)-q['pressure']
        from prototype.quadratic import W
        before=np.einsum('e,q,eq->',p.volumes,W,error**2)
        selected=np.argsort(np.max(abs(error),axis=1))[-40:]
        u,lam=p.refine_bulk(u,lam,selected)
        x=p.mesh.reference_xy
        np.testing.assert_allclose(u.reshape(-1,2),np.column_stack((.002*x[:,0]**2+.001*x[:,0]*x[:,1],.003*x[:,1]**2)),atol=1e-13)
        q=p.bulk_response(u);error=p.config.bulk_modulus_n_mm2*(q['J']-1)-q['pressure']
        after=np.einsum('e,q,eq->',p.volumes,W,error**2)
        self.assertLess(after,before)

    def test_nested_predictor_is_separate_from_physical_slip_reference(self):
        from prototype.initiation import ContactResolutionError,refine_resolution
        p=problem('quadratic_integral');old=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        def field(x):return np.column_stack((.002*x[:,0]**2,-.001*x[:,1]**2)).ravel()
        trial=field(p.mesh.reference_xy);lam=state+np.array([2.,.3])
        error=ContactResolutionError('verification',np.arange(5),u=trial,multipliers=lam,depth=.01)
        previous,previous_lam=refine_resolution(p,old,state,error)
        reference,seed,warm,depth=p._spatial_predictor
        np.testing.assert_array_equal(previous,0.)
        np.testing.assert_array_equal(previous_lam,0.)
        np.testing.assert_array_equal(reference,previous)
        np.testing.assert_allclose(seed,field(p.mesh.reference_xy),atol=1e-13)
        np.testing.assert_allclose(warm,np.tile([2.,.3],(len(warm),1)))
        self.assertEqual(depth,.01)

    def check_finite_compression(self,contact,volume='taylor_hood',branch_probe=False):
        from prototype.material import response
        c=FEMConfig(length_mm=1,height_mm=1,width_mm=1,drop_s_mm=.5,inclination_deg=0,mesh_size_mm=.25,max_indentation_mm=.3,friction_coefficient=0.,newton_tolerance_n=1e-8,contact_multiplier_tolerance_n_mm2=1e-8)
        p=QuadraticProblem(c,InitiationOptions(refinement_mm=.25,surface_spacing_mm=.125,mesh_grading='graded',contact_discretization=contact,volumetric_discretization=volume))
        select=np.all(np.isclose(p.mesh.reference_xy[p.edges,1],1),axis=1)
        p.mesh.boundary_midpoints=p.mesh.boundary_midpoints[select];p.mesh.macro_boundary_edges=p.edges[select]
        p._rebuild_boundary(p.edges[select])
        p.fixed=np.unique(np.r_[2*np.arange(p.mesh.node_count),2*p.mesh.bottom_nodes+1]);p.free=np.setdiff1d(np.arange(p.mesh.dof_count),p.fixed)
        def plane(x,c,o,depth,with_tangent=False):
            result=(x[:,1]-1+depth,np.tile([0.,-1.],(len(x),1)))
            return result+(np.zeros((len(x),2,2)),) if with_tangent else result
        u=np.zeros(p.mesh.dof_count);lam=p.initial_contact_state();old=0.
        with patch('prototype.initiation.rounded_blade_distance',plane),contextlib.redirect_stdout(io.StringIO()):
            for depth in (.1,.2,.3):
                u,lam,d=p.step(u,old,depth,.1,lam)
                old=depth
                expected=-response(np.array([[[1.,0.],[0.,1-depth]]]),c,tangent=False)['P'][0,1,1]
                self.assertAlmostEqual(d['reaction_fz_n'],expected,places=9)
                np.testing.assert_allclose(p.fields(u)['J'],1-depth,atol=1e-10)
                np.testing.assert_allclose(u.reshape(-1,2)[:,1],-depth*p.mesh.reference_xy[:,1],atol=1e-10)
            if branch_probe:
                from dataclasses import replace
                from prototype.contact_branches import integral_branch_predictors
                p.config=replace(p.config,newton_tolerance_n=.001)
                predicted=list(integral_branch_predictors(p,u,old,old,.1,u,lam+np.array([.005,0.]),max_predictors=1))
                self.assertEqual(len(predicted),1)
                seed,warm,_=predicted[0]
                residual=p.assemble(seed,u,old,old,.1,warm[:,0],warm[:,1],with_tangent=False)[0]
                self.assertLess(np.linalg.norm(residual[p.free]),1e-8)
                np.testing.assert_allclose(seed,u,atol=1e-10)
                perturbed=u.copy();perturbed[2*p.contact_nodes+1]+=1e-5*np.cos(np.arange(len(p.contact_nodes)))
                seed,warm=p.feasible_contact_predictor(u,old,old,perturbed,lam)
                np.testing.assert_allclose(seed,u,atol=1e-10)
                np.testing.assert_allclose(warm,lam,atol=1e-10)


if __name__=='__main__':unittest.main()
