import unittest
import tempfile
import json
import contextlib
import io
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prototype.model_config import FEMConfig
from prototype.initiation import IntactProblem, InitiationOptions, rounded_blade_distance, intact_mesh
from prototype.model_config import local_to_lab, _rotation_vectors
from prototype.run_initiation import run


class InitiationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = FEMConfig(mesh_size_mm=1.5)
        cls.options = InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2)
        cls.problem = IntactProblem(cls.config,cls.options)

    def test_intact_mesh_area_and_no_internal_boundary(self):
        p = self.problem
        self.assertEqual(len(p.mesh.seam_pairs),0)
        self.assertAlmostEqual(p.mesh.bulk_data[0].sum(),150)
        self.assertEqual(len(np.unique(p.mesh.reference_xy,axis=0)),p.mesh.node_count)
        sn = p.mesh.local_sn[p.edges]
        on_edge = (np.all(np.isclose(sn[:,:,0],0),axis=1) |
                   np.all(np.isclose(sn[:,:,0],30),axis=1) |
                   np.all(np.isclose(sn[:,:,1],5),axis=1))
        self.assertTrue(on_edge.all())

    def test_initial_contact_stress_free(self):
        p = self.problem
        u = np.zeros(p.mesh.dof_count)
        r,k,contact,d = p.assemble(u,u,0,0,.1,np.zeros(len(p.weights)))
        self.assertLess(np.linalg.norm(r),1e-9)
        self.assertLess(d['max_penetration_mm'],1e-10)

    def test_patch_interface_has_one_grid_without_almost_coincident_vertices(self):
        config=FEMConfig.from_json(Path(__file__).resolve().parents[1]/'prototype/initiation_config.json')
        mesh,_=intact_mesh(config,InitiationOptions())
        sn=mesh.local_sn
        row=sn[np.isclose(sn[:,1],2.5)&(sn[:,0]>=3)&(sn[:,0]<=14),0]
        self.assertGreater(np.diff(np.sort(row)).min(),.14)
        self.assertAlmostEqual(mesh.bulk_data[0].sum(),150)

    def test_residual_only_assembly_matches_full_assembly_exactly(self):
        p=self.problem;previous=np.zeros(p.mesh.dof_count)
        u=np.random.default_rng(31).normal(size=len(previous))*1e-5
        lam=p.initial_contact_state();lam[:,0]=.2;lam[:,1]=.02
        for formulation in ('alart_curnier','jean_moreau'):
            for mixed in (False,True):
                a=p.assemble(u,previous,.0173,0,.1,lam[:,0],lam[:,1],mixed=mixed,formulation=formulation)
                b=p.assemble(u,previous,.0173,0,.1,lam[:,0],lam[:,1],mixed=mixed,formulation=formulation,with_tangent=False)
                np.testing.assert_array_equal(a[0],b[0]);self.assertIsNone(b[1]);self.assertEqual(a[3],b[3])

    def test_jean_moreau_tangents_and_common_translation(self):
        p=self.problem;c=p.config;options=p.options
        _,en=_rotation_vectors(c.inclination_deg)
        center=local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]+options.tip_radius_mm*en
        radial=np.array([np.cos(-1.1),np.sin(-1.1)]);tangent=np.array([-radial[1],radial[0]])
        for gap in (-.002,.0005):
            points=np.tile(center+(options.tip_radius_mm-gap)*radial,(len(p.weights),1))
            for slip in (1e-5,.003):
                previous=points-slip*tangent
                normal=np.full(len(p.weights),.3);shear=np.full(len(p.weights),.02)
                args=(previous,0,np.zeros(2),normal,shear)
                q=p.contact(points,*args,formulation='jean_moreau')
                for axis in range(2):
                    shift=np.zeros_like(points);shift[:,axis]=1e-7
                    a=p.contact(points+shift,*args,formulation='jean_moreau')
                    b=p.contact(points-shift,*args,formulation='jean_moreau')
                    np.testing.assert_allclose(q['jac'][:,:,axis],(a['force']-b['force'])/2e-7,rtol=1e-5,atol=1e-5)
                    projected_a=np.column_stack((a['pressure'],a['shear']))
                    projected_b=np.column_stack((b['pressure'],b['shear']))
                    np.testing.assert_allclose(q['projection_u'][:,:,axis],(projected_a-projected_b)/2e-7,rtol=1e-5,atol=1e-5)
                    dn=np.full(len(normal),1e-7) if axis==0 else np.zeros(len(normal))
                    ds=np.full(len(normal),1e-7) if axis==1 else np.zeros(len(normal))
                    a=p.contact(points,previous,0,np.zeros(2),normal+dn,shear+ds,formulation='jean_moreau')
                    b=p.contact(points,previous,0,np.zeros(2),normal-dn,shear-ds,formulation='jean_moreau')
                    np.testing.assert_allclose(q['dforce_dlambda'][:,:,axis],(a['force']-b['force'])/2e-7,rtol=1e-5,atol=1e-6)
                    projected_a=np.column_stack((a['pressure']-normal-dn,a['shear']-shear-ds))
                    projected_b=np.column_stack((b['pressure']-normal+dn,b['shear']-shear+ds))
                    np.testing.assert_allclose(q['projection_lambda'][:,:,axis],(projected_a-projected_b)/2e-7,rtol=1e-5,atol=1e-6)
        shift=np.array([0.,-.003])
        for formulation in ('alart_curnier','jean_moreau'):
            a=p.contact(points,points,0,np.zeros(2),normal,shear,formulation=formulation)
            b=p.contact(points+shift,points,.003,shift,normal,shear,formulation=formulation)
            np.testing.assert_allclose(b['slip'],0,atol=1e-14)
            np.testing.assert_allclose(a['force'],b['force'],atol=1e-10)

    def test_contact_predictor_removes_nodal_overlap_without_committing_history(self):
        p=self.problem
        old=np.zeros(p.mesh.dof_count)
        seed=p.contact_predictor(old,.001)
        gap,_=rounded_blade_distance((p.mesh.reference_xy+seed.reshape(-1,2))[p.contact_nodes],p.config,p.options,.001)
        self.assertLessEqual(gap.max(),1e-10)
        np.testing.assert_array_equal(old,0)
        np.testing.assert_array_equal(seed[p.fixed],old[p.fixed])

    def test_mixed_equilibrium_tangent_with_signed_trial_multipliers(self):
        p=self.problem
        old=np.zeros(p.mesh.dof_count)
        state=p.initial_contact_state()
        state[:,0]=np.linspace(-.1,.1,len(state));state[:,1]=.02
        _,k,contact,_=p.assemble(old,old,.0173,0,.1,state[:,0],state[:,1],mixed=True)
        direction=np.random.default_rng(5).normal(size=len(old))
        eps=1e-7
        fd=(p.assemble(old+eps*direction,old,.0173,0,.1,state[:,0],state[:,1],mixed=True)[0]-
            p.assemble(old-eps*direction,old,.0173,0,.1,state[:,0],state[:,1],mixed=True)[0])/(2*eps)
        np.testing.assert_allclose(k@direction,fd,rtol=2e-5,atol=1e-4)

    def test_nominal_stress_matches_homogeneous_uniaxial_and_rotation(self):
        p = self.problem
        # Plane strain extension F=diag(lambda,1/lambda,1), J=1.
        stretch = 1.3
        F = np.diag([stretch,1/stretch])
        u = (p.mesh.reference_xy@(F-np.eye(2)).T).ravel()
        fields = p.fields(u)
        expected_cauchy = p.config.shear_modulus_n_mm2*(stretch**2-(stretch**2+stretch**-2+1)/3)
        np.testing.assert_allclose(fields['nominal_tensile_mpa'],expected_cauchy/stretch,atol=1e-10)
        np.testing.assert_allclose(fields['averaged_nominal_tensile_mpa'],fields['nominal_tensile_mpa'],atol=1e-10)
        angle = .6
        R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotated = (p.mesh.reference_xy@(R@F-np.eye(2)).T).ravel()
        np.testing.assert_allclose(p.fields(rotated)['nominal_tensile_mpa'],fields['nominal_tensile_mpa'],atol=1e-10)

    def test_compression_does_not_trigger_tensile_criterion(self):
        p = self.problem
        u = -.05*p.mesh.reference_xy.ravel()
        np.testing.assert_allclose(p.fields(u)['nominal_tensile_mpa'],0,atol=1e-10)

    def test_contact_tangent_and_force_balance(self):
        p = self.problem
        u = np.zeros(p.mesh.dof_count)
        lam = np.zeros(len(p.weights))
        # Nonzero depth avoids exactly switching contact branches.
        r,k,g,d = p.assemble(u,u,.0173,0,.2,lam)
        direction = np.random.default_rng(4).normal(size=len(u))
        eps = 2e-7
        fd = (p.assemble(u+eps*direction,u,.0173,0,.2,lam)[0]-
              p.assemble(u-eps*direction,u,.0173,0,.2,lam)[0])/(2*eps)
        np.testing.assert_allclose(k@direction,fd,rtol=2e-4,atol=2e-4)
        self.assertGreater(d['reaction_fz_n'],0)

    def test_distance_normal_direction(self):
        c,o = self.config,self.options
        contact = local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]
        _,en = _rotation_vectors(c.inclination_deg)
        points = np.array([contact+.001*en,contact-.001*en])
        distance,normal = rounded_blade_distance(points,c,o,0)
        np.testing.assert_allclose(distance,[.001,-.001],atol=1e-12)
        np.testing.assert_allclose(normal,np.tile(-en,(2,1)),atol=1e-12)

    def test_contact_return_map_tangent_on_curved_nose(self):
        p = self.problem
        c = p.config
        contact = local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]
        _,en = _rotation_vectors(c.inclination_deg)
        points = np.tile(contact+.001*en,(len(p.weights),1))
        tangent = np.array([en[1],-en[0]])
        lam = np.zeros(len(p.weights))
        for slip in (1e-7,1e-3):
            previous = points-slip*tangent
            data = p.contact(points,previous,0,np.zeros(2),lam,np.zeros(len(lam)))
            eps = 1e-9
            for axis in range(2):
                shift = np.zeros(2); shift[axis] = eps
                fd = (p.contact(points+shift,previous,0,np.zeros(2),lam,np.zeros(len(lam)))['force']-
                      p.contact(points-shift,previous,0,np.zeros(2),lam,np.zeros(len(lam)))['force'])/(2*eps)
                np.testing.assert_allclose(data['jac'][:,:,axis],fd,rtol=2e-5,atol=.02)

    def test_flank_normal_stays_exact_as_gap_approaches_zero(self):
        from prototype.initiation import _blade_boundary_vertices
        c,o = self.config,self.options
        vertices = _blade_boundary_vertices(c,o,.41328125)
        a,b = vertices[1:3]
        edge = b-a
        normal = np.array([edge[1],-edge[0]])/np.linalg.norm(edge)
        points = (a+b)/2 + np.array([-1e-7,-1e-10,1e-10,1e-7])[:,None]*normal
        _,computed = rounded_blade_distance(points,c,o,.41328125)
        np.testing.assert_allclose(computed,np.tile(normal,(4,1)),atol=1e-13,rtol=0)

    def test_failed_run_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run'
            with patch.object(IntactProblem,'step',side_effect=RuntimeError('deliberate failure')):
                with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        run(self.config,self.options,output)
            status = json.loads((output/'status.json').read_text(encoding='utf-8'))
            self.assertFalse(status['completed'])
            self.assertEqual(status['state'],'failed')

    def test_unique_boundary_nodes_and_lumped_area(self):
        p = self.problem
        self.assertEqual(len(p.contact_nodes),len(np.unique(p.contact_nodes)))
        self.assertAlmostEqual(p.weights.sum(),p.config.width_mm*(p.config.length_mm+2*p.config.height_mm))
        self.assertTrue(np.all(p.weights > 0))
        # Each normal constraint acts on a distinct displacement node.
        self.assertEqual(len(np.unique(p.cdofs.ravel())),p.cdofs.size)

    def test_failed_step_keeps_history_and_parameters(self):
        p = self.problem
        u = np.zeros(p.mesh.dof_count)
        state = p.initial_contact_state()
        before = p.config.to_dict()
        with patch.object(p,'assemble',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                p.step(u,0,.01,.05,state)
        np.testing.assert_array_equal(u,0)
        np.testing.assert_array_equal(state,0)
        self.assertEqual(p.config.to_dict(),before)

    def test_old_quadrature_history_is_rejected(self):
        p = self.problem
        with self.assertRaisesRegex(ValueError,'incompatible'):
            p.step(np.zeros(p.mesh.dof_count),0,.01,.05,np.zeros(len(p.weights)))

    def test_pressure_and_shear_multiplier_tangents(self):
        p=self.problem
        contact=local_to_lab(np.array([[p.config.drop_s_mm,p.config.height_mm]]),p.config)[0]
        _,en=_rotation_vectors(p.config.inclination_deg)
        points=np.tile(contact+.001*en,(len(p.weights),1))
        tangent=np.array([en[1],-en[0]])
        for slip in (1e-7,1e-2):
            old=points-slip*tangent
            multipliers=np.zeros((len(p.weights),2));multipliers[:,0]=.1
            data=p.contact(points,old,0,np.zeros(2),multipliers[:,0],multipliers[:,1])
            for axis in range(2):
                shift=np.zeros_like(multipliers);shift[:,axis]=1e-6
                plus=multipliers+shift;minus=multipliers-shift
                a=p.contact(points,old,0,np.zeros(2),plus[:,0],plus[:,1])
                b=p.contact(points,old,0,np.zeros(2),minus[:,0],minus[:,1])
                np.testing.assert_allclose(data['dforce_dlambda'][:,:,axis],(a['force']-b['force'])/2e-6,atol=1e-7)
                pa=np.column_stack((a['pressure'],a['shear']))-plus
                pb=np.column_stack((b['pressure'],b['shear']))-minus
                np.testing.assert_allclose(data['projection_lambda'][:,:,axis],(pa-pb)/2e-6,atol=1e-7)

    def test_boundary_refinement_preserves_geometry_and_affine_history(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        gradient=np.array([[.02,.01],[.0,-.01]])
        u=(p.mesh.reference_xy@gradient.T).ravel()
        state=p.initial_contact_state()
        state[:,0]=2+.01*p.mesh.reference_xy[p.contact_nodes,0]
        state[:,1]=.1
        original_u=u.copy();original_state=state.copy()
        original_resultant=p.weights@state
        count=p.mesh.node_count
        edge=int(np.flatnonzero(np.all(np.isclose(p.mesh.local_sn[p.edges,1],5),axis=1))[0])
        mapped,history=p.refine_contact(u,state,[edge])
        self.assertEqual(p.mesh.node_count,count+1)
        self.assertAlmostEqual(p.mesh.bulk_data[0].sum(),150)
        self.assertTrue(np.all(p.mesh.bulk_data[0]>0))
        all_edges=np.sort(np.concatenate([p.mesh.triangles[:,[0,1]],p.mesh.triangles[:,[1,2]],p.mesh.triangles[:,[2,0]]]),axis=1)
        unique,counts=np.unique(all_edges,axis=0,return_counts=True)
        self.assertTrue(np.all(counts<=2))
        boundary=unique[counts==1]
        boundary=boundary[~np.all(np.isclose(p.mesh.local_sn[boundary,1],0),axis=1)]
        self.assertEqual({tuple(edge) for edge in boundary},{tuple(sorted(edge)) for edge in p.edges})
        np.testing.assert_allclose(mapped,(p.mesh.reference_xy@gradient.T).ravel(),atol=1e-12)
        np.testing.assert_allclose(p.weights@history,original_resultant,atol=1e-10)
        np.testing.assert_array_equal(u,original_u)
        np.testing.assert_array_equal(state,original_state)


if __name__ == '__main__':
    unittest.main()
