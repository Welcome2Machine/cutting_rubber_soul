import tempfile
from pathlib import Path
import unittest
import contextlib
import io
from unittest.mock import patch
import numpy as np
from prototype.model_config import FEMConfig,FEMMesh
from prototype.initiation import IntactProblem,InitiationOptions
from prototype.contact_study import save_state,load_failed_problem
from run_support import write_json


class MacroElementTests(unittest.TestCase):
    def test_sparse_pressure_blocks_have_exact_condensed_tangent(self):
        from scipy.sparse import diags
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.1,mesh_grading='graded',contact_discretization='macro',volumetric_discretization='macro_mixed'))
        rng=np.random.default_rng(35);u=rng.normal(size=p.mesh.dof_count)*1e-5;lam=p.initial_contact_state()
        r,k,_,d=p.assemble(u,u,0.,0.,.1,lam[:,0],lam[:,1],mixed=True)
        rr,local,q,dd=p.assemble(u,u,0.,0.,.1,lam[:,0],lam[:,1],mixed=True,pressure_blocks=True)
        H=q['pressure_coupling'];condensed=local+H.T@diags(1/q['pressure_mass'])@H
        np.testing.assert_array_equal(r,rr)
        self.assertLess(np.max(abs((k-condensed).data),initial=0.),1e-10)
        self.assertEqual(d['bulk_energy_n_mm'],dd['bulk_energy_n_mm'])

    def test_finite_confined_compression_contact_matches_homogeneous_solution(self):
        from prototype.material import response
        c=FEMConfig(length_mm=1,height_mm=1,width_mm=1,drop_s_mm=.5,inclination_deg=0,mesh_size_mm=.25,max_indentation_mm=.3,friction_coefficient=0.,newton_tolerance_n=1e-8,contact_multiplier_tolerance_n_mm2=1e-8)
        p=IntactProblem(c,InitiationOptions(refinement_mm=.25,surface_spacing_mm=.125,mesh_grading='graded',contact_discretization='macro',volumetric_discretization='macro_mixed'))
        p.mesh.macro_boundary_edges=p.mesh.macro_boundary_edges[np.all(np.isclose(p.mesh.reference_xy[p.mesh.macro_boundary_edges,1],1),axis=1)]
        p._rebuild_boundary(p.edges[np.all(np.isclose(p.mesh.reference_xy[p.edges,1],1),axis=1)])
        p.fixed=np.unique(np.r_[2*np.arange(p.mesh.node_count),2*p.mesh.bottom_nodes+1])
        p.free=np.setdiff1d(np.arange(p.mesh.dof_count),p.fixed)
        def plane(x,c,o,depth,with_tangent=False):
            q=(x[:,1]-1+depth,np.tile([0.,-1.],(len(x),1)))
            return q+(np.zeros((len(x),2,2)),) if with_tangent else q
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state();old=0.
        with patch('prototype.initiation.rounded_blade_distance',plane),contextlib.redirect_stdout(io.StringIO()):
            for depth in (.1,.2,.3):
                u,state,d=p.step(u,old,depth,.1,state);old=depth
                expected=-response(np.array([[[1.,0.],[0.,1-depth]]]),c,tangent=False)['P'][0,1,1]
                self.assertAlmostEqual(d['reaction_fz_n'],expected,places=10)
                np.testing.assert_allclose(u.reshape(-1,2)[:,1],-depth*p.mesh.reference_xy[:,1],atol=1e-10)
                np.testing.assert_allclose(p.fields(u)['J'],1-depth,atol=1e-10)

    def test_linearized_pressure_space_has_no_spurious_null_modes(self):
        from scipy.sparse import coo_matrix,diags
        from scipy.sparse.linalg import splu
        from prototype.macro_element import refine_macro_mesh
        from prototype.material import NodalVolume
        positive=[]
        for n in (3,6):
            xy=np.array([(i/n,j/n) for j in range(n+1) for i in range(n+1)])
            triangles=[]
            for j in range(n):
                for i in range(n):
                    a=j*(n+1)+i;b=a+1;d=a+n+1;e=d+1
                    triangles.extend([[a,b,e],[a,e,d]])
            base=FEMMesh(xy,xy,np.array(triangles),np.empty((0,2),int),np.empty(0),np.empty(0,int),np.empty(0,int),np.zeros(len(xy),int))
            mesh,_=refine_macro_mesh(base,np.empty((0,2),int))
            area,grad,dofs=mesh.bulk_data;nv=NodalVolume(mesh,FEMConfig(width_mm=1.))
            divergence=coo_matrix((grad.reshape(-1,6).ravel(),(np.repeat(np.arange(len(area)),6),dofs.ravel())),shape=(len(area),mesh.dof_count)).tocsr()
            B=nv.incidence@divergence
            local=np.einsum('e,eai,ebi,jk->eajbk',area,grad,grad,np.eye(2)).reshape(-1,6,6)
            A=coo_matrix((local.ravel(),(np.broadcast_to(dofs[:,:,None],local.shape).ravel(),np.broadcast_to(dofs[:,None,:],local.shape).ravel())),shape=(mesh.dof_count,)*2).tocsc()
            xy=mesh.reference_xy;inner=np.flatnonzero(np.all((xy>1e-10)&(xy<1-1e-10),axis=1))
            free=(2*inner[:,None]+np.arange(2)).ravel()
            C=(diags(1/np.sqrt(nv.nodal_volumes))@B)[:,free].toarray()
            eigenvalues=np.linalg.eigvalsh(C@splu(A[free][:,free]).solve(C.T))
            # Fully fixed boundary leaves only the physical constant-pressure
            # nullspace. This is a two-mesh linearized check, not a finite-strain
            # contact stability proof.
            self.assertEqual(np.sum(eigenvalues<1e-10),1)
            positive.append(eigenvalues[1])
        self.assertGreater(min(positive),.03)
        self.assertGreater(positive[1]/positive[0],.9)

    def test_refinement_preserves_parent_pressure_affine_field_and_checkpoint(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.05,mesh_grading='graded',contact_discretization='macro',volumetric_discretization='macro_mixed'))
        gradient=np.array([[.02,.01],[-.01,.03]])
        u=(p.mesh.reference_xy@gradient).ravel();state=p.initial_contact_state();state[:,0]=2.;state[:,1]=.3
        force=p.weights@state;pressure_count=p.mesh.macro_node_count
        for _ in range(2):
            centers=p.mesh.local_sn[p.edges].mean(axis=1)
            index=np.linalg.norm(centers-[p.config.drop_s_mm,p.config.height_mm],axis=1).argmin()
            u,state=p.refine_contact(u,state,[index])
            np.testing.assert_allclose(u.reshape(-1,2),p.mesh.reference_xy@gradient,atol=1e-13)
            np.testing.assert_allclose(p.weights@state,force,atol=1e-11)
            self.assertEqual(p.mesh.macro_node_count,pressure_count)
            self.assertAlmostEqual(p.mesh.bulk_data[0].sum(),150.)
            q=p.nodal_volume.S@p.fields(u)['J']
            np.testing.assert_allclose(q,q[0],atol=1e-12)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);save_state(out/'last_state.npz',p,u,state,.1)
            write_json(out/'result.json',dict(config=p.config.to_dict(),options=vars(p.options)))
            other,v,lam,depth=load_failed_problem(out)
            np.testing.assert_array_equal(v,u);np.testing.assert_array_equal(lam,state)
            np.testing.assert_array_equal(other.mesh.macro_parents,p.mesh.macro_parents)
            np.testing.assert_allclose(other.fields(v)['cauchy_stress_xzy_mpa'],p.fields(u)['cauchy_stress_xzy_mpa'])
            self.assertEqual(depth,.1)


if __name__=='__main__':unittest.main()
