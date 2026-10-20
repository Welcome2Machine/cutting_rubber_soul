"""Independent consistency, geometry, bending and unchanged-contact checks."""
import unittest
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from prototype.ipc_p2 import InteriorP2, triangle_rule
from prototype.model_config import FEMConfig, FEMMesh
from prototype.material import bulk, response


def rectangle(h=.5, length=2.):
    nx,ny=round(length/h),round(1/h)
    xy=np.array([(x,y) for y in np.linspace(0,1,ny+1) for x in np.linspace(0,length,nx+1)])
    tri=[]
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i; b=a+1; d=a+nx+1; e=d+1
            tri.extend([[a,b,e],[a,e,d]])
    return FEMMesh(xy,xy,np.array(tri),np.empty((0,2),int),np.empty(0),
                   np.flatnonzero(xy[:,1]==0),np.empty(0,int),np.zeros(len(xy),int))


def assembled(result, n):
    r,k,dofs,E,J=result
    g=np.zeros(n);np.add.at(g,dofs.ravel(),r.ravel())
    H=None if k is None else coo_matrix((k.ravel(),(np.broadcast_to(dofs[:,:,None],k.shape).ravel(),
        np.broadcast_to(dofs[:,None,:],k.shape).ravel())),shape=(n,n)).tocsr()
    return g,H,E,J


def bending(h, p2):
    base=rectangle(h,20.);c=FEMConfig(width_mm=1.,poisson_ratio=.49)
    p=InteriorP2(base,c) if p2 else None
    mesh=base if p is None else p.mesh
    n=mesh.dof_count;xy=mesh.reference_xy
    r=(bulk(mesh,np.zeros(n),c) if p is None else p.bulk(np.zeros(n)))
    H=assembled(r,n)[1]
    fixed=(2*np.flatnonzero(xy[:,0]==0)[:,None]+np.arange(2)).ravel()
    free=np.setdiff1d(np.arange(n),fixed)
    tip=np.flatnonzero(xy[:,0]==20);weights=np.ones(len(tip));weights[[0,-1]]=.5;weights/=weights.sum()
    force=np.zeros(n);force[2*tip+1]=-1e-6*weights
    u=np.zeros(n);u[free]=spsolve(H[free][:,free],force[free])
    G=c.shear_modulus_n_mm2
    reference=1e-6*(20**3/(3*(2*G/(1-c.poisson_ratio))*(1/12))+20/((5/6)*G))
    return float(-(weights@u[2*tip+1])/reference)


class InteriorP2Tests(unittest.TestCase):
    def test_rule_monomials(self):
        import math
        l,w=triangle_rule(4)
        for i in range(6):
            for j in range(6-i):
                self.assertAlmostEqual(w@(l[:,1]**i*l[:,2]**j),
                    2*math.factorial(i)*math.factorial(j)/math.factorial(i+j+2),places=13)

    def test_affine_energy_stress_and_objectivity(self):
        base=rectangle();c=FEMConfig(width_mm=20.)
        p=InteriorP2(base,c)
        for F in (np.array([[1.2,.3],[-.1,.85]]),np.array([[0.,-1.],[1.,0.]])):
            u=p.mesh.reference_xy@(F-np.eye(2)).T
            actual=p.bulk(u)[3:]
            exact=response(F[None],c,tangent=False)
            self.assertAlmostEqual(actual[0],float(base.bulk_data[0].sum()*c.width_mm*exact['energy'][0]),places=10)
            self.assertAlmostEqual(actual[1],float(np.linalg.det(F)),places=12)
            np.testing.assert_allclose(p.deformation(u),np.broadcast_to(F,p.deformation(u).shape),atol=2e-14)

    def test_energy_gradient_tangent_and_quadrature(self):
        p=InteriorP2(rectangle(),FEMConfig(width_mm=1.))
        rng=np.random.default_rng(417);u=rng.normal(size=p.mesh.dof_count)*.007
        v=rng.normal(size=len(u));v/=np.linalg.norm(v)
        g,H,E,_=assembled(p.bulk(u),len(u));eps=1e-6
        plus=assembled(p.bulk(u+eps*v,with_tangent=False),len(u))
        minus=assembled(p.bulk(u-eps*v,with_tangent=False),len(u))
        self.assertAlmostEqual((plus[2]-minus[2])/(2*eps),float(g@v),delta=2e-8)
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps),H@v,rtol=3e-7,atol=1e-7)
        self.assertLess(np.max(np.abs((H-H.T).data),initial=0),1e-11)
        q=InteriorP2(rectangle(),p.config,6)
        g6,_,E6,_=assembled(q.bulk(u,with_tangent=False),len(u))
        self.assertLess(abs(E/E6-1),2e-6)
        self.assertLess(np.linalg.norm(g-g6)/np.linalg.norm(g6),2e-5)

    def test_p1_subspace_energy_and_force_pullback(self):
        base=rectangle();p=InteriorP2(base,FEMConfig(width_mm=1.))
        rng=np.random.default_rng(6);v=rng.normal(size=(base.node_count,2))*.01
        u=np.vstack((v,v[p.interior_edges].mean(axis=1)))
        g,_,E,_=assembled(p.bulk(u,with_tangent=False),u.size)
        g=g.reshape(-1,2);pull=g[:base.node_count].copy()
        for side in (0,1):np.add.at(pull,p.interior_edges[:,side],.5*g[base.node_count:])
        g1,_,E1,_=assembled(bulk(base,v.ravel(),p.config,with_tangent=False),v.size)
        np.testing.assert_allclose(pull.ravel(),g1,atol=1e-12)
        self.assertAlmostEqual(E,E1,places=12)

    def test_boundary_trace_is_exactly_original_linear(self):
        p=InteriorP2(rectangle(),FEMConfig())
        rng=np.random.default_rng(9);u=rng.normal(size=(p.mesh.node_count,2))*.01
        from prototype.quadratic import shape
        for k,(a,b) in enumerate(((0,1),(1,2),(2,0))):
            ids=np.flatnonzero(p.boundary_slots[:,k]);l=np.zeros((1,3));l[0,a]=.37;l[0,b]=.63
            element_u=u[p.elements[ids]].copy()
            for j,(v,w) in enumerate(((0,1),(1,2),(2,0))):
                mask=p.boundary_slots[ids,j]
                element_u[mask,3+j]=.5*(element_u[mask,v]+element_u[mask,w])
            np.testing.assert_allclose(np.einsum('a,eai->ei',shape(l)[0],element_u),
                .37*u[p.elements[ids,a]]+.63*u[p.elements[ids,b]],atol=1e-15)

    def test_path_rejects_hidden_inversion(self):
        p=InteriorP2(rectangle(),FEMConfig())
        u=np.zeros_like(p.mesh.reference_xy);d=p.mesh.reference_xy@np.diag([-4.,-2.])
        self.assertGreater(p.minimum_j(d),0)
        alpha=p.safe_step(u,d)
        self.assertLess(alpha,.25)
        for t in np.linspace(0,alpha,25):self.assertGreater(p.minimum_j(t*d),0)

    def test_same_native_contact_on_p1_subspace(self):
        from prototype.ipc_reference import IPCProblem
        opts=dict(case='patch',mu=.35,local_spacing=.2,surface_spacing=.2)
        a=IPCProblem(**opts);b=IPCProblem(**opts,element='p2-interior')
        xa=a.rest.copy();xa[a.n:,1]-=.003
        xb=b.rest.copy();xb[b.n:]=xa[a.n:]
        np.testing.assert_array_equal(xa[a.surface],xb[b.surface])
        E1,g1,_,_=a.evaluate(xa,a.lag(xa),tangent=False)
        E2,g2,_,_=b.evaluate(xb,b.lag(xb),tangent=False)
        self.assertAlmostEqual(E1,E2,places=11)
        np.testing.assert_allclose(g1.reshape(-1,2)[a.n:],g2.reshape(-1,2)[b.n:],atol=1e-12)

    def test_bending_improvement_and_refinement(self):
        p1=bending(.5,False);coarse=bending(.5,True);fine=bending(.25,True)
        print(dict(p1_bending_ratio=p1,p2_coarse_ratio=coarse,p2_fine_ratio=fine),flush=True)
        self.assertGreater(coarse,2*p1)
        self.assertLess(abs(fine-1),abs(coarse-1))
        self.assertGreater(fine,.9)

    def test_no_extra_zero_energy_modes(self):
        p=InteriorP2(rectangle(),FEMConfig(width_mm=1.))
        H=assembled(p.bulk(np.zeros(p.mesh.dof_count)),p.mesh.dof_count)[1].toarray()
        eig=np.linalg.eigvalsh(H)
        self.assertEqual(int(np.count_nonzero(np.abs(eig)<1e-9)),3)
        self.assertGreater(eig[3],1e-3)

    def test_native_finite_compression_path_and_reaction(self):
        import contextlib,io,time
        from prototype.ipc_reference import IPCProblem
        from threadpoolctl import threadpool_limits
        p=IPCProblem('patch',mu=.35,local_spacing=.2,surface_spacing=.2,
                     element='p2-interior',max_iterations=200)
        x=p.rest.copy();old=p.start
        with contextlib.redirect_stdout(io.StringIO()),threadpool_limits(limits=2):
            for depth in (0.,.01,.02,.03):
                x,info=p.step(x,old,depth,time.monotonic()+20);old=depth
        g=p.evaluate(x,p.lag(x),tangent=False)[1].reshape(-1,2)
        np.testing.assert_allclose(g.sum(axis=0),0,atol=1e-9)
        self.assertLess(info['residual_n'],1e-4)
        self.assertGreater(info['min_j'],0)
        top=np.flatnonzero(np.isclose(p.mesh.reference_xy[:,1],1.))
        stretch=x[top,1].mean()
        expected=-response(np.diag([1.,stretch])[None],p.config,tangent=False)['P'][0,1,1]*p.config.width_mm
        self.assertLess(abs(info['reaction_fz_n']/expected-1),.002)

    def test_runner_checkpoints_p2_and_rejects_p1_renderer(self):
        import tempfile,contextlib,io
        from pathlib import Path
        from prototype.ipc_reference import parse_args,run
        from tools.visualize_ipc import load_frames
        with tempfile.TemporaryDirectory() as folder:
            args=parse_args(['--output',folder,'--case','patch','--depth','.01','--increment','.01',
                '--friction','.35','--local-spacing','.2','--surface-spacing','.2','--element','p2-interior',
                '--no-visualize','--threads','2','--seconds','20'])
            with contextlib.redirect_stdout(io.StringIO()):self.assertTrue(run(args))
            with np.load(Path(folder)/'states/0001.npz',allow_pickle=False) as state:
                self.assertEqual(state['p2_elements'].shape[1],6)
                self.assertEqual(int(state['p2_quadrature_order']),4)
            with self.assertRaisesRegex(ValueError,'P1 stress renderer'):load_frames(Path(folder))

    def test_study_rejects_changed_physics_and_missing_steps(self):
        import copy
        from tools.run_ipc_element_study import accepted_prefix,check_pair
        good=dict(depth_mm=0.,reaction_fz_n=1.,residual_n=1e-6,min_j=.9,
                  minimum_distance_mm=.001,friction_work_n_mm=0.)
        self.assertEqual(len(accepted_prefix(dict(steps=[good]))),1)
        bad=dict(good,depth_mm=.02)
        with self.assertRaises(ValueError):accepted_prefix(dict(steps=[good,bad]))
        keys=('config','mesh_settings','dhat_mm','barrier_stiffness_mpa','ipc_binary_sha256',
              'loading_increment_mm','force_tolerance_n','friction_solver','friction_solver_settings','hessian_mode')
        a={k:1 for k in keys};a['config']={'shear_modulus_n_mm2':.66}
        a['simplifications']={k:1 for k in (
            'tip_radius_mm','nose_facets','fracture','material','friction_smoothing_mm')}
        a['source_sha256']={k:'hash' for k in ('prototype/material.py','mechanics/finite_strain.py',
            'prototype/model_config.py','prototype/initiation.py')}
        b=copy.deepcopy(a);check_pair(a,b);b['config']=2
        with self.assertRaises(ValueError):check_pair(a,b)
        b=copy.deepcopy(a);b['source_sha256']['prototype/material.py']='newhash'
        with self.assertRaises(ValueError):check_pair(a,b)
        equivalence=dict(passed=True,sources={'prototype/material.py':dict(baseline='hash',current='newhash')})
        check_pair(a,b,equivalence)
        equivalence['passed']=False
        with self.assertRaises(ValueError):check_pair(a,b,equivalence)
        b=copy.deepcopy(a);b['config']['ogden_alpha']=2.;b['contact_formulation']='improved-max'
        check_pair(a,b)
        b['contact_formulation']='offset-geometric'
        with self.assertRaisesRegex(ValueError,'contact_formulation'):check_pair(a,b)


if __name__=='__main__':unittest.main()
