import unittest
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from prototype.model_config import FEMConfig,FEMMesh
from prototype.material import bulk,NodalVolume
from prototype.initiation import IntactProblem,InitiationOptions


def bending_compliance(h,nu,averaged,stabilization=0.):
    """Small-load slender plane-strain cantilever; includes shear correction."""
    if averaged=='macro':h*=2
    nx,ny=round(20/h),round(1/h)
    xy=np.array([(x,y) for y in np.linspace(0,1,ny+1) for x in np.linspace(0,20,nx+1)])
    triangles=[]
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i;b=a+1;d=a+nx+1;e=d+1
            triangles.extend([[a,b,e],[a,e,d]])
    tri=np.array(triangles)
    mesh=FEMMesh(xy,xy,tri,np.empty((0,2),int),np.empty(0),np.empty(0,int),np.empty(0,int),np.zeros(len(xy),int))
    if averaged=='macro':
        from prototype.macro_element import refine_macro_mesh
        mesh,_=refine_macro_mesh(mesh,np.empty((0,2),int))
        xy=mesh.reference_xy
    c=FEMConfig(poisson_ratio=nu,width_mm=1.)
    if averaged=='quadratic':
        from prototype.quadratic import quadratic_mesh,L,W
        from prototype.material import response
        mesh=quadratic_mesh(mesh,np.empty((0,2),int));xy=mesh.reference_xy
        area,grad,_=mesh.bulk_data
        grad=np.concatenate(((4*L[None,:,:,None]-1)*grad[:,None,:,:],4*(L[None,:,[1,2,0],None]*grad[:,None,[0,1,2],:]+L[None,:,[0,1,2],None]*grad[:,None,[1,2,0],:])),axis=2)
        C=response(np.eye(2)[None],c)['tangent'].reshape(2,2,2,2)
        k=np.einsum('e,q,eqaJ,iJkL,eqbL->eaibk',area,W,grad,C,grad,optimize=True).reshape(-1,12,12)
        dofs=(2*mesh.elements[:,:,None]+np.arange(2)).reshape(-1,12)
    else:r,k,dofs,energy,_=bulk(mesh,np.zeros(mesh.dof_count),c)
    extra=None
    if averaged and averaged!='quadratic':r,k,extra,energy=NodalVolume(mesh,c,stabilization).correct(np.zeros(mesh.dof_count),r,k,energy)
    K=coo_matrix((k.ravel(),(np.broadcast_to(dofs[:,:,None],k.shape).ravel(),np.broadcast_to(dofs[:,None,:],k.shape).ravel())),shape=(mesh.dof_count,)*2).tocsr()
    if extra is not None:K=K+extra
    fixed=(2*np.flatnonzero(xy[:,0]==0)[:,None]+np.arange(2)).ravel()
    free=np.setdiff1d(np.arange(mesh.dof_count),fixed)
    force=np.zeros(mesh.dof_count);tip=np.flatnonzero(xy[:,0]==20)
    weights=np.ones(len(tip));weights[[0,-1]]=.5;weights/=weights.sum()
    if averaged=='quadratic':
        tip=tip[np.argsort(xy[tip,1])];weights=np.ones(len(tip));weights[1:-1:2]=4;weights[2:-1:2]=2;weights/=weights.sum()
    force[2*tip+1]=-1e-6*weights
    u=np.zeros_like(force);u[free]=spsolve(K[free][:,free],force[free])
    actual=-(weights@u[2*tip+1]);G=c.shear_modulus_n_mm2
    reference=1e-6*(20**3/(3*(2*G/(1-nu))*(1/12))+20/((5/6)*G))
    return actual/reference


class NodalVolumeTests(unittest.TestCase):
    def test_full_quadratic_bending_avoids_original_linear_element_locking(self):
        coarse=bending_compliance(.5,.49,'quadratic');fine=bending_compliance(.25,.49,'quadratic')
        self.assertGreater(coarse,.97);self.assertGreater(fine,coarse);self.assertLess(abs(fine-1),.02)
    def problem(self,kind,stabilization=0.):
        return IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2,
            contact_discretization='integral',volumetric_discretization=kind,volume_stabilization_shear=stabilization))

    def test_homogeneous_material_and_stress_preserved(self):
        a,b=self.problem('element'),self.problem('nodal_volume')
        u=(a.mesh.reference_xy@np.array([[.12,.03],[-.02,-.06]])).ravel()
        zero=a.initial_contact_state()
        ra,_,_,da=a.assemble(u,u,0.,0.,.1,zero[:,0],zero[:,1],mixed=True)
        rb,_,_,db=b.assemble(u,u,0.,0.,.1,zero[:,0],zero[:,1],mixed=True)
        np.testing.assert_allclose(ra,rb,atol=1e-10)
        self.assertAlmostEqual(da['bulk_energy_n_mm'],db['bulk_energy_n_mm'],places=10)
        np.testing.assert_allclose(a.fields(u)['cauchy_stress_xzy_mpa'],b.fields(u)['cauchy_stress_xzy_mpa'],atol=1e-10)

    def test_nonlocal_energy_gradient_and_tangent(self):
        self.check_derivatives(self.problem('nodal_volume'))

    def test_macro_pressure_condensation_derivatives_and_homogeneous_response(self):
        p=self.problem('macro_mixed');self.check_derivatives(p)
        nv=p.nodal_volume
        self.assertEqual(nv.incidence.shape[0],p.mesh.macro_node_count)
        np.testing.assert_allclose(nv.S@np.ones(len(p.mesh.triangles)),1.,atol=1e-13)
        np.testing.assert_allclose(nv.pressure(np.full(len(p.mesh.triangles),1.1)),.1*p.config.bulk_modulus_n_mm2,atol=1e-11)

    def test_macro_bending_converges_without_volume_stabilization(self):
        coarse=bending_compliance(.25,.49,'macro')
        fine=bending_compliance(.125,.49,'macro')
        self.assertGreater(coarse,.8)
        self.assertLess(abs(fine-1),abs(coarse-1))
        self.assertLess(abs(fine-1),.1)

    def test_stabilized_energy_gradient_tangent_and_stress(self):
        self.check_derivatives(self.problem('nodal_volume',2.))

    def check_derivatives(self,p):
        rng=np.random.default_rng(21)
        u=rng.normal(size=p.mesh.dof_count)*1e-4;v=rng.normal(size=len(u));v/=np.linalg.norm(v)
        zero=p.initial_contact_state()
        def evaluate(x):return p.assemble(x,x,0.,0.,.1,zero[:,0],zero[:,1],mixed=True)
        r,k,_,d=evaluate(u);h=1e-6;a=evaluate(u+h*v);b=evaluate(u-h*v)
        self.assertAlmostEqual((a[3]['bulk_energy_n_mm']-b[3]['bulk_energy_n_mm'])/(2*h),float(r@v),delta=1e-7)
        np.testing.assert_allclose((a[0]-b[0])/(2*h),k@v,rtol=1e-5,atol=1e-5)
        self.assertLess(np.max(np.abs((k-k.T).data),initial=0.),1e-9)
        fields=p.fields(u);areas,grad,dofs=p.mesh.bulk_data
        internal=np.einsum('e,eiJ,eaJ->eai',areas*p.config.width_mm,
            fields['first_piola_stress_xzy_mpa'][:,:2,:2],grad).reshape(-1,6)
        reconstructed=np.zeros_like(r);np.add.at(reconstructed,dofs.ravel(),internal.ravel())
        np.testing.assert_allclose(reconstructed,r,atol=1e-10)

    def test_stabilization_is_nonnegative_patch_variance_and_preserves_uniform_response(self):
        a,b=self.problem('nodal_volume'),self.problem('nodal_volume',2.)
        rng=np.random.default_rng(34);u=rng.normal(size=a.mesh.dof_count)*1e-4
        def energy(p,x):
            zero=p.initial_contact_state()
            return p.assemble(x,x,0.,0.,.1,zero[:,0],zero[:,1],mixed=True,with_tangent=False)[3]['bulk_energy_n_mm']
        J=a.fields(u)['J'];nv=a.nodal_volume;Jbar=nv.S@J
        variance=0.
        for i,nodes in enumerate(a.mesh.triangles):
            variance+=nv.volumes[i]/3*np.sum((J[i]-Jbar[nodes])**2)
        self.assertGreater(variance,0.)
        self.assertAlmostEqual(energy(b,u)-energy(a,u),a.config.shear_modulus_n_mm2*variance,delta=1e-11)
        affine=(a.mesh.reference_xy@np.array([[.12,.03],[-.02,-.06]])).ravel()
        self.assertAlmostEqual(energy(a,affine),energy(b,affine),places=10)
        np.testing.assert_allclose(a.fields(affine)['cauchy_stress_xzy_mpa'],b.fields(affine)['cauchy_stress_xzy_mpa'],atol=1e-10)

    def test_nearly_incompressible_bending_improves_with_refinement(self):
        locked=bending_compliance(.25,.49,False)
        for stabilization in (0.,2.):
            coarse=bending_compliance(.25,.49,True,stabilization)
            fine=bending_compliance(.125,.49,True,stabilization)
            self.assertLess(locked,.4)
            self.assertGreater(coarse,locked*2)
            self.assertLess(abs(fine-1),abs(coarse-1))
            self.assertLess(abs(fine-1),.15)


if __name__=='__main__':unittest.main()
