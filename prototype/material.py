"""CB245NE plane-strain adapter. No unmeasured Maxwell branch is invented."""
import numpy as np
from mechanics.finite_strain import Material,elastic_response


def from_config(config):
    # The shared kernel uses G=sum(mu_i*alpha_i)/2. Preserve the measured
    # small-strain G and K when varying the large-strain shape, not mu_i.
    # alpha=2 exactly retains the previous neo-Hookean law.
    alpha = config.ogden_alpha
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ogden exponent must be finite and positive')
    return Material(mu_eq=(2*config.shear_modulus_n_mm2/alpha,),alpha_eq=(alpha,),
                    poisson=config.poisson_ratio,volumetric='quadratic_j')


def response(F,config,*,tangent=True,diagnostics=True):
    return elastic_response(F,from_config(config),kinematics='plane_strain',tangent=tangent,
                            diagnostics=diagnostics)


def bulk(mesh,u,config,*,with_tangent=True):
    areas,gradients,dofs=mesh.bulk_data
    F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[mesh.triangles],gradients)
    # P1 assembly consumes only P, energy, J and (optionally) the tangent.
    # Keep public response() diagnostics available for stress/field outputs,
    # without constructing unused Cauchy tensors and eigensystems per trial.
    q=response(F,config,tangent=with_tangent,diagnostics=False);volumes=areas*config.width_mm
    r=np.einsum('e,eiJ,eaJ->eai',volumes,q['P'],gradients).reshape(-1,6)
    k=None
    if with_tangent:
        C=q['tangent'].reshape(-1,2,2,2,2)
        k=np.einsum('e,eaJ,eiJkL,ebL->eaibk',volumes,gradients,C,gradients,optimize=True).reshape(-1,6,6)
    return r,k,dofs,float(volumes@q['energy']),float(q['J'].min())


class NodalVolume:
    """Energy-consistent nodal averaging of quadratic volumetric energy.

    Isochoric energy remains elementwise. E_vol = K/2 sum_i V_i (Jbar_i-1)^2.
    This changes the discrete volume constraint, not G, K or the homogeneous
    material law. Nonlocal tangent terms and stress corrections are required.
    """
    def __init__(self,mesh,config,stabilization_shear=0.):
        from scipy.sparse import coo_matrix,diags
        self.mesh,self.config=mesh,config
        self.volumes=mesh.bulk_data[0]*config.width_mm
        rows=mesh.triangles.ravel();cols=np.repeat(np.arange(len(self.volumes)),3)
        self.incidence=coo_matrix((np.repeat(self.volumes/3,3),(rows,cols)),
                                 shape=(mesh.node_count,len(self.volumes))).tocsr()
        if hasattr(mesh,'macro_triangles'):
            from .macro_element import pressure_incidence
            self.incidence=pressure_incidence(mesh,self.volumes)
        self.nodal_volumes=np.asarray(self.incidence.sum(axis=1)).ravel()
        self.S=diags(1/self.nodal_volumes)@self.incidence
        self.K=config.bulk_modulus_n_mm2
        if not np.isfinite(stabilization_shear) or not 0<=stabilization_shear<=self.K/config.shear_modulus_n_mm2:
            raise ValueError('Stabilization must lie between zero and K/G')
        # A positive shear-scale penalty on unresolved volume variance:
        # a*G/2 * [sum_e V_e (J_e-1)^2 - sum_i V_i (Jbar_i-1)^2].
        # The bracket is a sum of squared patch fluctuations and vanishes for
        # homogeneous deformation. This is a numerical stabilization, not a
        # changed homogeneous bulk modulus. Zero retains the comparison model.
        self.projected_fraction=max(0.,1-stabilization_shear*config.shear_modulus_n_mm2/self.K)

    def pressure(self,J):
        nodal=self.K*(self.S@J-1)
        projected=np.asarray(self.S.T@(self.nodal_volumes*nodal)).ravel()/self.volumes
        return self.projected_fraction*projected+(1-self.projected_fraction)*self.K*(J-1)

    def correct(self,u,r,k,energy,*,condensed=True):
        from scipy.sparse import coo_matrix,diags
        mesh=self.mesh;_,grad,dofs=mesh.bulk_data
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[mesh.triangles],grad)
        J=np.linalg.det(F);cofactor=np.empty_like(F)
        cofactor[:,0,0]=F[:,1,1];cofactor[:,1,1]=F[:,0,0]
        cofactor[:,0,1]=-F[:,1,0];cofactor[:,1,0]=-F[:,0,1]
        dj=np.einsum('eiJ,eaJ->eai',cofactor,grad).reshape(-1,6)
        difference=self.pressure(J)-self.K*(J-1)
        r=r+self.volumes[:,None]*difference[:,None]*dj
        energy+=.5*self.K*self.projected_fraction*(self.nodal_volumes@((self.S@J-1)**2)-self.volumes@((J-1)**2))
        nonlocal_k=None
        if k is not None:
            h=np.zeros((2,2,2,2));h[0,0,1,1]=h[1,1,0,0]=1.;h[0,1,1,0]=h[1,0,0,1]=-1.
            hessian=np.einsum('eaJ,iJkL,ebL->eaibk',grad,h,grad).reshape(-1,6,6)
            k=k+self.volumes[:,None,None]*(difference[:,None,None]*hessian-self.projected_fraction*self.K*dj[:,:,None]*dj[:,None,:])
            D=coo_matrix((dj.ravel(),(np.repeat(np.arange(len(J)),6),dofs.ravel())),shape=(len(J),mesh.dof_count)).tocsr()
            self.pressure_coupling=self.incidence@D
            self.pressure_mass=self.nodal_volumes/self.K
            if condensed:
                nonlocal_k=self.pressure_coupling.T@diags(self.projected_fraction/self.pressure_mass)@self.pressure_coupling
        return r,k,nonlocal_k,float(energy)
