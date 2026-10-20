"""Finite Ogden-Maxwell material, paper equations 23-24 and 58-61.

Reference units N, mm, s. Supports plane strain and total-stress plane stress.
Material parameters are supplied explicitly; no EPDM parameters are defaults.
The apparent over-stress-only condition printed in equation 62 is not used.
Every trial integrates from the previous accepted inverse viscous metric.
"""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class Material:
    mu_eq: tuple
    alpha_eq: tuple
    mu_ov: tuple = ()
    alpha_ov: tuple = ()
    poisson: float = .49
    relaxation_s: float | None = None
    volumetric: str = "quadratic_j"
    viscous_flow: str = "paper2023"

    def __post_init__(self):
        if not 0<self.poisson<.5:raise ValueError('Poisson ratio must lie in (0, .5)')
        for mu,alpha in ((self.mu_eq,self.alpha_eq),(self.mu_ov,self.alpha_ov)):
            if len(mu)!=len(alpha) or not np.all(np.isfinite([*mu,*alpha])) or any(a==0 for a in alpha):
                raise ValueError('Ogden coefficients must be paired, finite, with nonzero exponents')
        if not self.mu_eq or self.moduli()[0]<=0 or not 0<self.poisson<.5:
            raise ValueError('Positive equilibrium shear and Poisson ratio in (0, .5) required')
        if self.mu_ov and (self.moduli(True)[0]<=0 or self.relaxation_s is None or not np.isfinite(self.relaxation_s) or self.relaxation_s<=0):
            raise ValueError('A viscous branch needs positive shear and measured positive relaxation time')
        if self.volumetric not in ('quadratic_j','paper2023'):raise ValueError('Unknown volumetric energy')
        if self.viscous_flow not in ('paper2023','isochoric_maxwell'):
            raise ValueError('Unknown viscous flow law')

    def moduli(self,over=False):
        mu,alpha=(self.mu_ov,self.alpha_ov) if over else (self.mu_eq,self.alpha_eq)
        shear=.5*np.dot(mu,alpha)
        bulk=2*shear*(1+self.poisson)/(3*(1-2*self.poisson))
        if over and self.viscous_flow=='isochoric_maxwell':bulk=0.
        return shear,bulk

    def to_dict(self):return asdict(self)


def principal(e,material,over=False):
    """Energy, principal Kirchhoff stresses, and d(tau)/d(log stretches)."""
    mu,alpha=(material.mu_ov,material.alpha_ov) if over else (material.mu_eq,material.alpha_eq)
    _,bulk=material.moduli(over)
    total=e.sum(axis=-1);dev=e-total[...,None]/3
    expj=np.exp(2*total)
    energy=bulk/4*(np.expm1(2*total)-2*total)
    tau=np.broadcast_to((bulk/2*np.expm1(2*total))[...,None],e.shape).copy()
    h=np.broadcast_to((bulk*expj)[...,None,None],e.shape[:-1]+(3,3)).copy()
    if material.volumetric=="quadratic_j":
        J=np.exp(total)
        energy=bulk/2*np.expm1(total)**2
        tau=np.broadcast_to((bulk*J*np.expm1(total))[...,None],e.shape).copy()
        h=np.broadcast_to((bulk*J*(2*J-1))[...,None,None],e.shape[:-1]+(3,3)).copy()
    identity=np.eye(3)
    for m,a in zip(mu,alpha):
        values=np.exp(a*dev);mean=values.mean(axis=-1)
        energy+=m/a*(values.sum(axis=-1)-3)
        tau+=m*(values-mean[...,None])
        h+=m*a*(identity*values[...,None,:]-(values[...,None,:]+values[...,:,None])/3+mean[...,None,None]/3)
    return energy,tau,h


def elastic_response(F,material,*,tangent=True,kinematics='plane_strain',diagnostics=True):
    if kinematics=='plane_strain' and material.alpha_eq==(1.,) and material.volumetric=='quadratic_j':
        return _ogden_one_plane_strain(F,material,tangent=tangent,diagnostics=diagnostics)
    # Closed-form specialization of the SAME Ogden alpha=2 law. This avoids
    # spectral decompositions in the intact CB245NE elastic limit; all other
    # laws and plane-stress calculations retain the general spectral kernel.
    if kinematics!='plane_strain' or material.alpha_eq!=(2.,) or material.volumetric!='quadratic_j':
        return _spectral_elastic_response(F,material,tangent=tangent,kinematics=kinematics)
    F=np.asarray(F,dtype=float)
    if F.ndim!=3 or F.shape[1:]!=(2,2) or not np.isfinite(F).all():raise ValueError('Expected finite F(n,2,2)')
    J=F[:,0,0]*F[:,1,1]-F[:,0,1]*F[:,1,0]
    if np.any(J<=0):raise FloatingPointError('Inverted deformation')
    cofactor=np.empty_like(F)
    cofactor[:,0,0]=F[:,1,1];cofactor[:,1,1]=F[:,0,0]
    cofactor[:,0,1]=-F[:,1,0];cofactor[:,1,0]=-F[:,0,1]
    H=cofactor/J[:,None,None];I1=np.sum(F*F,axis=(1,2))+1
    G,K=material.moduli();a=J**(-2/3)
    P=G*a[:,None,None]*(F-I1[:,None,None]*H/3)+K*(J-1)[:,None,None]*cofactor
    result=dict(P=P,energy=G/2*(a*I1-3)+K/2*(J-1)**2,J=J)
    if diagnostics:
        # Assembly needs only P, energy, J and the tangent. Avoid allocating
        # diagnostic tensors/eigensystems at every quadrature point when the
        # caller explicitly opts out; the default preserves the public result.
        b=np.tile(np.eye(3),(len(F),1,1));b[:,:2,:2]=F@F.transpose(0,2,1)
        cauchy=G*(a/J)[:,None,None]*(b-I1[:,None,None]*np.eye(3)/3)+K*(J-1)[:,None,None]*np.eye(3)
        out=cauchy[:,2,2]*J
        result.update(lambda_out=np.ones(len(F)),cauchy=cauchy,plane_stress_residual=out,
            metric=np.tile(np.eye(3),(len(F),1,1)),viscous_rate=np.zeros(len(F)))
        # Preserve the diagnostic normalization used by the spectral kernel.
        eig=np.linalg.eigvalsh(b)
        values=a[:,None]*eig;mean=values.mean(axis=1)
        h=2*G*(np.eye(3)*values[:,None,:]-(values[:,None,:]+values[:,:,None])/3+mean[:,None,None]/3)
        h+=K*J[:,None,None]*(2*J-1)[:,None,None]
        result['plane_stress_scaled_residual']=np.abs(out)/np.maximum(1.,np.max(np.abs(h),axis=(1,2)))
    if tangent:
        outer=np.einsum('eiJ,ekL->eiJkL',H,H)
        cross=np.einsum('eiL,ekJ->eiJkL',H,H)
        identity=np.einsum('ik,JL->iJkL',np.eye(2),np.eye(2))
        C=G*a[:,None,None,None,None]*(identity-2/3*(np.einsum('eiJ,ekL->eiJkL',F,H)+np.einsum('eiJ,ekL->eiJkL',H,F))
            +2*I1[:,None,None,None,None]/9*outer+I1[:,None,None,None,None]/3*cross)
        C+=K*J[:,None,None,None,None]*((2*J-1)[:,None,None,None,None]*outer-(J-1)[:,None,None,None,None]*cross)
        result['tangent']=C.reshape(-1,4,4)
    return result


def _ogden_one_plane_strain(F,material,*,tangent=True,diagnostics=True):
    """Exact alpha=1 specialization; no principal-vector derivatives are needed.

    With F33=1 and J>0, the two in-plane stretches satisfy
    lambda1+lambda2 = sqrt(F:F+2*J). Thus the isochoric energy is
    2*G*(J**(-1/3)*(lambda1+lambda2+1)-3). The tangent below is its exact
    Hessian, including the unchanged K/2*(J-1)**2 volumetric energy.
    This is an optional constitutive law, not a fitted rubber model.
    """
    F=np.asarray(F,dtype=float)
    if F.ndim!=3 or F.shape[1:]!=(2,2) or not np.isfinite(F).all():
        raise ValueError('Expected finite F(n,2,2)')
    J=F[:,0,0]*F[:,1,1]-F[:,0,1]*F[:,1,0]
    if np.any(J<=0):raise FloatingPointError('Inverted deformation')
    cofactor=np.empty_like(F)
    cofactor[:,0,0]=F[:,1,1];cofactor[:,1,1]=F[:,0,0]
    cofactor[:,0,1]=-F[:,1,0];cofactor[:,1,0]=-F[:,0,1]
    H=cofactor/J[:,None,None]
    S=np.sqrt(np.sum(F*F,axis=(1,2))+2*J)
    total=S+1
    U=(F+cofactor)/S[:,None,None]
    G,K=material.moduli();a=J**(-1/3)
    P=2*G*a[:,None,None]*(U-total[:,None,None]*H/3)+K*(J-1)[:,None,None]*cofactor
    result=dict(P=P,energy=2*G*(a*total-3)+K/2*(J-1)**2,J=J)
    if diagnostics:
        cauchy=np.zeros((len(F),3,3))
        cauchy[:,:2,:2]=(P@F.transpose(0,2,1))/J[:,None,None]
        out=2*G*a*(1-total/3)+K*J*(J-1)
        cauchy[:,2,2]=out/J
        result.update(lambda_out=np.ones(len(F)),cauchy=cauchy,plane_stress_residual=out,
            metric=np.tile(np.eye(3),(len(F),1,1)),viscous_rate=np.zeros(len(F)))
        # Match the general kernel's diagnostic normalization; these spectral
        # values are intentionally absent from the assembly-only path.
        eig=np.linalg.eigvalsh(F@F.transpose(0,2,1))
        values=a[:,None]*np.column_stack((np.sqrt(eig),np.ones(len(F))))
        mean=values.mean(axis=1)
        h=2*G*(np.eye(3)*values[:,None,:]-(values[:,None,:]+values[:,:,None])/3+mean[:,None,None]/3)
        h+=K*J[:,None,None]*(2*J-1)[:,None,None]
        result['plane_stress_scaled_residual']=np.abs(out)/np.maximum(1.,np.max(np.abs(h),axis=(1,2)))
    if tangent:
        identity=np.einsum('ik,JL->iJkL',np.eye(2),np.eye(2))
        # Hessian of det(F), equivalently derivative of cofactor(F), in 2D.
        determinant_hessian=np.zeros((2,2,2,2))
        determinant_hessian[0,0,1,1]=determinant_hessian[1,1,0,0]=1.
        determinant_hessian[0,1,1,0]=determinant_hessian[1,0,0,1]=-1.
        outer=np.einsum('eiJ,ekL->eiJkL',H,H)
        cross=np.einsum('eiL,ekJ->eiJkL',H,H)
        du=(identity+determinant_hessian-np.einsum('eiJ,ekL->eiJkL',U,U))/S[:,None,None,None,None]
        C=2*G*a[:,None,None,None,None]*(du
            -(np.einsum('eiJ,ekL->eiJkL',U,H)+np.einsum('eiJ,ekL->eiJkL',H,U))/3
            +total[:,None,None,None,None]*(outer/9+cross/3))
        C+=K*J[:,None,None,None,None]*((2*J-1)[:,None,None,None,None]*outer-(J-1)[:,None,None,None,None]*cross)
        result['tangent']=C.reshape(-1,4,4)
    return result


def _neo_hookean_plane_stress_log_stretch(e2,material):
    """Safeguarded local thickness solve for positive neo-Hookean/quad-J.

    The isochoric tau33 vanishes at z^2=(lambda1^2+lambda2^2)/2;
    the volumetric term vanishes at z=1/det(F2). These two points
    bracket the unique positive thickness root. Curvature in log(z) can
    be negative before reaching that root, so clipped Newton is unsafe.
    This bracket is specific to alpha=2, not a general Ogden guarantee.
    """
    q_iso=.5*(np.logaddexp(2*e2[:,0],2*e2[:,1])-np.log(2.))
    q_vol=-e2.sum(axis=1)
    lo=np.minimum(q_iso,q_vol);hi=np.maximum(q_iso,q_vol)
    q=np.clip(np.zeros(len(e2)),lo,hi)
    for _ in range(80):
        _,t,h=principal(np.column_stack((e2,q)),material)
        tau=t[:,2];curvature=h[:,2,2]
        done=np.abs(tau)<=1e-9+64*np.finfo(float).eps*np.max(np.abs(h),axis=(1,2))
        if np.all(done):
            if np.any(curvature<=0):
                raise RuntimeError('Unstable elastic plane-stress thickness root')
            return q
        lo=np.where((tau<0)&~done,q,lo)
        hi=np.where((tau>0)&~done,q,hi)
        correction=np.zeros_like(q)
        np.divide(tau,curvature,out=correction,where=curvature>0)
        trial=q-correction
        usable=(curvature>0)&np.isfinite(trial)&(trial>lo)&(trial<hi)
        q=np.where(done,q,np.where(usable,trial,.5*(lo+hi)))
    raise RuntimeError('Elastic plane-stress bracket solve failed')


def _spectral_elastic_response(F,material,*,tangent=True,kinematics='plane_strain'):
    """Equilibrium Ogden response; plane strain fixes F33=1, not stress33=0."""
    F=np.asarray(F,dtype=float)
    if F.ndim!=3 or F.shape[1:]!=(2,2) or not np.isfinite(F).all():raise ValueError('Expected finite F(n,2,2)')
    det=np.linalg.det(F)
    if np.any(det<=0):raise FloatingPointError('Inverted deformation')
    if kinematics not in ('plane_strain','plane_stress'):raise ValueError('Unknown kinematics')
    eig,Q=np.linalg.eigh(F@F.transpose(0,2,1));e2=.5*np.log(eig)
    q=(_neo_hookean_plane_stress_log_stretch(e2,material)
       if kinematics=='plane_stress' and material.alpha_eq==(2.,)
       and material.mu_eq[0]>0 and material.volumetric=='quadratic_j'
       else np.zeros(len(F)))
    for _ in range(60):
        w,t,h=principal(np.column_stack((e2,q)),material)
        if kinematics=='plane_strain' or np.all(np.abs(t[:,2])<=1e-9+64*np.finfo(float).eps*np.max(np.abs(h),axis=(1,2))):break
        q-=np.clip(t[:,2]/h[:,2,2],-.4,.4)
    else:raise RuntimeError('Elastic plane-stress solve failed')
    tau=(Q*t[:,None,:2])@Q.transpose(0,2,1);inverse=np.linalg.inv(F);P=tau@inverse.transpose(0,2,1)
    J=det*np.exp(q);cauchy=np.zeros((len(F),3,3));cauchy[:,:2,:2]=tau/J[:,None,None];cauchy[:,2,2]=t[:,2]/J
    result=dict(P=P,energy=w,J=J,lambda_out=np.exp(q),cauchy=cauchy,
        plane_stress_residual=t[:,2],plane_stress_scaled_residual=np.abs(t[:,2])/np.maximum(1.,np.max(np.abs(h),axis=(1,2))),
        metric=np.tile(np.eye(3),(len(F),1,1)),viscous_rate=np.zeros(len(F)))
    if not tangent:return result
    directions=np.eye(4).reshape(4,2,2)
    db=directions[None]@F[:,None].transpose(0,1,3,2)+F[:,None]@directions.transpose(0,2,1)[None]
    dbq=Q[:,None].transpose(0,1,3,2)@db@Q[:,None]
    de2=np.diagonal(dbq,axis1=2,axis2=3)/(2*eig[:,None,:])
    dq=np.zeros((len(F),4)) if kinematics=='plane_strain' else -np.einsum('ni,nki->nk',h[:,2,:2],de2)/h[:,2,2,None]
    dt=np.einsum('nij,nkj->nki',h,np.concatenate((de2,dq[:,:,None]),axis=2))
    gap=eig[:,0]-eig[:,1];coefficient=(h[:,0,0]-h[:,0,1])/(2*eig[:,0])
    np.divide(t[:,0]-t[:,1],gap,out=coefficient,where=np.abs(gap)>1e-8*np.max(eig,axis=1))
    local=np.zeros((len(F),4,2,2));local[:,:,0,0]=dt[:,:,0];local[:,:,1,1]=dt[:,:,1]
    local[:,:,0,1]=coefficient[:,None]*dbq[:,:,0,1];local[:,:,1,0]=local[:,:,0,1]
    dtau=Q[:,None]@local@Q[:,None].transpose(0,1,3,2)
    dP=dtau@inverse[:,None].transpose(0,1,3,2)-P[:,None]@directions.transpose(0,2,1)[None]@inverse[:,None].transpose(0,1,3,2)
    result['tangent']=dP.reshape(len(F),4,4).transpose(0,2,1)
    return result


def response(F,previous_metric,dt,material,*,tangent=True,kinematics="plane_stress"):
    """Batched 2-D F -> plane-stress P, consistent tangent and trial history."""
    F=np.asarray(F,dtype=float);A=np.asarray(previous_metric,dtype=float)
    if F.ndim!=3 or F.shape[1:]!=(2,2) or A.shape!=(len(F),3,3) or dt<=0:
        raise ValueError('Expected batched F(n,2,2), inverse Cv(n,3,3), positive dt')
    if kinematics not in ("plane_stress","plane_strain"):raise ValueError("Unknown kinematics")
    if not material.mu_ov:return elastic_response(F,material,tangent=tangent,kinematics=kinematics)
    determinant=np.linalg.det(F)
    if np.any(determinant<=0):raise FloatingPointError('Inverted in-plane deformation')
    b=F@F.transpose(0,2,1);bv=F@A[:,:2,:2]@F.transpose(0,2,1)
    eig,Q=np.linalg.eigh(b);ev,V=np.linalg.eigh(bv)
    if np.any(ev<=0):raise FloatingPointError('Nonpositive viscous trial metric')
    eq2=.5*np.log(eig);tr2=.5*np.log(ev)
    q=-np.log(determinant) if kinematics=="plane_stress" else np.zeros(len(F))
    e=np.column_stack((tr2,q+.5*np.log(A[:,2,2])))
    shear,bulk=material.moduli(True)
    if material.viscous_flow=='isochoric_maxwell':
        # Positive deviatoric mobility only: no unmeasured bulk relaxation.
        # Here relaxation_s is the actual infinitesimal shear relaxation time:
        # tau_dev=2*G*e_dev, de_vis/dt=tau_dev/(2*G*relaxation_s).
        L=dt*(np.eye(3)-np.ones((3,3))/3)/(2*shear*material.relaxation_s)
    else:
        # Preserve the original 2023 reproduction and its parameter convention.
        eta_i=2*shear*material.relaxation_s;eta_v=bulk*material.relaxation_s
        L=dt*((np.eye(3)-np.ones((3,3))/3)/(2*eta_i)+np.ones((3,3))/(9*eta_v))
    local=np.empty((len(F),4,4));r=np.empty((len(F),4))
    for iteration in range(35):
        eeq=np.column_stack((eq2,q))
        weq,teq,heq=principal(eeq,material)
        wov,tov,hov=principal(e,material,True)
        trial=np.column_stack((tr2,q+.5*np.log(A[:,2,2])))
        r[:,0]=teq[:,2]+tov[:,2]
        r[:,1:]=e-trial+np.einsum('ij,nj->ni',L,tov)
        local[:,0,0]=heq[:,2,2];local[:,0,1:]=hov[:,2,:]
        local[:,1:,0]=np.array([0.,0.,-1.])
        local[:,1:,1:]=np.eye(3)+L@hov
        if kinematics=="plane_strain":
            r[:,0]=q;local[:,0,:]=0.;local[:,0,0]=1.
        # Crack-band stretches can amplify the branch tangents by many orders
        # of magnitude. Account for attainable floating-point cancellation;
        # a fixed 1e-9 MPa threshold falsely rejects already converged trials.
        stress_scale=np.maximum(np.max(np.abs(heq),axis=(1,2)),np.max(np.abs(hov),axis=(1,2)))
        stress_tolerance=1e-9+64*np.finfo(float).eps*stress_scale
        strain_tolerance=1e-11*np.maximum(1.,np.max(np.abs(trial),axis=1))
        if np.all(np.abs(r[:,0])<=stress_tolerance) and np.all(np.max(np.abs(r[:,1:]),axis=1)<=strain_tolerance):break
        delta=np.linalg.solve(local,-r[...,None])[...,0]
        fraction=np.minimum(1.,.4/np.maximum(np.max(np.abs(delta),axis=1),1e-30))
        q+=fraction*delta[:,0];e+=fraction[:,None]*delta[:,1:]
    else:raise RuntimeError(f'Local material solve failed, max residual={np.max(np.abs(r)):.6g}')
    tau=(Q*teq[:,None,:2])@Q.transpose(0,2,1)+(V*tov[:,None,:2])@V.transpose(0,2,1)
    inverse=np.linalg.inv(F);P=tau@inverse.transpose(0,2,1)
    elastic_b=(V*np.exp(2*e[:,:2])[:,None,:])@V.transpose(0,2,1)
    new_metric=np.zeros_like(A)
    new_metric[:,:2,:2]=inverse@elastic_b@inverse.transpose(0,2,1)
    new_metric[:,2,2]=np.exp(2*(e[:,2]-q))
    viscous_rate=np.einsum('ni,ij,nj->n',tov,L/dt,tov)
    result=dict(P=P,energy=weq+wov,metric=new_metric,lambda_out=np.exp(q),
        J=determinant*np.exp(q),plane_stress_residual=teq[:,2]+tov[:,2],viscous_rate=viscous_rate,
        plane_stress_scaled_residual=np.abs(teq[:,2]+tov[:,2])/np.maximum(1.,stress_scale))
    # Reduced local incremental potential: elastic energy plus quadratic
    # viscous distance. Its F derivative is P; stored energy alone is not.
    # On the converged return map e-trial=-L*tau, the distance is tau.L.tau/2.
    result['incremental_potential']=weq+wov+.5*dt*viscous_rate
    result['equilibrium_energy']=weq
    result['overstress_energy']=wov
    if not tangent:return result
    directions=np.eye(4).reshape(4,2,2)
    db=directions[None]@F[:,None].transpose(0,1,3,2)+F[:,None]@directions.transpose(0,2,1)[None]
    dbv=directions[None]@A[:,None,:2,:2]@F[:,None].transpose(0,1,3,2)+F[:,None]@A[:,None,:2,:2]@directions.transpose(0,2,1)[None]
    dbq=Q[:,None].transpose(0,1,3,2)@db@Q[:,None]
    dbv_local=V[:,None].transpose(0,1,3,2)@dbv@V[:,None]
    deq2=np.diagonal(dbq,axis1=2,axis2=3)/(2*eig[:,None,:])
    detr2=np.diagonal(dbv_local,axis1=2,axis2=3)/(2*ev[:,None,:])
    forcing=np.zeros((len(F),4,4))
    forcing[:,0,:]=-np.einsum('ni,nki->nk',heq[:,2,:2],deq2)
    if kinematics=="plane_strain":forcing[:,0,:]=0.
    forcing[:,1:3,:]=detr2.transpose(0,2,1)
    changes=np.linalg.solve(local,forcing)
    deq=np.concatenate((deq2,changes[:,0,:,None]),axis=2)
    dt_eq=np.einsum('nij,nkj->nki',heq,deq)
    dt_ov=np.einsum('nij,njk->nki',hov,changes[:,1:,:])
    # Spectral divided differences, with analytic limits at repeated eigenvalues.
    gap=eig[:,0]-eig[:,1];gapv=ev[:,0]-ev[:,1]
    ceq=(heq[:,0,0]-heq[:,0,1])/(2*eig[:,0])
    gov=hov@np.linalg.inv(np.eye(3)+L@hov)
    cov=(gov[:,0,0]-gov[:,0,1])/(2*ev[:,0])
    np.divide(teq[:,0]-teq[:,1],gap,out=ceq,where=np.abs(gap)>1e-8*np.max(eig,axis=1))
    np.divide(tov[:,0]-tov[:,1],gapv,out=cov,where=np.abs(gapv)>1e-8*np.max(ev,axis=1))
    matrices=[]
    for basis,diagonal,off in [(Q,dt_eq,ceq[:,None]*dbq[:,:,0,1]),(V,dt_ov,cov[:,None]*dbv_local[:,:,0,1])]:
        local_stress=np.zeros((len(F),4,2,2))
        local_stress[:,:,0,0]=diagonal[:,:,0];local_stress[:,:,1,1]=diagonal[:,:,1]
        local_stress[:,:,0,1]=off;local_stress[:,:,1,0]=off
        matrices.append(basis[:,None]@local_stress@basis[:,None].transpose(0,1,3,2))
    dtau=matrices[0]+matrices[1]
    dP=dtau@inverse[:,None].transpose(0,1,3,2)-P[:,None]@directions.transpose(0,2,1)[None]@inverse[:,None].transpose(0,1,3,2)
    result['tangent']=dP.reshape(len(F),4,4).transpose(0,2,1)
    return result
