"""Q1 plane-stress SENT and AT2 history-field reproduction (approach A).

Half-domain symmetry represents the 75 x 30 x 2 mm specimen with a 20 mm
geometrical slit. Applied full grip opening is U, half-domain grip motion U/2.
Q1 quadrilaterals differ from the paper's quadratic triangles; h is explicit.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .material import Material,response


@dataclass(frozen=True)
class Config:
    mesh_h_mm: float = .275
    ell_mm: float = .275
    gc_n_mm: float = 10.7
    thickness_mm: float = 2.
    residual_stiffness: float = 1e-10
    phase_viscosity: float = 1e-4
    speed_mm_min: float = 25.
    end_opening_mm: float = 22.
    increment_mm: float = .25
    minimum_increment_mm: float = .0005
    force_tolerance_n: float = 2e-5
    stagger_tolerance: float = 2e-4
    max_stagger: int = 1000

    def to_dict(self):return asdict(self)


def graded_axis(length,h,uniform_length):
    values=list(np.linspace(0,uniform_length,int(np.ceil(uniform_length/h))+1))
    step=h
    while values[-1]<length:
        step=min(step*1.25,2.)
        values.append(min(length,values[-1]+step))
    return np.array(values)


class Problem:
    def __init__(self,config=Config(),material=Material()):
        self.c,self.material=config,material
        x=graded_axis(37.5,config.mesh_h_mm,1.1)
        upper=np.linspace(18.9,30,int(np.ceil(11.1/config.mesh_h_mm))+1)
        lower=18.9-graded_axis(18.9,config.mesh_h_mm,1.1)[::-1]
        y=np.unique(np.r_[lower,upper,20.])
        xx,yy=np.meshgrid(x,y,indexing='ij');self.xy=np.column_stack((xx.ravel(),yy.ravel()))
        ids=np.arange(len(self.xy)).reshape(len(x),len(y))
        self.cells=np.stack((ids[:-1,:-1],ids[1:,:-1],ids[1:,1:],ids[:-1,1:]),axis=-1).reshape(-1,4)
        xi=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])/np.sqrt(3)
        signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        self.N=np.prod(1+xi[:,None,:]*signs[None,:,:],axis=2)/4
        grad=np.empty((4,4,2))
        grad[:,:,0]=signs[None,:,0]*(1+xi[:,None,1]*signs[None,:,1])/4
        grad[:,:,1]=signs[None,:,1]*(1+xi[:,None,0]*signs[None,:,0])/4
        jac=np.einsum('eai,qaJ->eqiJ',self.xy[self.cells],grad)
        self.G=np.einsum('qak,eqkJ->eqaJ',grad,np.linalg.inv(jac))
        self.w=np.linalg.det(jac)*config.thickness_mm
        assert np.all(self.w>0)
        self.dofs=(2*self.cells[:,:,None]+np.arange(2)).reshape(-1,8)
        self.right=np.flatnonzero(np.isclose(self.xy[:,0],37.5))
        symmetry=np.flatnonzero(np.isclose(self.xy[:,0],0)&(self.xy[:,1]>=20.-1e-9))
        self.fixed=np.unique(np.r_[2*symmetry,2*self.right,2*self.right+1])
        self.free=np.setdiff1d(np.arange(2*len(self.xy)),self.fixed)
        self.rr=np.broadcast_to(self.dofs[:,:,None],(len(self.cells),8,8)).ravel()
        self.cc=np.broadcast_to(self.dofs[:,None,:],(len(self.cells),8,8)).ravel()
        self.dr=np.broadcast_to(self.cells[:,:,None],(len(self.cells),4,4)).ravel()
        self.dc=np.broadcast_to(self.cells[:,None,:],(len(self.cells),4,4)).ravel()

    def initial(self):
        return dict(u=np.zeros(2*len(self.xy)),damage=np.zeros(len(self.xy)),
            history=np.zeros(self.w.shape),metric=np.tile(np.eye(3),self.w.shape+(1,1)),
            opening=0.,time=0.,viscous_dissipation=0.,phase_viscous_dissipation=0.)

    def mechanics(self,u,damage,old,dt,*,tangent=True):
        F=np.eye(2)+np.einsum('eai,eqaJ->eqiJ',u.reshape(-1,2)[self.cells],self.G)
        q=response(F.reshape(-1,2,2),old['metric'].reshape(-1,3,3),dt,self.material,tangent=tangent)
        dg=np.einsum('qa,ea->eq',self.N,damage[self.cells]);g=(1-self.c.residual_stiffness)*(1-dg)**2+self.c.residual_stiffness
        P=q['P'].reshape(self.w.shape+(2,2))*g[:,:,None,None]
        re=np.einsum('eq,eqiJ,eqaJ->eai',self.w,P,self.G).reshape(-1,8)
        r=np.zeros(len(u));np.add.at(r,self.dofs.ravel(),re.ravel())
        k=None
        if tangent:
            C=q['tangent'].reshape(self.w.shape+(2,2,2,2))
            ke=np.einsum('eq,eqaJ,eqiJkL,eqbL->eaibk',self.w*g,self.G,C,self.G,optimize=True).reshape(-1,8,8)
            k=coo_matrix((ke.ravel(),(self.rr,self.cc)),shape=(len(u),len(u))).tocsr()
        q['g']=g;q['damage_q']=dg
        return r,k,q

    def equilibrate(self,u,damage,old,opening,dt):
        u=u.copy();u[2*self.right]=opening/2;u[2*self.right+1]=0
        for iteration in range(40):
            r,k,q=self.mechanics(u,damage,old,dt)
            norm=np.linalg.norm(r[self.free])
            if norm<self.c.force_tolerance_n:return u,r,q,iteration
            du=np.zeros(len(u));du[self.free]=spsolve(k[self.free][:,self.free],-r[self.free])
            if not np.isfinite(du).all():raise RuntimeError('Nonfinite equilibrium direction')
            for backtrack in range(18):
                candidate=u+du*(.5**backtrack)
                try:
                    rr,_,_=self.mechanics(candidate,damage,old,dt,tangent=False)
                    if np.linalg.norm(rr[self.free])<norm*(1-1e-4*.5**backtrack):
                        u=candidate;break
                except (RuntimeError,FloatingPointError,np.linalg.LinAlgError):pass
            else:raise RuntimeError(f'Equilibrium line search failed U={opening:g}, residual={norm:g}')
        raise RuntimeError(f'Equilibrium iteration limit U={opening:g}, residual={norm:g}')

    def phase(self,history,old,dt):
        c=self.c;viscosity=c.phase_viscosity/(c.gc_n_mm*dt)
        H=2*(1-c.residual_stiffness)*history
        mass=np.einsum('eq,qa,qb->eab',self.w*(H+1/(2*c.ell_mm)+viscosity),self.N,self.N)
        grad=2*c.ell_mm*np.einsum('eq,eqai,eqbi->eab',self.w,self.G,self.G)
        previous=np.einsum('qa,ea->eq',self.N,old['damage'][self.cells])
        rhs=np.einsum('eq,qa->ea',self.w*(H+viscosity*previous),self.N)
        matrix=coo_matrix(((mass+grad).ravel(),(self.dr,self.dc)),shape=(len(self.xy),len(self.xy))).tocsr()
        b=np.zeros(len(self.xy));np.add.at(b,self.cells.ravel(),rhs.ravel())
        damage=spsolve(matrix,b)
        # Box-constrained solve prevents discretization overshoots/healing.
        lower=old['damage'];fixed_low=damage<lower;fixed_high=damage>1
        for _ in range(30):
            damage[fixed_low]=lower[fixed_low];damage[fixed_high]=1
            fixed=fixed_low|fixed_high;free=~fixed
            if fixed.any():damage[free]=spsolve(matrix[free][:,free],b[free]-matrix[free][:,fixed]@damage[fixed])
            residual=matrix@damage-b
            low=(damage<lower-1e-10)|(fixed_low&(residual>0))
            high=(damage>1+1e-10)|(fixed_high&(residual<0))
            if np.array_equal(low,fixed_low) and np.array_equal(high,fixed_high):break
            fixed_low,fixed_high=low,high
        else:raise RuntimeError('Phase bound active set failed')
        return np.clip(damage,lower,1.)

    def step(self,old,opening):
        dt=(opening-old['opening'])/(self.c.speed_mm_min/60)
        u=old['u'].copy();damage=old['damage'].copy();iterations=0
        for stagger in range(self.c.max_stagger):
            previous_u=u.copy()
            u,r,q,it=self.equilibrate(u,damage,old,opening,dt);iterations+=it
            H=np.maximum(old['history'],q['energy'].reshape(self.w.shape)/self.c.gc_n_mm)
            new_damage=self.phase(H,old,dt)
            change=float(np.max(np.abs(new_damage-damage)))
            uchange=float(np.max(np.abs(u-previous_u)))
            damage=new_damage
            if stagger%20==0:print(f'  stagger={stagger} U={opening:.7g} dd={change:.3g} du={uchange:.3g}',flush=True)
            if change<self.c.stagger_tolerance and uchange<self.c.stagger_tolerance:
                # Both original blocks must be satisfied at the committed state.
                u,r,q,it=self.equilibrate(u,damage,old,opening,dt);iterations+=it
                H=np.maximum(old['history'],q['energy'].reshape(self.w.shape)/self.c.gc_n_mm)
                checked=self.phase(H,old,dt)
                if np.max(np.abs(checked-damage))<self.c.stagger_tolerance:break
        else:raise RuntimeError(f'Stagger limit U={opening:g}, dd={change:g}')
        dq=np.einsum('qa,ea->eq',self.N,damage[self.cells])
        dd=np.einsum('eqaJ,ea->eqJ',self.G,damage[self.cells])
        fracture=2*self.c.gc_n_mm*np.sum(self.w*(dq*dq/(4*self.c.ell_mm)+self.c.ell_mm*(dd*dd).sum(axis=2)))
        bulk=2*np.sum(self.w*q['g']*q['energy'].reshape(self.w.shape))
        viscous=old['viscous_dissipation']+2*dt*np.sum(self.w*q['g']*q['viscous_rate'].reshape(self.w.shape))
        phase_increment=np.einsum('qa,ea->eq',self.N,(damage-old['damage'])[self.cells])
        phase_viscous=old.get('phase_viscous_dissipation',0.)+2*self.c.phase_viscosity/dt*np.sum(self.w*phase_increment**2)
        new=dict(u=u,damage=damage,history=H,metric=q['metric'].reshape(self.w.shape+(3,3)),
            opening=opening,time=old['time']+dt,viscous_dissipation=viscous,phase_viscous_dissipation=phase_viscous)
        row=dict(opening_mm=opening,time_s=new['time'],force_n=float(r[2*self.right].sum()),
            max_damage=float(damage.max()),fracture_energy_n_mm=float(fracture),bulk_energy_n_mm=float(bulk),
            viscous_dissipation_n_mm=float(viscous),equilibrium_residual_n=float(np.linalg.norm(r[self.free])),
            phase_viscous_dissipation_n_mm=float(phase_viscous),
            plane_stress_residual_mpa=float(np.max(np.abs(q['plane_stress_residual']))),
            plane_stress_scaled_residual=float(np.max(q['plane_stress_scaled_residual'])),
            min_J=float(q['J'].min()),stagger_iterations=stagger+1,newton_iterations=iterations,
            phase_fixed_point_error=float(np.max(np.abs(checked-damage))))
        return new,row
