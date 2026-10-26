import unittest
import numpy as np
from reproductions.dammass2023.material import Material,principal,response
from reproductions.dammass2023.solver import Config,Problem


class Paper2023Tests(unittest.TestCase):
    def test_crack_band_precision_regression(self):
        # Fixed large-stretch return-map regression, independent of local data.
        F=np.array([[[4.056473675647802,-84.32662864044048],
                     [-0.13744870883239102,17.46267124249576]]])
        A=np.array([[[1.0635476362706662,.6637632208369567,0.],
                     [.6637632208369567,.644419892648346,0.],
                     [0.,0.,1.3169280455511032]]])
        dt=.0046875
        q=response(F,A,dt)
        self.assertGreater(np.max(np.abs(F)),70.)
        self.assertLess(np.max(q['plane_stress_scaled_residual']),1e-12)
        for i in range(4):
            perturbation=np.eye(4)[i].reshape(2,2)*1e-5
            fd=(response(F+perturbation,A,dt,tangent=False)['P']-response(F-perturbation,A,dt,tangent=False)['P'])/2e-5
            np.testing.assert_allclose(fd.reshape(-1,4),q['tangent'][:,:,i],rtol=1e-4,atol=1e-3)

    def test_energy_stress_and_hessian(self):
        m=Material();e=np.array([[.1,-.2,.03],[0.,0.,0.]])
        for over in (False,True):
            w,t,h=principal(e,m,over)
            for i in range(3):
                d=np.eye(3)[i]*1e-6
                a,ta,_=principal(e+d,m,over);b,tb,_=principal(e-d,m,over)
                np.testing.assert_allclose((a-b)/2e-6,t[:,i],atol=1e-8)
                np.testing.assert_allclose((ta-tb)/2e-6,h[:,:,i],atol=1e-7)

    def test_plane_stress_algorithmic_tangent(self):
        F=np.array([np.eye(2),[[1.3,.2],[-.1,.85]],[[.9,0],[0,.9]]])
        A=np.tile(np.eye(3),(3,1,1));A[1,:2,:2]=[[.9,.1],[.1,1.2]]
        result=response(F,A,.7)
        self.assertLess(np.max(np.abs(result['plane_stress_residual'])),1e-9)
        for i in range(4):
            d=np.eye(4)[i].reshape(2,2)*1e-6
            fd=(response(F+d,A,.7,tangent=False)['P']-response(F-d,A,.7,tangent=False)['P'])/2e-6
            np.testing.assert_allclose(fd.reshape(3,4),result['tangent'][:,:,i],atol=2e-7,rtol=1e-6)

    def test_rigid_rotation_objectivity(self):
        F=np.array([[[1.3,.2],[-.1,.85]]]);A=np.eye(3)[None]
        theta=.73;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a=response(F,A,.3);b=response(Q@F,A,.3)
        np.testing.assert_allclose(b['P'],Q@a['P'],atol=1e-10)
        np.testing.assert_allclose(b['energy'],a['energy'],atol=1e-10)
        np.testing.assert_allclose(b['metric'],a['metric'],atol=1e-10)

    def test_relaxation_reduces_stress(self):
        F=np.array([[[1.5,0],[0,.85]]]);A=np.eye(3)[None]
        first=response(F,A,.01)
        current=first
        for _ in range(50):current=response(F,current['metric'],.5,tangent=False)
        self.assertLess(current['P'][0,0,0],first['P'][0,0,0])
        self.assertGreater(current['P'][0,0,0],0)

    def test_uniform_phase_solution_and_irreversibility(self):
        p=Problem(Config(mesh_h_mm=1.));old=p.initial();H=np.full(p.w.shape,.2);dt=.1
        d=p.phase(H,old,dt);c=p.c
        expected=.4*(1-c.residual_stiffness)/(.4*(1-c.residual_stiffness)+1/(2*c.ell_mm)+c.phase_viscosity/(c.gc_n_mm*dt))
        np.testing.assert_allclose(d,expected,atol=1e-12)
        old['damage']=d
        np.testing.assert_allclose(p.phase(H/10,old,dt),d,atol=1e-12)

    def test_assembled_mechanical_jacobian_and_reaction(self):
        p=Problem(Config(mesh_h_mm=1.));old=p.initial()
        u=np.column_stack((.1*p.xy[:,0],-.03*p.xy[:,1])).ravel()
        damage=.1+.05*p.xy[:,1]/30
        r,k,_=p.mechanics(u,damage,old,.2)
        direction=np.random.default_rng(481).normal(size=len(u))
        direction/=np.max(np.abs(direction));epsilon=1e-7
        plus=p.mechanics(u+epsilon*direction,damage,old,.2,tangent=False)[0]
        minus=p.mechanics(u-epsilon*direction,damage,old,.2,tangent=False)[0]
        np.testing.assert_allclose((plus-minus)/(2*epsilon),k@direction,rtol=2e-6,atol=1e-6)
        self.assertLess(np.linalg.norm((k-k.T).data),1e-8)
        np.testing.assert_allclose(r.reshape(-1,2).sum(axis=0),0,atol=1e-10)


if __name__=='__main__':unittest.main()
