"""True failed predictors and coordinate-roundoff perturbations; raw acceptance."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import numpy as np
from prototype.model_config import FEMConfig,FEMMesh
from prototype.initiation import IntactProblem,InitiationOptions


class ContactRoundoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Path(__file__).parent/'fixtures/contact_predictor_roundoff.npz').is_file():
            raise unittest.SkipTest('Local-only roundoff state is not distributed; these regressions were not run')
        with np.load(Path(__file__).parent/'fixtures/contact_predictor_roundoff.npz') as z:
            cls.data={k:z[k] for k in z.files}
        z=cls.data;p=IntactProblem.__new__(IntactProblem)
        p.config=FEMConfig(**json.loads(str(z['config_json'])))
        p.options=InitiationOptions(**json.loads(str(z['options_json'])));p.strength_surface=None
        xy,sn,tri=z['reference_xy'],z['local_sn'],z['triangles']
        p.mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
            np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
        p._rebuild_boundary(z['boundary_edges']);cls.problem=p

    def test_real_predictors_and_small_perturbations_solve_original_equations(self):
        p,z=self.problem,self.data;forces=[]
        for source,perturb in (('main',0.),('diagnostic',0.),('main',1.),('main',-1.)):
            with self.subTest(source=source,perturb=perturb):
                previous=z[source+'_previous'].copy();history=z[source+'_multipliers'].copy()
                saved_u=previous.copy();saved_lam=history.copy();seed=z[source+'_seed'].copy()
                # Perturb the actual boundary node that flipped the old solver.
                node=3285;point=p.mesh.reference_xy[node]+seed[2*node:2*node+2]
                seed[2*node:2*node+2]+=perturb*4*np.spacing(np.maximum(1.,np.abs(point)))
                old,depth,dt=(float(z[k]) for k in ('old_depth','depth','dt'))
                with contextlib.redirect_stdout(io.StringIO()):
                    new,lam,d=p._mixed_step(previous,old,depth,dt,history,initial=seed)
                r,_,q,_=p.assemble(new,previous,depth,old,dt,lam[:,0],lam[:,1],roundoff_guard=False)
                self.assertLessEqual(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
                self.assertLessEqual(np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-lam)),p.config.contact_multiplier_tolerance_n_mm2)
                self.assertLessEqual(p.edge_penetration(new,depth),p.config.contact_penetration_tolerance_mm)
                self.assertGreaterEqual(lam[:,0].min(),0.)
                self.assertTrue(np.all(np.abs(lam[:,1])<=p.config.friction_coefficient*lam[:,0]))
                self.assertGreaterEqual(d['friction_dissipation_n_mm'],-d['friction_dissipation_tolerance_n_mm'])
                np.testing.assert_array_equal(previous,saved_u);np.testing.assert_array_equal(history,saved_lam)
                forces.append(d['reaction_fz_n'])
        self.assertLess(np.ptp(forces),1e-8)

    def test_roundoff_branch_does_not_suppress_resolved_penetration(self):
        p,z=self.problem,self.data;u=z['main_seed'];previous=z['main_previous'];lam=z['main_multipliers']
        points=(p.mesh.reference_xy+u.reshape(-1,2))[p.contact_nodes]
        old_points=(p.mesh.reference_xy+previous.reshape(-1,2))[p.contact_nodes]
        depth,old=float(z['depth']),float(z['old_depth']);increment=np.array([0.,old-depth])
        guarded=p.contact(points,old_points,depth,increment,lam[:,0],lam[:,1])
        raw=p.contact(points,old_points,depth,increment,lam[:,0],lam[:,1],roundoff_guard=False)
        index=int(np.flatnonzero(p.contact_nodes==3285)[0])
        self.assertGreater(raw['pressure'][index],0.)
        self.assertEqual(guarded['pressure'][index],0.)
        self.assertLess(guarded['gap_roundoff_mm'].max(),1e-10)
        penetrated=points.copy();penetrated[index]-=1e-6*raw['normal'][index]
        a=p.contact(penetrated,old_points,depth,increment,lam[:,0],lam[:,1])
        b=p.contact(penetrated,old_points,depth,increment,lam[:,0],lam[:,1],roundoff_guard=False)
        self.assertGreater(a['pressure'][index],0.)
        self.assertEqual(a['pressure'][index],b['pressure'][index])


if __name__=='__main__':unittest.main()
