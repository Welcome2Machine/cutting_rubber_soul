import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from prototype.contact_study import bounded_path
from prototype.model_config import FEMConfig
from prototype.initiation import IntactProblem,InitiationOptions,ContactSolveError


class ContactStudyTests(unittest.TestCase):
    def test_iteration_limit_reports_saved_trial_residual(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,max_contact_iterations=2),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ContactSolveError) as caught:
            p._mixed_step(u,0.,.01,.05,state)
        exc=caught.exception
        r,_,q,_=p.assemble(exc.u,u,.01,0.,.05,exc.multipliers[:,0],exc.multipliers[:,1],mixed=True)
        norm=np.linalg.norm(r[p.free])
        defect=np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-exc.multipliers))
        self.assertIn(f'residual={norm:.6g} N, projection={defect:.6g} MPa',str(exc))

    def test_saved_failure_merges_with_scheduled_float_without_tiny_step(self):
        from prototype.loading import loading_depths,recovery_targets
        c=FEMConfig(approach_steps=481)
        scheduled=[d for d,_ in loading_depths(c,.00625,stop_mm=.4) if d>.375]
        merged=recovery_targets(c,.00625,.375,.4,.38125)
        self.assertEqual(merged,scheduled)
        self.assertGreater(min(np.diff(np.r_[.375,merged])),.006)
        inserted=recovery_targets(c,.00625,.375,.4,.378)
        self.assertEqual(inserted[0],.378)
    def test_zero_time_or_backward_loading_targets_are_rejected_before_output(self):
        for targets in ([],[.1],[np.nextafter(.1,1.)],[.2,.15],[float('nan')]):
            with self.assertRaisesRegex(ValueError,'strictly exceed'):
                bounded_path(None,None,None,.1,targets,Path('unused'),adaptive=True)

    def test_failed_snapshot_labels_actual_recursive_increment(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,min_adaptive_increment_mm=.001),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        def solve(previous,old,depth,dt,history):
            if depth>.00625:
                raise ContactSolveError(f'synthetic failure depth={depth}',previous,history)
            return previous.copy(),history.copy(),dict(reaction_fz_n=0.,residual_n=0.)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(p,'step',side_effect=solve),contextlib.redirect_stdout(io.StringIO()):
                report=bounded_path(p,u,state,0.,[.01],out,adaptive=True)
            self.assertFalse(report['completed'])
            self.assertAlmostEqual(report['last_depth_mm'],.00625)
            self.assertAlmostEqual(report['failed_depth_mm'],.0075)
            with np.load(out/'failed_trial.npz') as z:
                self.assertAlmostEqual(float(z['indentation_mm']),.0075)


if __name__=='__main__':unittest.main()
