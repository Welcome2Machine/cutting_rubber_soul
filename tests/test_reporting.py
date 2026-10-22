"""Reusable saved-history reporting checks; synthetic records are not FE results."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from mechanics.finite_strain import Material, response
from prototype.model_config import FEMConfig
from tools.qualify_intact_baseline import sha
from tools.visualize_ipc import load_frames
from tools.plot_experiment_angles import comparison


def saved_maxwell_fixture(folder):
    """Two affine triangles, with exact local Maxwell history and stored hashes."""
    (folder/'states').mkdir()
    material=Material((.14,),(2.,),(1.24,),(2.,),.49,1.7,
                      viscous_flow='isochoric_maxwell')
    reference=np.array([[0.,0.],[1.,0.],[0.,1.],[-.1,1.01],[1.1,1.01],[1.1,1.2]])
    triangles=np.array([[0,1,2]])
    metric=np.eye(3)[None]
    steps=[];audits=[];expected=[]
    for i,F in enumerate((np.eye(2),np.array([[.9,.2],[0.,1.05]]))):
        q=response(F[None],metric,.5,material,kinematics='plane_strain',tangent=False)
        positions=reference.copy();positions[:3]=reference[:3]@F.T
        path=folder/'states'/f'{i:04d}.npz'
        np.savez(path,reference_xy=reference,triangles=triangles,body_nodes=3,
                 positions=positions,depth_mm=i*.1,previous_material_metric=metric,
                 material_metric=q['metric'],material_dt_s=.5)
        steps.append(dict(depth_mm=i*.1,reaction_fz_n=.5+i,min_j=float(q['J'][0])))
        audits.append(dict(depth_mm=i*.1,passed=True,checkpoint_sha256=sha(path)))
        stress=np.zeros((3,3));stress[:2,:2]=q['P'][0]@F.T/q['J'][0]
        stress[2,2]=q['plane_stress_residual'][0]/q['J'][0]
        expected.append(max(float(np.linalg.eigvalsh(stress)[-1]),0.))
        metric=q['metric']
    root=Path(__file__).resolve().parents[1]
    report=dict(completed=True,status='completed',target_mm=.1,settings=dict(increment=.1),steps=steps,
        signature=dict(config=FEMConfig(inclination_deg=57).to_dict(),material=material.to_dict(),loading_rate=.2),
        source_sha256={'mechanics/finite_strain.py':sha(root/'mechanics/finite_strain.py')})
    (folder/'result.json').write_text(json.dumps(report),encoding='utf-8')
    (folder/'audit.json').write_text(json.dumps(dict(passed=True,states=2,rows=audits)),encoding='utf-8')
    return expected


class ReportingTests(unittest.TestCase):
    def test_maxwell_contours_use_saved_metric_and_transverse_stress(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);expected=saved_maxwell_fixture(folder)
            frames=load_frames(folder)[5]
            np.testing.assert_allclose([f['peak_mpa'] for f in frames],expected,rtol=1e-12,atol=1e-12)
            self.assertFalse((folder/'source').exists())

    def test_changed_checkpoint_is_rejected_before_stress_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);saved_maxwell_fixture(folder)
            with (folder/'states/0001.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed case input'):load_frames(folder)

    def test_explicit_comparison_preserves_original_zero_and_common_window(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);saved_maxwell_fixture(folder)
            experiment=folder/'experimental.csv'
            with experiment.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['Depth (μm)','Fz (N)','depth_unit','force_unit','curve_type','component','speed_um_s'])
                writer.writerows([[0,.7,'μm','N','experiment','Fz',200],[100,1.7,'μm','N','experiment','Fz',200]])
            summary,values=comparison(57,folder,experiment,2.)
            self.assertAlmostEqual(summary['comparison_end_mm'],.1)
            self.assertAlmostEqual(summary['rmse_n'],.2)
            self.assertEqual(values[-1][0],.7)
            self.assertEqual(values[-2][0],.5)
            with self.assertRaisesRegex(ValueError,'angle'):comparison(33,folder,experiment,2.)


if __name__=='__main__':unittest.main()
