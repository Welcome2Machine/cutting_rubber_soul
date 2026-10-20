import json
from pathlib import Path
import tempfile
import unittest

from tools.run_material_queue import make_command, compare_mesh_case


class NativeDiagnosticTests(unittest.TestCase):
    def test_mesh_command_cannot_silently_extend_scope(self):
        m=dict(python='python',solve_seconds_per_case=3600,experiment='wrong_angle.csv')
        j=dict(diagnostic='native_mesh',angle=33,target_mm=.5,level=1,radius_mm=1.,output='new',solve_seconds=7200)
        cmd=make_command(m,j)
        self.assertIn('tools.check_native_mesh',cmd)
        self.assertEqual(cmd[cmd.index('--seconds')+1],'7200')
        self.assertNotIn('--experiment',cmd)
        with self.assertRaises(ValueError):make_command(m,dict(j,target_mm=2.))

    def test_partial_mesh_result_keeps_missing_contact_and_same_angle(self):
        sig=dict(config=dict(inclination_deg=33),material={'G':.14},loading_rate=.2,
                 contact_formulation='improved-max',dhat=.0025,physical_force_tolerance_n=1e-4)
        def row(d,f,status):
            return dict(depth_mm=d,reaction_fz_n=f,min_j=.5,
                        tip_contact=dict(status=status,reference_offset_mm=.1 if status=='unique' else None))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';candidate=root/'candidate';base.mkdir();candidate.mkdir()
            reference=dict(signature=sig,completed=True,steps=[row(0,.05,'unique'),row(.5,9.,'unique')])
            new=dict(signature=sig,completed=False,steps=[row(0,.05,'unique'),row(.25,4.,'no_tip_contact')])
            (base/'result.json').write_text(json.dumps(reference))
            (candidate/'result.json').write_text(json.dumps(new))
            job=dict(output=str(candidate),baseline=str(base),target_mm=.5)
            answer=compare_mesh_case(job)
            self.assertFalse(answer['full_target_available'])
            saved=json.loads((candidate/'mesh_comparison.json').read_text())
            self.assertIsNone(saved['rows'][-1]['reference_offset_difference_mm'])
            new['signature']=dict(sig,config=dict(inclination_deg=57))
            (candidate/'result.json').write_text(json.dumps(new))
            with self.assertRaises(ValueError):compare_mesh_case(job)


if __name__=='__main__':unittest.main()
