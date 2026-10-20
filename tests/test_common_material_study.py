"""Physical-comparison and bounded-selection safeguards, no FE truth surrogate."""
import unittest
from unittest.mock import Mock
import numpy as np
from scipy.sparse import eye
from tools.run_common_material_study import (BASE,PROBES,ANGLES,coordinates,decode,
    joint_proposal,features,metrics,promising,schedule,ComputeMaterial)


class CommonMaterialTests(unittest.TestCase):
    def test_log_coordinates_keep_positive_common_moduli_and_roundtrip(self):
        for value in (BASE,*PROBES.values()):
            np.testing.assert_allclose(decode(coordinates(value)),value,rtol=2e-15)
        for value in ((0.,1.,1.),(1.,-1.,1.),(1.,1.,float('nan'))):
            with self.assertRaises(ValueError):coordinates(value)

    def test_proposal_recovers_joint_interior_weights_without_angle_parameters(self):
        control=np.zeros(6)
        probes={k:np.eye(6)[:,i] for i,k in enumerate(PROBES)}
        expected=np.array([.2,.3,.1]);target=sum(expected[i]*probes[k] for i,k in enumerate(PROBES))
        result=joint_proposal(control,probes,target)
        np.testing.assert_allclose(result['weights'],expected,atol=1e-7)
        self.assertLessEqual(sum(result['weights']),1.+1e-10)
        self.assertEqual(len(result['parameters']),3)
        self.assertLess(result['predicted_residual_norm'],1e-6)

    def test_unrepresentable_target_stays_in_sampled_tetrahedron(self):
        control=np.zeros(6);probes={k:np.eye(6)[:,i] for i,k in enumerate(PROBES)}
        result=joint_proposal(control,probes,np.ones(6)*10)
        self.assertTrue(np.all(np.asarray(result['weights'])>=-1e-8))
        self.assertLessEqual(sum(result['weights']),1.+1e-8)
        self.assertGreater(result['predicted_residual_norm'],10)

    def test_equal_angle_force_weighting_and_explicit_increment_features(self):
        d=np.arange(81)/100;curves={a:(d,d*10+a/10) for a in ANGLES}
        base=features(curves)
        for a in ANGLES:
            changed=dict(curves);changed[a]=(d,curves[a][1]+1)
            self.assertAlmostEqual(float(np.linalg.norm(features(changed)-base)**2),1/3,places=12)
        short=dict(curves);short[57]=(d[:30],d[:30])
        with self.assertRaises(ValueError):features(short)

    def test_no_missing_angle_or_depth_extrapolation(self):
        d=np.arange(51)/100;curves={a:(d,d*10) for a in ANGLES}
        self.assertAlmostEqual(metrics(curves,curves)['joint_rmse_n'],0.)
        missing=dict(curves);missing[33]=(d[:-1],d[:-1]*10)
        with self.assertRaises(ValueError):metrics(missing,curves)

    def test_gate_allows_57_error_to_change_but_limits_every_angle(self):
        def value(errors):
            return dict(joint_rmse_n=float(np.sqrt(np.mean(np.asarray(errors)**2))),worst_angle_rmse_n=max(errors),
                angles={str(a):dict(rmse_n=e,max_abs_n=e*1.1,maximum_secant_relative_error=.1) for a,e in zip(ANGLES,errors)})
        baseline=value([3.,3.,.5])
        self.assertTrue(promising(value([1.,1.,1.]),baseline))
        self.assertFalse(promising(value([.2,.2,2.]),baseline))
        candidate=value([1.,1.,1.]);candidate['angles']['33']['maximum_secant_relative_error']=.21
        self.assertFalse(promising(candidate,baseline))

    def test_fixed_schedule_never_adds_cutbacks_or_extends_beyond_bound(self):
        for end,count in ((.01,2),(.5,51),(.8,81)):
            d=schedule(end);self.assertEqual(len(d),count)
            self.assertEqual(d[-1],end);self.assertTrue(np.all(np.diff(d)>0))
        for end in (0.,1.,float('nan')):
            with self.assertRaises(ValueError):schedule(end)

    def test_linear_backend_never_accepts_nonfinite_or_wrong_physical_solution(self):
        p=object.__new__(ComputeMaterial);p.parallel_linear=Mock()
        for x in (np.array([float('nan'),0]),np.array([0.,0.])):
            p.parallel_linear.solve.return_value=x
            with self.assertRaisesRegex(RuntimeError,'Physical linear'):p.linear_direction(eye(2),np.ones(2))
        p.parallel_linear.solve.return_value=np.ones(2)
        np.testing.assert_array_equal(p.linear_direction(eye(2),np.ones(2)),np.ones(2))

    def test_preflights_never_count_as_complete_three_angle_training_set(self):
        from tools.run_common_material_study import training_jobs
        jobs=[dict(group='soft',stage=stage,angle=a) for stage in ('train','preflight','depth_check') for a in ANGLES]
        self.assertEqual(len(training_jobs(jobs,'soft')),3)
        self.assertTrue(all(j['stage']=='train' for j in training_jobs(jobs,'soft')))

    def test_reporting_revision_keeps_physical_resume_protection(self):
        from tools.run_material_candidate import resume_compatibility
        old=dict(signature={'material':'.098/.868'},source_sha256={
            'prototype/ipc_reference.py':'physical','tools/run_common_material_study.py':'report_v1',
            'tools/run_material_candidate.py':'adapter_v1'})
        new=dict(signature=old['signature'],source_sha256={**old['source_sha256'],
            'tools/run_common_material_study.py':'report_v2','tools/run_material_candidate.py':'adapter_v2'})
        self.assertTrue(resume_compatibility(old,new)['physical_sources_identical'])
        new['source_sha256']['prototype/ipc_reference.py']='changed_physics'
        with self.assertRaisesRegex(ValueError,'physical source'):resume_compatibility(old,new)


if __name__=='__main__':unittest.main()
