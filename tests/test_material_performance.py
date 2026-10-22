"""Execution-policy guards and numerical equivalence; no FE loading runs."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csr_matrix

from mechanics.finite_strain import Material, response
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.material_performance import (performance_policy, require_same_performance,
    enable_performance, close_performance, native_audit)


class BaseDirection:
    def linear_direction(self, matrix, rhs):
        return np.linalg.solve(matrix.toarray(), rhs)


class MaterialPerformanceTests(unittest.TestCase):
    def test_native_default_keeps_original_dispatch(self):
        p = BaseDirection()
        enable_performance(p, performance_policy())
        self.assertIs(type(p), BaseDirection)
        self.assertEqual(p.material_batch_size, 0)
        close_performance(p)

    def test_resume_cannot_change_batch_backend_or_threads(self):
        policy = performance_policy(material_batch_size=2048, linear_backend='pardiso')
        old = dict(performance_policy=policy)
        require_same_performance(old, copy.deepcopy(policy))
        for key, value in (('material_batch_size',1024), ('linear_backend','native'),
                           ('linear_threads',8), ('ipc_threads',4), ('blas_threads',2)):
            changed = dict(policy, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_same_performance(old, changed)
        require_same_performance({}, performance_policy())
        with self.assertRaises(ValueError):require_same_performance({}, policy)

    def test_invalid_policy_is_rejected(self):
        for values in (dict(material_batch_size=-1),dict(material_batch_size=2.5),
                       dict(linear_backend='unknown'),dict(linear_threads=0),dict(ipc_threads=0)):
            with self.subTest(values=values), self.assertRaises(ValueError):performance_policy(**values)

    def test_backend_original_matrix_check_includes_fallback(self):
        A = csr_matrix([[2.,3.],[-1.,4.]])
        rhs = np.array([5.,3.])
        backend = Mock()
        backend.solve.return_value = np.linalg.solve(A.toarray(),rhs)
        p = BaseDirection();p.ipc = Mock()
        policy = performance_policy(material_batch_size=2048,linear_backend='pardiso')
        with patch('prototype.pardiso.PardisoDirection',return_value=backend) as factory:
            enable_performance(p,policy)
            factory.assert_called_once_with(4)
        p.ipc.set_num_threads.assert_called_once_with(2)
        np.testing.assert_allclose(A@p.linear_direction(A,rhs),rhs,atol=1e-12)
        for invalid in (np.zeros(2),np.full(2,np.nan)):
            backend.solve.return_value = invalid
            with self.assertRaises(RuntimeError):p.linear_direction(A,rhs)
        close_performance(p);backend.close.assert_called_once()

    def test_native_audit_uses_separate_unbatched_problem_and_propagates_failure(self):
        report = dict(signature={'material':'same'},steps=[{}])
        separate = SimpleNamespace(material_batch_size=0)
        make = Mock(return_value=separate)
        with patch('prototype.material_case.signature',return_value=report['signature']),\
             patch('prototype.material_case.audit',return_value=dict(passed=True,states=1)) as check:
            native_audit(make,Path('unused'),report)
            check.assert_called_once_with(separate,Path('unused'),report)
            self.assertTrue(report['independent_audit_policy']['fresh_problem'])
            check.side_effect=ValueError('bad material history')
            with self.assertRaisesRegex(ValueError,'bad material history'):
                native_audit(make,Path('unused'),report)
        separate.material_batch_size=2048
        with self.assertRaisesRegex(ValueError,'unbatched native'):
            native_audit(make,Path('unused'),report)

    def test_batched_updated_response_preserves_each_points_history_and_tangent(self):
        p = ViscoelasticProblem.__new__(ViscoelasticProblem)
        p.material = Material((.14,),(2.,),(1.24,),(2.,),.49,1.7,
                              viscous_flow='isochoric_maxwell')
        F = np.array([[[1.1,.2],[0.,.9]],[[.72,-.3],[.02,1.2]],
                      [[1.5,.1],[0.,.7]],[[1.,.001],[0.,1.]],[[.85,.4],[.1,.95]]])
        p.previous_material_metric=response(F,np.tile(np.eye(3),(len(F),1,1)),
            .15,p.material,kinematics='plane_strain')['metric']
        before=p.previous_material_metric.copy();p.material_dt=.05
        target=F+np.array([[.04,-.07],[.01,.03]])
        p.material_batch_size=0;native=p.constitutive_response(target,tangent=True)
        p.material_batch_size=2;batched=p.constitutive_response(target,tangent=True)
        for key in ('P','tangent','metric','incremental_potential','energy','viscous_rate','J'):
            with self.subTest(key=key):
                np.testing.assert_allclose(batched[key],native[key],rtol=2e-10,atol=2e-10)
        np.testing.assert_array_equal(p.previous_material_metric,before)


if __name__ == '__main__':unittest.main()
