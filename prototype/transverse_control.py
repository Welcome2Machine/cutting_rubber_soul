"""Explicit free-transverse-traction diagnostic; not a finite-width 3-D model.

Use the existing total-stress plane-stress Maxwell return and its condensed
tangent. In-plane IPC geometry and reference-width integration stay unchanged.
Both the equilibrium and viscous stresses enter the eliminated stress equation.
"""
import numpy as np
from mechanics.finite_strain import response


class PlaneStressControl:
    material_kinematics='plane_stress'

    def constitutive_response(self,F,*,tangent):
        return response(F,self.previous_material_metric,self.material_dt,self.material,
                        tangent=tangent,kinematics=self.material_kinematics)

    def material_acceptance(self,x,depth,accepted):
        accepted=super().material_acceptance(x,depth,accepted)
        q=self.material_response(x)
        if (not np.isfinite(q['lambda_out']).all() or np.min(q['lambda_out'])<=0
                or np.min(q['J'])<=0 or np.max(q['plane_stress_scaled_residual'])>1e-8):
            raise RuntimeError('Total plane-stress local acceptance failed')
        w=self.material_integration_weights()
        accepted['transverse']=dict(
            kinematics=self.material_kinematics,
            lambda_min=float(q['lambda_out'].min()),lambda_max=float(q['lambda_out'].max()),
            lambda_reference_volume_mean=float(w@q['lambda_out']/w.sum()),
            min_volume_j=float(q['J'].min()),
            max_scaled_total_stress_residual=float(q['plane_stress_scaled_residual'].max()),
            reference_width_mm=self.config.width_mm)
        return accepted

    def _audit_extra_acceptance(self,fresh,record):
        values=super()._audit_extra_acceptance(fresh,record)
        if fresh['transverse']['kinematics']!=record['transverse']['kinematics']:
            raise RuntimeError('Changed transverse constraint')
        for key,value in fresh['transverse'].items():
            if key!='kinematics' and not np.isclose(value,record['transverse'][key],rtol=1e-9,atol=1e-10):
                raise RuntimeError('Transverse state does not reproduce: '+key)
        return dict(values,transverse=fresh['transverse'])
