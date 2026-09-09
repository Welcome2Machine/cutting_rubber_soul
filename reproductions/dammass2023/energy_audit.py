"""Locate saved energy-accounting defects without modifying source results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory):
    d=np.genfromtxt(directory/'results.csv',delimiter=',',names=True,ndmin=1)
    cfg=json.loads((directory/'config.json').read_text())['config']
    u=np.r_[0.,d['opening_mm']];force=np.r_[0.,d['force_n']];du=np.diff(u)
    energy=d['bulk_energy_n_mm']+d['fracture_energy_n_mm']+d['viscous_dissipation_n_mm']
    if 'phase_viscous_dissipation_n_mm' in d.dtype.names:energy+=d['phase_viscous_dissipation_n_mm']
    missing=.5*(force[1:]+force[:-1])*du-np.diff(np.r_[0.,energy])
    i=int(np.argmax(np.abs(missing)));dt=du[i]/(cfg['speed_mm_min']/60)
    # Irreversibility and 0<=d<=1 imply |d_new-d_old|<=1 everywhere.
    # This deliberately loose whole-specimen bound includes both half domains.
    upper=cfg['phase_viscosity']*(75*30*cfg['thickness_mm'])/dt
    quadrature=.5*abs(force[i+1]-force[i])*du[i]
    return dict(directory=str(directory),status=json.loads((directory/'status.json').read_text()),
        total_unaccounted_work_n_mm=float(missing.sum()),largest_increment=dict(
            opening_bracket_mm=u[i:i+2].tolist(),force_bracket_n=force[i:i+2].tolist(),
            missing_work_n_mm=float(missing[i]),dt_s=float(dt),
            whole_specimen_phase_viscosity_upper_bound_n_mm=float(upper),
            trapezoid_vs_endpoint_work_difference_n_mm=float(quadrature)),
        phase_dissipation_recorded='phase_viscous_dissipation_n_mm' in d.dtype.names,
        csv_sha256=hashlib.sha256((directory/'results.csv').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sent',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(scope='Saved increment audit; missing work is not assigned to physical dissipation',
        assumptions='Phase viscosity bound uses backward Euler rate, irreversible phase in [0,1], reference volume',
        cases=[audit(p) for p in args.sent])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
