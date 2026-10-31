"""Independent accepted-prefix and strict interior entry checks for pure controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_precrack_entry as entry
from . import run_fracture_route as base
from .audit_zero_gap_cut import blade_contacts
from .frozen_scene import digest
from .face_witness import native
from run_support import write_json


def strict_contacts(p,x,precrack_mm):
    """Reconstruct geometrically, irrespective of ANY producer qualification flag."""
    verified=[]
    for contact in blade_contacts(p,x):
        ids=contact['body_nodes'];y=p.rest[ids,1]
        if (contact['body_reference_group']=='seam' and np.all(y<p.config.height_mm-1e-9)
            and np.all(y>p.config.height_mm-precrack_mm+1e-9)
            and np.all(x[ids,0]>p.config.drop_s_mm+1e-10)):
            verified.append(dict(contact=contact,reference_y_mm=y.tolist(),current_positions_mm=x[ids].tolist()))
    return verified


def check_entry_flag(contacts,claimed):
    if bool(contacts)!=bool(claimed):
        raise ValueError('Strict entry flag mismatch: independently reconstructed='+str(bool(contacts))+', producer='+str(bool(claimed)))


@runtime_entry
def inspect(case,out,seconds=540,*,runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic()
    if out.exists() or not 0<seconds<=540:raise ValueError('New bounded audit required')
    raw=json.loads((case/'result.json').read_text());prefix=out.with_suffix('.prefix.json')
    base.problem=entry.problem;base.sources=entry.sources
    path=base.audit(case,prefix,seconds,**runtime['requested'],physical_core_affinity=runtime['one_hardware_thread_per_core'])
    report=dict(schema='softcut-pure-entry-audit/1',passed=False,result_sha256=digest(case/'result.json'),
        auditor_sha256=digest(Path(__file__)),prefix_audit=str(prefix),prefix_audit_sha256=digest(prefix),
        source_hashes=raw['source_hashes'],thread_runtime=runtime,strict_entry_states=[],
        time_converged=False,mesh_converged=False,physical_prediction_validated=False)
    try:
        if not path['passed']:raise ValueError('Full accepted-prefix audit failed: '+str(path.get('error')))
        p=entry.problem(raw['config']);initial_history=p.history.copy();initial_faces=p.active_faces.copy()
        free_crack=p.seam[p.released];initial_gap=2*(p.rest[free_crack,0]-p.config.drop_s_mm)
        report['opening_history']=[]
        initial_contacts=strict_contacts(p,p.rest,raw['config']['precrack_mm'])
        if initial_contacts:raise ValueError('Initial state already entered; no prior nonentry bracket')
        previous=dict(time_s=0.,depth_mm=p.start,state_file='initial_state.npz')
        for row in raw['steps']:
            if time.monotonic()-start>seconds:raise TimeoutError('Entry audit budget')
            with np.load(case/row['state_file'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(s['history'],initial_history)
            np.testing.assert_array_equal(s['active_faces'],initial_faces)
            if abs(row['energy']['fracture_dissipation_n_mm'])>1e-12:raise ValueError('Precrack paid new Gc')
            if row.get('face_events'):raise ValueError('Pure control activated a face')
            gap=2*(s['positions'][free_crack,0]-p.config.drop_s_mm)
            if gap.min() < -1e-12:raise ValueError('Mirror crack penetration')
            report['opening_history'].append(dict(depth_mm=row['depth_mm'],time_s=row['time_s'],
                minimum_gap_mm=float(gap.min()),maximum_gap_mm=float(gap.max()),
                minimum_change_from_initial_mm=float(np.min(gap-initial_gap)),
                maximum_change_from_initial_mm=float(np.max(gap-initial_gap))))
            # Prefix already verifies equations; reconstruction here is unconditional.
            base.restore_audit(p,s,row)
            verified=strict_contacts(p,s['positions'],raw['config']['precrack_mm'])
            if verified:
                if not report['strict_entry_states']:
                    report['first_entry_bracket']=dict(previous_nonentry=previous,first_entry=dict(time_s=row['time_s'],depth_mm=row['depth_mm'],state_file=row['state_file']))
                report['strict_entry_states'].append(dict(depth_mm=row['depth_mm'],time_s=row['time_s'],
                    state_file=row['state_file'],state_sha256=row['state_sha256'],contacts=verified))
            else:previous=dict(time_s=row['time_s'],depth_mm=row['depth_mm'],state_file=row['state_file'])
            check_entry_flag(verified,row['blade_entered_surface_channel'])
        report.update(passed=True,history_frozen=True,new_fracture_work_n_mm=0.,
            strict_interior_entry_verified=bool(report['strict_entry_states']),
            new_activation=False,new_connected_crack_growth=False)
    except (ValueError,RuntimeError,AssertionError,TimeoutError) as exc:report['error']=str(exc)
    report.update(elapsed_s=time.monotonic()-start,source_unchanged=entry.sources()==raw['source_hashes'])
    report['passed'] &= report['source_unchanged'];write_json(out,native(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--seconds',type=float,default=540);add_thread_arguments(p);a=p.parse_args()
    r=inspect(a.case,a.out,a.seconds,**thread_arguments(a));print(json.dumps({k:r.get(k) for k in ('passed','strict_interior_entry_verified','elapsed_s','error')}))
if __name__=='__main__':main()
