"""Bounded independent reconstruction of saved cutting states; no loading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import write_json
from .run_zero_gap_cut import problem, audit, hashes


def inspect(directory, seconds):
    if not np.isfinite(seconds) or not 1 <= seconds <= 3600:
        raise ValueError('Saved-state audit budget must be 1..3600 seconds')
    directory=Path(directory).resolve()
    result_path=directory/'result.json'
    result=json.loads(result_path.read_text(encoding='utf-8'))
    if result['schema'] != 'softcut-zero-gap-result/1':
        raise ValueError('Not a zero-gap cutting result')
    started=time.monotonic()
    source=hashes()
    report=dict(schema='softcut-zero-gap-independent-audit/1',loading_started=False,
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        source_hashes=source,original_source_hashes=result['source_hashes'],
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_source_matches_current=source==result['source_hashes'],
        interpreter=sys.executable,budget_s=seconds,native_threads=2,blas_threads=2,
        passed=False,endpoint_history_passed=False,accepted_chord_ccd_verified=False,audits=[])
    from prototype.ipc_reference import toolkit
    toolkit().set_num_threads(2)
    with threadpool_limits(limits=2):
        p=problem(result['config'])
        previous=dict(positions=p.rest,velocity=p.velocity,material_metric=p.material_metric,
                      history=p.history,released=p.released,depth_mm=p.start)
        try:
            for rec in result['steps']:
                if time.monotonic()-started >= seconds:
                    raise TimeoutError('Independent reconstruction wall budget')
                path=(directory/rec['state_file']).resolve()
                if path.parent != directory or path.suffix != '.npz':
                    raise ValueError('State path must remain directly inside the result directory')
                if hashlib.sha256(path.read_bytes()).hexdigest() != rec['state_sha256']:
                    raise ValueError('Saved state hash differs')
                with np.load(path,allow_pickle=False) as data:
                    state={k:data[k].copy() for k in data.files}
                for a,b in [('previous_positions','positions'),('previous_velocity','velocity'),
                            ('previous_material_metric','material_metric'),
                            ('previous_history','history'),('previous_released','released')]:
                    np.testing.assert_array_equal(state[a],previous[b])
                dt=(float(state['depth_mm'])-float(previous['depth_mm']))/p.loading_rate
                if dt <= 0:
                    raise ValueError('Saved time must increase')
                np.testing.assert_allclose(float(state['material_dt_s']),dt,rtol=0,atol=1e-12)
                np.testing.assert_array_equal(state['mass'],p.mass)
                report['audits'].append(audit(p,state,rec))
                previous=state
            report['endpoint_history_passed']=bool(result['steps'])
            report['accepted_chord_ccd_verified']=bool(result['steps']) and all(
                row['accepted_chord_ccd_safe'] for row in report['audits'])
        except (AssertionError,RuntimeError,ValueError,TimeoutError,OSError) as exc:
            report['error']=str(exc)
    report['source_unchanged']=source==hashes()
    report['passed']=(report['endpoint_history_passed'] and
                      report['accepted_chord_ccd_verified'] and report['source_unchanged'])
    report['elapsed_s']=time.monotonic()-started
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output; preserve previous evidence')
    report=inspect(args.directory,args.seconds)
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in
        ('source_hashes','original_source_hashes','audits')},ensure_ascii=False))


if __name__ == '__main__':
    main()
