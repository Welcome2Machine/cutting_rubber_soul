"""Start a detached calculation and record its logs and source fingerprints."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from run_support import source_hashes,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='prototype/initiation_config.json')
    parser.add_argument('--mode',choices=('cycle','to_candidate'),default='to_candidate')
    parser.add_argument('--candidate-criterion',choices=('local_nominal','multiaxial'),default='local_nominal')
    parser.add_argument('--candidate-tolerance',type=float,default=.0001)
    parser.add_argument('--tangential-stiffness',type=float,default=250.)
    parser.add_argument('--contact-discretization',choices=('nodal','integral','macro','quadratic_nodes','quadratic_integral'),default='quadratic_integral')
    parser.add_argument('--mesh-grading',choices=('legacy','graded'),default='graded')
    parser.add_argument('--volumetric-discretization',choices=('element','nodal_volume','macro_mixed','taylor_hood','quadratic_full'),default='quadratic_full')
    parser.add_argument('--volume-stabilization-shear',type=float,default=0.)
    parser.add_argument('--strength-calibration')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = (root/args.config).resolve(strict=True)
    record_path = root/'outputs/initiation_latest_run.json'
    name = 'initiation_'+args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = root/'outputs'/name
    log,error = output.with_suffix('.log'),output.with_suffix('.stderr.log')
    record_path.parent.mkdir(parents=True,exist_ok=True)
    startup,flags = None,0
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    command=[sys.executable,'-u','-m','prototype.run_initiation','--config',str(config),'--output',str(output),
             '--mode',args.mode,'--candidate-criterion',args.candidate_criterion,'--candidate-tolerance',str(args.candidate_tolerance),
             '--tangential-stiffness',str(args.tangential_stiffness),
             '--contact-discretization',args.contact_discretization,'--mesh-grading',args.mesh_grading,
             '--volumetric-discretization',args.volumetric_discretization,
             '--volume-stabilization-shear',str(args.volume_stabilization_shear)]
    if args.strength_calibration:command.extend(['--strength-calibration',str((root/args.strength_calibration).resolve(strict=True))])
    if args.candidate_criterion=='multiaxial' and not args.strength_calibration:parser.error('Multiaxial stopping requires a calibration file')
    with log.open('wb') as stdout,error.open('wb') as stderr:
        process = subprocess.Popen(command,
                                    cwd=root,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                    startupinfo=startup,creationflags=flags,
                                    start_new_session=os.name != 'nt')
    record = dict(pid=process.pid,output=str(output.relative_to(root)),log=str(log),stderr=str(error),
                  config=str(config),command=command,run_mode=args.mode,started_at=datetime.now().astimezone().isoformat(),
                  source_hashes=source_hashes())
    write_json(record_path,record)
    write_json(output.with_suffix('.run.json'),record)
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
