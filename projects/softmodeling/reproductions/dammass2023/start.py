"""Start the two SENT reproduction cases with persistent detached logs."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mesh-h',type=float,default=.275)
    parser.add_argument('--increment',type=float,default=.25)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a new empty suite directory')
    args.output.mkdir(parents=True,exist_ok=True)
    processes=[]
    for speed in (25,200):
        stem=args.output/f'sent_v{speed}'
        command=[sys.executable,'-u','-m','reproductions.dammass2023.run','--case','sent',
            '--output',str(stem),'--mesh-h',str(args.mesh_h),'--speed',str(speed),'--increment',str(args.increment)]
        stdout=stem.with_suffix('.stdout.log');stderr=stem.with_suffix('.stderr.log')
        flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        with stdout.open('wb') as out,stderr.open('wb') as err:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=flags,start_new_session=os.name!='nt',cwd=Path.cwd())
        record=dict(pid=proc.pid,command=command,stdout=str(stdout),stderr=str(stderr),
            started_utc=datetime.now(timezone.utc).isoformat())
        processes.append(record)
        stem.with_suffix('.run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(processes,indent=2))


if __name__=='__main__':main()
