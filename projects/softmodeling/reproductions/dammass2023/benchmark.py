"""Detached suite worker: run four SENT cases, then write the final comparison.

Use --detach for a persistent master log. Waiting happens in this worker,
without requiring an agent to monitor cases. Every return code is retained.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
from .run import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    # Fail before launching long calculations when the local-only comparison
    # inputs have not been prepared on a fresh code/documentation checkout.
    from .report import require_reference_data
    require_reference_data()
    flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
    if args.detach:
        if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty suite directory')
        root.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','reproductions.dammass2023.benchmark','--output',str(root)]
        with (root/'suite.stdout.log').open('wb') as stdout,(root/'suite.stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=flags,start_new_session=os.name!='nt')
        record=dict(pid=proc.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'suite.run.json',record)
        print(f'Suite PID {proc.pid}; logs and final comparison: {root}')
        return
    root.mkdir(parents=True,exist_ok=True)
    status=dict(state='running',completed=False,cases=[])
    write_json(root/'suite_status.json',status)
    try:
        material=root/'material'
        subprocess.run([sys.executable,'-u','-m','reproductions.dammass2023.run',
            '--case','material','--output',str(material)],check=True)
        workers=[]
        for h,name in [(.275,'h0275'),(.275/1.5,'h01833')]:
            for speed in (25,200):
                directory=root/f'{name}_v{speed}'
                command=[sys.executable,'-u','-m','reproductions.dammass2023.run','--case','sent',
                    '--output',str(directory),'--mesh-h',str(h),'--speed',str(speed)]
                with directory.with_suffix('.stdout.log').open('wb') as stdout,directory.with_suffix('.stderr.log').open('wb') as stderr:
                    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        creationflags=flags,start_new_session=os.name!='nt')
                record=dict(pid=proc.pid,command=command,directory=str(directory),returncode=None)
                write_json(directory.with_suffix('.run.json'),record)
                status['cases'].append(record);workers.append((proc,record))
                print(f'START {directory.name} PID {proc.pid}',flush=True)
        write_json(root/'suite_status.json',status)
        for proc,record in workers:
            record['returncode']=proc.wait()
            print(f'EXIT {Path(record["directory"]).name} code={record["returncode"]}',flush=True)
            write_json(root/'suite_status.json',status)
        subprocess.run([sys.executable,'-m','reproductions.dammass2023.report','--material',str(material),
            '--sent',*[r['directory'] for r in status['cases']],'--output',str(root/'comparison')],check=True)
        status.update(state='complete' if all(r['returncode']==0 for r in status['cases']) else 'failed',
            completed=all(r['returncode']==0 for r in status['cases']))
    except Exception as error:
        status.update(state='failed',error=str(error));raise
    finally:write_json(root/'suite_status.json',status)


if __name__=='__main__':main()
