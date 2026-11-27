"""Local ownership, restart accounting and status for the existing finite material queue."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import socket
import sys
import time
import uuid

import psutil
from run_support import write_json


class QueuePaused(Exception):
    """A safe control boundary was reached before the next history commit."""


def checkpoint(request=None):
    request = request or os.environ.get('SOFTMODELING_QUEUE_PAUSE')
    if request and Path(request).is_file():
        raise QueuePaused('Pause requested by queue owner')


def read(path):
    return json.loads(Path(path).read_bytes())


@contextmanager
def control_lock(root):
    """An OS lock is released even on abrupt exit. Never unlink a live lock."""
    with (root/'control.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream,fcntl.LOCK_UN)


def active(root):
    link = root/'active_attempt.json'
    return Path(read(link)['directory']) if link.exists() else root


def identity(pid):
    p = psutil.Process(pid)
    return dict(pid=pid,created_epoch=p.create_time(),executable=p.exe(),command=p.cmdline(),cwd=p.cwd(),
                hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                python=sys.executable)


def matches(entry):
    if not entry:
        return False
    if ((entry.get('hostname') and entry['hostname'] != socket.gethostname()) or
            entry.get('slurm_job_id') != os.environ.get('SLURM_JOB_ID')):
        return None  # A PID is meaningful only on its original host and allocation.
    try:
        current=identity(entry['pid'])
    except psutil.NoSuchProcess:
        return False
    return (abs(current['created_epoch']-entry['created_epoch'])<.001 and
        current['command']==entry['command'] and os.path.normcase(current['cwd'])==os.path.normcase(entry['cwd']) and
        os.path.normcase(current['executable'])==os.path.normcase(entry['executable']))


def status(root):
    out=active(root)
    state=read(out/'queue_status.json') if (out/'queue_status.json').exists() else {'status':'prepared','jobs':[]}
    receipt=read(out/'launch.json') if (out/'launch.json').exists() else {}
    state['directory']=str(out)
    owner=receipt.get('identity') or {}
    state['owner']={k:owner.get(k) for k in ('hostname','slurm_job_id','pid','created_epoch','python')}
    exit_file=out/'worker_exit.json'
    exited=read(exit_file) if exit_file.exists() else {}
    reaped=bool(owner and exited.get('identity')==owner and exited.get('directory')==str(out))
    state['worker_reaped']=reaped
    state['worker_identity_matches']=False if reaped else matches(owner)
    state['pause_requested']=(out/'pause_requested.json').is_file()
    for job in state['jobs']:
        source=Path(job['output']);result=source/'result.json'
        if result.is_file():
            r=read(result);steps=r.get('steps',[])
            job.update(phase=r.get('phase'),last_complete_depth_mm=steps[-1]['depth_mm'] if steps else None,
                       accepted_records=len(steps),last_write_epoch=result.stat().st_mtime)
        if (source/'audit.json').is_file():
            rows=read(source/'audit.json').get('rows',[])
            valid=[]
            for row in rows:
                if not row.get('passed'):break
                valid.append(row)
            job['last_audited_depth_mm']=valid[-1]['depth_mm'] if valid else None
        if job.get('process_identity'):
            job['child_identity_matches']=(False if reaped and exited['returncode']==0
                                           else matches(job['process_identity']))
    if state['status']=='running' and state['worker_identity_matches'] is not True:
        state['status']='remote_reported_running' if state['worker_identity_matches'] is None else 'worker_missing'
    return state


def pause(root):
    with control_lock(root):
        out=active(root)
        state=status(root)
        if state['status'] in ('paused','finished','system_error'):
            return state
        remote=state['worker_identity_matches'] is None
        if remote:
            receipt=read(out/'launch.json')
            if (not state['owner']['slurm_job_id'] or not receipt.get('foreground') or
                    receipt.get('manifest')!=str(out/'manifest.json')):
                raise RuntimeError('Cannot bind remote pause to this Slurm attempt')
        elif not state['worker_identity_matches']:
            raise RuntimeError('Cannot confirm active worker identity')
        marker=out/'pause_requested.json'
        if not marker.exists():write_json(marker,dict(requested_epoch=time.time(),owner=state['owner']))
        if remote:
            return status(root)  # Cooperative request; the owning worker acknowledges exit.
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        state=status(root)
        if state['status']=='paused' and not state['worker_identity_matches'] and not any(
                j.get('child_identity_matches') for j in state['jobs']):
            return state
        time.sleep(.25)
    # The worker owns the child and will reap it. Terminate only an identified
    # solver whose last committed checkpoint is readable and matches its parent.
    state=status(root)
    manifest=read(out/'manifest.json')
    if state['worker_identity_matches']:
        for job in state['jobs']:
            entry=job.get('process_identity')
            if not matches(entry) or job['status']!='running':
                continue
            command=entry['command']
            if ('tools.run_material_candidate' not in command or '--output' not in command or
                    Path(command[command.index('--output')+1]).resolve()!=Path(job['output']).resolve() or
                    Path(entry['cwd']).resolve()!=Path(manifest['source']).resolve()):
                continue
            folder=Path(job['output']); record=folder/'result.json'
            if not record.is_file():
                continue
            r=read(record);steps=r.get('steps',[])
            if steps:
                import numpy as np
                i=len(steps)-1
                with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as saved:
                    if abs(float(saved['depth_mm'])-steps[i]['depth_mm'])>1e-10:continue
                    if not {'positions','previous_positions','material_metric','previous_material_metric'}<=set(saved.files):continue
                    if i:
                        with np.load(folder/'states'/f'{i-1:04d}.npz',allow_pickle=False) as parent:
                            if not np.array_equal(saved['previous_positions'],parent['positions']):continue
                            if not np.array_equal(saved['previous_material_metric'],parent['material_metric']):continue
            if matches(entry):
                write_json(out/'forced_pause.json',dict(identity=entry,accepted_records=len(steps),
                    forced_epoch=time.time(),audit_pending=True))
                process=psutil.Process(entry['pid'])
                if abs(process.create_time()-entry['created_epoch'])<.001:
                    process.terminate()
                    try:process.wait(timeout=3)
                    except psutil.TimeoutExpired:pass
    return status(root)


def restart(root, launch):
    """Preserve all old attempts. Earlier completed jobs remain referenced in state."""
    with control_lock(root):
        prior=active(root);state=status(root)
        old_job=state['owner'].get('slurm_job_id')
        if old_job and (not state['worker_reaped'] or os.environ.get('SLURM_JOB_ID') in (None,old_job)):
            raise RuntimeError('Slurm resume requires the reaped paused worker and a new allocation')
        if state['status']!='paused' or state['worker_identity_matches'] is not False or any(
                j.get('process_identity') and j.get('child_identity_matches') is not False for j in state['jobs']):
            raise RuntimeError('Resume requires an exited, identified paused attempt')
        manifest=read(prior/'manifest.json')
        dest=root/'attempts'/uuid.uuid4().hex
        dest.mkdir(parents=True)
        previous={j['name']:j for j in state['jobs']}
        for job in manifest['jobs']:
            old=previous[job['name']]
            if old.get('status') in ('completed','failed','budget_exhausted','export_failed','evidence_failed','process_timeout'):
                job.update(old);job['reused']=True
                continue
            if old.get('status') not in ('paused','queued','pending'):
                raise ValueError('Cannot implicitly rerun terminal failed case '+job['name'])
            source=Path(old['output'])
            saved=read(source/'result.json') if (source/'result.json').is_file() else {}
            if saved.get('steps'):
                job['resume_source']=str(source)
            spent=old.get('active_wall_s',0.)
            job['remaining_wall_s']=job.get('remaining_wall_s',job['solve_seconds']+manifest['audit_allowance_s'])-spent
            job['solve_seconds']=max(0.,job['solve_seconds']-old.get('active_solve_s',0.))
            if job['solve_seconds']==0 and abs(saved.get('last_depth_mm',-1)-job['target_mm'])<1e-10:
                job['solve_seconds']=1.  # all targets already covered; runner performs audit and export only
            if job['remaining_wall_s']<=0 or job['solve_seconds']<=0:
                raise RuntimeError('Finite attempt budget exhausted for '+job['name'])
            job.update(output=str(dest/job['name']),status='queued',parent_attempt=str(prior))
        manifest.update(parent_attempt=str(prior),batch_root=str(root),case_state_path=str(dest/'queue_status.json'),
                        summary_path=str(dest/'summary.csv'))
        write_json(dest/'manifest.json',manifest)
        launch(dest,manifest)
        write_json(root/'active_attempt.json',dict(directory=str(dest)))
        return dict(directory=str(dest),parent=str(prior))
