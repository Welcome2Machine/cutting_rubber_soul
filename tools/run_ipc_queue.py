"""Frozen IPC queue; advance after success, failure, crash or timeout."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_support import source_hashes, write_json

CASES = ((30, 0.), (0, 0.), (60, .35), (30, .35), (0, .35))


def now():
    return datetime.now(timezone.utc).isoformat()


def hidden_options():
    if os.name == 'nt':
        return dict(creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(start_new_session=True)


def run_jobs(output, state, env):
    """The queue records process outcome independently of mechanical completion."""
    state.update(status='running', pid=os.getpid(), started=now())
    write_json(output/'queue.json', state)
    for job in state['jobs']:
        job.update(status='running', started=now())
        folder = Path(job['output'])
        print(f"START {job['name']} angle={job.get('angle')} mu={job.get('friction')}", flush=True)
        write_json(output/'queue.json', state)
        try:
            # A decimal mu in the directory name is not a file extension.
            with Path(str(folder)+'.stdout.log').open('wb') as stdout, Path(str(folder)+'.stderr.log').open('wb') as stderr:
                process = subprocess.Popen(job['command'], cwd=state['source_root'], env=env,
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           **hidden_options())
                job['pid'] = process.pid
                write_json(output/'queue.json', state)
                try:
                    job['exit_code'] = process.wait(timeout=state['process_timeout_s'])
                except subprocess.TimeoutExpired:
                    # Terminate only this queue-owned process tree, including an encoder.
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=stderr, stderr=stderr, check=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        import signal
                        os.killpg(process.pid, signal.SIGKILL)
                    job['exit_code'] = process.wait(timeout=30)
                    job['status'] = 'timed_out'
            report_path = folder/'result.json'
            report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
            job['mechanics_completed'] = bool(report.get('completed'))
            job['last_depth_mm'] = report.get('last_depth_mm')
            job['visualization'] = report.get('visualization')
            if job['status'] != 'timed_out':
                job['status'] = ('completed' if job['exit_code'] == 0 and report.get('completed')
                                 and report.get('status') == 'completed' else 'failed')
            if report.get('error'):
                job['error'] = report['error']
        except Exception as exc:
            # A native crash, invalid/missing result or launch failure cannot skip later cases.
            job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        job['finished'] = now()
        print(f"END {job['name']} status={job['status']} last_depth_mm={job.get('last_depth_mm')}", flush=True)
        write_json(output/'queue.json', state)
    state.update(status='finished', finished=now(),
                 successful_cases=sum(j['status'] == 'completed' for j in state['jobs']))
    write_json(output/'queue.json', state)


def prepare(output, threads, seconds, *, friction_only=False, resume_first=None):
    if resume_first is not None and not friction_only:
        raise ValueError('First-case continuation requires --friction-only')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Queue output must be empty; existing queues are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/'source'
    fingerprints = source_hashes(snapshot=frozen)
    for relative in ('prototype/ipc_baseline.json', 'tools/visualize_ipc.py',
                     'tools/run_ipc_queue.py', 'requirements.txt'):
        import hashlib
        target = frozen/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        fingerprints[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    common = ['--hessian-mode', 'regularized', '--tip-radius', '.02',
              '--local-spacing', '.05', '--surface-spacing', '.005',
              '--fine-halfwidth', '3', '--nose-facets', '64', '--dhat', '.0025',
              '--increment', '.01', '--depth', '3',
              '--threads', str(threads), '--seconds', str(seconds)]
    jobs = []
    selected = [(a, mu) for a, mu in CASES if mu > 0] if friction_only else CASES
    for index, (angle, friction) in enumerate(selected, 1):
        name = f'{index:02d}_angle{angle}_mu{friction:g}'
        folder = output/name
        command = [sys.executable, '-u', '-m', 'prototype.ipc_reference', '--output', str(folder),
                   '--inclination', str(angle), '--friction', str(friction),
                   '--max-iterations', '500' if friction > 0 else '200', *common]
        if index == 1 and resume_first is not None:
            command += ['--resume', str(Path(resume_first).resolve())]
        jobs.append(dict(name=name, angle=angle, friction=friction, output=str(folder),
                         command=command, status='pending'))
    state = dict(status='pending', created=now(), source_root=str(frozen),
                 source_sha256=fingerprints, python=sys.executable, threads=threads,
                 solve_budget_s=seconds, process_timeout_s=seconds+1800 if seconds > 0 else None,
                 native_python_path=str(ROOT/'.runtime/ipc-python'), jobs=jobs)
    write_json(output/'queue.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=21600.,
                        help='Per-case wall budget; 0 disables solver and process time limits')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--friction-only', action='store_true',
                        help='Run 60, 30, 0 degrees at mu=0.35; retain completed mu=0 results')
    parser.add_argument('--resume-first', type=Path,
                        help='Continue the 60-degree case from an accepted prefix; later cases start fresh')
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.seconds < float('inf'):
        parser.error('Nonnegative finite budget and positive threads required')
    if args.resume_first is not None and not args.friction_only:
        parser.error('--resume-first requires --friction-only')
    output = args.output.resolve()
    state = (json.loads((output/'queue.json').read_text(encoding='utf-8')) if args.worker
             else prepare(output, args.threads, args.seconds, friction_only=args.friction_only,
                          resume_first=args.resume_first))
    if args.worker and state['status'] != 'pending':
        parser.error('This queue has already started; refusing duplicate execution')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([state['source_root'], state['native_python_path']])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = str(state['threads'])
    if args.detach and not args.worker:
        command = [sys.executable, '-u', str(output/'source/tools/run_ipc_queue.py'),
                   '--output', str(output), '--worker']
        with (output/'queue.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=state['source_root'], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       **hidden_options())
        write_json(output/'launch.json', dict(pid=process.pid, command=command, started=now()))
        print(json.dumps(dict(queue_pid=process.pid, output=str(output))))
    else:
        run_jobs(output, state, env)


if __name__ == '__main__':
    main()
