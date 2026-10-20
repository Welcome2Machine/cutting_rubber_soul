"""Frozen IPC queue; advance after success, failure, crash or timeout."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_support import source_hashes, write_json

CASES = ((30, 0.), (0, 0.), (60, .35), (30, .35), (0, .35))


def parse_case(value):
    """One explicit case; no inference of oil properties from its coefficient."""
    try:
        angle, friction = (float(v) for v in value.split(':'))
    except (TypeError, ValueError):
        raise ValueError('Cases must be ANGLE:MU, for example 60:0.1') from None
    if not math.isfinite(angle) or angle not in (0., 30., 60.):
        raise ValueError('Case angle must be 0, 30 or 60 degrees')
    if not math.isfinite(friction) or friction < 0:
        raise ValueError('Friction must be finite and nonnegative')
    return int(angle), friction


def process_identity(pid):
    """Read process birth identity without signalling it (especially on Windows)."""
    pid = int(pid)
    if pid <= 0:
        return dict(status='missing', pid=pid)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            return dict(status='missing' if error == 87 else 'unavailable', pid=pid, os_error=error)
        try:
            wait = kernel.WaitForSingleObject(handle, 0)
            if wait == 0:
                return dict(status='missing', pid=pid)
            if wait != 258:
                return dict(status='unavailable', pid=pid, os_error=ctypes.get_last_error())
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in times)):
                return dict(status='unavailable', pid=pid, os_error=ctypes.get_last_error())
            birth = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            return dict(status='alive', pid=pid, birth=str(birth), platform='windows')
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith('linux'):
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return dict(status='missing', pid=pid)
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            return dict(status='alive', pid=pid, birth=boot+':'+fields[19], platform='linux')
        except FileNotFoundError:
            return dict(status='missing', pid=pid)
        except (OSError, ValueError, IndexError) as exc:
            return dict(status='unavailable', pid=pid, error=str(exc))
    return dict(status='unavailable', pid=pid, error='Process birth identity unsupported on this platform')


def mechanical_completion(report, folder, expected_target=3.):
    """Audit saved acceptance records, not merely the process return code."""
    try:
        if report.get('status') != 'completed' or report.get('completed') is not True:
            return False
        target = float(report['target_mm'])
        increment = float(report['loading_increment_mm'])
        tolerance = float(report['force_tolerance_n'])
        if not (math.isfinite(expected_target) and expected_target > 0
                and target == expected_target and math.isfinite(increment) and increment > 0
                and 0 < tolerance <= 1e-4):
            return False
        steps = report['steps']
        expected = math.ceil(target/increment-1e-12)+1
        control = report.get('load_control') or {'mode': 'fixed'}
        mode = control.get('mode')
        if mode not in ('fixed', 'adaptive_bisection'):
            return False
        adaptive = mode == 'adaptive_bisection'
        if adaptive:
            minimum = float(control['minimum_increment_mm'])
            maximum = float(control['maximum_increment_mm'])
            output_increment = float(control['output_grid_increment_mm'])
            if (not all(math.isfinite(v) for v in (minimum, maximum, output_increment))
                    or not 0 < minimum <= maximum
                    or abs(maximum-increment) > 1e-10
                    or abs(output_increment-increment) > 1e-10):
                return False
        if ((not adaptive and len(steps) != expected)
                or not steps or abs(float(report['last_depth_mm'])-target) > 1e-9):
            return False
        previous_depth, next_output = None, 0
        for index, step in enumerate(steps):
            depth = float(step['depth_mm'])
            residual = float(step['residual_n'])
            jacobian = float(step['min_j'])
            force = float(step['reaction_fz_n'])
            if (not all(math.isfinite(v) for v in (depth, residual, jacobian, force))
                    or (not adaptive and abs(depth-min(index*increment, target)) > 1e-9)
                    or not 0 <= residual <= tolerance or jacobian <= 0
                    or not (Path(folder)/'states'/f'{index:04d}.npz').is_file()):
                return False
            if adaptive:
                gap = float(step['minimum_distance_mm'])
                work = float(step['friction_work_n_mm'])
                if (not math.isfinite(gap) or gap <= 0 or not math.isfinite(work) or work < -1e-12
                        or not -1e-10 <= depth <= target+1e-10
                        or (previous_depth is None and abs(depth) > 1e-10)
                        or (previous_depth is not None
                            and not 0 < depth-previous_depth <= increment+1e-10)):
                    return False
                # Every original output depth must itself be an accepted state.
                # Intermediate accepted states remain independently audited above.
                anchor = min(next_output*increment, target)
                if next_output < expected and depth > anchor+1e-10:
                    return False
                if next_output < expected and abs(depth-anchor) <= 1e-10:
                    next_output += 1
                previous_depth = depth
        if adaptive and (next_output != expected
                         or not math.isfinite(float(report['last_depth_mm']))):
            return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def run_comparison(output, state, env, reason):
    if not state.get('comparison_command'):
        return
    result = dict(reason=reason, started=now())
    try:
        with (output/'comparison.log').open('ab') as log:
            log.write(f'\nCOMPARISON {reason} {now()}\n'.encode())
            log.flush()
            process = subprocess.run(state['comparison_command'], cwd=state['source_root'], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     timeout=300, check=False, **hidden_options())
        result.update(exit_code=process.returncode,
                      status='completed' if process.returncode == 0 else 'failed')
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['finished'] = now()
    state.setdefault('comparison_attempts', []).append(result)
    write_json(output/'queue.json', state)


def observe_existing(run):
    """Read an external run only; never terminate, restart or edit its report."""
    folder = Path(run['output'])
    try:
        report = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        report = {}
        run['report_error'] = f'{type(exc).__name__}: {exc}'
    if report and report.get('pid') != run.get('pid'):
        run.update(status='external_report_identity_mismatch', mechanics_completed=False)
        return
    if report.get('status') in ('completed', 'failed'):
        complete = mechanical_completion(report, folder)
        run.update(status='completed' if complete else 'failed', mechanics_completed=complete,
                   last_depth_mm=report.get('last_depth_mm'), error=report.get('error'))
        return
    identity = process_identity(run['pid'])
    original = run.get('process_identity', {})
    if identity.get('status') == 'missing':
        status = 'external_process_missing'
    elif original.get('status') != 'alive' or identity.get('status') != 'alive':
        status = 'external_identity_unavailable'
    elif identity.get('birth') != original.get('birth') or identity.get('platform') != original.get('platform'):
        status = 'external_pid_reused'
    else:
        status = 'running'
    if not report:
        run['unreadable_report_checks'] = run.get('unreadable_report_checks', 0)+1
        if status == 'running' and run['unreadable_report_checks'] >= 3:
            status = 'external_report_unreadable'
    else:
        run['unreadable_report_checks'] = 0
    run.update(status=status, mechanics_completed=False,
               last_depth_mm=report.get('last_depth_mm'), last_checked=now())


def wait_observed(output, state, env):
    runs = state.get('observed_runs', [])
    while runs:
        for run in runs:
            if run['status'] in ('pending', 'running'):
                observe_existing(run)
                print(f"OBSERVED pid={run['pid']} status={run['status']} "
                      f"last_depth_mm={run.get('last_depth_mm')}", flush=True)
        write_json(output/'queue.json', state)
        if not any(run['status'] == 'running' for run in runs):
            break
        state['status'] = 'waiting_for_observed_runs'
        write_json(output/'queue.json', state)
        time.sleep(60)


def now():
    return datetime.now(timezone.utc).isoformat()


def hidden_options():
    if os.name == 'nt':
        return dict(creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(start_new_session=True)


def write_outcome(job, state, report):
    """Summarize this attempt without turning a numerical failure into physics."""
    folder = Path(job['output'])
    folder.mkdir(parents=True, exist_ok=True)
    report = report if isinstance(report, dict) else {}
    error = job.get('error') or report.get('error')
    error_text = str(error or '')
    if job.get('status') == 'completed' and job.get('mechanics_completed'):
        status = 'completed'
        reason = '已完成目标深度，保存的力学接受记录通过审核。'
    elif job.get('status') == 'timed_out' or error_text.startswith('TimeoutError:'):
        status = 'wall_budget_exhausted'
        reason = '本次配置在规定墙钟预算内未完成全部运行流程。'
    elif (report.get('completed') is True or report.get('status') == 'completed') and not job.get('mechanics_completed'):
        status = 'completion_audit_failed'
        reason = '计算报告声称完成，但保存记录的完整性或力学接受条件审核未通过。'
    elif error_text.startswith(('LoadStepFailure:', 'RuntimeError:')) and any(
            known in error_text for known in (
                'IPC adaptive load exhausted',
                'IPC friction pseudo-transient model mismatch',
                'IPC friction pseudo-transient stagnation',
                'IPC friction pseudo-transient iteration budget',
                'IPC inner iteration budget',
                'IPC Newton directions/energy line search failed',
                'Unbalanced IPC acceptance',
                'Prescribed-motion predictor is not collision free')):
        status = 'nonlinear_nonconvergence'
        reason = '本次配置在指定非线性迭代和加载回退限制内未收敛。'
    else:
        status = 'execution_error'
        reason = ('运行、输出读取或进程退出异常；现有证据不足以归因为非线性不收敛。'
                  if report else '未取得可读的求解报告；请检查标准错误日志和进程退出码。')

    def finite(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None

    steps = report.get('steps')
    steps = steps if isinstance(steps, list) else []
    # This is deliberately a saved-record audit, not a fresh FE evaluation.
    # Stop at the first invalid/missing record rather than reporting a later
    # unverified depth from a tampered completion claim.
    last_accepted, accepted_count = None, 0
    tolerance = finite(report.get('force_tolerance_n'))
    target = finite(state.get('target_mm', 3.))
    increment = finite(report.get('loading_increment_mm'))
    adaptive = (report.get('load_control') or {}).get('mode') == 'adaptive_bisection'
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            break
        depth, residual, jacobian, force = (
            finite(step.get(key)) for key in ('depth_mm', 'residual_n', 'min_j', 'reaction_fz_n'))
        if (None in (depth, residual, jacobian, force) or tolerance is None
                or not 0 < tolerance <= 1e-4 or not 0 <= residual <= tolerance or jacobian <= 0
                or target is None or not -1e-9 <= depth <= target+1e-9
                or increment is None or increment <= 0
                or (not adaptive and abs(depth-min(index*increment, target)) > 1e-9)
                or (last_accepted is None and abs(depth) > 1e-9)
                or (last_accepted is not None and not 0 < depth-last_accepted <= increment+1e-9)
                or not (folder/'states'/f'{index:04d}.npz').is_file()):
            break
        # Older fixed-step reports omit these fields. Do not invent them.
        geometry_keys = ('minimum_distance_mm', 'friction_work_n_mm')
        if adaptive or any(key in step for key in geometry_keys):
            gap, work = (finite(step.get(key)) for key in geometry_keys)
            if gap is None or work is None or gap <= 0 or work < -1e-12:
                break
        last_accepted, accepted_count = depth, accepted_count+1
    tracking = report.get('tip_contact_tracking') or {}
    rejected = report.get('rejected_load_attempts') or []
    counts = Counter((step.get('tip_contact') or {}).get('status', 'not_recorded')
                     for step in steps[:accepted_count])
    observed = sum(counts.get(name, 0) for name in ('unique', 'no_tip_contact', 'ambiguous'))
    outcome = dict(status=status, termination_reason=reason,
        scope='本次配置及预算的运行结果；数值失败不证明物理过程不可实现。',
        target_mm=target,
        last_accepted_depth_mm=last_accepted, accepted_record_count=accepted_count,
        reported_last_depth_mm=finite(report.get('last_depth_mm')),
        attempted_depth_mm=finite(report.get('attempted_depth_mm')),
        rejected_load_attempt_count=len(rejected),
        last_rejected_load_attempt=rejected[-1] if rejected else None,
        mechanics_completed=bool(job.get('mechanics_completed')),
        acceptance_level='保存的接受记录与检查点存在性审核；未全程重新装配，未核验全部 NPZ 内容。',
        validation_limit='力学完成不代表接触迁移验证或实验合格。',
        job_status=job.get('status'), exit_code=job.get('exit_code'),
        elapsed_s=finite(job.get('elapsed_s')), solver_elapsed_s=finite(report.get('elapsed_s')),
        solve_budget_s=finite(state.get('solve_budget_s')),
        process_timeout_s=finite(state.get('process_timeout_s')),
        original_error=error_text or None,
        source_root=state.get('source_root'), source_sha256=state.get('source_sha256', {}),
        solver_source_sha256=report.get('source_sha256', {}),
        command=job.get('command'), configuration=report.get('config'),
        requested_parameters=dict(angle_deg=job.get('angle'), friction=job.get('friction'),
                                  shear_modulus_mpa=state.get('shear_modulus_mpa')),
        process_policy=state.get('process_policy'),
        friction_globalization=state.get('friction_globalization'),
        tip_tracking_requested=bool(state.get('track_tip_contact')),
        tip_observations_available=observed > 0,
        tip_observation_count=observed, tip_status_counts=dict(counts),
        tip_tracking_errors=tracking.get('errors', []),
        tip_observation_scope='有效状态含 unique/no_tip_contact/ambiguous；不表示每个深度都有唯一切点，缺测不填值。',
        logs=dict(solver=str(folder/'run.log'), stdout=str(folder)+'.stdout.log',
                  stderr=str(folder)+'.stderr.log'),
        result_path=str(folder/'result.json'), finished=job.get('finished'))
    write_json(folder/'outcome.json', outcome)
    display = lambda value: '未取得' if value is None else str(value)
    lines = [f'# 本次计算结果：{status}', '', reason, '',
        outcome['scope'], '',
        f'- 目标深度：{display(outcome["target_mm"])} mm',
        f'- 最后接受深度（保存记录审核）：{display(last_accepted)} mm；记录数：{accepted_count}',
        f'- 报告声称的最后深度：{display(outcome["reported_last_depth_mm"])} mm',
        f'- 终止时尝试深度：{display(outcome["attempted_depth_mm"])} mm；加载拒步数：{len(rejected)}',
        f'- 最后拒步：{json.dumps(outcome["last_rejected_load_attempt"], ensure_ascii=False)}',
        f'- 力学完成：{"是" if outcome["mechanics_completed"] else "否"}',
        f'- 验收级别：{outcome["acceptance_level"]}',
        f'- 耗时：{display(outcome["elapsed_s"])} s；求解预算：{display(outcome["solve_budget_s"])} s；进程截止：{display(outcome["process_timeout_s"])} s（0/空表示相应限制未启用）',
        f'- 求解报告记录的耗时：{display(outcome["solver_elapsed_s"])} s（与包含初始化和后处理的进程耗时分开）',
        f'- 进程退出码：{display(outcome["exit_code"])}；原始错误：{error_text or "未报告"}',
        f'- 切点观测可用：{"是（至少一个有效状态）" if observed else "否"}；状态统计：{json.dumps(dict(counts), ensure_ascii=False)}',
        f'- 切点追踪错误：{json.dumps(outcome["tip_tracking_errors"], ensure_ascii=False)}', '',
        outcome['tip_observation_scope'], outcome['validation_limit'], '',
        '源码指纹和冻结命令完整记录在同目录 `outcome.json`。',
        f'源码目录：`{outcome["source_root"]}`', '',
        '日志（未创建的文件可能不存在）：', '',
        *(f'- {name}：`{path}`' for name, path in outcome['logs'].items()), '']
    (folder/'outcome.md').write_text('\n'.join(lines), encoding='utf-8')
    return outcome


def run_jobs(output, state, env):
    """The queue records process outcome independently of mechanical completion."""
    state.update(status='running', pid=os.getpid(), started=now())
    write_json(output/'queue.json', state)
    run_comparison(output, state, env, 'queue_started')
    for job in state['jobs']:
        job.update(status='running', started=now())
        job_started = time.monotonic()
        report = {}
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
                        killed = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                                stdout=stderr, stderr=stderr, check=False,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
                        if killed.returncode and process.poll() is None:
                            # A restricted Windows token can deny taskkill even
                            # though our existing Popen handle can stop OUR child.
                            # This fallback makes no claim about descendants.
                            process.kill()
                            job['termination_note'] = ('taskkill denied; owned child terminated '
                                                       'using Popen handle; descendants unverified')
                    else:
                        import signal
                        os.killpg(process.pid, signal.SIGKILL)
                    job['exit_code'] = process.wait(timeout=30)
                    job['status'] = 'timed_out'
            report_path = folder/'result.json'
            report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
            job['mechanics_completed'] = mechanical_completion(report, folder, state.get('target_mm', 3.))
            job['last_depth_mm'] = report.get('last_depth_mm')
            job['visualization'] = report.get('visualization')
            if job['status'] != 'timed_out':
                job['status'] = ('completed' if job['exit_code'] == 0
                                 and job['mechanics_completed'] else 'failed')
            if report.get('error'):
                job['error'] = report['error']
        except Exception as exc:
            # A native crash, invalid/missing result or launch failure cannot skip later cases.
            job.update(status='failed', mechanics_completed=False, error=f'{type(exc).__name__}: {exc}')
        job['finished'] = now()
        job['elapsed_s'] = time.monotonic()-job_started
        print(f"END {job['name']} status={job['status']} last_depth_mm={job.get('last_depth_mm')}", flush=True)
        write_json(output/'queue.json', state)
        try:
            outcome = write_outcome(job, state, report)
            job['outcome_status'] = outcome['status']
        except Exception as exc:
            # Reporting is separate from mechanics and cannot skip later jobs.
            job['outcome_error'] = f'{type(exc).__name__}: {exc}'
        write_json(output/'queue.json', state)
        run_comparison(output, state, env, 'case_finished:'+job['name'])
    wait_observed(output, state, env)
    run_comparison(output, state, env, 'queue_finished')
    state.update(status='finished', finished=now(),
                 successful_cases=sum(j['status'] == 'completed' for j in state['jobs']),
                 all_mechanics_completed=all(j.get('mechanics_completed', False)
                                             for j in state['jobs']+state.get('observed_runs', [])))
    write_json(output/'queue.json', state)


def prepare(output, threads, seconds, *, friction_only=False, resume_first=None,
            cases=None, track_tip_contact=False, comparison_plan=None, observed_runs=None,
            max_iterations=None, predictor='affine', adaptive_load=False,
            minimum_increment=.0003125, accuracy_reference=None, shear_modulus=None,
            friction_globalization='model-error', process_policy='compute'):
    output = Path(output).resolve()
    if threads < 1 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Nonnegative finite budget and positive threads required')
    if friction_globalization not in ('model-error', 'residual-merit'):
        raise ValueError('Friction globalization must be model-error or residual-merit')
    if process_policy not in ('system', 'compute'):
        raise ValueError('Process policy must be system or compute')
    if shear_modulus is None:
        from prototype.model_config import FEMConfig
        shear_modulus = FEMConfig().shear_modulus_n_mm2
    if not math.isfinite(shear_modulus) or shear_modulus <= 0:
        raise ValueError('Shear modulus must be finite and positive')
    if max_iterations is not None and (not isinstance(max_iterations, int) or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')
    if predictor not in ('affine', 'secant'):
        raise ValueError('Predictor must be affine or secant')
    if adaptive_load and (not math.isfinite(minimum_increment) or not 0 < minimum_increment <= .01):
        raise ValueError('Adaptive minimum increment must be finite, positive and no larger than .01 mm')
    if accuracy_reference is not None and comparison_plan is not None:
        raise ValueError('--accuracy-reference and --comparison-plan are mutually exclusive')
    if cases is not None and (friction_only or resume_first is not None):
        raise ValueError('Explicit cases cannot be combined with friction-only or resume-first')
    if resume_first is not None and not friction_only:
        raise ValueError('First-case continuation requires --friction-only')
    selected = [(a, mu) for a, mu in CASES if mu > 0] if friction_only else CASES
    if cases is not None:
        selected = [parse_case(c if isinstance(c, str) else f'{c[0]}:{c[1]}') for c in cases]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Explicit cases must be nonempty and unique')
    reference = None
    if accuracy_reference is not None:
        if len(selected) != 1:
            raise ValueError('Accuracy comparison requires exactly one case')
        reference = Path(accuracy_reference).resolve()
        reference_report = json.loads((reference/'result.json').read_text(encoding='utf-8-sig'))
        try:
            depths = [float(step['depth_mm']) for step in reference_report['steps']]
            valid = (reference_report.get('status') == 'completed'
                     and reference_report.get('completed') is True
                     and abs(float(reference_report['target_mm'])-3.) <= 1e-10
                     and abs(float(reference_report['last_depth_mm'])-3.) <= 1e-10
                     and len(depths) >= 2 and all(math.isfinite(d) for d in depths)
                     and abs(depths[0]) <= 1e-10 and abs(depths[-1]-3.) <= 1e-10
                     and all(b > a for a, b in zip(depths, depths[1:])))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('Accuracy reference must report a completed accepted history from 0 to 3 mm')
        reference_config = reference_report.get('config', {})
        if (('shear_modulus_n_mm2' in reference_config
             and reference_config['shear_modulus_n_mm2'] != shear_modulus)
                or reference_config.get('ogden_alpha', 2.) != 2.
                or reference_report.get('contact_formulation', 'improved-max') != 'improved-max'):
            raise ValueError('Accuracy reference material/contact differs; request the matching legacy G explicitly')
        if not (ROOT/'tools/compare_ipc_accuracy.py').is_file():
            raise ValueError('Accuracy comparison script is missing')
    plan = None
    if comparison_plan is not None:
        plan = json.loads(Path(comparison_plan).read_text(encoding='utf-8'))
        if not isinstance(plan, dict):
            raise ValueError('Comparison plan must be a JSON object')
        if not (ROOT/'tools/compare_ipc_study.py').is_file():
            raise ValueError('Comparison script is missing')
    observed = []
    for requested in observed_runs or []:
        item = dict(requested) if isinstance(requested, dict) else dict(output=str(requested))
        folder = Path(item['output']).resolve()
        report = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        pid = int(item.get('pid', report.get('pid', 0)))
        if pid <= 0 or report.get('pid') != pid:
            raise ValueError('Observed run PID does not match its report')
        if str(folder) in [r['output'] for r in observed]:
            raise ValueError('Duplicate observed run')
        observed.append(dict(output=str(folder), pid=pid, process_identity=process_identity(pid),
                             status='pending', ownership='external_read_only'))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Queue output must be empty; existing queues are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/'source'
    fingerprints = source_hashes(snapshot=frozen)
    for relative in ('prototype/ipc_baseline.json', 'tools/visualize_ipc.py',
                     'tools/run_ipc_queue.py', 'requirements.txt',
                     *(('tools/compare_ipc_study.py',) if plan is not None else ()),
                     *(('tools/compare_ipc_accuracy.py',) if reference is not None else ())):
        import hashlib
        target = frozen/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        fingerprints[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    common = ['--hessian-mode', 'regularized', '--tip-radius', '.02',
              '--local-spacing', '.05', '--surface-spacing', '.005',
              '--fine-halfwidth', '3', '--nose-facets', '64', '--dhat', '.0025',
              '--increment', '.01', '--depth', '3',
              '--threads', str(threads), '--seconds', str(seconds), '--predictor', predictor,
              '--shear-modulus', str(shear_modulus), '--ogden-alpha', '2',
              '--contact-formulation', 'improved-max',
              '--friction-globalization', friction_globalization,
              '--process-policy', process_policy]
    if adaptive_load:
        common += ['--adaptive-load', '--minimum-increment', str(minimum_increment)]
    if track_tip_contact:
        common.append('--track-tip-contact')
    jobs = []
    for index, (angle, friction) in enumerate(selected, 1):
        name = f'{index:02d}_angle{angle}_mu{friction:g}'
        folder = output/name
        command = [sys.executable, '-u', '-m', 'prototype.ipc_reference', '--output', str(folder),
                   '--inclination', str(angle), '--friction', str(friction),
                   '--max-iterations', str(max_iterations if max_iterations is not None
                                          else 500 if friction > 0 else 200), *common]
        if index == 1 and resume_first is not None:
            command += ['--resume', str(Path(resume_first).resolve())]
        jobs.append(dict(name=name, angle=angle, friction=friction, output=str(folder),
                         command=command, status='pending'))
    state = dict(status='pending', created=now(), source_root=str(frozen),
                 source_sha256=fingerprints, python=sys.executable, threads=threads,
                 target_mm=3.,
                 solve_budget_s=seconds, process_timeout_s=seconds+1800 if seconds > 0 else None,
                 native_python_path=str(ROOT/'.runtime/ipc-python'), jobs=jobs,
                 observed_runs=observed, track_tip_contact=track_tip_contact,
                 predictor=predictor, adaptive_load=adaptive_load,
                 friction_globalization=friction_globalization, process_policy=process_policy,
                 shear_modulus_mpa=shear_modulus, ogden_alpha=2., contact_formulation='improved-max',
                 minimum_increment_mm=minimum_increment if adaptive_load else None)
    if plan is not None:
        import hashlib
        plan_path = output/'study_plan.json'
        shutil.copyfile(comparison_plan, plan_path)
        state.update(comparison_plan=str(plan_path),
                     comparison_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                     comparison_output=str(output/'comparison'),
                     comparison_command=[sys.executable, '-u', str(frozen/'tools/compare_ipc_study.py'),
                                         '--plan', str(plan_path), '--output', str(output/'comparison')])
    if reference is not None:
        import hashlib
        state.update(accuracy_reference=str(reference),
                     accuracy_reference_result_sha256=hashlib.sha256((reference/'result.json').read_bytes()).hexdigest(),
                     comparison_output=str(output/'comparison'),
                     comparison_command=[sys.executable, '-u', str(frozen/'tools/compare_ipc_accuracy.py'),
                                         '--reference', str(reference), '--candidate', jobs[0]['output'],
                                         '--output', str(output/'comparison'), '--allow-pending'])
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
    parser.add_argument('--cases', nargs='+', help='Explicit unique ANGLE:MU pairs, e.g. 60:0 60:.05')
    parser.add_argument('--max-iterations', type=int, help='Override the per-case iteration budget')
    parser.add_argument('--shear-modulus', type=float,
                        help='G in MPa; defaults to current catalogue value. Use .66 explicitly for a legacy study')
    parser.add_argument('--friction-globalization', choices=('model-error', 'residual-merit'),
                        default='model-error', help='Explicit frozen friction solver globalization')
    parser.add_argument('--process-policy', choices=('system', 'compute'), default='compute',
                        help='Explicit frozen process runtime policy; compute requests Windows HighQoS')
    parser.add_argument('--track-tip-contact', action='store_true')
    parser.add_argument('--predictor', choices=('affine', 'secant'), default='affine')
    parser.add_argument('--adaptive-load', action='store_true', help='Retry and bisect failed loading increments')
    parser.add_argument('--minimum-increment', type=float, default=.0003125,
                        help='Smallest adaptive cutback increment in mm; ignored without --adaptive-load')
    comparisons = parser.add_mutually_exclusive_group()
    comparisons.add_argument('--comparison-plan', type=Path, help='Existing JSON comparison plan to freeze')
    comparisons.add_argument('--accuracy-reference', type=Path,
                             help='Completed 3 mm reference for one-case exact-depth accuracy comparison')
    parser.add_argument('--observe-existing', type=Path, nargs='+', default=[],
                        help='Observe existing runs to final status without restarting or terminating them')
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.seconds < float('inf'):
        parser.error('Nonnegative finite budget and positive threads required')
    if args.resume_first is not None and not args.friction_only:
        parser.error('--resume-first requires --friction-only')
    if args.cases is not None and (args.friction_only or args.resume_first is not None):
        parser.error('--cases cannot be combined with --friction-only or --resume-first')
    output = args.output.resolve()
    try:
        state = (json.loads((output/'queue.json').read_text(encoding='utf-8')) if args.worker
                 else prepare(output, args.threads, args.seconds, friction_only=args.friction_only,
                              resume_first=args.resume_first, cases=args.cases,
                              track_tip_contact=args.track_tip_contact, comparison_plan=args.comparison_plan,
                              observed_runs=args.observe_existing, max_iterations=args.max_iterations,
                              predictor=args.predictor, adaptive_load=args.adaptive_load,
                              minimum_increment=args.minimum_increment, accuracy_reference=args.accuracy_reference,
                              shear_modulus=args.shear_modulus,
                              friction_globalization=args.friction_globalization,
                              process_policy=args.process_policy))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
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
