"""Shared outputs, source provenance, and reversible process runtime policy."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def background_child_options():
    """Run supervised compute children without a Windows console window.

    A detached supervisor alone does not specify its children's console flags.
    Keep explicit terminate()/wait() available; do not detach from supervision.
    """
    import subprocess
    if sys.platform == 'win32':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        return dict(startupinfo=startup,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    return dict(start_new_session=True)


def classify_process_exit(returncode, *, supervisor_timeout=False):
    """Process outcome only: zero exit is not proof of FE acceptance/audit."""
    if supervisor_timeout:
        return 'supervisor_timeout'
    if returncode is None:
        return 'running'
    if int(returncode) & 0xffffffff == 0xc000013a:
        return 'interrupted_control_event'
    return 'exited_normally' if returncode == 0 else 'process_failed'


class _WindowsProcessPowerPolicy:
    """Documented ProcessPowerThrottling API, scoped to this process only."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG),
                        ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]

        self.ctypes, self.State = ctypes, State
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.process = self.kernel.GetCurrentProcess()
        for name in ('GetProcessInformation', 'SetProcessInformation'):
            function = getattr(self.kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                 ctypes.c_void_p, wintypes.DWORD]
            function.restype = wintypes.BOOL

    def read(self):
        value = self.State(1, 0, 0)
        if not self.kernel.GetProcessInformation(
                self.process, 4, self.ctypes.byref(value), self.ctypes.sizeof(value)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return dict(version=value.Version, control_mask=value.ControlMask,
                    state_mask=value.StateMask)

    def write(self, state):
        value = self.State(state['version'], state['control_mask'], state['state_mask'])
        if not self.kernel.SetProcessInformation(
                self.process, 4, self.ctypes.byref(value), self.ctypes.sizeof(value)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())


@contextmanager
def compute_process_policy(mode='compute'):
    """Prefer performance cores on Windows; restore the original policy on exit.

    ``compute`` explicitly disables execution-speed throttling for this process.
    ``system`` and non-Windows platforms leave runtime policy untouched. Neither
    mode changes affinity, priority, system power settings, or numerical inputs.
    API failures are recorded in the yielded, mutable metadata dictionary; they
    must not replace an exception raised by the computation. ``active`` records
    the policy read back at entry, while ``restored`` is updated at exit.

    This is process-wide: use around the run, not independently in worker threads.
    See Microsoft's SetProcessInformation / PROCESS_POWER_THROTTLING_STATE docs.
    """
    if mode not in ('system', 'compute'):
        raise ValueError("process policy must be 'system' or 'compute'")
    metadata = dict(requested=mode, platform=sys.platform, applied=False,
                    previous=None, active=None, restored=None, error=None)
    if mode == 'system' or sys.platform != 'win32':
        metadata['reason'] = 'system_policy_unchanged' if mode == 'system' else 'non_windows'
        yield metadata
        return

    api, previous, attempted = None, None, False

    def record_error(stage, exc):
        message = f'{stage}: {type(exc).__name__}: {exc}'
        metadata['error'] = '; '.join(filter(None, (metadata['error'], message)))

    def restore():
        try:
            api.write(previous)
            observed = api.read()
            if observed != previous:
                raise RuntimeError(f'policy restoration readback differs: {observed}')
            metadata['restored'] = True
        except Exception as exc:
            metadata['restored'] = False
            record_error('restore', exc)

    try:
        try:
            api = _WindowsProcessPowerPolicy()
            previous = api.read()
            metadata['previous'] = previous.copy()
            wanted = dict(previous, control_mask=previous['control_mask'] | 1,
                          state_mask=previous['state_mask'] & ~1)
            attempted = True
            api.write(wanted)
            metadata['active'] = api.read()
            if metadata['active'] != wanted:
                raise RuntimeError(f'policy application readback differs: {metadata["active"]}')
            metadata['applied'] = True
        except Exception as exc:
            record_error('apply', exc)
            # A successful setter followed by failed readback may have changed
            # the policy. Roll back before computation. The outer finally also
            # covers KeyboardInterrupt/SystemExit during this setup phase.
            if attempted:
                restore()
                metadata['active'] = previous.copy() if metadata['restored'] else None
        yield metadata
    finally:
        if attempted and metadata['restored'] is not True:
            restore()


def replace_with_retry(temporary, path, *, timeout_s=2.):
    """Bounded retry for Windows sharing/access conflicts on atomic replace.

    Readers (dashboard, editor, antivirus) can briefly deny delete-sharing.
    Never remove the destination first, never report an unwritten result as
    saved, and never swallow persistent permissions or other I/O errors.
    """
    temporary, path = Path(temporary), Path(path)
    deadline = time.monotonic()+timeout_s
    delay = .01
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0., deadline-time.monotonic())))
            delay = min(.1, 2*delay)


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload, encoding='utf-8')
    replace_with_retry(temporary, path)


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    replace_with_retry(temporary, path)


def source_hashes(packages=('prototype', 'mechanics'), *, snapshot=None):
    sources = [Path(__file__).resolve()]
    for package in packages:
        sources.extend((ROOT / package).glob('*.py'))
    result = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        result[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        if snapshot is not None:
            target = Path(snapshot) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return result
