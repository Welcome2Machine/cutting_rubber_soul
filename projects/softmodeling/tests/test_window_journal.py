import json,os,subprocess,sys
from pathlib import Path
import pytest
from tools.window_journal import Journal,inspect
from tools.frozen_scene import digest


def test_forced_exit_retains_last_accepted_record_without_certifying_completion(tmp_path):
    script='''import os,sys
from pathlib import Path
from tools.window_journal import Journal
from tools.frozen_scene import digest
p=Path(sys.argv[1]);j=Journal(p,dict(parent_sha256='parent',status='running'))
f=p/'state_00000.npz';f.write_bytes(b'exact-state')
j.append('accepted',dict(record=dict(state_file=f.name,state_sha256=digest(f),time_s=.25),partition_decision=dict(status='accepted',right_s=.25)))
os._exit(7)
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    q=subprocess.run([sys.executable,'-c',script,str(tmp_path)],env=env)
    assert q.returncode==7
    a=inspect(tmp_path)
    assert len(a['records'])==2 and not a['terminal_record_present']
    assert not a['physical_qualification'] and not a['production_result_reconstructed']
    assert a['records'][1]['payload']['partition_decision']['right_s']==.25


def test_rejection_and_terminal_do_not_grant_qualification(tmp_path):
    j=Journal(tmp_path,dict(status='running'));f=tmp_path/'rejected.npz';f.write_bytes(b'trial')
    j.append('rejected',dict(record=dict(trial_file=f.name,trial_sha256=digest(f),failure_kind='physical_energy_resolution',rollback_verified=True),partition_decision=dict(bisected=True)))
    j.append('terminal',dict(status='global_compute_budget_exhausted'))
    q=inspect(tmp_path);assert q['terminal_record_present'] and not q['event_complete']
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed journal state'):inspect(tmp_path)


@pytest.mark.parametrize('damage',['edit','truncate','delete'])
def test_corruption_or_truncated_tail_is_rejected(tmp_path,damage):
    j=Journal(tmp_path,dict(status='running'));j.append('terminal',dict(status='completed_window'))
    lines=j.path.read_bytes().splitlines(keepends=True)
    if damage=='edit':lines[0]=lines[0].replace(b'running',b'stopped')
    elif damage=='truncate':lines[-1]=lines[-1][:-1]
    else:lines=lines[1:]
    j.path.write_bytes(b''.join(lines))
    with pytest.raises(ValueError):inspect(tmp_path)
