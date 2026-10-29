import os
import pytest
from threadpoolctl import threadpool_info
from numerical_runtime import numerical_runtime, runtime_entry
from prototype.ipc_reference import toolkit


def test_explicit_threads_effective_and_restore_on_exception():
    ipc=toolkit();previous=ipc.get_num_threads()
    pools=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    with pytest.raises(RuntimeError,match='test failure'):
        with numerical_runtime(1,1) as info:
            assert ipc.get_num_threads()==1
            assert all(p['num_threads']==1 for p in info['threadpools_effective'] if p['user_api']=='blas')
            raise RuntimeError('test failure')
    assert ipc.get_num_threads()==previous
    assert pools==[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    assert info['wall_s']>0 and info['process_cpu_s']>=0


def test_nested_thread_budget_cannot_be_exceeded(monkeypatch):
    monkeypatch.setenv('SOFTCUT_THREAD_BUDGET','8')
    with pytest.raises(ValueError,match='nested'):
        with numerical_runtime(8,2):pass


@pytest.mark.parametrize('value',[0,65,1.5,True])
def test_invalid_threads_rejected(value):
    with pytest.raises(ValueError):
        with numerical_runtime(value,1):pass


def test_entry_records_overrides_without_changing_positional_arguments():
    @runtime_entry
    def call(value,*,runtime):
        return dict(value=value,observed=toolkit().get_num_threads())
    result=call(17,ipc_threads=1,blas_threads=1)
    assert result['value']==17 and result['observed']==1
    assert result['thread_runtime']['requested']==dict(ipc_threads=1,blas_threads=1)


def test_continuation_thread_options_cannot_bypass_source_identity(tmp_path):
    import json
    from tools.continue_zero_gap_cut import continue_run
    from prototype.ipc_reference import toolkit
    (tmp_path/'result.json').write_text(json.dumps(dict(source_hashes={})))
    old=toolkit().get_num_threads()
    with pytest.raises(ValueError,match='identical-source'):
        continue_run(tmp_path,tmp_path/'new',30,ipc_threads=1,blas_threads=1)
    assert toolkit().get_num_threads()==old and not (tmp_path/'new').exists()
