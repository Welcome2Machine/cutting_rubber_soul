import json
import time
import numpy as np
import pytest
from test_energy_cut import candidate
from tools.run_zero_gap_cut import problem, snapshot
from tools.dynamic_release import sources
from tools.frozen_scene import digest
from tools.audit_contact_time import inspect, combine
from stiff_contact import install, momentum_ledger


def test_bounded_chunks_cannot_certify_gaps_duplicates_or_changed_states(tmp_path):
    cfg=candidate();p=install(problem(cfg),'endpoint_be');x=p.rest.copy();depth=p.start;rows=[]
    for i,target in enumerate((-.015,-.0145)):
        released=p.released.copy();x,rec=p.step(x,depth,target,time.monotonic()+20);depth=target
        state=snapshot(p,x,released,depth);fields,rec['impulse']=momentum_ledger(p,x);state.update(fields)
        path=tmp_path/f'state_{i}.npz';np.savez_compressed(path,**state)
        rec.update(local_time_s=(depth-p.start)/p.loading_rate,state_file=path.name,state_sha256=digest(path));rows.append(rec)
    raw=dict(config=cfg,settings=dict(integrator='endpoint_be'),source_hashes=sources(),steps=rows)
    (tmp_path/'result.json').write_text(json.dumps(raw))
    paths=[]
    for i in range(2):
        report=inspect(tmp_path,20,state_start=i,state_stop=i+1,ipc_threads=2,blas_threads=1)
        assert report['chunk_passed'] and not report['passed'] and not report['full_path_coverage'],report
        path=tmp_path/f'chunk_{i}.json';path.write_text(json.dumps(report));paths.append(path)
    assert combine(tmp_path,paths)['passed']
    assert combine(tmp_path,paths[::-1])['full_path_coverage']
    with pytest.raises(ValueError,match='gap or duplicate'):combine(tmp_path,paths[:1])
    with pytest.raises(ValueError,match='gap or duplicate'):combine(tmp_path,paths+paths[:1])
    with (tmp_path/'state_0.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='state hash'):combine(tmp_path,paths)
