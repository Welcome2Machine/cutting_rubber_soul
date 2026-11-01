import pytest
from tools.audit_coupled_chunks import validate_coverage


def part(a,b,**changes):
    return dict(passed=True,range=[a,b],binding={'result':'fixed'},states=[{}]*(b-a),witnesses_checked=a==0,**changes)


def test_full_disjoint_coverage_and_single_witness_pass():
    assert validate_coverage([part(2,4),part(0,2)],4)=={'result':'fixed'}


@pytest.mark.parametrize('parts',[[part(0,1),part(2,4)],[part(0,3),part(2,4)],[part(0,2)],[part(0,5)]])
def test_missing_duplicate_truncated_or_outside_ranges_rejected(parts):
    with pytest.raises(ValueError):validate_coverage(parts,4)


def test_changed_input_failed_chunk_or_unchecked_witness_rejected():
    for field,value in [('binding',{'result':'other'}),('passed',False),('witnesses_checked',True)]:
        a=part(2,4);a[field]=value
        with pytest.raises(ValueError):validate_coverage([part(0,2),a],4)


def test_real_window_chunk_composition_matches_whole_audit(tmp_path,monkeypatch):
    import json,numpy as np
    from test_coupled_growth_window import control
    from tools import run_coupled_growth as growth,run_fracture_route as route,branch_coupled_window as branch,audit_coupled_chunks as chunks
    c=control();parent=tmp_path/'parent';proof=tmp_path/'parent.audit.json'
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    r=route.run(c,parent,30,stop_depth=.002,ipc_threads=1,blas_threads=1)
    a=growth.audit(parent,proof,30,ipc_threads=1,blas_threads=1);assert a['passed']
    plan=tmp_path/'plan.json';index=len(r['steps'])-1;assert index>=1
    plan.write_text(json.dumps({'coupled_window':dict(parent_state_index=index,dt_s=[.0005],window_s=.001,solve_tolerance_n=1e-7)}))
    out=tmp_path/'window';q=branch.run(parent,proof,index,plan,out,.0005,30,ipc_threads=1,blas_threads=1);assert q['status']=='completed_window'
    whole=branch.audit(out,tmp_path/'whole.json',30,ipc_threads=1,blas_threads=1);assert whole['passed'],whole.get('error')
    parts=[tmp_path/'part0.json',tmp_path/'part1.json']
    for i,path in enumerate(parts):
        a=chunks.run(out,path,i,i+1,30,ipc_threads=1,blas_threads=1);assert a['passed'],a.get('error')
    composed=chunks.compose(out,parts,tmp_path/'composed.json');assert composed['passed'] and composed['complete_requested_window']
    assert composed['time_chain']==whole['time_chain'] and composed['states']==whole['states']
    path=out/q['steps'][0]['state_file'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed audited state'):chunks.compose(out,parts,tmp_path/'tampered.json')
