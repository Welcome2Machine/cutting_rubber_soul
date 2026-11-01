import copy
import pytest
from tools.report_staged_c1_event import verify_link

def test_only_complete_accepted_parent_and_identical_equations_can_link_phases():
    p=dict(steps=[dict(state_sha256='state',depth_mm=.4)],source_hashes={'kernel':'same'},config={'closed':True},material_layer='maxwell',solve_tolerance_n=1e-8,initial_time_s=1.,window_s=.1)
    q=dict(parent=dict(result_sha256='result',audit_sha256='audit',state_sha256='state'),source_hashes=p['source_hashes'],config=p['config'],material_layer='maxwell',solve_tolerance_n=1e-8,initial_time_s=1.1,initial_depth_mm=.4,conditional_comparison=True,production_continuation=False)
    verify_link(p,q,'result','audit')
    for key,value in [('source_hashes',{'kernel':'changed'}),('initial_time_s',1.2),('production_continuation',True),('parent',dict(result_sha256='result',audit_sha256='wrong',state_sha256='state'))]:
        bad=copy.deepcopy(q);bad[key]=value
        with pytest.raises(ValueError):verify_link(p,bad,'result','audit')
