import time
import numpy as np
import pytest
from tools.step_doubling import branch, compare, read_state


def test_two_half_steps_commit_actual_maxwell_and_opening_history(tmp_path):
    from test_coupled_cohesive import config
    from tools.run_coupled_growth import problem
    from tools.run_fracture_route import snapshot
    c=dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-7))
    p=problem(c);old=p.released.copy();x,row=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    s=snapshot(p,x,old,p.start+1e-4);predictors=[(float(s['depth_mm']),s['positions'])]
    a=branch(c,s,predictors,row,tmp_path/'1',1e-4,1,time.monotonic()+20)
    b=branch(c,s,predictors,row,tmp_path/'2',1e-4,2,time.monotonic()+20)
    assert a['completed'] and b['completed'] and len(b['independent_audit'])==2
    mid=read_state(tmp_path/'2'/b['steps'][0]['state_file']);end=read_state(tmp_path/'2'/b['steps'][1]['state_file'])
    for key in ('positions','velocity','material_metric','history','active_faces'):
        np.testing.assert_array_equal(end['previous_'+key],mid[key])
    assert not np.array_equal(end['previous_material_metric'],s['material_metric'])
    assert compare(tmp_path/'1',tmp_path/'2')['comparable']

