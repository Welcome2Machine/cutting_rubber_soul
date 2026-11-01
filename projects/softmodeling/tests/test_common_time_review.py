import json
from pathlib import Path
import numpy as np
import pytest
from tools import report_coupled_contact as report


def test_common_scale_keeps_large_finer_peaks_and_old_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'common_inputs',lambda cases:{'verified_by_test_fixture':True})
    monkeypatch.setattr(report,'registered_offsets',lambda q,a:np.array([s['time_s'] for s in a['states']]))
    cases=[]
    for level,times in enumerate(([2.,4.],[1.,2.,3.,4.],[.5,1.,1.5,2.,2.5,3.,3.5,4.])):
        path=tmp_path/str(level);path.mkdir();rows=[];old=0.
        for k,t in enumerate(times):
            value=10 if level==0 else (11 if t in (2.,4.) else 100) if level==1 else (11.5 if t in (2.,4.) else 110)
            f=f'{k}.npz';np.savez(path/f,mass=[1.],velocity=[[value,0.]],impulse_normal=[[t-old,0.]],impulse_friction=[[0.,0.]])
            rows.append(dict(time_s=t,state_file=f));old=t
        cases.append((path,{},dict(states=rows)))
    legacy=dict(passed=False,gates=dict(velocity_difference_decreases=False))
    q=report.common_time_review(cases,legacy)
    assert q['common_sample_count']==2 and q['common_velocity_scale_sqrt_nmm']==11.5
    assert [p['maximum_velocity_difference_sqrt_nmm'] for p in q['pairs']]==[1.,1.5,.5]
    assert q['velocity_difference_decreases'] and q['fine_pair_velocity_5percent']
    assert q['original_comparison']==legacy and not q['original_comparison']['passed']
    assert q['extra_middle_fine_common_times']['maximum_velocity_difference_sqrt_nmm']==10
    assert q['all_sample_peaks'][2]['all_velocity_norm_peak']['value']==110
    assert q['pairs'][0]['largest_dof_contributions'][0]['node']==0
    assert q['original_pairs_reconstructed'][1]['sample_count']==4
    assert q['original_pairs_reconstructed'][1]['original_velocity_relative']>.05
