import numpy as np
from src.analysis.contact_zero_audit import hinge_onset


def example():
    t=np.arange(300)*.0125
    gap=-35+20*t
    good=np.ones(len(t),bool)
    return t,gap,good


def test_motion_hinge_recovers_known_contact_with_slow_baseline_drift():
    t,gap,good=example();onset=1.4
    y=np.column_stack((.08*t+3*np.maximum(t-onset,0),-.03*t+7*np.maximum(t-onset,0)))
    r=hinge_onset(t,y,gap,good)
    assert r['accepted']
    assert abs(t[r['frame']]-onset)<.03


def test_slow_precontact_drift_is_not_contact():
    t,gap,good=example();y=np.column_stack((.2*t,.1*t))
    assert not hinge_onset(t,y,gap,good)['accepted']


def test_missing_precontact_baseline_does_not_create_new_zero():
    t,gap,good=example();gap+=30
    y=np.column_stack((3*t,7*t))
    assert not hinge_onset(t,y,gap,good)['accepted']


def test_brief_optical_glitch_does_not_establish_contact():
    t,gap,good=example();y=np.zeros((len(t),2));y[150:152]=4
    assert not hinge_onset(t,y,gap,good)['accepted']
