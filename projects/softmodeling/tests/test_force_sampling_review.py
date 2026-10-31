import pytest
from tools.force_sampling_review import mechanism_gate


def test_shared_times_keep_extra_peak_and_strict_failure():
    def a(rows):return dict(passed=True,states=[dict(time_s=t,independent_fz_n=f) for t,f in rows])
    audits=[a([(1.,1.),(2.,2.)]),a([(1.,1.0000002),(1.5,3.),(2.,2.0000002)]),a([(1.,1.0000005),(1.5,3.1),(2.,2.0000005)])]
    strict=dict(gates=dict(force_difference_decreases=False,mass_velocity=True,event=True),passed=False)
    r=mechanism_gate(strict,audits,1e-5)
    assert r['common_times_s']==[1.,2.] and r['peaks'][2]['extra_samples_peak']['fz_n']==3.1
    assert r['mechanism_force_consistency_passed'] and not r['strict_passed']
    strict['gates']['event']=False
    assert not mechanism_gate(strict,audits,1e-5)['mechanism_force_consistency_passed']
    audits[2]['states'][-1]['independent_fz_n']=2.001
    assert not mechanism_gate(dict(gates=dict(force_difference_decreases=True),passed=True),audits,1e-5)['mechanism_force_consistency_passed']
    with pytest.raises(ValueError):mechanism_gate(strict,audits,1e-4)
