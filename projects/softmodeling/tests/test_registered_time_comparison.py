import numpy as np
import pytest
from tools.coupled_audit_contract import registered_offsets,common_offset_indices


def test_same_scheduled_time_straddling_decimal_clock_rounding_boundary():
    t=3.3336409057615293;other=3.3336409057615
    assert round(t,12)!=round(other,12)
    raw=dict(initial_time_s=t-.001,dt_requested_s=.001)
    a=registered_offsets(raw,dict(states=[dict(time_s=t)]))
    b=registered_offsets(dict(raw,dt_requested_s=.0005),dict(states=[dict(time_s=t-.0005),dict(time_s=other)]))
    assert common_offset_indices(b,a)==[(0,1)]
    with pytest.raises(ValueError):registered_offsets(raw,dict(states=[dict(time_s=t+1e-5)]))
    with pytest.raises(ValueError):common_offset_indices(np.array([.0006,.0012]),a)
