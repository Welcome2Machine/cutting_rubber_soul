import copy,json,time
import numpy as np
import pytest
from test_coupled_cohesive import patch,config
from tools.face_witness import WitnessStore,accepted_fields,native
from tools.audit_face_witness import restore,inspect_one
from tools.run_coupled_growth import sources
from tools.frozen_scene import digest
from coupled_cohesive import problem
from face_cut import FaceCutProblem
from prototype.zero_gap_cut import EnergyResolutionRequired


def test_strong_inertia_rejection_saves_final_ledger_and_legacy_is_explicit(tmp_path):
    p=patch();x=p.rest.copy();x[p.seam[44],0]+=.001
    p.history=p.cohesive_response(x,p.history)['history'].copy();p.previous_history=p.history.copy();p.velocity[p.seam[44],0]=100.
    parent=tmp_path/'parent.npz';np.savez_compressed(parent,**accepted_fields(p,x));src=sources()
    store=WitnessStore(tmp_path,dict(config=config(),source_hashes=src));h=1e-8
    store.begin(p,x,dict(parent=dict(file=parent.name,sha256=digest(parent)),from_depth_mm=p.start,target_depth_mm=p.start+p.loading_rate*h,physical_dt_s=h))
    with pytest.raises(EnergyResolutionRequired) as caught:p.step(x,p.start,p.start+p.loading_rate*h,time.monotonic()+30)
    store.rejected(p,x,caught.value,'physical_energy_resolution');ref=next(iter(store.signatures.values()))['latest']
    factory=lambda c:problem(c,solve_tolerance_n=1e-7)
    checked=inspect_one(tmp_path,ref,src,time.monotonic()+30,factory=factory)
    assert checked['passed'] and checked['recorded_energy_is_final_coupled_ledger']
    assert checked['energy']['backward_euler_inertia_loss_n_mm']>1e-12
    with np.load(tmp_path/ref['file'],allow_pickle=False) as z:fields={k:z[k].copy() for k in z.files if k!='metadata_json'};meta=json.loads(str(z['metadata_json']))
    model=restore(config(),fields,factory=factory);legacy=copy.deepcopy(src)
    legacy['tools/face_witness.py']='6601c1768f0c799107bd3248828c74ca57ae5e27f01f7a98f563e590dd49a675'
    legacy['coupled_contact.py']='44dacd93e04275579eb16c11c609402bac0f36ae2967df36f140b58a5528f2ae'
    meta['identity']['source_hashes']=legacy;meta['energy']=FaceCutProblem.energy_ledger(model,fields['positions'])
    path=tmp_path/'witnesses/rejected/legacy.npz'
    def write():
        np.savez_compressed(path,**fields,metadata_json=np.array(json.dumps(native(meta))))
        return dict(file=str(path.relative_to(tmp_path)),sha256=digest(path))
    old=inspect_one(tmp_path,write(),legacy,time.monotonic()+30,factory=factory)
    assert old['passed'] and not old['recorded_energy_is_final_coupled_ledger']
    assert old['energy']['backward_euler_inertia_loss_n_mm']==checked['energy']['backward_euler_inertia_loss_n_mm']
    meta['energy']['external_work_n_mm']+=.001
    with pytest.raises(AssertionError):inspect_one(tmp_path,write(),legacy,time.monotonic()+30,factory=factory)
