import copy,json
import numpy as np
import pytest
from face_cut import eligible_nodes
from tools.face_witness import array_hash
from tools.frozen_scene import digest
from tools.face_activation_chain import transition


def scene(tmp_path,changes=((1,2),)):
    before=dict(positions=np.zeros((4,2)),velocity=np.zeros((4,2)),history=np.zeros((3,2)),
        material_metric=np.ones((1,2,2)),active_faces=np.array([False,False,False]),active_contact=np.zeros(4,bool))
    before['released']=eligible_nodes(before['active_faces'])
    np.savez(tmp_path/'parent.npz',**before)
    parent=dict(file='parent.npz',sha256=digest(tmp_path/'parent.npz'))
    row=dict(time_s=1.,face_events=[],activation_witnesses=[])
    active=before['active_faces'].copy();seam=np.arange(4)
    (tmp_path/'witnesses').mkdir()
    for i,faces in enumerate(changes):
        s={'previous_'+k:before[k] for k in ('positions','velocity','history','material_metric','active_faces')}
        s.update(active_faces=active.copy(),released=eligible_nodes(active),material_dt_s=.1)
        old=eligible_nodes(active);active[list(faces)]=True
        event=dict(new_faces=list(faces),new_free_nodes=seam[eligible_nodes(active)&~old].tolist())
        m=dict(kind='activation',identity=dict(config={},source_hashes={}),attempt=dict(parent=parent,
            from_depth_mm=.1,target_depth_mm=.12,physical_dt_s=.1),parent_accepted_array_sha256=array_hash(before),
            decision=dict(event,target_depth_mm=.12))
        f=tmp_path/f'witnesses/{i}.npz';np.savez(f,**s,metadata_json=json.dumps(m))
        row['face_events'].append(event);row['activation_witnesses'].append(dict(file=str(f.relative_to(tmp_path)),sha256=digest(f)))
    state=dict(previous_active_faces=before['active_faces'],active_faces=active,released=eligible_nodes(active),depth_mm=.12,material_dt_s=.1)
    return before,state,row,parent,seam


def check(tmp_path,args):
    before,state,row,parent,seam=args
    return transition(tmp_path,before,state,row,parent,.1,{}, {},seam)


def test_multiface_event_and_two_ordered_events(tmp_path):
    assert check(tmp_path,scene(tmp_path))['new_faces']==[1,2]
    other=tmp_path/'two';other.mkdir()
    assert len(check(other,scene(other,((2,),(1,))))['events'])==2


@pytest.mark.parametrize('fault',['missing_witness','declared_without_change','hidden_change','wrong_parent','wrong_faces','wrong_time','reordered'])
def test_incomplete_or_misbound_chain_rejected(tmp_path,fault):
    args=scene(tmp_path,((2,),(1,)));before,state,row,parent,seam=args
    if fault=='missing_witness':row['activation_witnesses'].pop()
    elif fault=='declared_without_change':state.update(active_faces=before['active_faces'].copy(),released=before['released'].copy())
    elif fault=='hidden_change':row.update(face_events=[],activation_witnesses=[])
    elif fault=='reordered':
        row['face_events'].reverse();row['activation_witnesses'].reverse()
    else:
        ref=row['activation_witnesses'][0];f=tmp_path/ref['file']
        with np.load(f,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        m=json.loads(str(s.pop('metadata_json')))
        if fault=='wrong_parent':m['attempt']['parent']=dict(parent,file='other.npz')
        elif fault=='wrong_faces':m['decision']['new_faces']=[1]
        elif fault=='wrong_time':m['attempt']['physical_dt_s']=.2
        np.savez(f,**s,metadata_json=json.dumps(m));ref['sha256']=digest(f)
    with pytest.raises((ValueError,AssertionError)):check(tmp_path,args)
