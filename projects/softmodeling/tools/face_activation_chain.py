"""Bind ordered activation evidence to actual accepted NPZ topology changes.

One graph event may activate several faces. Mechanical witness replay remains
an additional check; this module proves completeness and parent/step binding.
"""
import json
from pathlib import Path
import numpy as np
from face_cut import eligible_nodes
from .face_witness import array_hash
from .frozen_scene import digest
from .coupled_audit_contract import close


def transition(directory, previous, state, row, parent_ref, from_depth, config, source, seam):
    directory=Path(directory)
    before=np.asarray(previous['active_faces'],bool)
    after=np.asarray(state['active_faces'],bool)
    if before.shape!=after.shape or np.any(before & ~after):
        raise ValueError('Activation chain: healing or incompatible topology')
    np.testing.assert_array_equal(state['previous_active_faces'],before)
    np.testing.assert_array_equal(previous['released'],eligible_nodes(before))
    np.testing.assert_array_equal(state['released'],eligible_nodes(after))
    events=row.get('face_events',[]);refs=row.get('activation_witnesses',[])
    if len(events)!=len(refs):raise ValueError('Activation chain: event missing witness')
    active=before.copy();records=[];used=set()
    parent_fields={k:previous[k] for k in ('positions','velocity','history','material_metric','released','active_contact','active_faces')}
    for order,(event,ref) in enumerate(zip(events,refs)):
        path=directory/ref['file']
        if not path.resolve().is_relative_to((directory/'witnesses').resolve()) or digest(path)!=ref['sha256'] or ref['sha256'] in used:
            raise ValueError('Activation chain: invalid or duplicate witness')
        used.add(ref['sha256'])
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata_json']));s={k:z[k].copy() for k in z.files if k!='metadata_json'}
        if m['kind']!='activation' or m['identity']['config']!=config or m['identity']['source_hashes']!=source:
            raise ValueError('Activation chain: witness identity mismatch')
        attempt=m['attempt'];decision=m['decision']
        if attempt['parent']!=parent_ref or digest(directory/parent_ref['file'])!=parent_ref['sha256'] or m['parent_accepted_array_sha256']!=array_hash(parent_fields):
            raise ValueError('Activation chain: wrong parent')
        close(attempt['from_depth_mm'],from_depth,'Activation parent depth')
        close(attempt['target_depth_mm'],float(state['depth_mm']),'Activation target depth')
        close(decision['target_depth_mm'],float(state['depth_mm']),'Activation decision depth')
        close(attempt['physical_dt_s'],float(state['material_dt_s']),'Activation physical dt')
        close(float(s['material_dt_s']),float(state['material_dt_s']),'Activation saved dt')
        for key in ('positions','velocity','history','material_metric','active_faces'):
            np.testing.assert_array_equal(s['previous_'+key],previous[key])
        np.testing.assert_array_equal(s['active_faces'],active)
        np.testing.assert_array_equal(s['released'],eligible_nodes(active))
        faces=event['new_faces']
        if not faces or len(set(faces))!=len(faces) or any(type(f) is not int or f<0 or f>=len(active) or active[f] for f in faces):
            raise ValueError('Activation chain: invalid new face set')
        if sorted(faces)!=sorted(decision['new_faces']):raise ValueError('Activation chain: wrong witness face set')
        for key,value in event.items():
            if key not in decision or decision[key]!=value:raise ValueError('Activation chain: event/decision disagreement: '+key)
        old_free=eligible_nodes(active);active[faces]=True
        expected=np.asarray(seam)[eligible_nodes(active)&~old_free].tolist()
        if sorted(event['new_free_nodes'])!=sorted(expected):raise ValueError('Activation chain: released topology mismatch')
        records.append(dict(order=order,new_faces=faces,witness=ref,parent=parent_ref,
            from_depth_mm=from_depth,depth_mm=float(state['depth_mm']),time_s=row['time_s'],dt_s=float(state['material_dt_s'])))
    if not np.array_equal(active,after):raise ValueError('Activation chain: NPZ changes not exactly covered by ordered events and witnesses')
    return dict(new_faces=np.flatnonzero(after&~before).tolist(),events=records,complete=True)
