"""Bounded, content-addressed activation and rejected-attempt evidence.

Observers do not decide acceptance. Rejection retention is first/worst/latest per
signature, at most eight signatures; obsolete unreferenced representative files
are removed only inside this store's newly created rejected/ directory.
"""
import hashlib,json,re
from pathlib import Path
import numpy as np


def native(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v


def array_hash(fields):
    h=hashlib.sha256()
    for name,value in sorted(fields.items()):
        a=np.ascontiguousarray(value);h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def accepted_fields(p,x):
    d=dict(positions=x,velocity=p.velocity,history=p.history,material_metric=p.material_metric,released=p.released,active_contact=p.active_contact,active_faces=p.active_faces)
    return {k:np.array(v,copy=True) for k,v in d.items()}


def trial_fields(p,x):
    d=dict(positions=x,increment=p.step_increment(x),previous_positions=p.previous,previous_velocity=p.previous_velocity,
       previous_history=p.previous_history,previous_material_metric=p.previous_material_metric,active_faces=p.active_faces,
       previous_active_faces=p.previous_active_faces,released=p.released,active_contact=p.active_contact,
       velocity=p.endpoint_velocity(x),material_dt_s=p.material_dt,previous_barrier_energy_n_mm=p.previous_barrier_energy,
       reference_xy=p.rest,mass=p.mass,triangles=p.mesh.triangles,precracked_faces=p.precracked_faces)
    for i,(depth,position) in enumerate(getattr(p,'_accepted_predictor_history',[])):
        d[f'predictor_{i}_positions']=position;d[f'predictor_{i}_depth_mm']=depth
    return {k:np.array(v,copy=True) for k,v in d.items()}


def rejection_measurement(error,energy,p):
    """Order only a measured violation of the SAME gate, never missing data as 0."""
    measurement=getattr(error,'failure_measurement',None)
    if measurement is not None:
        m=dict(measurement)
    elif energy is not None and 'Negative' in str(error) and 'friction' in str(error):
        m=dict(quantity='minimum_friction_work',value=min(energy['friction_work_n_mm'],energy['friction_algorithmic_work_n_mm']),limit=-1e-12,units='N mm',comparison='lower')
    elif energy is not None and ('defect' in str(error) or 'energy' in str(error).lower()):
        from release_dynamics import DEFECT_KEYS
        value=sum(abs(energy.get(k,0.)) for k in (*DEFECT_KEYS,'friction_time_defect_n_mm'))
        limit=p.energy_absolute_tolerance+(p.energy_tolerance or 0)*energy['mechanical_reservoir_n_mm']
        m=dict(quantity='absolute_temporal_defect',value=value,limit=limit,units='N mm',comparison='upper')
    else:
        # A narrow parser covers original nonlinear diagnostics without inventing
        # residuals for timeouts or iteration budgets that did not record one.
        match=re.search(r'residual=([+\-0-9.eE]+) N',str(error))
        if match is None:return dict(comparable=False,reason='No measured gate violation recorded')
        m=dict(quantity='free_force_residual_half_n',value=float(match[1]),limit=p.force_tolerance_n,units='N',comparison='upper')
    value,limit=float(m['value']),float(m['limit'])
    if not np.isfinite(value+limit) or m['comparison'] not in ('upper','lower'):
        raise ValueError('Invalid rejection measurement')
    excess=(value-limit) if m['comparison']=='upper' else (limit-value)
    m.update(comparable=True,signed_gate_excess=excess,severity=excess/max(abs(limit),1e-30),severity_units='dimensionless excess / absolute gate')
    return m


class WitnessStore:
    def __init__(self,directory,identity,max_signatures=8):
        self.root=Path(directory)/'witnesses';self.root.mkdir(exist_ok=False)
        (self.root/'activation').mkdir();(self.root/'rejected').mkdir()
        self.identity=native(identity);self.max_signatures=max_signatures;self.signatures={};self.omitted_signatures=0
        self.activations=[];self.trial=None

    def begin(self,p,x,context):
        self.context=native(context);self.before=accepted_fields(p,x);self.before_hash=array_hash(self.before)
        self.activations=[];self.trial=None;self.last_energy=None;self.energy_fields=None;p.evidence_hook=self.observe

    def observe(self,kind,p,x,data):
        if kind=='activation':
            self.activations.append((trial_fields(p,x),native(data)))
        elif kind=='activation_after':
            if self.activations:self.activations[-1][1].update(native(data))
        elif kind=='energy':
            self.last_energy=native(data['ledger']);self.energy_fields={k:np.array(v,copy=True) for k,v in data.items() if k!='ledger'}
            self.energy_positions=x.copy()
        elif kind=='failed':
            fields=trial_fields(p,x)
            fields['released']=p.failed_released.copy();fields['active_contact']=p.failed_active_contact.copy()
            if self.energy_fields is not None and np.array_equal(x,self.energy_positions):fields.update(self.energy_fields)
            else:self.last_energy=None
            self.trial=(fields,native(data))

    def _save(self,subdir,fields,metadata):
        metadata=dict(identity=self.identity,attempt=self.context,parent_accepted_array_sha256=self.before_hash,**metadata)
        fields={**fields,'metadata_json':np.array(json.dumps(native(metadata),sort_keys=True))}
        name=array_hash(fields)+'.npz';path=self.root/subdir/name
        if not path.exists():np.savez_compressed(path,**fields)
        return dict(file=str(path.relative_to(self.root.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def accepted(self,record):
        refs=[]
        for fields,metadata in self.activations:
            refs.append(self._save('activation',fields,dict(kind='activation',decision=metadata)))
        record['activation_witnesses']=refs
        record['activation_witness_verified']=False # only the independent sidecar may certify
        return refs

    def rejected(self,p,x,error,kind):
        after=accepted_fields(p,x);rollback=array_hash(after)==self.before_hash
        if not rollback:raise RuntimeError('Rejected face step changed accepted state')
        if self.trial is None:return dict(saved=False,reason='No actual trial reached',rollback_verified=True)
        fields,stage=self.trial
        energy=self.last_energy
        signature=kind+'|'+str(stage.get('stage'))+'|'+re.sub(r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?','#',str(error).split(';')[0])
        measurement=rejection_measurement(error,energy,p)
        # Different quantities/gates cannot compete for one worst representative.
        signature+='|'+str(measurement.get('quantity','unmeasured'))+'|'+str(measurement.get('comparison'))+'|'+str(measurement.get('units'))
        if signature not in self.signatures and len(self.signatures)>=self.max_signatures:
            self.omitted_signatures+=1;return dict(saved=False,reason='Signature cap',rollback_verified=True)
        ref=self._save('rejected',fields,dict(kind='rejected',error=str(error),failure_kind=kind,stage=stage,
            rejection_measurement=measurement,energy=energy,negative_work_gate_n_mm=-1e-12,rollback_array_sha256=array_hash(after),rollback_verified=rollback))
        item=self.signatures.setdefault(signature,dict(first=ref,worst=None,worst_score=None,worst_status='not_comparable'))
        if measurement['comparable'] and (item['worst_score'] is None or measurement['severity']>item['worst_score']):
            item.update(worst=ref,worst_score=measurement['severity'],worst_status='measured_gate_violation')
        item['latest']=ref
        retained={q[k]['file'] for q in self.signatures.values() for k in ('first','worst','latest') if q.get(k) is not None}
        for path in (self.root/'rejected').glob('*.npz'):
            if str(path.relative_to(self.root.parent)) not in retained:path.unlink()
        manifest=dict(signatures=self.signatures,maximum_signatures=self.max_signatures,maximum_files=3*self.max_signatures,
                      omitted_new_signatures=self.omitted_signatures,scope='first/worst/latest retained; attempt rows reference this index, not mutable latest filenames')
        (self.root/'representatives.json').write_text(json.dumps(manifest,indent=2))
        return dict(saved=True,representative_index='witnesses/representatives.json',signature=signature,rollback_verified=True)


def qualification(rows,precrack_mm):
    model=any(r.get('face_events') or r.get('insertion_events') for r in rows)
    events=[r for r in rows if r.get('face_events')]
    return dict(preexisting_connected_length_mm=float(precrack_mm),
        new_connected_failed_length_mm=max([0.,*[float(r.get('surface_connected_failed_length_mm',0))-precrack_mm for r in rows]]),
        model_activation_occurred=bool(model),activation_witness_verified=bool(events) and all(r.get('activation_witness_verified',False) for r in events),
        blade_contacted_strict_crack_face=any(r.get('blade_entered_surface_channel',False) for r in rows),
        blade_contacted_strict_new_face=any(r.get('blade_contacted_strict_new_face',False) for r in rows),
        physical_initiation_prediction_validated=False)
