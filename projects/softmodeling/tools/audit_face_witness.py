"""Reconstruct face decisions from pre-activation arrays, not saved face_events."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .frozen_scene import digest
from .face_events import endpoint_forces
from .face_witness import array_hash,accepted_fields
from . import run_endpoint_face_route as route
from face_cut import eligible_nodes
from endpoint_friction import velocity_increment
from run_support import write_json
from .audit_zero_gap_cut import blade_contacts


def restore(config,state):
    if 'initial_slit_halfwidth_mm' in config:
        from .run_precrack_entry import problem
        p=problem(config)
    else:p=route.problem(config)
    for key in ('previous_positions','previous_velocity','previous_history','previous_material_metric'):
        setattr(p,'previous' if key=='previous_positions' else key,state[key].copy())
    p.velocity=p.previous_velocity.copy();p.history=p.previous_history.copy();p.material_metric=p.previous_material_metric.copy()
    p.active_faces=state['active_faces'].copy();p.previous_active_faces=state['previous_active_faces'].copy()
    p.rebuild_contact();p.active_contact=state['active_contact'].copy();p.set_free()
    np.testing.assert_array_equal(p.released,state['released'])
    np.testing.assert_array_equal(p.rest,state['reference_xy']);np.testing.assert_array_equal(p.mass,state['mass'])
    p.material_dt=float(state['material_dt_s']);p.previous_barrier_energy=float(state['previous_barrier_energy_n_mm'])
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    p.set_primary_increment(state['positions'],state['increment'])
    np.testing.assert_allclose(p.endpoint_velocity(state['positions']),state['velocity'],atol=1e-13,rtol=0)
    return p


def inspect_one(directory,reference,source,deadline):
    if time.monotonic()>deadline:raise TimeoutError('Witness audit budget')
    path=directory/reference['file']
    if not path.resolve().is_relative_to((directory/'witnesses').resolve()) or digest(path)!=reference['sha256']:raise ValueError('Witness identity mismatch')
    with np.load(path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files if k!='metadata_json'};m=json.loads(str(z['metadata_json']))
    if m['identity']['source_hashes']!=source:raise ValueError('Witness producer source mismatch')
    parent=directory/m['attempt']['parent']['file']
    if digest(parent)!=m['attempt']['parent']['sha256']:raise ValueError('Parent file changed')
    with np.load(parent,allow_pickle=False) as z:
        fields={k:z[k].copy() for k in ('positions','velocity','history','material_metric','released','active_contact','active_faces')}
    if array_hash(fields)!=m['parent_accepted_array_sha256']:raise ValueError('Wrong accepted parent')
    for a,b in [('previous_positions','positions'),('previous_velocity','velocity'),('previous_history','history'),('previous_material_metric','material_metric'),('previous_active_faces','active_faces')]:np.testing.assert_array_equal(s[a],fields[b])
    np.testing.assert_allclose(s['material_dt_s'],m['attempt']['physical_dt_s'],rtol=0,atol=1e-14)
    p=restore(m['identity']['config'],s);x=s['positions'];out=dict(file=reference['file'],kind=m['kind'],passed=False)
    if m['kind']=='activation':
        g=p.evaluate(x,p.lag(x),tangent=False)[1];d=m['decision'];outgoing=np.zeros(len(p.seam))
        for face in np.flatnonzero(~p.active_faces):
            for q,weight in enumerate(p.rule['weights_mm2'][face]):
                outgoing[face:face+2]+=weight*p.strength*p.rule['shape'][q]
        excess=-g[2*p.seam]-outgoing
        candidates=[i for i in range(len(p.seam)) if not p.released[i] and p.rest[p.seam[i],1]>0 and excess[i]>0]
        ranking=sorted(candidates,key=lambda i:(-excess[i],i))
        if ranking!=d['candidate_ranking'] or ranking[0]!=d['candidate_seam_index']:raise ValueError('Candidate ranking changed')
        np.testing.assert_allclose(g,d['algorithmic_gradient'],rtol=1e-11,atol=1e-12)
        endpoint=endpoint_forces(p,x)
        for key,value in endpoint.items():np.testing.assert_allclose(value,d['endpoint_force_components'][key],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(-sum(endpoint.values())[2*p.seam]-outgoing,d['endpoint_excess_half_n'],rtol=1e-11,atol=1e-12)
        i=ranking[0];faces=p.active_faces.copy();faces[max(0,i-1):min(len(faces),i+1)]=True
        newfree=eligible_nodes(faces)&~p.released
        if np.max(excess[newfree],initial=0)>p.force_tolerance_n:raise ValueError('Handoff force gate failed')
        np.testing.assert_array_equal(np.flatnonzero(faces&~p.active_faces),d['new_faces'])
        before=2*float(p.barrier(p.collisions(x),p.cmesh,x[p.surface]))
        out['blade_contacts_before']=blade_contacts(p,x)
        p.active_faces=faces;p.rebuild_contact()
        after=2*float(p.barrier(p.collisions(x),p.cmesh,x[p.surface]))
        out['blade_contacts_after']=blade_contacts(p,x)
        np.testing.assert_allclose(after-before,d['contact_topology_energy_jump_n_mm'],rtol=0,atol=1e-12)
        if abs(after-before)>1e-10:raise ValueError('Unfunded topology jump')
        out.update(candidate_seam_index=i,ranking=ranking,maximum_excess_half_n=float(np.max(excess[newfree])),topology_jump_n_mm=after-before)
    else:
        if m['rollback_array_sha256']!=m['parent_accepted_array_sha256'] or not m['rollback_verified']:raise ValueError('Rollback not exact')
        if m['energy'] is not None:
            e=p.energy_ledger(x)
            for k,v in m['energy'].items():np.testing.assert_allclose(e[k],v,rtol=1e-10,atol=1e-12)
            fu=velocity_increment(p,p.step_increment(x));lag=p.lag(x)
            fg=np.zeros(2*len(p.surface)) if lag is None else p.friction.gradient(lag,p.cmesh,fu[p.surface])
            np.testing.assert_array_equal(p.surface,s['friction_surface_nodes'])
            np.testing.assert_allclose(fu[p.surface],s['friction_velocity_increment_surface'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(fg,s['friction_gradient_surface'],rtol=1e-10,atol=1e-12)
            out['energy']=e
            out['friction_fields_verified']=True
        out.update(failure_kind=m['failure_kind'],stage=m['stage'],rollback_verified=True,scope='Saved actual trial, reconstructed equations; nonlinear iteration replay is a separate check')
    out['passed']=True;return out


@runtime_entry
def inspect(directory,out,seconds=540,*,runtime=None):
    directory=Path(directory);out=Path(out);start=time.monotonic()
    if not 0<seconds<=540 or out.exists():raise ValueError('Bounded new audit output required')
    raw=json.loads((directory/'result.json').read_text());producer=route
    if 'initial_slit_halfwidth_mm' in raw['config']:
        from . import run_precrack_entry as producer
    source=producer.sources()
    report=dict(passed=False,activation_witness_verified=False,source_hashes=source,thread_runtime=runtime,audits=[],result_sha256=digest(directory/'result.json'),auditor_sha256=digest(Path(__file__)))
    try:
        if raw['source_hashes']!=source:raise ValueError('Source changed')
        activation=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])]
        expected=sum(len(row.get('face_events',[])) for row in raw['steps'])
        if len(activation)!=expected:raise ValueError('Missing event witnesses')
        refs=list(activation);index=directory/'witnesses/representatives.json'
        if index.exists():
            for item in json.loads(index.read_text())['signatures'].values():refs.extend(item[k] for k in ('first','worst','latest'))
        unique={r['sha256']:r for r in refs}
        for ref in unique.values():report['audits'].append(inspect_one(directory,ref,source,start+seconds))
        report.update(passed=bool(unique),activation_witness_verified=bool(activation),activation_count=expected,representative_count=len(unique)-len(activation))
    except (ValueError,RuntimeError,AssertionError,TimeoutError,KeyError) as exc:report['error']=str(exc)
    report.update(elapsed_s=time.monotonic()-start,source_unchanged=source==producer.sources());write_json(out,report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=540);add_thread_arguments(p);a=p.parse_args()
    r=inspect(a.directory,a.out,a.seconds,**thread_arguments(a));print(json.dumps({k:r[k] for k in r if k not in ('source_hashes','thread_runtime','audits')}))
if __name__=='__main__':main()
