"""Bounded saved-field qualification of an experimental contact chart.

No FE loading, material fitting or modification of archived histories. Native
physics is imported from each case's verified source; the candidate is loaded
from an explicitly hashed diagnostic snapshot. Partial contact forces are NOT
equilibrated reactions. One process per physical source snapshot is required.
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from scipy.interpolate import CubicSpline
from numpy.polynomial import polynomial as poly

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def cubic_polygon_check(parameters,nodes,polygon):
    """All cubic/line intersections by polynomial roots and Bezier box filter.

    Finite-precision static geometry check, NOT swept CCD. Max deviation from
    the corresponding P1 edge is evaluated at polynomial stationary points.
    """
    c=CubicSpline(parameters,nodes,axis=0,bc_type='natural');hits=[];maximum=0.;where=None
    pa,pb=polygon,np.roll(polygon,-1,axis=0);edge=pb-pa
    emin,emax=np.minimum(pa,pb),np.maximum(pa,pb)
    for i,h in enumerate(np.diff(parameters)):
        a,b,v,d=c.c[:,i,:]
        controls=np.array([d,d+v*h/3,d+2*v*h/3+b*h*h/3,d+v*h+b*h*h+a*h**3])
        # Normalized segment parameter avoids ill-scaled tiny-edge polynomials.
        coeff=np.array([d,v*h,b*h*h,a*h**3]);dev=coeff.copy()
        dev[0]-=nodes[i];dev[1]-=nodes[i+1]-nodes[i]
        squared=poly.polyadd(poly.polymul(dev[:,0],dev[:,0]),poly.polymul(dev[:,1],dev[:,1]))
        roots=poly.polyroots(poly.polyder(squared));ts=[0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-9 and 0<z.real<1]
        lengths=np.linalg.norm(poly.polyval(ts,dev),axis=0);j=int(np.argmax(lengths))
        if lengths[j]>maximum:maximum=float(lengths[j]);where=dict(segment=i,fraction=ts[j],point=c(parameters[i]+ts[j]*h))
        candidates=np.flatnonzero(np.all(emax>=controls.min(axis=0)-1e-12,axis=1)&np.all(emin<=controls.max(axis=0)+1e-12,axis=1))
        for k in candidates:
            q=coeff.copy();q[0]-=pa[k];cross=q[:,0]*edge[k,1]-q[:,1]*edge[k,0]
            if np.max(abs(cross))<1e-20:raise ValueError('Coincident curve/line requires a separate overlap check')
            for root in poly.polyroots(cross):
                if abs(root.imag)>1e-9 or not -1e-10<=root.real<=1+1e-10:continue
                t=float(np.clip(root.real,0,1));point=c(parameters[i]+t*h)
                alpha=float((point-pa[k])@edge[k]/(edge[k]@edge[k]))
                if -1e-10<=alpha<=1+1e-10:
                    diff=point-pa[k]
                    error=float(abs(diff[0]*edge[k,1]-diff[1]*edge[k,0])/np.linalg.norm(edge[k]))
                    if error>1e-9:raise ValueError('Polynomial intersection residual too large')
                    hits.append(dict(segment=i,fraction=t,polygon_edge=int(k),edge_fraction=alpha,point=point,residual_mm=error))
    return dict(intersections=hits,intersection_count=len(hits),maximum_same_parameter_p1_deviation_mm=maximum,
                maximum_deviation_location=where,scope='Static interpolated top surface versus native blade polygon; no global curved self-contact CCD')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def snapshot(out):
    files=[Path(__file__),ROOT/'prototype/surface_history.py',ROOT/'prototype/surface_contact_transport.py',
           ROOT/'prototype/curve_contact.py',ROOT/'tests/test_surface_contact_transport.py',ROOT/'tests/test_contact_chart_geometry.py']
    manifest={}
    for p in files:
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        manifest[str(p.relative_to(ROOT))]=sha(dest)
    save(out/'candidate_sources.json',manifest)


def progress(out,event,**data):
    row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S %z'),event=event,**data)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    (out/'run_status.html').write_text('<!doctype html><meta charset="utf-8"><title>接触资格核查</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>接触资格核查</h1><p>有界保存场诊断，新增加载步数为零。</p><pre>'+
        html.escape((out/'events.jsonl').read_text(encoding='utf-8'))+'</pre>',encoding='utf-8')


def patch_checks(p,x,topids,chart_module,kernel,out,label,deadline):
    lag=p.lag(x);excluded={};eligible=[];rows=[];topmap={int(v):i for i,v in enumerate(topids)}
    for kind in ('vv','ev'):
        for s in getattr(lag,kind+'_collisions'):
            ids=p.surface[np.array([v for v in s.vertex_ids(p.cmesh.edges,p.cmesh.faces) if v>=0],int)]
            reason=None
            if kind!='ev':reason='VV_not_assembled_by_local_candidate'
            elif s.weight<=0:reason='signed_native_stencil_not_substitutable'
            elif np.all(ids[1:]>=p.n):
                ia,ib=ids[1:]-p.n
                if abs(ia-ib)!=1 or min(ia,ib)<4 or max(ia,ib)+4>=len(p.rest)-p.n:reason='blade_patch_endpoint'
                else:master=np.arange(min(ia,ib)-3,max(ia,ib)+4)+p.n
            elif all(int(v) in topmap for v in ids[1:]):
                ia,ib=[topmap[int(v)] for v in ids[1:]]
                if abs(ia-ib)!=1 or min(ia,ib)<3 or max(ia,ib)+4>len(topids):reason='body_patch_endpoint'
                else:master=topids[min(ia,ib)-3:max(ia,ib)+4]
            else:reason='outside_top_chart'
            if reason:excluded[reason]=excluded.get(reason,0)+1;continue
            capacity=abs(float(s.weight*s.normal_force_magnitude*s.mu_s_effective_lagged))
            y=x[ids];v=y[2]-y[1];b=float((y[0]-y[1])@v/(v@v))
            native_gap=float(np.linalg.norm(y[0]-y[1]-b*v))
            eligible.append((capacity,ids[0],master,float(s.weight),native_gap))
    active_tangents=0
    for rank,(capacity,slave,master,weight,native_gap) in enumerate(sorted(eligible,key=lambda v:-v[0])[:8]):
        if time.monotonic()>deadline:raise TimeoutError('Bounded saved-field qualification budget exhausted')
        ids=np.r_[slave,master];old=p.previous[ids];now=x[ids]
        u=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p.rest[master],axis=0),axis=1))]
        chart=chart_module.SurfaceChart(u);row=dict(rank=rank,ids=ids,master='blade' if master[0]>=p.n else 'body',
            original_absolute_stencil_capacity_n=capacity,native_gap_mm=native_gap,
            scope='Independent local reconstruction, not native stencil substitution or Fz')
        try:
            raw=chart.project(now[0],now[1:]);side=1 if raw['gap']>0 else -1
            # Baseline barrier stiffness is fixed by this diagnostic's scope;
            # epsilon and mu come from the exact restored physical case.
            kw=dict(weight=weight,dhat=p.dhat,stiffness=6.6*p.config.width_mm,
                    mu=p.config.friction_coefficient,epsilon=float(p.friction.eps_v),side=side)
            history=kernel.initial_history(chart,old,dhat=kw['dhat'],side=side)
            q=kernel.evaluate_contact(chart,now,history,**kw)
            row.update(status='evaluated',gap_mm=q['gap_mm'],slip_mm=q['slip_mm'],contact_state=q['status'],
                path_proof=q['path_proof'],activation_fraction=None if q['activation'] is None else q['activation']['fraction'],
                point_coulomb_metric=q.get('slave_tangential_metric'),friction_work_nmm=q['friction_work_nmm'],
                gap_change_mm=q['gap_mm']-native_gap,
                force_balance_n=float(np.linalg.norm(q['gradient'].sum(axis=0))),
                moment_balance_nmm=float(abs(np.sum(q['gradient']*np.column_stack((-now[:,1],now[:,0]))))))
            theta=.4;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            rigid=kernel.evaluate_contact(chart,old@Q.T+[.1,-.2],history,**kw)
            row['rigid_rotation_slip_mm']=rigid['slip_mm']
            row['rigid_rotation_friction_force_n']=float(np.linalg.norm(rigid['friction_gradient']))
            if q['active'] and active_tangents<4:
                active_tangents+=1
                rng=np.random.default_rng(71+rank);d=rng.normal(size=now.shape);d/=np.max(abs(d))
                action=q['jacobian']@d.ravel();trials=[]
                for h in (1e-7,1e-8,1e-9):
                    plus=kernel.evaluate_contact(chart,now+h*d,history,**kw)
                    minus=kernel.evaluate_contact(chart,now-h*d,history,**kw)
                    fd=((plus['gradient']-minus['gradient'])/(2*h)).ravel()
                    trials.append(dict(h_mm=h,relative_error=float(np.linalg.norm(fd-action)/max(np.linalg.norm(action),1e-12))))
                row['tangent_trials']=trials;row['tangent_verified']=any(a['relative_error']<1e-3 and b['relative_error']<1e-3 for a,b in zip(trials,trials[1:]))
            np.savez_compressed(out/f'patch_{label}_{rank}.npz',positions=now,previous_positions=old,parameters=u,ids=ids,
                jacobian=q['jacobian'],gradient=q['gradient'],side=side,weight=weight)
        except (ValueError,RuntimeError) as exc:row.update(status='rejected_local_reconstruction',error=repr(exc))
        rows.append(row)
    return dict(total_eligible=len(eligible),sampled=len(rows),unsupported_native_counts=excluded,patches=rows,
        caveat='Positive EV subset only, eight greatest absolute capacities; omitted VV/signed stencils prohibit a global replacement claim.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--kind',choices=('common','controls'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.case.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot(out);started=time.monotonic();deadline=started+600
    sys.path.insert(0,str(ROOT/'.runtime/ipc-python'));sys.path.insert(0,str(folder/'source'))
    from tools.run_common_material_study import verify_case
    from prototype.material_case import signature
    from threadpoolctl import threadpool_limits
    if args.kind=='common':from tools.run_common_material_study import make
    else:from tools.run_contact_controls import make
    chart=load_module('prototype._qualified_surface_history',out/'source/prototype/surface_history.py')
    # Explicit candidate's barrier helper as well; never silently use a changed worktree helper.
    load_module('prototype.curve_contact',out/'source/prototype/curve_contact.py')
    kernel=load_module('prototype._qualified_surface_contact_transport',out/'source/prototype/surface_contact_transport.py')
    summary=dict(status='running',source_case=str(folder),new_accepted_states=0,rows=[],inputs={},
        scope='Conditional saved-field contact and static geometry diagnostics, not new equilibria or material fits')
    try:
        with threadpool_limits(1):
            progress(out,'verify_frozen_sources')
            summary['inputs'].update(verify_case(folder));r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            job=r['job'];p=make(job['angle'],job['parameters']) if args.kind=='common' else make(job['angle'],job['parameters'],job['group'])
            if json.loads(json.dumps(signature(p)))!=r['signature']:raise ValueError('Frozen physical signature mismatch')
            topids=np.flatnonzero(abs(p.mesh.local_sn[:,1]-p.config.height_mm)<1e-9)
            topids=topids[np.argsort(p.mesh.local_sn[topids,0])];u=p.mesh.local_sn[topids,0]
            selected=[next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-d)<1e-10) for d in (.25,.4,.5)] if args.kind=='common' else [len(r['steps'])-1]
            targets=[(folder/'states'/f'{i:04d}.npz',r['steps'][i],f'{i:04d}') for i in selected]
            if args.kind=='controls':targets.append((folder/'failed_trial.npz',None,'failed'))
            for file,record,label in targets:
                if time.monotonic()>deadline:raise TimeoutError('600 s budget exhausted')
                summary['inputs'][str(file)]=sha(file)
                with np.load(file,allow_pickle=False) as f:state=dict(f)
                progress(out,'field_begin',field=label)
                audit=p.audit_state(state,record) if record is not None else None
                x=state['positions'];p.previous=state['previous_positions'];p.previous_material_metric=state['previous_material_metric']
                p.material_dt=float(state['material_dt_s']);p._contact_initial_preparation=bool(state.get('contact_initial_preparation',False))
                collision=bool(p.ipc.has_intersections(p.cmesh,x[p.surface]))
                geom=cubic_polygon_check(u,x[topids],x[p.n:])
                np.savez_compressed(out/f'geometry_{label}.npz',parameters=u,top_ids=topids,top_positions=x[topids],
                    blade_positions=x[p.n:],reference_top=p.rest[topids],reference_blade=p.rest[p.n:])
                patches=patch_checks(p,x,topids,chart,kernel,out,label,deadline)
                depth=float(state['depth_mm']) if record is not None else float(r['rejected_load_attempts'][-1]['depth_mm'])
                if record is None:
                    prior_depth=float(r['steps'][-1]['depth_mm'])
                    increment=p.previous[p.n:,1]-x[p.n:,1]
                    if not np.allclose(increment,depth-prior_depth,rtol=0,atol=1e-10):
                        raise ValueError('Failed target metadata does not match prescribed blade displacement')
                row=dict(label=label,angle=job['angle'],depth_mm=depth,accepted=record is not None,
                    fresh_physical_audit=audit,native_intersection=collision,geometry=geom,local_contact=patches)
                summary['rows'].append(row);save(out/'summary.json',summary)
                progress(out,'field_end',field=label,spline_intersections=geom['intersection_count'],
                    local_evaluated=sum(z['status']=='evaluated' for z in patches['patches']))
            summary.update(status='completed',elapsed_s=time.monotonic()-started)
    except Exception as exc:
        summary.update(status='diagnostic_failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        save(out/'summary.json',summary);progress(out,'ended',status=summary['status'])


if __name__=='__main__':main()
