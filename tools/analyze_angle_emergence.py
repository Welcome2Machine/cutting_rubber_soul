"""Saved intact paths: depth-dependent angular difference, no load solves.

Use frozen native physical modules. Screenshot anchors are manual visual ranges,
not a digitized experimental series, uncertainty estimates, or calibration data.
"""
from pathlib import Path
import sys
import json
import csv
import time
import argparse
from dataclasses import replace
import hashlib
import os
import subprocess
import importlib.util
from html import escape

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/{
    '0':'outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0/source',
    '33':'outputs/native_angles_2mm_20261016/01_angle33/source',
    '57':'outputs/native_angles_2mm_20261016/02_angle57/source',
}[os.environ.get('SOFTMODELING_ANALYSIS_ANGLE','33')]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime/ipc-python'))
sys.path.insert(0,str(FROZEN))
import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.finite_strain import elastic_response
from prototype.model_config import _rotation_vectors
# The old0 snapshot predates the shared constructor helper. Use the current
# constructor adapter, while all physical imports resolve to that case's frozen
# package; exact source checks and fresh equilibrium checks follow below.
adapter_spec=importlib.util.spec_from_file_location('tools.run_material_candidate',ROOT/'tools/run_material_candidate.py')
adapter=importlib.util.module_from_spec(adapter_spec)
sys.modules[adapter_spec.name]=adapter
adapter_spec.loader.exec_module(adapter)
from tools.check_contact_transport import BASELINES, inspect_case, construct, sha, read

DEPTHS=(.5,.75,1.,1.25,1.5,1.75,2.)
EXPERIMENT=[
    # Coarse reading of the user's latest paired 1008 plot. No fitted points.
    # Direct gap ranges preserve pairing; they are read separately, not treated
    # as independent statistical errors of two force measurements.
    dict(depth_mm=.5,f0_n=7.7,f33_n=7.5,force_reading_halfwidth_n=.5,gap_n=.2,gap_range_n=[-.2,.6]),
    dict(depth_mm=1.,f0_n=18.,f33_n=16.8,force_reading_halfwidth_n=.6,gap_n=1.2,gap_range_n=[.6,1.8]),
    dict(depth_mm=1.5,f0_n=26.8,f33_n=24.6,force_reading_halfwidth_n=.7,gap_n=2.2,gap_range_n=[1.6,2.8]),
    dict(depth_mm=2.,f0_n=34.2,f33_n=31.5,force_reading_halfwidth_n=.7,gap_n=2.7,gap_range_n=[2.,3.4]),
]


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stress_field(p,state,record):
    """Work on last actual increment; bookkeeping, not causal sensitivities."""
    x=state['positions'];p.previous=state['previous_positions'].copy()
    p.previous_material_metric=state['previous_material_metric'].copy()
    p.material_dt=float(state['material_dt_s'])
    area,grad,dofs=p.mesh.bulk_data
    F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:p.n]-p.rest[:p.n])[p.mesh.triangles],grad)
    q=p.material_response(x)
    eq=elastic_response(F,replace(p.material,mu_ov=(),alpha_ov=()),tangent=False)
    J=q['J'];K=p.material.moduli()[1]
    pv=(K*J*(J-1))[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
    parts={'volume':pv,'equilibrium_shear':eq['P']-pv,'maxwell':q['P']-eq['P']}
    np.testing.assert_allclose(sum(parts.values()),q['P'],rtol=1e-12,atol=1e-12)
    dd=p.loading_rate*p.material_dt
    v=(x-p.previous)/dd
    dF=np.einsum('eai,eaJ->eiJ',v[:p.n][p.mesh.triangles],grad)
    work={key:area*p.config.width_mm*np.einsum('eiJ,eiJ->e',P,dF) for key,P in parts.items()}
    # Whole-system virtual-power identity using freshly refreshed contact.
    c=p.collisions(x);sx=x[p.surface];delta=(x-p.previous)[p.surface]
    ng=p.barrier.gradient(c,p.cmesh,sx)
    fg=p.friction.gradient(p.lag(x),p.cmesh,delta)
    normal_power=float(ng@v[p.surface].ravel())
    friction_power=float(fg@v[p.surface].ravel())
    blade_g=(ng+fg).reshape(-1,2)[p.surface>=p.n]
    blade_power=float(np.sum(blade_g*v[p.n:]))
    mismatch=sum(float(w.sum()) for w in work.values())+normal_power+friction_power-blade_power
    _,g,_,_=p.evaluate(x,p.lag(x),tangent=False)
    np.testing.assert_allclose(mismatch,g[p.free]@v.ravel()[p.free],atol=1e-9,rtol=1e-6)
    es,en=_rotation_vectors(p.config.inclination_deg);R=np.column_stack((es,en))
    # Spatial Hencky strain in specimen axes (both modes exist at both angles).
    B=F@F.transpose(0,2,1);eig,U=np.linalg.eigh(B)
    h=(U*(.5*np.log(eig))[:,None,:])@U.transpose(0,2,1)
    h=R.T@h@R
    X=p.rest[:p.n];centroid=X[p.mesh.triangles].mean(axis=1)
    loc=np.column_stack((centroid@es-p.config.drop_s_mm,centroid@en-p.config.height_mm))
    radius=np.linalg.norm(loc,axis=1)
    near=radius<1.;aw=area[near]/area[near].sum()
    stress=(q['P']@F.transpose(0,2,1))/J[:,None,None]
    pres=-.5*np.trace(stress,axis1=1,axis2=2)
    # in-plane mean compression only, not 3D hydrostatic pressure
    minimum=int(np.argmin(J))
    row=dict(depth_mm=record['depth_mm'],min_j=float(J.min()),
        min_j_reference_local_mm=loc[minimum].tolist(),
        low_j_area_fraction_within_1mm=float(area[near & (J<.5)].sum()/area[near].sum()),
        normal_hencky_rms_1mm=float(np.sqrt(aw@(h[near,1,1]**2))),
        shear_hencky_rms_1mm=float(np.sqrt(aw@((2*h[near,0,1])**2))),
        internal_virtual_work_n={key:float(w.sum()) for key,w in work.items()},
        near_1mm_virtual_work_n={key:float(w[near].sum()) for key,w in work.items()},
        bottom_half_virtual_work_n={key:float(w[(centroid@en)<p.config.height_mm/2].sum()) for key,w in work.items()},
        contact_normal_virtual_work_n=normal_power,friction_virtual_work_n=friction_power,
        blade_virtual_work_n=blade_power,power_closure_error_n=float(mismatch),
        note='Last-increment virtual power, not derivative of stored energy or causal fractions')
    arrays=dict(reference_local=np.column_stack((X@es-p.config.drop_s_mm,X@en-p.config.height_mm)),
        current_local=np.column_stack((x[:p.n]@es-p.config.drop_s_mm,x[:p.n]@en-p.config.height_mm)),
        triangles=p.mesh.triangles,J=J,normal_hencky=h[:,1,1],shear_hencky=2*h[:,0,1],
        in_plane_mean_compression_mpa=pres)
    return row,arrays


def analyze_case(out,a,end=2.):
    if not 0 < end <= 2.:raise ValueError('Saved-path review limited to 2 mm')
    depths=tuple(d for d in DEPTHS if d<=end+1e-10)
    with threadpool_limits(1):
        for a in (a,):
            folder=ROOT/BASELINES[a];r=read(folder/'result.json');audit=read(folder/'audit.json')
            assert r['completed'] and audit['passed'] and audit['states']==len(r['steps'])==201
            p=construct(r)
            np.testing.assert_equal(p.material.to_dict(),r['signature']['material'])
            for name in ('prototype/ipc_reference.py','prototype/ipc_viscoelastic.py','mechanics/finite_strain.py'):
                if sha(FROZEN/name)!=r['source_sha256'][name]:
                    raise ValueError('Audit physical source mismatch: '+name)
            c=inspect_case(folder,f'angle{a}',end=end,anchor_depths=depths)
            c['input_sha256'][str(folder/'audit.json')]=sha(folder/'audit.json')
            fields=[]
            for d in depths:
                i=next(i for i,row in enumerate(r['steps']) if abs(row['depth_mm']-d)<1e-10)
                with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as z:
                    row,arr=stress_field(p,z,r['steps'][i]);fields.append(row)
                    np.savez_compressed(out/f'field_{a}_{d:.2f}.npz',**arr)
                print(f'FIELD angle={a} depth={d} minJ={row["min_j"]:.6f} power_closure={row["power_closure_error_n"]:.3g}',flush=True)
            c['work_fields']=fields;c['physical_root']=str(FROZEN)
            force=[dict(depth_mm=s['depth_mm'],fz_n=s['reaction_fz_n'],
                fx_n=s.get('reaction_x_n'),normal_fz_n=s['reaction_normal_fz_n'],
                friction_fz_n=s['reaction_friction_fz_n'],min_j=s['min_j']) for s in r['steps'] if s['depth_mm']<=end+1e-10]
    save(out/f'case{a}.json',dict(case=c,forces=force))


def analyze(out):
    if (out/'evidence.json').exists():raise ValueError('Do not overwrite completed analysis')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cases=[];fingerprints={};force_by_angle={}
    for a in (0,33):
        env=os.environ.copy();env['SOFTMODELING_ANALYSIS_ANGLE']=str(a)
        with (out/f'case{a}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),
                '--output',str(out),'--case-angle',str(a)],env=env,check=True,
                stdout=log,stderr=subprocess.STDOUT,timeout=240)
        saved=read(out/f'case{a}.json');c=saved['case'];cases.append(c)
        fingerprints.update(c['input_sha256']);force_by_angle[a]=saved['forces']
        print(f'CASE_DONE angle={a} states={len(c["rows"])} fields={len(c["fields"])}',flush=True)
    rows=[]
    for r0,r33 in zip(force_by_angle[0],force_by_angle[33]):
        assert abs(r0['depth_mm']-r33['depth_mm'])<1e-10
        f0,f33=r0['fz_n'],r33['fz_n']
        rows.append(dict(depth_mm=r0['depth_mm'],f0_n=f0,f33_n=f33,mean_n=(f0+f33)/2,
            delta_n=f0-f33,relative_reduction=(f0-f33)/f0,
            delta_normal_n=r0['normal_fz_n']-r33['normal_fz_n'],
            delta_friction_n=r0['friction_fz_n']-r33['friction_fz_n']))
    exp=[]
    for e in EXPERIMENT:
        s=min(rows,key=lambda r:abs(r['depth_mm']-e['depth_mm']))
        exp.append(dict(e,fe_f0_n=s['f0_n'],fe_f33_n=s['f33_n'],fe_delta_n=s['delta_n'],
            mean_experiment_n=(e['f0_n']+e['f33_n'])/2,
            common_mean_excess_n=s['mean_n']-(e['f0_n']+e['f33_n'])/2))
    slopes=[]
    for e0,e1 in zip(EXPERIMENT[:-1],EXPERIMENT[1:]):
        d=e1['depth_mm']-e0['depth_mm'];r0=rows[round(e0['depth_mm']*100)];r1=rows[round(e1['depth_mm']*100)]
        for a in (0,33):
            key=f'f{a}_n';half=(e0['force_reading_halfwidth_n']+e1['force_reading_halfwidth_n'])/d
            center=(e1[key]-e0[key])/d
            slopes.append(dict(angle=a,start_mm=e0['depth_mm'],end_mm=e1['depth_mm'],
                fe_n_mm=(r1[key]-r0[key])/d,visual_n_mm=center,
                visual_range_n_mm=[center-half,center+half]))
    for path,h in fingerprints.items():
        if sha(path)!=h:raise ValueError('Input changed during analysis')
    code=['tools/analyze_angle_emergence.py','tools/check_contact_transport.py','tools/run_material_candidate.py']
    save(out/'evidence.json',dict(cases=cases,forces=force_by_angle,force_difference=rows,
        experiment=exp,slopes=slopes,input_sha256=fingerprints,
        analysis_source_sha256={s:sha(ROOT/s) for s in code},physical_roots={str(c['angle']):c['physical_root'] for c in cases},
        elapsed_s=time.monotonic()-start,scope='No new loading; 402 saved states, 14 fresh exact-history anchor audits',
        experiment_provenance='Manual coarse readings of latest user screenshot 1008n0a1 / 1008n33a1. Not digitized raw data, not CI; no depth/force shifts; no calibration.',
        frozen_invariants='Same common material, mu, boundary, tip/mesh settings; actual angles0/33 retained'))
    write_csv(out/'force_difference.csv',rows)
    save(out/'experiment_visual_readings.json',dict(source='Latest user-supplied paired screenshot',
        values=EXPERIMENT,method='Manual visual anchors and conservative reading ranges; not statistically estimated uncertainty',
        used_for_calibration=False))
    print(f'COMPLETED seconds={time.monotonic()-start:.3f} files={len(fingerprints)}',flush=True)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    a=read(out/'evidence.json');r=a['force_difference'];d=np.array([v['depth_mm'] for v in r])
    colors={0:'#0072B2',33:'#D55E00'}
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    exp=a['experiment'];ed=[e['depth_mm'] for e in exp]
    for angle,style in ((0,'-'),(33,'--')):
        key=f'f{angle}_n';color=colors[angle]
        axes[0,0].plot(d,[v[key] for v in r],style,color=color,label=f'FE {angle} deg')
        axes[0,0].errorbar(ed,[e[key] for e in exp],yerr=[e['force_reading_halfwidth_n'] for e in exp],
            fmt='s' if angle==0 else '^',mfc='white',color=color,capsize=4,label=f'Image {angle}: visual range')
    axes[0,0].set(title='A  Absolute force: FE and latest paired experiment',ylabel='Fz (N)',ylim=(0,46));axes[0,0].legend(fontsize=8)
    axes[0,1].plot(d,[v['delta_n'] for v in r],color='#222222',label='FE F0 - F33')
    y=np.array([e['gap_n'] for e in exp]);low=np.array([e['gap_range_n'][0] for e in exp]);high=np.array([e['gap_range_n'][1] for e in exp])
    axes[0,1].errorbar(ed,y,yerr=[y-low,high-y],fmt='o',mfc='white',color='#009E73',capsize=4,label='Paired image gap: direct visual range')
    axes[0,1].axhline(0,c='gray',lw=.8);axes[0,1].set(title='B  Angular gap evolves with depth',ylabel='F0 - F33 (N)',ylim=(-1,4.5));axes[0,1].legend(fontsize=8)
    for angle,off in ((0,-.035),(33,.035)):
        slopes=[s for s in a['slopes'] if s['angle']==angle];mid=np.array([(s['start_mm']+s['end_mm'])/2 for s in slopes])+off
        y=np.array([s['visual_n_mm'] for s in slopes]);lo=np.array([s['visual_range_n_mm'][0] for s in slopes]);hi=np.array([s['visual_range_n_mm'][1] for s in slopes])
        axes[1,0].errorbar(mid,y,yerr=[y-lo,hi-y],fmt='s',mfc='white',color=colors[angle],capsize=4,label=f'Image {angle}: visual range')
        axes[1,0].plot(mid,[s['fe_n_mm'] for s in slopes],'x',ms=9,color=colors[angle],label=f'FE {angle}')
    axes[1,0].set(title='C  Half-mm secant stiffness (no smoothing)',ylabel='Secant stiffness (N/mm)',ylim=(0,28));axes[1,0].legend(fontsize=8,ncol=2)
    axes[1,1].plot(d,[v['mean_n'] for v in r],color='#222222',label='FE mean (F0 + F33)/2')
    axes[1,1].errorbar(ed,[e['mean_experiment_n'] for e in exp],yerr=[e['force_reading_halfwidth_n'] for e in exp],fmt='o',mfc='white',color='#009E73',capsize=4,label='Image mean: visual range')
    axes[1,1].set(title='D  Shared force excess is distinct from angular gap',ylabel='Mean Fz (N)',ylim=(0,46));axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.set(xlabel='Vertical depth (mm)',xlim=(0,2.05));ax.grid(alpha=.2)
    fig.suptitle('Intact FE vs experiment including fracture | onset estimates: 0 deg ~1 mm; 33 deg 1.2-1.3 mm\nImage points: manual readings, not raw data / CI. Late range is NOT intact-model calibration.',fontsize=11)
    fig.savefig(out/'force_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for c in a['cases']:
        angle=int(c['angle']);color=colors[angle];style='-' if angle==0 else '--'
        rr=c['rows'];ff=c['fields'];ww=c['work_fields']
        xx=[v['depth_mm'] for v in rr];fd=[v['depth_mm'] for v in ff]
        axes[0,0].plot(xx,[np.nan if v['unique_tip_offset_mm'] is None else v['unique_tip_offset_mm'] for v in rr],style,c=color,label=f'{angle} deg')
        axes[0,1].plot(xx,[v['force_support_depth_mm'] for v in rr],style,c=color)
        axes[0,2].plot(fd,[v['reference_normal_force_quantiles_mm'][1] for v in ff],style+'o',c=color)
        axes[1,0].plot(fd,[v['fresh_min_j'] for v in ff],style+'o',c=color)
        axes[1,1].plot(fd,[v['internal_virtual_work_n']['maxwell'] for v in ww],style+'o',c=color)
        axes[1,2].plot(fd,[v['deformation_reference_radius_quantiles_mm'][1] for v in ff],style+'o',c=color)
    labels=[('Unique pole identity (missing = gaps)','Reference offset (mm)'),('Continuous nodal normal-force support','Loading-depth interval (mm)'),('Normal-force weighted reference median','Reference offset (mm)'),('Local compression remains severe','Minimum J'),('Material branch work, not causal share','Maxwell virtual work / depth (N)'),('Deformation extends beyond the tip','90% reference radius (mm)')]
    for ax,(title,ylabel) in zip(axes.flat,labels):ax.set(title=title,ylabel=ylabel,xlabel='Vertical depth (mm)',xlim=(0,2.05));ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Contact and deformation evolution | intact FE, not experimental truth')
    fig.savefig(out/'field_evolution.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained',sharex=True,sharey=True)
    for i,angle in enumerate((0,33)):
        for j,depth in enumerate((.5,1.,1.5)):
            with np.load(out/f'field_{angle}_{depth:.2f}.npz',allow_pickle=False) as z:
                xy=z['current_local'];tri=Triangulation(xy[:,0],xy[:,1],z['triangles'])
                im=axes[i,j].tripcolor(tri,facecolors=z['J'],cmap='viridis',vmin=0,vmax=1.1,shading='flat',rasterized=True)
                axes[i,j].set(title=f'{angle} deg | depth {depth:g} mm',xlim=(-1.6,2.6),ylim=(-2.7,.7),aspect='equal',xlabel='Current specimen-tangent coordinate (mm)',ylabel='Current specimen-normal coordinate (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='J (same scale, deformed body only; blade not drawn)',shrink=.85)
    fig.suptitle('Local deformation / volume ratio | same specimen axes, not a lab-frame overlay')
    fig.savefig(out/'deformation_j.png',dpi=160);plt.close(fig)
    print('PLOTS_WRITTEN',flush=True)


def review_three(out):
    """Publish the user-selected Maxwell review, using existing loading only.

    Keep the original two-angle evidence immutable. Onset annotations below are
    user estimates, never mechanically detected transitions or fitted values.
    """
    from tools.render_solver_dashboard import shell, atomic_write, stamp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evidence=read(out/'evidence.json');cases={};inputs=dict(evidence['input_sha256'])
    signatures={};summaries=[];wide=[];contact=[];end=1.5
    for angle in (0,33,57):
        saved=read(out/f'case{angle}.json');c=saved['case'];cases[angle]=c
        if angle in (0,33):
            assert c==next(v for v in evidence['cases'] if v['angle']==angle)
        folder=ROOT/BASELINES[angle];r=read(folder/'result.json');au=read(folder/'audit.json')
        assert r['completed'] and au['passed'] and au['states']==len(r['steps'])
        prefix=[s for s in r['steps'] if s['depth_mm']<=end+1e-10]
        assert len(prefix)==151 and abs(prefix[-1]['depth_mm']-end)<1e-10
        np.testing.assert_allclose([s['depth_mm'] for s in prefix],np.arange(151)/100,atol=1e-12)
        for i,s in enumerate(prefix):
            ar=au['rows'][i];path=folder/'states'/f'{i:04d}.npz'
            assert ar['passed'] and abs(ar['depth_mm']-s['depth_mm'])<1e-10
            inputs[str(path.resolve())]=ar['checkpoint_sha256']
        inputs.update(c['input_sha256'])
        inputs[str((out/f'case{angle}.json').resolve())]=sha(out/f'case{angle}.json')
        sig=json.loads(json.dumps(r['signature']));sig.pop('mesh_sha256');sig['config'].pop('inclination_deg')
        signatures[angle]=sig
        fresh=[v for v in c['fields'] if v['depth_mm']<=end+1e-10]
        assert len(fresh)==5 and all(v['fresh_residual_n']<=1e-4 and v['fresh_min_j']>0 for v in fresh)
        for v in fresh:
            field=out/f'field_{angle}_{v["depth_mm"]:.2f}.npz'
            inputs[str(field.resolve())]=sha(field)
        summaries.append(dict(angle=angle,states=151,target_mm=end,
            fz_end_n=prefix[-1]['reaction_fz_n'],end_min_j=prefix[-1]['min_j'],
            path_min_j=min(v['min_j'] for v in prefix),
            fresh_anchor_count=len(fresh),fresh_max_residual_n=max(v['fresh_residual_n'] for v in fresh),
            old_terminal_audit_prefix_max_residual_n=max(v['residual_n'] for v in au['rows'][:151]),
            source=str(folder),physical_root=c['physical_root']))
        for i,s in enumerate(prefix):
            if angle==0:wide.append(dict(depth_mm=s['depth_mm']))
            wide[i][f'fz_{angle}_n']=s['reaction_fz_n']
            obs=c['rows'][i];assert abs(obs['depth_mm']-s['depth_mm'])<1e-10
            contact.append(dict(angle_deg=angle,depth_mm=s['depth_mm'],fz_n=s['reaction_fz_n'],
                min_j=s['min_j'],unique_reference_offset_mm=obs['unique_tip_offset_mm'],
                tip_status=obs['tip_status'],continuous_nodal_support_depth_mm=obs['force_support_depth_mm']))
    assert signatures[0]==signatures[33]==signatures[57], 'Common physical signature mismatch'
    exp_path=ROOT/'docs/1014n57a1__Fz__curve_0.csv'
    ed,ef,speed=adapter.load_force_csv(exp_path);assert speed==.2
    inputs[str(exp_path.resolve())]=sha(exp_path)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Changed input: '+path)
    d=np.array([v['depth_mm'] for v in wide]);colors={0:'#0072B2',33:'#D55E00',57:'#009E73'}
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for angle,style in ((0,'-'),(33,'--'),(57,'-.')):
        f=[v[f'fz_{angle}_n'] for v in wide]
        axs[0,0].plot(d,f,style,c=colors[angle],label=f'Intact FE {angle} deg')
        rr=[v for v in cases[angle]['rows'] if v['depth_mm']<=end+1e-10]
        axs[1,1].plot(d,[np.nan if v['unique_tip_offset_mm'] is None else v['unique_tip_offset_mm'] for v in rr],style,c=colors[angle],label=f'{angle} deg')
    axs[0,0].set(title='A  Three completed intact paths',ylabel='Fz (N)');axs[0,0].legend()
    for angle,marker in ((0,'s'),(33,'^')):
        axs[0,1].plot(d,[v[f'fz_{angle}_n'] for v in wide],'-' if angle==0 else '--',c=colors[angle],label=f'FE {angle}')
        for e in evidence['experiment']:
            if e['depth_mm']>end:continue
            earlier=e['depth_mm']<(1. if angle==0 else 1.2)
            axs[0,1].errorbar(e['depth_mm'],e[f'f{angle}_n'],yerr=e['force_reading_halfwidth_n'],
                fmt=marker,mfc='white' if earlier else '#777777',mec=colors[angle],ecolor=colors[angle],capsize=4)
    axs[0,1].axvline(1,c=colors[0],lw=1,ls=':');axs[0,1].axvspan(1.2,1.3,color=colors[33],alpha=.12)
    axs[0,1].set(title='B  0/33 image readings + estimated fracture onsets',ylabel='Fz (N)')
    axs[0,1].legend(fontsize=9)
    axs[1,0].plot(d,[v['fz_57_n'] for v in wide],c=colors[57],label='Intact FE 57')
    mask=ed<=end;axs[1,0].plot(ed[mask],ef[mask],c='#333333',lw=1,label='1014n57a1 experiment (selected trial)')
    axs[1,0].set(title='C  57 deg: fracture onset unknown',ylabel='Fz (N)');axs[1,0].legend(fontsize=9)
    axs[1,1].set(title='D  Unique tip material identity: missing stays missing',ylabel='Reference surface offset (mm)');axs[1,1].legend()
    for ax in axs.flat:ax.set(xlabel='Vertical depth (mm)',xlim=(0,1.52));ax.grid(alpha=.2)
    fig.suptitle('Existing common-Maxwell solutions to 1.5 mm | no new loading\nImage markers: reading ranges, not CI; gray fill = near/after estimated onset. Estimates are not exact crack detection.',fontsize=11)
    fig.savefig(out/'three_angles_15mm.png',dpi=160);plt.close(fig)
    write_csv(out/'fz_three_angles_15mm.csv',wide);write_csv(out/'contact_three_angles_15mm.csv',contact)
    onset={'0':dict(estimate_mm=1.,status='user_rough_estimate',uncertainty_range_mm=None),
           '33':dict(estimate_range_mm=[1.2,1.3],status='user_rough_estimate'),
           '57':dict(status='unknown')}
    data=dict(status='completed_saved_path_review',created=stamp(),target_mm=end,new_loading_solves=0,
        model='common finite isochoric Maxwell, native improved-max IPC, intact P1',
        onset=onset,fracture_in_late_experiment_confirmed=True,full_range_calibration=False,
        cases=summaries,reviewed_states=453,fresh_anchor_states=15,signature_equal_excluding_angle_and_mesh_hash=True,
        input_sha256=inputs,original_two_angle_evidence_sha256=sha(out/'evidence.json'),
        report_source_sha256=sha(Path(__file__)),
        caveat='Pre-estimate windows are provisional, not independently verified intact. 57 CSV is a selected different trial, not the paired 1008 experiment.')
    save(out/'review_15mm.json',data)
    table='<table><tr><th>角度</th><th>范围 / 接受态</th><th>Fz(1.5 mm) N</th><th>末端 / 路径最小J</th><th>本轮锚点最大残差 N</th></tr>'
    for s in summaries:
        table+=f'<tr><td>{s["angle"]}°</td><td>0–1.5 mm / 151</td><td>{s["fz_end_n"]:.4f}</td><td>{s["end_min_j"]:.5f} / {s["path_min_j"]:.5f}</td><td>{s["fresh_max_residual_n"]:.3g}</td></tr>'
    table+='</table>'
    force_table='<table><tr><th>d / mm</th><th>0° / N</th><th>33° / N</th><th>57° / N</th><th>F0−F33 / N</th></tr>'
    for depth in (.5,.75,1.,1.25,1.5):
        r=wide[round(depth*100)]
        force_table+=f'<tr><td>{depth:g}</td><td>{r["fz_0_n"]:.3f}</td><td>{r["fz_33_n"]:.3f}</td><td>{r["fz_57_n"]:.3f}</td><td>{r["fz_0_n"]-r["fz_33_n"]:+.3f}</td></tr>'
    force_table+='</table>'
    work_table='<table><tr><th>深度 mm</th><th>体积项差 N</th><th>平衡剪切项差 N</th><th>Maxwell项差 N</th></tr>'
    for i,depth in enumerate(DEPTHS):
        if depth>end:continue
        w0=cases[0]['work_fields'][i]['internal_virtual_work_n'];w33=cases[33]['work_fields'][i]['internal_virtual_work_n']
        work_table+='<tr><td>'+str(depth)+'</td>'+''.join(f'<td>{w0[k]-w33[k]:+.4f}</td>' for k in ('volume','equilibrium_shear','maxwell'))+'</tr>'
    work_table+='</table>'
    body=f'''<h1>共同Maxwell：0° / 33° / 57°至1.5 mm复核</h1><p>{escape(stamp())} · 已完成已有路径复核 · 新增加载求解：0</p>
<div style="border-left:5px solid #a85e00;padding:12px;background:#fff3d7"><b>本轮判断必须考虑实验已断裂。</b>用户估计0°约1 mm起裂、33°约1.2–1.3 mm起裂，57°未指定。这些是粗略估计，不是视频同步确定的起裂位置；不把更早区间自动认证为无裂。实验后段不用于完整体材料/摩擦标定，也不把降斜率唯一归因于本构错误。</div>
<p>按用户选择，采用已有共同Maxwell主线，不续跑无松弛对照、不重启旧队列。三个历史算例均已计算至2 mm；本报告只取0–1.5 mm。核查453个该范围检查点的原审查与哈希，本轮在每角度.5/.75/1/1.25/1.5 mm重新装配，共15态通过原1e−4 N力残差及原状态审查。0/33的前置诊断另覆盖到2 mm；未把“15态新审查”说成“453态全新重算”。</p>
{table}
<p>核对共同材料/配置/接触签名（仅入侵角和旋转网格哈希不同）。每角度使用自己的冻结物理模块及准确材料、摩擦历史；旧0°缺失的Fx字段保留缺失。数值平衡通过不代表空间收敛、滑移准确或实验验证。</p>
<h2>三工况力与接触材料身份</h2><img src="three_angles_15mm.png" alt="三个角度到1.5毫米的完整体反力、与各自实验的比较及唯一刀尖材料身份偏移" style="width:100%;max-width:1250px">
<p>图B竖虚线是0°约1 mm的用户估计，橙色带是33°的1.2–1.3 mm估计范围。灰填充读图点处于估计起裂附近/之后。误差棒仅表示人工读图范围，不是置信区间；不自动移动深度、归零或缩放力。图C的57°来自已授权1014n57a1 CSV，保留初始.51 N；它参与过探索选参，而且不是1008配对试验，不能据三图拼出精确实验角度比。</p>
{force_table}
<h2>对当前问题的修正结论</h2>
<p><b>浅层接近这一点，模型与实验一致。</b>0.5 mm处FE的0/33°仅差−.157 N，符合用户指出的“500 μm本来就没什么差异”。因此撤回“必须使浅层角度差变大”这一目标。完整体FE在1.25–1.5 mm逐渐产生正角度差；这是完整体自身的演化，不能直接解释已发生断裂的实验差异。</p>
<p><b>仍有独立于后段断裂的早期幅值问题值得检查。</b>0.5 mm处两角度FE约11.49/11.64 N，而最新配对图粗读约7.7/7.5 N（各±.5 N读图范围）。按用户估计，该深度早于起裂，故可作为暂定裂前核查窗口；但仍需保留起裂估计误差。0.5–1 mm的割线刚度FE为18.91/18.76 N/mm，图中粗读20.6/18.6 N/mm，落在读图范围内。当前证据更指向较早已累积的共同幅值偏差，不能再笼统说整个前段形状都错了。0°在1 mm附近可能已起裂，使这个区间比较仍需谨慎。</p>
<p><b>估计起裂后的0/33实验差异不能承担完整体校准任务。</b>按用户估计，1.5 mm时0/33均已裂。完整体计算仍继续抵抗压入，而实验多了裂面形成、接触重分配等过程；二者斜率差不能用来直接调软共同本构。不同角度的起裂深度也不同，所以实验后段角度差混合了变形、摩擦和断裂状态。既不能宣布“本构已正确”，也不能宣布“断裂已解释全部偏差”。</p>
<h2>保存场给出的机制证据</h2>
<p>沿各自最后真实加载增量，将体内虚功分成体积、平衡剪切和Maxwell项。下表为0°减33°，单位是虚功除深度后的N；它是已有平衡路径的记账，不是参数敏感性，也不是重新求解后的贡献比例。接触虚功另参与总平衡闭合，未隐藏在材料项中。</p>
{work_table}
<p>在1 mm，平衡剪切差约+.219 N，Maxwell差约−.209 N，彼此抵消；到1.5 mm二者均为正，完整体角度差开始增加。两角度近尖端同时有法向与剪切变形，其比例随深度演化，不能只把斜切映射为更小的法向深度。该分解解释数值模型内部“怎样形成差值”，尚未证明材料律是实验差异的唯一原因。</p>
<p>0/33°承载的原表面材料中位位置仅缓慢移动：.5→1.5 mm分别约0→−.015 mm和+.020→+.025 mm；连续法向承载深度随加载增长，没有看到角度差出现时大批新材料替换承载的证据。33°唯一最低刀尖身份偏移约+.02068→+.02131 mm。缺失身份不能填零，也不能把“没有唯一最低点接触”说成刀具完全不接触材料；这些离散观测尚不是实验滑移真值。</p>
<p>局部压缩仍然显著。1.5 mm时0/33/57的最小J见上表；极小值集中于尖端附近很小的单元区域。0/33在参考半径1 mm内的J&lt;.5面积占比分别约0.01%/0.03%，不能把minJ解释成整体体积塌缩；也不能因为范围小就忽略局部应力和接触身份的离散误差。</p>
<h2>后续工作应如何收敛到任务</h2>
<ol><li>保持共同Maxwell和本轮完整体基线，使用0/33共同较早窗口（例如0–.5 mm）检查幅值、曲率与表面运动。这个窗口是基于当前起裂估计的暂定窗口，不是新的未裂证明。</li><li>只有在该窗口内成立的偏差才用于共同材料/界面/离散的有界诊断。先前去松弛使.5 mm力升到约19.3 N，远离读图幅值，且没有必要放大此处角度差；因此不优先续跑无松弛长线。</li><li>不从1–1.5 mm的角度差反拟合材料，不强求完整体复现裂后降斜率。以后若恢复起裂预测，须另给断裂准则及验证证据；本轮没有启用断裂。</li></ol>
<h2>数据与复现</h2>
<p><a href="fz_three_angles_15mm.csv">三角度深度–Fz CSV</a> · <a href="contact_three_angles_15mm.csv">接触材料身份和J CSV</a> · <a href="review_15mm.json">核查、起裂估计与输入哈希</a> · <a href="evidence.json">原0/33场诊断证据</a> · <a href="audit_log.html">本轮锚点审查日志HTML</a></p>
<p><a href="field_evolution.png">0/33完整体场演化（补充至2 mm）</a> · <a href="deformation_j.png">0/33局部变形与J</a> · <a href="force_comparison.png">原0/33至2 mm描述性比较，已标明裂后限制</a>。补充图的深段仅作完整体反事实分析，不是实验未裂假设。</p>
<p>代码：tools/analyze_angle_emergence.py。使用 --review-three 重新核验现有分析输入并生成本报告，不进行新的加载求解。保留原evidence.json及其分析版本哈希，另记录本次报告版本，不伪称旧证据由修改后的代码生成。</p>'''
    atomic_write(out/'index.html',shell('共同Maxwell三角度1.5 mm复核',body))
    log_body='<h1>已有路径审查日志</h1><p>这是保存场重装配，没有新的加载求解。0/33前置诊断至2 mm，本报告只取至1.5 mm；57新审查至1.5 mm。完整结果见<a href="index.html">分析</a>。</p>'
    for angle in (0,33,57):
        raw=(out/f'case{angle}.log').read_bytes()
        # Windows PowerShell redirection writes UTF-16; Python child logs UTF-8.
        log_text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        log_body+=f'<h2>{angle}°</h2><pre>'+escape(log_text)+'</pre>'
    log_body+='<h2>本轮15态 / 453历史态摘要</h2><pre>'+escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'audit_log.html',shell('保存场审查日志',log_body))
    print(json.dumps(dict(status=data['status'],states=453,fresh=15,checked_files=len(inputs),cases=summaries),ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--plot-only',action='store_true');ap.add_argument('--review-three',action='store_true');ap.add_argument('--case-angle',type=int,choices=(0,33,57));ap.add_argument('--end-depth',type=float,default=2.);args=ap.parse_args()
    if args.case_angle is not None:analyze_case(args.output.resolve(),args.case_angle,args.end_depth)
    elif args.review_three:plot(args.output.resolve());review_three(args.output.resolve())
    else:
        if not args.plot_only:analyze(args.output.resolve())
        plot(args.output.resolve())
