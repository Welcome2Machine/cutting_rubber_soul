"""Report raw accepted paths, event windows, and explicit qualification gates."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root=Path(root);cases={};energy=[];events=[]
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for name in ('activation_witness_pilot','pure_notch_entry','pure_wide_notch_entry','pure_closed_entry'):
        path=root/name/'result.json'
        if not path.exists():continue
        raw=json.loads(path.read_text());rows=raw['steps'];audit_path=root/(name+'.audit.json')
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        cases[name]=dict(summary=raw['summary'],status=raw['status'],error=raw.get('error'),
            failure_kind=raw.get('failure_kind'),rejection_classes=dict(Counter(r['failure_kind'] for r in raw['attempts'])),
            negative_algorithmic_friction_rejections=sum('Negative endpoint friction work' in r['error'] for r in raw['attempts']),
            input=raw['config'],source_hashes=raw['source_hashes'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            independent_audit_passed=audit.get('passed',False),strict_entry_verified=audit.get('strict_interior_entry_verified',False),
            audit_scope='Saved equations, histories, J/CCD and empirical primary-chord quadrature; not physical-time or event convergence',
            first_entry=(audit.get('strict_entry_states') or [None])[0],accepted_state_count=len(rows),rejected_attempt_count=len(raw['attempts']))
        if audit.get('strict_entry_states'):
            first=audit['strict_entry_states'][0]
            with np.load(root/name/first['state_file'],allow_pickle=False) as z:
                x=z['positions'];tri=z['triangles'];n=len(z['mass']);reference=z['reference_xy']
            from .run_precrack_entry import problem
            p=problem(raw['config']);np.testing.assert_array_equal(p.rest,reference)
            blade_edges=p.all_boundary_edges[np.all(p.all_boundary_edges>=n,axis=1)]
            f,ax=plt.subplots(figsize=(6,6))
            for sign in (-1,1):
                ax.triplot(sign*(x[:n,0]-15),x[:n,1],tri,color='.6',lw=.4)
                for edge in blade_edges:ax.plot(sign*(x[edge,0]-15),x[edge,1],color='black',lw=1.)
                for q in first['contacts']:
                    xy=np.asarray(q['current_positions_mm']);ax.plot(sign*(xy[:,0]-15),xy[:,1],'-o',color='green')
                    xy=x[q['contact']['blade_nodes']];ax.plot(sign*(xy[:,0]-15),xy[:,1],'x',color='blue')
            ax.plot([],[],'-o',color='green',label='active EV body-edge support')
            ax.plot([],[],'x',color='blue',label='active EV blade vertex')
            ax.axhline(5.,ls='--',color='gray',label='original top')
            ax.axhline(4.5,ls=':',color='red',label='initial crack tip')
            ax.set(xlim=(-.32,.32),ylim=(4.35,5.08),xlabel='x - 15 mm',ylabel='y mm',
                title=f'{name}; first audited strict entry d={first["depth_mm"]:.8g} mm')
            ax.set_aspect('equal');ax.legend();f.tight_layout();f.savefig(root/(name+'_entry_geometry.png'),dpi=160);plt.close(f)
        if not rows:continue
        if name=='pure_closed_entry':
            from . import run_precrack_entry as entry
            from .run_fracture_route import restore_audit
            from .face_events import endpoint_forces
            if entry.sources()!=raw['source_hashes']:raise ValueError('Closed-state force report needs exact producer source')
            p=entry.problem(raw['config'])
            with np.load(root/name/rows[-1]['state_file'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            restore_audit(p,s,rows[-1]);fields=endpoint_forces(p,s['positions']);top=int(p.seam[-1])
            forces={k:float(-v[2*top]) for k,v in fields.items()}
            cases[name]['last_top_diagnostic']=dict(node=top,gap_mm=float(2*(s['positions'][top,0]-p.config.drop_s_mm)),
                released_from_intact_faces=bool(s['released'][-1]),unilateral_mirror_constraint_active=bool(s['active_contact'][top]),
                outward_force_components_half_n=forces,outward_force_sum_half_n=sum(forces.values()),
                scope='Instantaneous force at accepted endpoint, excludes constraint reaction and inertia; not proof of non-entry at uncomputed depths')
        t=np.array([r['time_s'] for r in rows]);fz=np.array([r['reaction_fz_n'] for r in rows])
        axes[0].plot(t,fz,label=name)
        axes[1].plot(t,[r['instantaneous_fz_n'] for r in rows],label=name)
        axes[2].plot(t,[r['energy']['kinetic_energy_n_mm'] for r in rows],label=name)
        peak=0;pair=(0,0);drop=0.
        for i in range(len(rows)):
            if fz[i]>fz[peak]:peak=i
            if fz[peak]-fz[i]>drop:drop=float(fz[peak]-fz[i]);pair=(peak,i)
        a,b=pair;window=rows[a+1:b+1];keys=rows[0]['energy']
        event=dict(case=name,complete_event=False,scope='Raw maximum peak-to-later-trough subwindow; no stable post-event proof',
            peak_index=a,trough_index=b,peak_time_s=float(t[a]),trough_time_s=float(t[b]),
            peak_depth_mm=rows[a]['depth_mm'],trough_depth_mm=rows[b]['depth_mm'],drop_n=drop,
            normal_drop_n=rows[a]['reaction_normal_fz_n']-rows[b]['reaction_normal_fz_n'],
            friction_drop_n=rows[a]['reaction_friction_fz_n']-rows[b]['reaction_friction_fz_n'],
            energy_n_mm={k:sum(r['energy'][k] for r in window) for k in keys if k not in ('kinetic_energy_n_mm','mechanical_reservoir_n_mm')})
        events.append(event)
        energy.append(dict(case=name,scope='whole accepted prefix',**raw['summary']['ledger_n_mm'],
            absolute_numerical_work_n_mm=raw['summary']['absolute_numerical_work_n_mm']))
        f,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
        lo=max(0,a-3);hi=min(len(rows),b+4);selected=rows[lo:hi];tt=[r['time_s'] for r in selected]
        for key in ('reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n'):axs[0].plot(tt,[r[key] for r in selected],'.-',label=key)
        axs[1].plot(tt,[r['energy']['kinetic_energy_n_mm'] for r in selected],'.-',label='K')
        axs[0].set(ylabel='raw force N',title=name+'; incomplete event window');axs[1].set(xlabel='absolute physical time s',ylabel='K N mm')
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        f.tight_layout();f.savefig(root/(name+'_event.png'),dpi=140);plt.close(f)
    for ax,title in zip(axes,['Algorithmic blade reaction N','Instantaneous blade force N','Kinetic energy N mm']):
        ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('absolute physical time s; no event alignment or smoothing')
    fig.tight_layout();fig.savefig(root/'raw_fz_and_kinetic.png',dpi=140);plt.close(fig)
    with (root/'energy_table.csv').open('w') as f:
        keys=list(dict.fromkeys(k for row in energy for k in row));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(energy)
    (root/'event_windows.json').write_text(json.dumps(events,indent=2))
    summary=dict(cases=cases,event_windows=events,completed_2mm=False,physical_prediction_validated=False,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification_note='Precrack input, model activation, new crack growth, strict entry, complete event, energy/time/mesh validation are distinct.')
    (root/'qualification.json').write_text(json.dumps(summary,indent=2))
    links=''.join(f'<li><a href="{name}/index.html">{name}</a> <a href="{name}_event.png">raw event window</a>'+
        (f' <a href="{name}_entry_geometry.png">audited interior contact</a>' if (root/(name+'_entry_geometry.png')).exists() else '')+'</li>' for name in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Round 7 fracture evidence</title><h1>第七轮：接触参考与纯预裂</h1><p>输入预裂、模型激活、裂纹增长、严格内部进入、完整事件、时间／网格资格分别审查。所有曲线保留原绝对时间与原始力，失败后缀不补造。</p><img width="1000" src="raw_fz_and_kinetic.png"><ul>'+links+'</ul><p><a href="energy_table.csv">完整前缀能量表</a> <a href="event_windows.json">峰谷子窗账</a> <a href="qualification.json">资格与输入身份</a> <a href="accounting.json">作业核账</a> <a href="planar_reference_corrected/index.html">三档解析基准</a></p><p>条件硬接触参考并未通过完整有限变形动力学；本报告不升级2 mm、物理起裂或完整事件资格。</p>')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    from numerical_runtime import numerical_runtime
    with numerical_runtime(2,1,True):report(a.root)
if __name__=='__main__':main()
