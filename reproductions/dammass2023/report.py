"""Compare saved calculations with independently digitized paper model curves."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write_json


def table(path):
    with path.open(encoding='utf-8') as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def require_reference_data():
    ref=Path(__file__).parent/'reference'
    required=('fig5.csv','fig7_stroke_envelopes.csv','fig7_landmarks.json','provenance.json')
    missing=[name for name in required if not (ref/name).is_file()]
    if missing:
        raise FileNotFoundError('Paper digitization data is local-only and is not distributed. '
            'Restore the source PDF and run python -m reproductions.dammass2023.extract_reference. '
            'Missing: '+', '.join(missing))
    return ref


def export(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=170)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material',type=Path,required=True)
    parser.add_argument('--sent',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ref=require_reference_data()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent
    criteria=json.loads((root/'acceptance.json').read_text())
    material_status=json.loads((args.material/'status.json').read_text())
    report=dict(scope=criteria['scope'],criteria=criteria,material=[],sent=[],all_accepted=False,
        material_completed=material_status['completed'],
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_provenance=json.loads((ref/'provenance.json').read_text()),
        evidence='Calculated accepted states vs PDF model-line digitization, not raw experimental validation')
    colors=['#0072B2','#D55E00','#009E73'];styles=['-','--','-.'];markers=['o','s','^']
    references=table(ref/'fig5.csv');computed=table(args.material/'results.csv')
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
        for i,rate in enumerate([.0056,.044,.3]):
            rr=[r for r in references if r['stretch_rate_s']==rate];cc=[r for r in computed if r['stretch_rate_s']==rate]
            rx=np.array([r['stretch'] for r in rr]);ry=np.array([r['stress_mpa'] for r in rr])
            x=np.array([r['stretch'] for r in cc]);y=np.array([r['cauchy_stress_mpa'] for r in cc])
            sample=np.linspace(max(1.02,rx.min()),min(rx.max(),x.max()),100)
            truth=np.interp(sample,rx,ry);prediction=np.interp(sample,x,y)
            error=float(np.sqrt(np.mean((prediction-truth)**2))/max(truth))
            covered=bool(x.min()<=rx.min()+.005 and x.max()>=rx.max()-.005)
            report['material'].append(dict(rate_s=rate,relative_rmse=error,covered=covered,
                accepted=bool(material_status['completed'] and covered and error<=criteria['material_curve_relative_rmse_max'])))
            ax.plot(x,y,color=colors[i],linestyle=styles[i],label=f'Computed {rate:g} /s')
            samples=np.linspace(rx.min(),rx.max(),15)
            ax.plot(samples,np.interp(samples,rx,ry),linestyle='none',marker=markers[i],mfc='none',color=colors[i],label=f'Paper {rate:g} /s')
        ax.set(xlabel='Stretch',ylabel='Cauchy stress (MPa)',title='Figure 5: finite viscoelastic material',xlim=(1,2.5),ylim=(0,23))
        ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
        export(fig,out/'material_comparison')
        references=table(ref/'fig7_stroke_envelopes.csv');landmarks=json.loads((ref/'fig7_landmarks.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
        for directory in args.sent:
            cfg=json.loads((directory/'config.json').read_text())['config'];rate=cfg['speed_mm_min']
            status=json.loads((directory/'status.json').read_text());data=table(directory/'results.csv')
            if not data:continue
            index=0 if rate==25 else 1;ax=axes[index]
            x=np.r_[0.,[r['opening_mm'] for r in data]];y=np.r_[0.,[r['force_n'] for r in data]]
            peak=int(np.argmax(y));target=next(r for r in landmarks if r['speed_mm_min']==rate)
            reference=[r for r in references if r['speed_mm_min']==rate]
            rx=np.array([r['opening_mm'] for r in reference]);ry=np.array([r['upper_force_n'] for r in reference])
            samples=np.linspace(1,.8*target['peak_opening_mm'],40)
            nrmse=float(np.sqrt(np.mean((np.interp(samples,x,y)-np.interp(samples,rx,ry))**2))/target['peak_force_n'])
            ferr=float(abs(y[peak]/target['peak_force_n']-1));uerr=float(abs(x[peak]/target['peak_opening_mm']-1))
            drop=float(1-y[peak:].min()/y[peak])
            covered=bool(x.max()>=samples.max())
            checks=dict(completed=status['completed'],prepeak_covered=covered,peak_force=ferr<=criteria['sent_peak_force_relative_error_max'],
                peak_opening=uerr<=criteria['sent_peak_displacement_relative_error_max'],
                prepeak_curve=nrmse<=criteria['sent_prepeak_normalized_rmse_max'],
                postpeak_drop=drop>=criteria['postpeak_force_drop_fraction_min'])
            work=float(trapezoid(y,x))
            accounted=sum(data[-1][name] for name in ('bulk_energy_n_mm','fracture_energy_n_mm','viscous_dissipation_n_mm'))
            accounted+=data[-1].get('phase_viscous_dissipation_n_mm',0.)
            report['sent'].append(dict(directory=str(directory),speed_mm_min=rate,h_mm=cfg['mesh_h_mm'],
                results_sha256=hashlib.sha256((directory/'results.csv').read_bytes()).hexdigest(),
                peak_force_n=float(y[peak]),peak_opening_mm=float(x[peak]),paper_peak=target,
                relative_peak_force_error=float(ferr),relative_peak_opening_error=float(uerr),prepeak_nrmse=nrmse,
                postpeak_drop_fraction=drop,checks=checks,accepted=all(checks.values()),status=status,
                external_work_n_mm=work,accounted_energy_n_mm=accounted,
                relative_energy_imbalance=float((accounted-work)/max(abs(work),1e-12)),
                phase_viscosity_recorded='phase_viscous_dissipation_n_mm' in data[-1],
                energy_note='Diagnostic only: trapezoidal force work, endpoint bulk viscosity; phase viscosity included only when recorded. History-field/numerical jump effects remain unresolved.',
                max_equilibrium_residual_n=max(r['equilibrium_residual_n'] for r in data),
                max_plane_stress_scaled_residual=max((r['plane_stress_scaled_residual'] for r in data if 'plane_stress_scaled_residual' in r),default=None),
                max_plane_stress_residual_mpa=max(r['plane_stress_residual_mpa'] for r in data)))
            ax.plot(x,y,label=f'Computed h={cfg["mesh_h_mm"]:g} mm ({status["state"]})',linewidth=1.5)
            if len(ax.lines)==1:
                sample=np.linspace(max(rx.min(),.1),rx.max(),35)
                ax.plot(sample,np.interp(sample,rx,ry),'ko',mfc='none',ms=3,label='Paper model stroke (digitized)')
                ax.plot([target['last_opening_mm']],[target['last_lower_force_n']],'kx',ms=6,label='Paper final drop')
            ax.set(xlabel='Full grip opening (mm)',title=f'Figure 7 approach A: {rate:g} mm/min',xlim=(0,23),ylim=(0,60))
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[0].set_ylabel('Force (N)');export(fig,out/'sent_comparison')
        # Keep every supplied run in acceptance; do not select only a favorable mesh.
        rates=sorted(set(r['speed_mm_min'] for r in report['sent']))
        rate_checks=[]
        for h in sorted(set(r['h_mm'] for r in report['sent'])):
            group={r['speed_mm_min']:r for r in report['sent'] if r['h_mm']==h}
            if 25 in group and 200 in group:
                rate_checks.append(dict(h_mm=h,force_increases=group[200]['peak_force_n']>group[25]['peak_force_n'],
                    peak_opening_decreases=group[200]['peak_opening_mm']<group[25]['peak_opening_mm']))
        report['rate_checks']=rate_checks
        report['all_accepted']=bool(len(report['sent'])==len(args.sent) and all(r['accepted'] for r in report['material']) and rates==[25.,200.] and
            all(r['accepted'] for r in report['sent']) and rate_checks and
            all(r['force_increases'] and r['peak_opening_decreases'] for r in rate_checks))
        report['mesh_comparisons']=[]
        for rate in rates:
            group=sorted((r for r in report['sent'] if r['speed_mm_min']==rate),key=lambda r:r['h_mm'],reverse=True)
            for coarse,fine in zip(group,group[1:]):
                report['mesh_comparisons'].append(dict(speed_mm_min=rate,coarse_h_mm=coarse['h_mm'],fine_h_mm=fine['h_mm'],
                    peak_force_change_fraction=fine['peak_force_n']/coarse['peak_force_n']-1,
                    peak_opening_change_fraction=fine['peak_opening_mm']/coarse['peak_opening_mm']-1,
                    both_completed=bool(coarse['checks']['completed'] and fine['checks']['completed'])))
        for directory in args.sent:
            cfg=json.loads((directory/'config.json').read_text())['config']
            with np.load(directory/'mesh.npz') as z:xy,cells=z['xy'],z['cells']
            fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
            tri=np.r_[cells[:,[0,1,2]],cells[:,[0,2,3]]]
            for ax,name,title in zip(axes,['peak_state.npz','last_state.npz'],['Peak force','Last accepted state']):
                with np.load(directory/name) as z:d=z['damage'];opening=float(z['opening'])
                im=ax.tripcolor(xy[:,0],xy[:,1],tri,d,shading='gouraud',cmap='cividis',vmin=0,vmax=1)
                ax.set(xlabel='Distance from symmetry plane (mm)',ylabel='Reference y (mm)',title=f'{title}, U={opening:.3f} mm',xlim=(0,12),ylim=(15,30),aspect='equal')
            fig.colorbar(im,ax=axes,label='Phase field d (0 intact, 1 broken)',shrink=.75)
            export(fig,out/f'damage_{directory.name}')
    write_json(out/'acceptance.json',report)
    print(json.dumps(dict(all_accepted=report['all_accepted'],material=report['material'],sent=[{k:r[k] for k in ('speed_mm_min','h_mm','peak_force_n','peak_opening_mm','accepted')} for r in report['sent']]),indent=2))


if __name__=='__main__':main()
