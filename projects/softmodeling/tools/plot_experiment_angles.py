"""Compare explicit experimental CSVs with saved accepted FE curves.

No solving, force scaling, smoothing or zero shifting. Example:
python -m tools.plot_experiment_angles --output outputs/comparison \
  --case 0 outputs/baseline/angle0 docs/1008n0a2__Fz__curve_1.csv
Repeat --case ANGLE FE_DIRECTORY EXPERIMENT_CSV for each angle.
"""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import re

import numpy as np
from prototype.material_case import load_force_csv
from run_support import write_json
from tools.qualify_intact_baseline import sha, verify_case
from tools.render_solver_dashboard import atomic_write, shell


def solver_summary(folder,report):
    """Summarize the existing trace, separating retries from accepted loading."""
    attempts=[];current=None
    path=Path(folder)/'run.log'
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.startswith('FRICTION_PTC depth_mm='):
                    values=dict(re.findall(r'(\w+)=([^ ]+)',line.strip()))
                    if int(values['iteration'])==0:
                        current=dict(depth_mm=float(values['depth_mm']),rows=[],status='unresolved')
                        attempts.append(current)
                    if current is not None:
                        current['final_residual_n']=float(values['fresh_residual_N'])
                elif line.startswith('PTC_DIAGNOSTIC ') and current is not None:
                    current['rows'].append(json.loads(line.split(' ',1)[1]))
                elif line.startswith('LOAD_REJECT ') and current is not None:
                    current.update(status='rejected',event=json.loads(line.split(' ',1)[1]))
                elif line.startswith('ACCEPT ') and current is not None:
                    current['status']='accepted'
    rows=[]
    for attempt in attempts:
        values=attempt.pop('rows');accepted=[v for v in values if v['criterion']!='rejected']
        window=accepted[-40:]
        attempt.update(trials=len(values),updates=len(accepted),
            first_residual_n=values[0]['residual_n'] if values else None,
            last40=dict(updates=len(window),
                actual_increases=sum(v['trial_residual_n']>v['residual_n'] for v in window),
                predicted_increases=sum(v['predicted_residual_n']>v['residual_n'] for v in window),
                minimum_alpha=min((v['alpha'] for v in window),default=None),
                minimum_geometric_fraction=min((v['geometric_fraction'] for v in window),default=None),
                first_shift=window[0]['shift_used'] if window else None,
                last_shift=window[-1]['shift_next'] if window else None,
                max_linear_relative_residual=max((v['linear_relative_residual'] for v in window),default=None)))
        rows.append(attempt)
    phases={}
    events=report.get('rejected_load_attempts',[])
    items=[r.get('load_step',{}) for r in report['steps']]+events
    terminal=report.get('failed_attempt')
    if terminal and (not events or any(terminal.get(k)!=events[-1].get(k) for k in
            ('from_depth_mm','attempted_depth_mm','predictor'))):
        items.append(terminal)
    for item in items:
        for key,value in item.get('phase_seconds',{}).items():phases[key]=phases.get(key,0.)+value
    return dict(attempts=rows,phase_seconds=phases,
        note='Exclusive recorded phase totals, including terminal attempt without double counting; failed/retried attempts are not physical progress')


def short_window_metrics(sd,sf,ed,ef,max_depth=.5):
    """Fixed shared-fit targets; an unavailable anchor remains explicitly missing."""
    end=min(.5,max_depth,float(sd[-1]),float(ed[-1]))
    anchors=[]
    for depth in (.1,.25,.5):
        available=max(float(sd[0]),float(ed[0]))<=depth<=end+1e-12
        sim=float(np.interp(depth,sd,sf)) if available else None
        exp=float(np.interp(depth,ed,ef)) if available else None
        anchors.append(dict(depth_mm=depth,simulation_n=sim,experiment_n=exp,
                            error_n=sim-exp if available else None))
    slopes=[]
    for a,b in zip(anchors[:-1],anchors[1:]):
        available=a['simulation_n'] is not None and b['simulation_n'] is not None
        span=b['depth_mm']-a['depth_mm']
        sim=(b['simulation_n']-a['simulation_n'])/span if available else None
        exp=(b['experiment_n']-a['experiment_n'])/span if available else None
        relative=(sim-exp)/abs(exp) if available and abs(exp)>1e-12 else None
        slopes.append(dict(start_mm=a['depth_mm'],end_mm=b['depth_mm'],
            simulation_n_per_mm=sim,experiment_n_per_mm=exp,
            relative_error=relative,within_20_percent=None if relative is None else abs(relative)<=.2))
    grid=np.arange(.1,end+1e-12,.01)
    error=np.interp(grid,sd,sf)-np.interp(grid,ed,ef) if len(grid) else np.array([])
    return dict(window_mm=[.1,end] if end>=.1 else None,full_window=bool(end>=.5-1e-12),
        rmse_n=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        anchors=anchors,secants=slopes,
        slope_screen='20 percent relative secant error: exploratory engineering screen, not uncertainty')


def joint_metrics(cases,series):
    """Use one actual common span for pooled error and signed angle differences."""
    material=[r['signature']['material'] for r in cases]
    shared=all(m==material[0] for m in material)
    settings=[]
    for case in cases:
        signature=case['signature']
        settings.append(dict(config={k:v for k,v in signature['config'].items() if k!='inclination_deg'},
            blade_spacing_mm=case.get('blade_discretization',{}).get('spacing_mm'),
            **{k:signature.get(k) for k in ('loading_rate','contact_formulation','dhat','physical_force_tolerance_n')}))
    shared_settings=all(s==settings[0] for s in settings)
    end=min(.5,*[min(float(v[2][-1]),float(v[0][-1])) for v in series])
    grid=np.arange(.1,end+1e-12,.01)
    errors={str(c['angle_deg']):float(np.sqrt(np.mean((np.interp(grid,v[2],v[3])-
             np.interp(grid,v[0],v[1]))**2))) for c,v in zip(cases,series)} if len(grid) else {}
    by_angle={c['angle_deg']:c['short_window']['anchors'] for c in cases}
    differences=[]
    for a,b in ((33.,0.),(57.,0.),(57.,33.)):
        for i,depth in enumerate((.1,.25,.5)):
            available=(a in by_angle and b in by_angle and
                all(by_angle[t][i]['simulation_n'] is not None for t in (a,b)))
            sim=by_angle[a][i]['simulation_n']-by_angle[b][i]['simulation_n'] if available else None
            exp=by_angle[a][i]['experiment_n']-by_angle[b][i]['experiment_n'] if available else None
            differences.append(dict(angle_a=a,angle_b=b,depth_mm=depth,simulation_n=sim,
                experiment_n=exp,error_n=sim-exp if available else None))
    return dict(same_material=shared,same_other_physical_settings=shared_settings,
        all_three_angles=all(a in by_angle for a in (0.,33.,57.)),
        window_mm=[.1,end] if len(grid) else None,per_angle_rmse_n=errors,
        pooled_rmse_n=float(np.sqrt(np.mean(np.array(list(errors.values()))**2))) if errors and shared and shared_settings else None,
        worst_angle_rmse_n=max(errors.values()) if errors and shared and shared_settings else None,
        angle_differences=differences,interpretation='Actual FE only; unavailable anchors are null; no force ordering imposed')


def comparison(angle, folder, experiment, max_depth):
    """Read original units/zeros, compare on a uniform common grid only."""
    folder=Path(folder);experiment=Path(experiment)
    files=verify_case(folder)
    report=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if report['signature']['config']['inclination_deg']!=angle:
        raise ValueError('Requested angle does not match the FE result')
    ed,ef,speed=load_force_csv(experiment)
    if not np.isclose(report['signature']['loading_rate'],speed,rtol=0,atol=1e-12):
        raise ValueError('Experiment and simulation have different loading rates')
    sd=np.asarray([r['depth_mm'] for r in report['steps']])
    sf=np.asarray([r['reaction_fz_n'] for r in report['steps']])
    if len(sd)<2 or not np.isfinite(sf).all() or not np.all(np.diff(sd)>0):
        raise ValueError('At least two increasing accepted states required')
    start=max(float(sd[0]),float(ed[0]));end=min(max_depth,float(sd[-1]),float(ed[-1]))
    if end<=start:raise ValueError('No common comparison interval')
    grid=np.unique(np.r_[start,np.arange(np.ceil(start/.01)*.01,end,.01),end])
    sim=np.interp(grid,sd,sf);exp=np.interp(grid,ed,ef)
    intervals=[]
    for lo,hi in zip(np.arange(0,max_depth,.5),np.arange(.5,max_depth+.5,.5)):
        if lo>=start and hi<=end+1e-12:
            intervals.append(dict(start_mm=float(lo),end_mm=float(hi),
                simulation_n_per_mm=float((np.interp(hi,sd,sf)-np.interp(lo,sd,sf))/(hi-lo)),
                experiment_n_per_mm=float((np.interp(hi,ed,ef)-np.interp(lo,ed,ef))/(hi-lo))))
    files[str(experiment.resolve())]=sha(experiment)
    summary=dict(angle_deg=angle,simulation=str(folder.resolve()),experiment=str(experiment.resolve()),
        simulation_completed=report['completed'],accepted_end_mm=float(sd[-1]),
        comparison_start_mm=start,comparison_end_mm=end,rmse_n=float(np.sqrt(np.mean((sim-exp)**2))),
        endpoint_simulation_n=float(sim[-1]),endpoint_experiment_n=float(exp[-1]),interval_slopes=intervals,
        experiment_is_independent_validation=False,
        fracture_limit='Experimental onset is not inferred from this curve comparison; late data may include fracture.',
        zero_handling='Original force and first-contact depth zeros preserved',input_sha256=files,
        signature=report['signature'],blade_discretization=report.get('blade_discretization',{}),
        short_window=short_window_metrics(sd,sf,ed,ef,max_depth),
        solver=solver_summary(folder,report))
    return summary,(ed,ef,sd,sf,grid,sim,exp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,metavar=('ANGLE','FE_DIRECTORY','EXPERIMENT_CSV'))
    parser.add_argument('--max-depth',type=float,default=2.)
    args=parser.parse_args()
    if not np.isfinite(args.max_depth) or args.max_depth<=0:parser.error('Positive finite depth required')
    angles=[float(item[0]) for item in args.case]
    if len(set(angles))!=len(angles):parser.error('Each angle must occur once')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[];series=[]
    for item,angle in zip(args.case,angles):
        result,values=comparison(angle,Path(item[1]),Path(item[2]),args.max_depth)
        cases.append(result);series.append(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    with (out/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['angle_deg','depth_mm','experimental_fz_n','simulation_fz_n','error_n'])
        for result,values in zip(cases,series):
            ed,ef,sd,sf,grid,sim,exp=values
            keep=ed<=args.max_depth;line,=ax.plot(ed[keep],ef[keep],label=f'{result["angle_deg"]:g} deg experiment')
            keep=sd<=args.max_depth;ax.plot(sd[keep],sf[keep],ls='--',color=line.get_color(),label=f'{result["angle_deg"]:g} deg FE')
            writer.writerows((result['angle_deg'],d,e,s,s-e) for d,s,e in zip(grid,sim,exp))
    ax.set(xlabel='Vertical blade travel (mm)',ylabel='Fz (N)',xlim=(0,args.max_depth))
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    joint=joint_metrics(cases,series)
    write_json(out/'comparison.json',dict(cases=cases,joint=joint,
        scope='Saved accepted curves; no parameter fitting or fracture inference'))
    rows=''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in
        ('angle_deg','comparison_end_mm','rmse_n','endpoint_simulation_n','endpoint_experiment_n'))+'</tr>' for r in cases)
    body='<h1>实验与完整体仿真对照</h1><p>原始零点与单位保持；仅对共同区间插值。未确定起裂深度时，后段差值不能作为完整体模型误差。</p>'
    body+='<img src="comparison.png" style="max-width:100%"><table><tr><th>角度</th><th>比较至 mm</th><th>RMSE N</th><th>FE端点 N</th><th>实验端点 N</th></tr>'+rows+'</table>'
    body+='<p><a href="comparison.csv">对齐CSV</a> · <a href="comparison.json">来源、分段斜率及原始指纹</a></p>'
    body+='<h2>共同参数短窗口</h2><p>0.10–0.50 mm；缺失项不外推。割线相对误差20%仅为工程筛选。原始57°的0.51 N初值保留。</p>'
    body+='<table><tr><th>角度</th><th>实际窗口 RMSE / N</th><th>Fz(.10/.25/.50) FE / N</th><th>实验 / N</th><th>两段割线 FE / N/mm</th><th>实验 / N/mm</th></tr>'
    for c in cases:
        m=c['short_window']
        fields=[c['angle_deg'],m['rmse_n'],[a['simulation_n'] for a in m['anchors']],
            [a['experiment_n'] for a in m['anchors']],[s['simulation_n_per_mm'] for s in m['secants']],
            [s['experiment_n_per_mm'] for s in m['secants']]]
        body+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in fields)+'</tr>'
    body+='</table><p>共同材料：'+str(joint['same_material'])+'；三角度齐全：'+str(joint['all_three_angles'])+'</p>'
    body+='<details><summary>同窗口总误差、最差角度与有符号角度差</summary><pre>'+escape(json.dumps(joint,indent=2))+'</pre></details>'
    body+='<details><summary>实际求解尝试与分阶段计时</summary><pre>'+escape(json.dumps(
        {str(c['angle_deg']):c['solver'] for c in cases},indent=2))+'</pre></details>'
    atomic_write(out/'index.html',shell('实验与仿真对照',body))


if __name__=='__main__':main()
