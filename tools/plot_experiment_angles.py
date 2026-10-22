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

import numpy as np
from prototype.material_case import load_force_csv
from run_support import write_json
from tools.qualify_intact_baseline import sha, verify_case
from tools.render_solver_dashboard import atomic_write, shell


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
        zero_handling='Original force and first-contact depth zeros preserved',input_sha256=files)
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
    write_json(out/'comparison.json',dict(cases=cases,scope='Saved accepted curves; no parameter fitting or fracture inference'))
    rows=''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in
        ('angle_deg','comparison_end_mm','rmse_n','endpoint_simulation_n','endpoint_experiment_n'))+'</tr>' for r in cases)
    body='<h1>实验与完整体仿真对照</h1><p>原始零点与单位保持；仅对共同区间插值。未确定起裂深度时，后段差值不能作为完整体模型误差。</p>'
    body+='<img src="comparison.png" style="max-width:100%"><table><tr><th>角度</th><th>比较至 mm</th><th>RMSE N</th><th>FE端点 N</th><th>实验端点 N</th></tr>'+rows+'</table>'
    body+='<p><a href="comparison.csv">对齐CSV</a> · <a href="comparison.json">来源、分段斜率及原始指纹</a></p>'
    atomic_write(out/'index.html',shell('实验与仿真对照',body))


if __name__=='__main__':main()
