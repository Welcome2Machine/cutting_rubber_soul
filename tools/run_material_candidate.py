"""Bounded intact material experiment; separate from default/legacy queues.

Do not edit solver sources while a case is running. Outputs retain exact pre-increment
friction AND material histories. Resumption requires matching sources/parameters
and reassembles every state. No damage and no force rescaling.
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import source_hashes,write_json,save_npz,compute_process_policy
from prototype.ipc_reference import BASELINE,adaptive_load_steps,record_tip_contact,export_contact_table
from prototype.material_case import (make_problem, signature, audit, resume,
                                     resume_compatibility, load_force_csv, failure_checkpoint)
from prototype.tip_contact import TipContactTracker
from prototype.material_performance import (performance_policy, require_same_performance,
    enable_performance, performance_summary, close_performance, native_audit)


def compare(out,report,reference=None,image_estimates=None,experiment=None,*,experiment_angle=60.,image_angle=60.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    ref=json.loads(reference.read_text(encoding='utf-8')) if reference else None
    rows=report['steps']
    if not rows:return
    d=np.array([r['depth_mm'] for r in rows]);f=np.array([r['reaction_fz_n'] for r in rows])
    angle=report['signature']['config']['inclination_deg']
    if ref:
        config=ref.get('config') or ref.get('signature',{}).get('config',{})
        if config.get('inclination_deg')!=angle:
            raise ValueError('Force reference must have the same inclination')
    if image_estimates and angle!=image_angle:
        raise ValueError('Screenshot estimates must be explicitly assigned to this angle')
    rd=np.array([r['depth_mm'] for r in ref['steps']]) if ref else None
    rf=np.array([r['reaction_fz_n'] for r in ref['steps']]) if ref else None
    h=min(float(d[-1]),float(rd[-1])) if ref else float(d[-1])
    f0=f[0];rf0=rf[0] if ref else None
    # Linear interpolation ONLY within accepted data, for common-depth metrics.
    def metrics(x,y):
        zero=float(y[0]);mid=float(np.interp(h/2,x,y));end=float(np.interp(h,x,y))
        first=(mid-zero)/(h/2) if h>0 else None
        second=(end-mid)/(h/2) if h>0 else None
        return dict(horizon_mm=h,half_depth_ratio=(mid-zero)/(end-zero) if end>zero else None,
                    stiffness_growth_ratio=second/first if first and first>0 else None,
                    endpoint_force_n=end)
    summary=dict(candidate=metrics(d,f),reference=metrics(rd,rf) if ref else None,
                 status=report['status'],first_mm_available=bool(d[-1]>=1.-1e-10),
                 experimental_validation=False,
                 interpretation='Actual independently integrated FE histories; parameters remain hypotheses',
                 interpolation='linear within accepted states for common-depth metrics only',
                 reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest() if ref else None)
    experimental=None
    if experiment:
        if angle!=experiment_angle:raise ValueError('Experimental CSV must be explicitly assigned to this angle')
        ed,ef,rate=load_force_csv(experiment)
        if not np.isclose(rate,report['signature']['loading_rate'],rtol=0,atol=1e-12):
            raise ValueError('Experimental and FE loading speeds differ')
        select=(d>=ed[0])&(d<=ed[-1]);sd=d[select];sf=f[select]
        observed=np.interp(sd,ed,ef)
        delta=sf-observed
        # Compare candidates on the same .01 mm grid; adaptive cutbacks must
        # not give difficult depth regions extra weight in the screening RMSE.
        grid=np.arange(0.,float(sd[-1])+1e-12,.01)
        gf=np.interp(grid,d,f);ge=np.interp(grid,ed,ef);gd=gf-ge
        intervals=[]
        for a,b in ((0.,.5),(.5,1.),(1.,1.5),(1.5,2.)):
            if b<=min(float(d[-1]),float(ed[-1]))+1e-12:
                intervals.append(dict(start_mm=a,end_mm=b,
                    simulation_n_per_mm=float((np.interp(b,d,f)-np.interp(a,d,f))/(b-a)),
                    experiment_n_per_mm=float((np.interp(b,ed,ef)-np.interp(a,ed,ef))/(b-a))))
        experimental=dict(source_sha256=hashlib.sha256(Path(experiment).read_bytes()).hexdigest(),
            rows=len(ed),common_end_mm=float(sd[-1]),rmse_n=float(np.sqrt(np.mean(gd**2))),
            rmse_grid_mm=.01,rmse_grid_end_mm=float(grid[-1]),
            accepted_sample_rmse_n=float(np.sqrt(np.mean(delta**2))),
            force_increment_rmse_n=float(np.sqrt(np.mean(((gf-f[0])-(ge-ef[0]))**2))),
            endpoint_simulation_n=float(sf[-1]),endpoint_experiment_n=float(observed[-1]),
            interval_stiffness=intervals,experimental_validation=False,
            handling='Raw N and contact-depth zero preserved; increment RMSE subtracts each own F(0). '
                     'CSV compares accepted FE depths; RMSE uses common .01 mm grid. No extrapolation. '
                     'This trial participated in exploratory selection: NOT independent validation.')
        first=grid<=1.+1e-12
        experimental['first_mm_available']=bool(sd[-1]>=1.-1e-12)
        experimental['first_mm_rmse_n']=float(np.sqrt(np.mean(gd[first]**2))) if sd[-1]>=1.-1e-12 else None
        summary['experiment']=experimental
        with (out/'experiment_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['depth_mm','simulation_fz_n','experiment_fz_n','difference_n'])
            writer.writerows(zip(sd,sf,observed,delta))
    image_rows=[]
    if image_estimates is not None:
        with image_estimates.open(encoding='utf-8-sig',newline='') as stream:image_rows=list(csv.DictReader(stream))
        summary['image_sha256']=hashlib.sha256(image_estimates.read_bytes()).hexdigest()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    if ref:
        select=rd<=min(float(d[-1]),float(rd[-1]))+1e-12
        axes[0].plot(rd[select],rf[select],color='#0072B2',ls='--',label='Elastic P1 reference')
    candidate_label=('Maxwell + shear-limited interface' if 'interface_law' in report['signature'] else 'Finite Maxwell candidate')
    axes[0].plot(d,f,color='#D55E00',marker='.',ms=3,label=candidate_label)
    if experimental:
        select=ed<=float(sd[-1])+1e-12
        axes[0].plot(ed[select],ef[select],color='#222222',lw=1.3,label='Experiment (selection trial)')
        if h<=ed[-1] and h>0:
            select=ed<=h+1e-12
            axes[1].plot(ed[select]/h,(ef[select]-ef[0])/(np.interp(h,ed,ef)-ef[0]),
                         color='#222222',ls='-.',label='Experiment')
    # Field names are verified explicitly; unknown experimental formats fail closed.
    if image_rows:
        required={'depth_mm','force_n','lower_n','upper_n'}
        if not required<=image_rows[0].keys():raise ValueError('Unexpected screenshot-estimate columns')
        summary['image_comparison']=dict(angle_deg=angle,uncertainty='manual visual ranges, not confidence intervals',rows=[])
        for j,row in enumerate(image_rows):
            z=float(row['depth_mm']);center=float(row['force_n'])
            if d[0]<=z<=h+1e-10:
                value=float(np.interp(z,d,f))
                summary['image_comparison']['rows'].append(dict(depth_mm=z,fe_n=value,estimate_n=center,
                    lower_n=float(row['lower_n']),upper_n=float(row['upper_n']),difference_n=value-center))
                axes[0].errorbar([z],[center],yerr=[[center-float(row['lower_n'])],
                    [float(row['upper_n'])-center]],fmt='s',color='#333333',capsize=4,
                    label='Screenshot reading range (not CI)' if j==0 else None)
    if h>0 and np.interp(h,d,f)>f0:
        if ref:
            select=rd<=h+1e-12
            axes[1].plot(rd[select]/h,(rf[select]-rf0)/(np.interp(h,rd,rf)-rf0),
                         color='#0072B2',ls='--',label='Elastic reference')
        select=d<=h+1e-12
        axes[1].plot(d[select]/h,(f[select]-f0)/(np.interp(h,d,f)-f0),color='#D55E00',label='Maxwell candidate')
        axes[1].plot([0,1],[0,1],color='gray',ls=':',label='Linear shape guide')
    axes[0].set(xlabel='Downward depth (mm)',ylabel='Fz (N)',xlim=(0,max(float(d[-1]),.01)),ylim=(0,None))
    axes[1].set(xlabel='Depth / common endpoint',ylabel='[Fz - Fz(0)] / [Fz(end) - Fz(0)]',
                xlim=(0,1))
    axes[1].set_ylim(0,max(1.05,axes[1].get_ylim()[1]))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    mu=report['signature']['config'].get('friction_coefficient',.35)
    fig.suptitle(f'Intact {angle:g} deg, mu={mu:g} | accepted through {d[-1]:.3f} mm | {report["status"]}')
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)
    write_json(out/'comparison.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maxwell-shear',type=float,required=True)
    parser.add_argument('--equilibrium-shear',type=float,default=.14)
    parser.add_argument('--bulk-modulus',type=float)
    parser.add_argument('--equilibrium-alpha',type=float,default=2.,help='Common Ogden shape; small-strain G unchanged')
    parser.add_argument('--maxwell-alpha',type=float,default=2.,help='Common relaxing-branch shape; Gv and small-shear tau unchanged')
    parser.add_argument('--relaxation-s',type=float,required=True)
    parser.add_argument('--loading-rate',type=float,required=True)
    parser.add_argument('--angle',type=float,choices=(0.,30.,33.,57.,60.),default=60.)
    parser.add_argument('--mu',type=float,default=.35,
        help='Common positive Coulomb coefficient; exact-history resume requires an identical value')
    parser.add_argument('--depth',type=float,default=2.)
    parser.add_argument('--increment',type=float,default=.01)
    parser.add_argument('--seconds',type=float,default=21600.)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--diagnostics',action='store_true',
        help='Log actual PTC directions, step limits and phase timings; save last paired direction on failure')
    parser.add_argument('--material-batch-size',type=int,default=0,
        help='Opt-in material point batches; 0 retains the original whole-array evaluation')
    parser.add_argument('--blade-spacing',type=float,
        help='Optional straight-blade subdivision (mm); preserves shape/nose and requires an independent history')
    parser.add_argument('--linear-backend',choices=('native','pardiso'),default='native')
    parser.add_argument('--linear-threads',type=int,default=4,
        help='Positive PARDISO thread count; unused by the native backend')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--image-estimates',type=Path)
    parser.add_argument('--image-angle',type=float,default=60.,help='Explicit angle assigned to the screenshot table')
    parser.add_argument('--experiment-angle',type=float,default=60.,help='Explicit angle assigned to the force CSV')
    parser.add_argument('--experiment',type=Path,
        help='Authorized force-depth CSV in N and micrometres; assign its angle explicitly')
    args=parser.parse_args()
    if not all(np.isfinite(v) and v>0 for v in (args.depth,args.increment,args.seconds,args.mu)) or args.depth>2:
        parser.error('positive finite controls and target depth <= 2 mm required')
    if args.threads<1:parser.error('threads must be positive')
    if args.blade_spacing is not None and (not np.isfinite(args.blade_spacing) or not 0<args.blade_spacing<=.05):
        parser.error('Blade spacing must be in (0, .05] mm')
    try:
        performance=performance_policy(material_batch_size=args.material_batch_size,
            linear_backend=args.linear_backend,linear_threads=args.linear_threads,ipc_threads=args.threads)
    except ValueError as exc:parser.error(str(exc))
    if args.resume:
        old=json.loads((args.resume/'result.json').read_text(encoding='utf-8'))
        try:require_same_performance(old,performance)
        except ValueError as exc:parser.error(str(exc))
    if args.experiment:
        _,_,rate=load_force_csv(args.experiment)
        if args.angle!=args.experiment_angle or not np.isclose(rate,args.loading_rate,rtol=0,atol=1e-12):
            parser.error('Experiment requires its assigned angle and matching loading speed')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    sources=source_hashes(packages=('prototype','mechanics'))
    tool=Path(__file__).resolve()
    sources['tools/'+tool.name]=hashlib.sha256(tool.read_bytes()).hexdigest()
    baseline=Path(__file__).resolve().parents[1]/'prototype/ipc_baseline.json'
    sources['prototype/ipc_baseline.json']=hashlib.sha256(baseline.read_bytes()).hexdigest()
    report=dict(status='initializing',completed=False,source_sha256=sources,steps=[],
        solver_policy=dict(version=1,predictor='secant',affine_stagnation_window=0),
        target_mm=args.depth,rejected_load_attempts=[],force_tolerance_n=BASELINE['force_tolerance_n'],
        parameter_status='conditional material hypothesis; not independently calibrated',
        loading_rate_status='conditional inherited 0.2 mm/s unless explicitly replaced',
        tip_contact_tracking=dict(errors=[]),settings=vars(args).copy(),performance_policy=performance,
        load_control=dict(mode='adaptive_bisection',minimum_increment_mm=.000625,
                          maximum_increment_mm=args.increment,output_grid_increment_mm=args.increment))
    report['settings']={k:str(v) if isinstance(v,Path) else v for k,v in report['settings'].items()}
    write_json(out/'result.json',report)
    if args.experiment:
        shutil.copy2(args.experiment,out/'experiment.csv')
        args.experiment=out/'experiment.csv'
        report['experiment_sha256']=hashlib.sha256(args.experiment.read_bytes()).hexdigest()
        report['loading_rate_status']='matches speed_um_s in authorized experimental export'
    with (out/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),\
         threadpool_limits(limits=args.threads),\
         (threadpool_limits(limits=1,user_api='blas') if performance['blas_threads']==1
          else contextlib.nullcontext()),compute_process_policy() as policy:
        report['runtime_process_policy']=policy
        print('RUNTIME_POLICY '+json.dumps(policy),flush=True)
        started=time.monotonic();p=None;x=None
        try:
            construction={k:getattr(args,k) for k in ('angle','equilibrium_shear','maxwell_shear',
                'relaxation_s','loading_rate','bulk_modulus','mu','equilibrium_alpha','maxwell_alpha','blade_spacing')}
            p=make_problem(**construction)
            p.solver_diagnostics=args.diagnostics
            # JSON round trip makes signatures stable across tuples in Material.
            report['signature']=json.loads(json.dumps(signature(p)))
            report['reference_contact_check']=p.reference_contact_check
            report['mesh']=dict(nodes=p.n,elements=len(p.mesh.triangles))
            report['blade_discretization']=dict(spacing_mm=args.blade_spacing,
                outline_vertex_count=p.nose_facets+5,collision_vertices=len(p.rest)-p.n,
                geometry='Same original polygon, canonical nose vertices retained; straight-edge splits only')
            x=p.rest.copy();old=p.start
            if args.resume:
                report['status']='resuming';write_json(out/'result.json',report)
                x,old=resume(p,args.resume,out,report)
            enable_performance(p,performance)
            print('PERFORMANCE_POLICY '+json.dumps(performance),flush=True)
            tracker=TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets)
            report['status']='running';write_json(out/'result.json',report)
            solve_start=time.monotonic();deadline=solve_start+args.seconds
            targets=np.r_[0.,np.arange(args.increment,args.depth-1e-12,args.increment),args.depth]
            def rejected(event):
                report['rejected_load_attempts'].append(event);write_json(out/'result.json',report)
            for depth in targets[targets>old+1e-12]:
                for x,record in adaptive_load_steps(p,x,old,float(depth),deadline,.000625,rejected):
                    index=len(report['steps']);old=record['depth_mm']
                    save_npz(out/'states'/f'{index:04d}.npz',reference_xy=p.rest,
                        triangles=p.mesh.triangles,positions=x,previous_positions=p.previous,
                        depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                    report['steps'].append(record);report['last_depth_mm']=old
                    record_tip_contact(tracker,x,record,report)
                    write_json(out/'result.json',report)
                    print(f'ACCEPT depth_mm={old:.6f} Fz_N={record["reaction_fz_n"]:.9g} '
                          f'residual_N={record["residual_n"]:.4g}',flush=True)
            report.update(status='completed',completed=True)
        except Exception as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',
                          error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc(file=log)
            if p is not None and x is not None and p.material_dt is not None:
                failed,context=failure_checkpoint(p,x)
                save_npz(out/'failed_trial.npz',**failed)
                report['failed_attempt']=context
                if getattr(p,'ptc_state',None):
                    state=dict(p.ptc_state,previous_positions=failed['previous_positions'],
                               **p.checkpoint_data())
                    save_npz(out/'last_ptc.npz',**state)
        report['elapsed_s']=time.monotonic()-started
        write_json(out/'result.json',report)
        try:
            if p is not None and report['steps']:
                if args.material_batch_size or args.linear_backend!='native':
                    native_audit(lambda:make_problem(**construction),out,report)
                else:audit(p,out,report)
        except Exception as exc:
            report.update(status='audit_failed',completed=False,audit_error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc(file=log)
        finally:
            if p is not None:
                report['performance_counters']=performance_summary(p)
                try:close_performance(p)
                except Exception as exc:
                    report['backend_cleanup_error']=f'{type(exc).__name__}: {exc}'
                    traceback.print_exc(file=log)
    write_json(out/'result.json',report)
    export_contact_table(out,report)
    with (out/'fz_depth.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['depth_mm','fz_n'])
        writer.writerows((r['depth_mm'],r['reaction_fz_n']) for r in report['steps'])
    try:compare(out,report,args.reference,args.image_estimates,args.experiment,
                experiment_angle=args.experiment_angle,image_angle=args.image_angle)
    except Exception as exc:report['comparison_error']=f'{type(exc).__name__}: {exc}'
    write_json(out/'result.json',report)
    write_json(out/'outcome.json',{k:report.get(k) for k in
        ('status','completed','last_depth_mm','elapsed_s','error','audit_error','comparison_error')})
    print(json.dumps({k:report.get(k) for k in ('status','last_depth_mm','elapsed_s','error')}))


if __name__=='__main__':main()
