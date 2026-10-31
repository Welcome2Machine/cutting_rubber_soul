"""Three preregistered resolutions versus independent decimal closed forms."""
import argparse,csv,hashlib,json,sys,time
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from mechanics.impulse_reference import advance,midpoint_stop_counterexample
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .face_witness import native
from .run_zero_gap_cut import execution_identity
from run_support import write_json


def exact(mass,normal,mu,velocity,position,wall,window):
    """Independent closed form for vertical constant load, one e=0 impact and stop."""
    with localcontext() as ctx:
        ctx.prec=50
        m,N,c=map(lambda x:Decimal(str(x)),(mass,normal,mu));v=list(map(lambda x:Decimal(str(x)),velocity));x=list(map(lambda x:Decimal(str(x)),position));w=list(map(lambda x:Decimal(str(x)),wall));T=Decimal(str(window))
        a=N/m;rel=v[1]-w[1]
        hit=(rel+(rel*rel+2*a*x[1]).sqrt())/a if x[1]>0 else Decimal(0)
        pn=-m*(rel-a*hit);pt=-min(m*(v[0]-w[0]),c*pn)
        post=v[0]+pt/m;slip=post-w[0];stop=hit+slip/(c*a)
        assert 0<=hit<=stop<T and slip>=0
        finalx=x[0]+v[0]*hit+w[0]*(T-hit)+slip*slip/(2*c*a)
        P=[m*(w[0]-v[0]),m*(w[1]-v[1])+N*T]
        Dimpact=pn*pn/(2*m);Dfric=-pt*((v[0]-w[0])+slip)/2+m*slip*slip/2
        Wwall=P[0]*w[0]+P[1]*w[1];Wexternal=-N*(w[1]*T-x[1])
        return native(dict(position=[float(finalx),float(w[1]*T)],velocity=list(map(float,w)),impact_time_s=float(hit),stop_time_s=float(stop),contact_impulse_n_s=list(map(float,P)),friction_dissipation_n_mm=float(Dfric),normal_impact_loss_n_mm=float(Dimpact),wall_work_n_mm=float(Wwall),external_work_n_mm=float(Wexternal)))


@runtime_entry
def run(out,preregister,seconds=120,*,runtime=None):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);start=time.monotonic();limits=json.loads(Path(preregister).read_text())['benchmark']
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'mechanics/impulse_reference.py')}
    report=dict(schema='softcut-planar-impulse-reference/1',source_hashes=sources,execution_identity=execution_identity(),interpreter=sys.executable,thread_runtime=runtime,
                preregistration_sha256=hashlib.sha256(Path(preregister).read_bytes()).hexdigest(),limits=limits,cases=[],passed=False,
                scope='Exact piecewise constant point-mass planar reference; no FE, Maxwell, nonlinear geometry, or 2 mm qualification')
    m=limits['mass_n_s2_mm'];N=limits['normal_load_n'];mu=limits['mu'];T=limits['window_s']
    for name,x0,v0,wall in [('stop',[0.,0.],[1.,0.],[0.,0.]),('moving_wall',[0.,.01],[1.,-.2],[.2,.05])]:
        analytic=exact(m,N,mu,v0,x0,wall,T)
        for h in limits['dt_s']:
            x=np.array(x0);v=np.array(v0);t=0.;rows=[]
            while t<T-1e-14:
                if time.monotonic()-start>seconds:raise TimeoutError('Reference benchmark budget')
                q=advance(x,v,t,min(h,T-t),mass=m,force=[0,-N],wall_velocity=wall,mu=mu)
                x,v,t=q['position'],q['velocity'],q['time_s'];rows.append(q)
            sums={k:sum(row[k] for row in rows) for k in ('contact_impulse_n_s','friction_dissipation_n_mm','normal_impact_loss_n_mm','wall_work_n_mm','external_work_n_mm','kinetic_delta_n_mm','numerical_work_n_mm')}
            stop=next(e['time_s'] for row in rows for e in row['events'] if e['kind']=='stick')
            errors=dict(endpoint_velocity=float(np.max(np.abs(v-analytic['velocity']))),endpoint_position=float(np.max(abs(x-analytic['position']))),stop_time=abs(stop-analytic['stop_time_s']),impulse=float(np.max(abs(sums['contact_impulse_n_s']-analytic['contact_impulse_n_s']))),
                work=max(abs(sums[k]-analytic[k]) for k in ('friction_dissipation_n_mm','normal_impact_loss_n_mm','wall_work_n_mm','external_work_n_mm')),
                numerical_absolute=sum(abs(row['numerical_work_n_mm']) for row in rows),momentum=max(float(np.linalg.norm(row['momentum_residual_n_s'])) for row in rows))
            passed=errors['endpoint_position']<=limits['position_abs_tol_mm'] and errors['momentum']<=limits['momentum_abs_tol_ns'] and errors['endpoint_velocity']<=limits['velocity_abs_tol'] and errors['stop_time']<=limits['stop_time_abs_tol_s'] and errors['impulse']<=limits['impulse_abs_tol_ns'] and errors['work']<=limits['work_abs_tol_nmm'] and errors['numerical_absolute']<=limits['work_abs_tol_nmm'] and min(row['minimum_gap_mm'] for row in rows)>=-limits['gap_tol_mm']
            report['cases'].append(native(dict(name=name,dt_s=h,analytic=analytic,rows=rows,sums=sums,errors=errors,passed=passed)))
    old=[]
    for h in limits['dt_s']:
        v=1.;t=0.;rows=[]
        while t<T-1e-14:
            q=midpoint_stop_counterexample(v,min(h,T-t),m,mu*N);v=q['velocity'];t+=min(h,T-t);rows.append(dict(time_s=t,**q))
        old.append(dict(dt_s=h,rows=rows,false_reversal=any(r['velocity']<0 for r in rows)))
    report.update(midpoint_counterexample=old,passed=all(c['passed'] for c in report['cases']),elapsed_s=time.monotonic()-start)
    write_json(out/'result.json',report)
    with (out/'errors.csv').open('w') as f:
        keys=['name','dt_s','passed',*report['cases'][0]['errors']];w=csv.DictWriter(f,keys);w.writeheader();w.writerows(dict(name=c['name'],dt_s=c['dt_s'],passed=c['passed'],**c['errors']) for c in report['cases'])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axes,['stop','moving_wall']):
        for c in report['cases']:
            if c['name']==name:ax.plot([r['time_s'] for r in c['rows']],[r['velocity'][0] for r in c['rows']],'.-',label=str(c['dt_s']))
        ax.set(xlabel='absolute time s',ylabel='endpoint tangent velocity mm/s',title=name);ax.legend()
    fig.tight_layout();fig.savefig(out/'endpoint_velocity.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Event-resolved planar impulse reference</h1><p>Point mass / fixed plane normals only. No FE or fracture qualification.</p><img src="endpoint_velocity.png"><p><a href="errors.csv">Three-resolution errors</a> <a href="result.json">Raw segments, impacts, impulses and work</a></p><pre>'+json.dumps(dict(passed=report['passed'],limits=limits),indent=2)+'</pre>')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--preregister',required=True,type=Path);p.add_argument('--seconds',default=120,type=float);add_thread_arguments(p);a=p.parse_args()
    q=run(a.out,a.preregister,a.seconds,**thread_arguments(a));print(json.dumps(dict(passed=q['passed'],elapsed_s=q['elapsed_s'])))
if __name__=='__main__':main()
