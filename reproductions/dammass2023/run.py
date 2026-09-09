"""Logged, restartable-by-new-directory paper benchmarks. No cutting inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
from run_support import ROOT,write_json,save_npz,source_hashes
from .material import Material,response
from .solver import Config,Problem


def prepare(out):
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'provenance.json',dict(paper='10.1007/s00466-023-02310-1',
        python=platform.python_version(),numpy=np.__version__,
        sources=source_hashes(('reproductions/dammass2023','mechanics'),snapshot=out/'source_snapshot'),
        paper_sha256=hashlib.sha256((ROOT/'references/updated phase.pdf').read_bytes()).hexdigest()))


def material_cases(out):
    rows=[];m=Material()
    for rate in [.0056,.044,.3]:
        A=np.eye(3)[None];lateral=1.;last=1.
        for stretch in np.linspace(1,2.5,601):
            dt=max((stretch-last)/rate,1e-8)
            for _ in range(25):
                F=np.array([[[stretch,0],[0,lateral]]]);q=response(F,A,dt,m)
                residual=q['P'][0,1,1]
                if abs(residual)<1e-10:break
                lateral-=residual/q['tangent'][0,3,3]
            else:raise RuntimeError('Uniaxial lateral equilibrium failed')
            rows.append(dict(stretch_rate_s=rate,stretch=stretch,cauchy_stress_mpa=float(q['P'][0,0,0]*stretch/q['J'][0]),
                nominal_stress_mpa=float(q['P'][0,0,0]),lateral_stretch=lateral,
                plane_stress_residual_mpa=float(q['plane_stress_residual'][0])))
            A=q['metric'];last=stretch
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'status.json',dict(completed=True,scope='Figure 5 homogeneous material response',material=m.to_dict(),rows=len(rows)))
    print('COMPLETE material curves',flush=True)


def sent(out,config):
    started=time.monotonic();p=Problem(config);old=p.initial();rows=[];cutbacks=0
    write_json(out/'config.json',dict(config=config.to_dict(),material=p.material.to_dict(),
        geometry=dict(full_length_mm=75,height_mm=30,notch_length_mm=20,thickness_mm=2,
            symmetry='half in loading direction; full opening U, grip motion U/2; reaction is not doubled'),
        nodes=len(p.xy),elements=len(p.cells),element='Q1, 2x2 Gauss; paper uses quadratic triangles'))
    save_npz(out/'mesh.npz',xy=p.xy,cells=p.cells)
    status=dict(completed=False,state='running',accepted_steps=0,last_opening_mm=0.)
    write_json(out/'status.json',status)
    stream=(out/'results.csv').open('w',newline='',encoding='utf-8');writer=None
    try:
        increment=config.increment_mm
        while old['opening']<config.end_opening_mm-1e-10:
            target=min(old['opening']+increment,config.end_opening_mm)
            try:new,row=p.step(old,target)
            except (RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                cutbacks+=1
                print(f'CUTBACK accepted_U={old["opening"]:.8g} trial_U={target:.8g}: {error}',flush=True)
                if increment/2<config.minimum_increment_mm:raise
                increment/=2;continue
            old=new;rows.append(row)
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush()
            save_npz(out/'last_state.npz',**old)
            status.update(accepted_steps=len(rows),last_opening_mm=old['opening'],last_force_n=row['force_n'],
                wall_seconds=time.monotonic()-started,cutbacks=cutbacks)
            write_json(out/'status.json',status)
            print('ACCEPT '+json.dumps(row),flush=True)
            if row['force_n']>=max(r['force_n'] for r in rows):save_npz(out/'peak_state.npz',**old)
            if row['stagger_iterations']<12:increment=min(config.increment_mm,increment*1.5)
        status.update(completed=True,state='complete')
        print('COMPLETE SENT',flush=True)
    except Exception as error:
        status.update(state='failed',error=str(error));traceback.print_exc();raise
    finally:
        stream.close();status['wall_seconds']=time.monotonic()-started;write_json(out/'status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['material','sent'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mesh-h',type=float,default=.275)
    parser.add_argument('--speed',type=float,default=25.)
    parser.add_argument('--end',type=float,default=22.)
    parser.add_argument('--increment',type=float,default=.25)
    args=parser.parse_args();prepare(args.output)
    if args.case=='material':material_cases(args.output)
    else:sent(args.output,Config(mesh_h_mm=args.mesh_h,speed_mm_min=args.speed,end_opening_mm=args.end,increment_mm=args.increment))


if __name__=='__main__':main()
