"""Three seam grids and fixed face quadratures on prescribed opening fields only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from face_cohesive_quadrature import face_rule,response
from prototype.zero_gap_cut import extrinsic_response
from run_support import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    strength,toughness,width=.2,.006,20.;amplitude=.08
    def exact(y):
        delta=amplitude*np.sin(np.pi*y);q=extrinsic_response(delta,delta,strength,toughness)
        return float(width*(q['stored']+q['dissipated']))
    transition=np.arcsin((2*toughness/strength)/amplitude)/np.pi
    reference,error=quad(exact,0.,1.,points=[transition,1-transition],epsabs=1e-12)
    rows=[]
    for h in (.1,.05,.025):
        y=np.linspace(0,1,round(1/h)+1);opening=amplitude*np.sin(np.pi*y)
        for order in (2,4,8,16):
            rule=face_rule(y,width,order);q=response(opening,np.zeros(rule['history_shape']),rule,strength,toughness)
            nodal=extrinsic_response(opening,opening,strength,toughness)
            lumped=float(rule['nodal_tributary_area_mm2']@(nodal['stored']+nodal['dissipated']))
            rows.append(dict(seam_spacing_mm=h,quadrature_order=order,face_potential_n_mm=q['incremental_potential_n_mm'],
                nodal_potential_n_mm=lumped,face_error_n_mm=q['incremental_potential_n_mm']-reference,
                total_area_mm2=float(rule['face_area_mm2'].sum()),history_shape=rule['history_shape']))
    report=dict(schema='softcut-face-kernel-control/1',loading_started=False,production_nodal_model_changed=False,
        physical_cutting_mesh_convergence=False,prescribed_field='opening(y)=0.08 sin(pi y), y in [0,1] mm',
        strength_mpa=strength,toughness_n_mm=toughness,width_mm=width,
        independent_continuous_reference_n_mm=reference,reference_quadrature_empirical_error=error,rows=rows,
        source_sha256=hashlib.sha256((Path(__file__).parents[1]/'face_cohesive_quadrature.py').read_bytes()).hexdigest())
    write_json(args.output/'validation.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for order in (2,4,8,16):
        selected=[q for q in rows if q['quadrature_order']==order]
        ax.loglog([q['seam_spacing_mm'] for q in selected],[abs(q['face_error_n_mm']) for q in selected],'o-',label=f'Face Gauss {order}')
    selected=[q for q in rows if q['quadrature_order']==2]
    ax.loglog([q['seam_spacing_mm'] for q in selected],[abs(q['nodal_potential_n_mm']-reference) for q in selected],'s--',label='Nodal lumping')
    ax.set(xlabel='Seam spacing [mm]',ylabel='Absolute potential error [N mm]',title='Prescribed field only: not a cutting-path mesh validation');ax.legend();ax.grid();fig.tight_layout();fig.savefig(args.output/'error.png');plt.close(fig)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Face kernel control</title><h1>Independent face kernel control</h1><p>Prescribed opening field only. Production nodal model unchanged; full cutting mesh qualification remains false.</p><img src="error.png"><p><a href="validation.json">Data and source identity</a></p>')
    print(json.dumps(dict(rows=len(rows),reference_n_mm=reference)))


if __name__=='__main__':main()
