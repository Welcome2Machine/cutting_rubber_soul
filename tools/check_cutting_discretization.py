"""Explicit actual-geometry constraint/local-P2 qualification, not production.

The common P1/PE constitutive law, blade, native friction and physical acceptance
are preserved. Every integration-point layout starts from rest; only its OWN
identically signed history may resume. PS is a constraint test, not finite 3-D.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from prototype.ipc_viscoelastic_p2 import QuadratureViscoelasticProblem
from tools import check_native_mesh as runner
from tools.run_material_candidate import signature as base_signature
from run_support import source_hashes, write_json


def make_problem(angle, kind='p2', kin='plane_strain', radius=.5, order=3):
    p = QuadratureViscoelasticProblem(maxwell_shear=1.24, relaxation_s=1.7,
        loading_rate=.2, bulk_modulus=23.34333333333331, case='oblique', mu=.35,
        shear_modulus=.14, ogden_alpha=2., dhat=.0025, local_spacing=.05,
        surface_spacing=.005, nose_facets=64, inclination=angle, tip_radius=.02,
        fine_halfwidth=3., max_iterations=500, predictor='secant', hessian_mode='regularized',
        element='p1', kinematics=kin, enrichment_radius=radius if kind=='p2' else None,
        quadrature_order=order)
    p.stagnation_window=40
    return p


def signature(p):
    r=base_signature(p)
    r['kinematics']=p.kinematics
    r['quadrature_layout']={key:hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                            for key,value in p.discretization_data().items()}
    r['enrichment_radius_mm']=p.enrichment_radius
    return r


def geometry_audit(p, base, **unused):
    assert p.config.to_dict()==base.config.to_dict() and p.material==base.material
    assert p.dhat==base.dhat and p.loading_rate==base.loading_rate and p.max_iterations==base.max_iterations
    np.testing.assert_array_equal(p.mesh.reference_xy[:base.n],base.mesh.reference_xy)
    np.testing.assert_array_equal(p.mesh.triangles,base.mesh.triangles)
    np.testing.assert_array_equal(p.mesh.bottom_nodes,base.mesh.bottom_nodes)
    np.testing.assert_array_equal(p.rest[p.surface],base.rest[base.surface])
    np.testing.assert_array_equal(p.cmesh.edges,base.cmesh.edges)
    np.testing.assert_array_equal(p.rest[p.n:],base.rest[base.n:])
    np.testing.assert_array_equal(p.previous,p.rest)
    np.testing.assert_array_equal(p.material_metric,np.tile(np.eye(3),(len(p.integration_weights),1,1)))
    return dict(passed=True,base_nodes=base.n,nodes=p.n,
        triangles=len(p.mesh.triangles),quadrature_points=len(p.integration_weights),
        enriched_triangles=len(p.active_elements),original_trace_and_blade_identical=True,
        fixed_bottom_identical=True,material_identical=True,kinematics=p.kinematics,
        fresh_rest_histories=True,not_finite_width_3d=True)


def snapshot(out):
    hashes=source_hashes(packages=('prototype','mechanics'),snapshot=out/'source')
    root=Path(__file__).resolve().parents[1]
    for name in ('tools/check_cutting_discretization.py','tools/check_native_mesh.py',
                 'tools/run_material_candidate.py','prototype/ipc_baseline.json'):
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,dst);hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return hashes


def export_surface(out, report):
    """All original top nodes, not only near-tip probes. Reference identity fixed."""
    import csv
    p=make_problem(report['settings']['angle'],report['settings']['kind'],
        report['settings']['kinematics'],report['settings']['radius'],report['settings']['order'])
    s=p.mesh.local_sn[:p.base_nodes]
    nodes=np.flatnonzero(np.isclose(s[:,1],p.config.height_mm,atol=1e-10))
    theta=np.deg2rad(p.config.inclination_deg)
    es=np.array([np.cos(theta),-np.sin(theta)]);en=np.array([np.sin(theta),np.cos(theta)])
    with (out/'surface_material_motion.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['depth_mm','reference_node_id','reference_s_from_contact_mm',
                                        'tangential_displacement_mm','normal_displacement_mm','x_mm','y_mm'])
        for i,record in enumerate(report['steps']):
            with np.load(out/'states'/f'{i:04d}.npz') as z:
                x=z['positions'][nodes];u=x-p.rest[nodes]
            w.writerows((record['depth_mm'],int(n),s[n,0]-p.config.drop_s_mm,
                float(du@es),float(du@en),float(y[0]),float(y[1])) for n,du,y in zip(nodes,u,x))


def main():
    a=argparse.ArgumentParser(__doc__)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--angle',type=float,choices=(0.,33.,57.),required=True)
    a.add_argument('--kind',choices=('p1','p2'),default='p2')
    a.add_argument('--kinematics',choices=('plane_strain','plane_stress'),default='plane_strain')
    a.add_argument('--radius',type=float,default=.5)
    a.add_argument('--order',type=int,choices=(3,4),default=3)
    a.add_argument('--depth',type=float,default=.01)
    a.add_argument('--increment',type=float,default=.01)
    a.add_argument('--seconds',type=float,default=300.)
    a.add_argument('--threads',type=int,default=2)
    a.add_argument('--resume',type=Path)
    a.add_argument('--phase',choices=('geometry','run'),default='run')
    args=a.parse_args();args.level=0
    if not all(np.isfinite(x) and x>0 for x in (args.radius,args.depth,args.increment,args.seconds)):
        a.error('Require positive finite radius/depth/increment/budget')
    if args.depth>.25 or args.threads<1:
        a.error('Bounded study maximum .25 mm; positive threads required')
    meta=dict(name='actual-geometry-constraint-local-P2',kind=args.kind,kinematics=args.kinematics,
        radius_mm=args.radius if args.kind=='p2' else None,order=args.order,
        fresh_start_required=True,no_cross_mesh_history_transfer=True)
    report=runner.run(args,problem_factory=lambda angle,level,radius:
        make_problem(angle,args.kind,args.kinematics,radius,args.order),
        geometry_checker=geometry_audit,source_snapshot=snapshot,problem_signature=signature,
        diagnostic_metadata=meta)
    if report['steps']:
        try:export_surface(args.output,report)
        except Exception as exc:
            report['surface_export_error']=repr(exc);write_json(args.output/'result.json',report)
            raise


if __name__=='__main__':main()
