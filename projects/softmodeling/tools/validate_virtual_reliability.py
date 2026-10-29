"""Bounded archived-field validation of work intervals, CCD and seam areas."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve
from numerical_runtime import numerical_runtime,add_thread_arguments,thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from .virtual_work_audit import audit_work
from virtual_friction_work import virtual_work,interval_decision
from virtual_path_geometry import native_partitioned_chord,ev_crossing_witnesses
from prototype.zero_gap_cut import extrinsic_response


def area_evidence(p,v,after):
    y=p.rest[p.seam,1];length=np.diff(y);area=p.seam_area
    nodes,weights=np.polynomial.legendre.leggauss(32);shape=np.column_stack(((1-nodes)/2,(1+nodes)/2))
    def energy(opening):
        nodal=extrinsic_response(opening,opening,p.strength,p.toughness)
        faces=opening[:-1,None]*shape[None,:,0]+opening[1:,None]*shape[None,:,1]
        response=extrinsic_response(faces,faces,p.strength,p.toughness)
        face=float((length*p.config.width_mm)@((response['stored']+response['dissipated'])@weights/2))
        lumped=float(area@(nodal['stored']+nodal['dissipated']))
        return dict(nodal_total_potential_n_mm=lumped,face_p1_total_potential_n_mm=face,ratio=face/lumped if lumped else None)
    controls=[]
    for index in (len(y)-3,len(y)-1):
        for factor in (1.,2.,10.):
            opening=np.zeros_like(y);opening[index]=factor*2*p.toughness/p.strength
            removed=opening>0;support=p.config.width_mm*sum(length[removed[:-1]|removed[1:]])
            controls.append(dict(seam_index=index,opening_over_delta_f=factor,tributary_area_mm2=float(area[index]),
                p1_support_area_mm2=float(support),finite_crack_work_n_mm=float(p.toughness*area[index]),**energy(opening)))
    opening=2*(after[p.seam,0]-p.config.drop_s_mm)
    failed=v.previous_history>=2*p.toughness/p.strength
    connected=0.
    for i in range(len(length)-1,-1,-1):
        if failed[i] and failed[i+1]:connected+=length[i]
        else:break
    return dict(reference_y_mm=y.tolist(),nodal_area_mm2=area.tolist(),
        total_nodal_area_mm2=float(sum(area)),total_face_area_mm2=float(sum(length)*p.config.width_mm),
        candidate48=dict(tributary_mm2=float(area[48]),p1_support_mm2=float(p.config.width_mm*(length[47]+length[48]))),
        opening_shape_controls=controls,surface_connected_fully_failed_length_mm=connected,
        archived_max_opening_mm=float(max(opening)),
        interpretation='Nodal and P1 face potentials differ for nonuniform opening; total reference area agrees. Controls do not change the physical model or establish the cause of surface nonconnection.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--endpoint',choices=['production','diagnostic'],default='diagnostic');parser.add_argument('--seconds',type=float,default=480);add_thread_arguments(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+args.seconds
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p,base,v,before,after,scene=load_scene(args.case,args.endpoint)
        result=dict(scene=scene,runtime=runtime,area=area_evidence(p,v,after))
        write_json(args.output/'partial.json',result)
        if scene['endpoint']['residual_n']<=1e-4:
            production=virtual_work(v,before,after,deadline);independent=audit_work(v,before,after,deadline)
            result.update(production=production,independent=independent,
                work_agreement=abs(production['work_n_mm']-independent['work_n_mm'])<=production['empirical_uncertainty_n_mm']+independent['empirical_uncertainty_n_mm']+1e-7)
            result['decision']=interval_decision(base.stored(before)-v.stored(after),p.toughness*float(p.seam_area[v.virtual_removed].sum()),production['work_n_mm'],production['empirical_uncertainty_n_mm'],production['converged'])
        result['geometry_cover']=native_partitioned_chord(v,after,256)
        v.set_primary_increment(after,after-v.previous);lag=v.lag(after);_,g,h,j=v.evaluate(after,lag,projected=False);v.update_constraint_free(after,g)
        jac=v.coupled_friction_tangent(after,lag,h)[v.free][:,v.free];q=spsolve(jac,-g[v.free]);d=np.zeros(v.total);d[v.free]=q;d=d.reshape(after.shape);trial=after+d
        v.set_primary_increment(trial,trial-v.previous);_,gt,_,jt=v.evaluate(trial,v.lag(trial),tangent=False)
        result['full_newton']=dict(residual_n=float(np.sqrt(2)*np.linalg.norm(gt[v.free])),min_j=float(jt),
            original_linear_residual_n=float(np.linalg.norm(jac@q+g[v.free])),crossing_witnesses=ev_crossing_witnesses(v,before,trial))
        np.savez_compressed(args.output/'fields.npz',before=before,after=after,newton_trial=trial,direction=d,gradient=g)
    write_json(args.output/'validation.json',result)
    print(json.dumps(dict(output=str(args.output),work_agreement=result.get('work_agreement'),decision=result.get('decision'),newton_residual=result['full_newton']['residual_n'],crossing_count=len(result['full_newton']['crossing_witnesses']))))


if __name__=='__main__':main()
