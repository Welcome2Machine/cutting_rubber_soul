"""Explicit pure-precrack IPC controls, with an optional prescribed V-notch.

The positive geometric control has a NEW stress-free reference shape and true
mass integrated on that shape. It is neither an opening imposed on actual C1,
nor a natural-fracture prediction. Closed C1 uses halfwidth zero unchanged.
The friction/time law remains the existing IPC candidate, not the planar hard
contact reference. No contact-model transfer is hidden in this entry.
"""
import copy,json
from pathlib import Path
import numpy as np
from . import run_endpoint_face_route as endpoint
from . import run_fracture_route as base
from .frozen_scene import digest
from mechanics.discrete_gradient import cohesive_discrete_response


class PurePrecrackProblem(endpoint.EndpointFaceCutProblem):
    def cohesive_algorithmic(self, x):
        u0=self.opening_at_points(self.previous);u1=self.opening_at_points(x)
        # A fully failed initial face has identically zero traction/storage.
        # Its frozen marker is delta_f, not a record of later maximum gap.
        # Satisfy the generic kernel's reload-history domain without changing
        # any saved history, damage, or fracture charge.
        effective=self.previous_history.copy()
        effective[self.precracked_faces]=np.maximum(effective[self.precracked_faces],u0[self.precracked_faces])
        return cohesive_discrete_response(u0,u1,effective,self.strength,self.toughness)

    def connected_length(self, failed):
        # Entry classification uses vertical depth; the notch's longer arc
        # length is already represented by its physical integration areas.
        length=0.
        for i in range(len(failed)-1,-1,-1):
            if not failed[i]:break
            length+=self.rest[self.seam[i+1],1]-self.rest[self.seam[i],1]
        return float(length)


def problem(config):
    c=copy.deepcopy(config);gap=c.pop('initial_slit_halfwidth_mm',None)
    if not isinstance(gap,(float,int)) or not np.isfinite(gap) or not 0<=gap<=.1:raise ValueError('Explicit geometric control halfwidth in [0,.1] mm required')
    if not c.get('pure_precrack') or c['precrack_mm']<=0:raise ValueError('Pure initial crack required')
    p=endpoint.problem(c)
    p.__class__=PurePrecrackProblem
    p.entry_reference_geometry='actual closed 0.5 mm precrack' if gap==0 else 'prescribed stress-free tapered notch; diagnostic geometry only'
    if gap:
        xy=p.mesh.reference_xy.copy();centre=p.config.drop_s_mm;tip=p.config.height_mm-c['precrack_mm']
        xy[:,0]+=gap*np.maximum(0.,(xy[:,1]-tip)/c['precrack_mm'])*np.maximum(0.,1-(xy[:,0]-centre)/.5)
        p.mesh.reference_xy=xy;p.mesh.local_sn=xy.copy();p.mesh.__dict__.pop('bulk_data',None)
        area,p.grad,p.dofs=p.mesh.bulk_data
        if np.any(area<=0):raise ValueError('Notch invalidates reference triangles')
        p.rest[:p.n]=xy;p.previous=p.rest.copy()
        p.mass=np.zeros(p.n);np.add.at(p.mass,p.mesh.triangles.ravel(),np.repeat(area*p.config.width_mm*p.density/3,3))
        p.mass_dof=np.r_[np.repeat(p.mass,2),np.zeros(p.total-2*p.n)]
        edge=p.rest[p.seam[1:]]-p.rest[p.seam[:-1]];length=np.linalg.norm(edge,axis=1)
        ratio=length/edge[:,1];p.rule['weights_mm2']*=ratio[:,None];p.rule['face_area_mm2']*=ratio
        p.rule['nodal_tributary_area_mm2'][:]=0.
        p.rule['nodal_tributary_area_mm2'][:-1]+=p.rule['face_area_mm2']/2
        p.rule['nodal_tributary_area_mm2'][1:]+=p.rule['face_area_mm2']/2
        p.initial_fracture_energy_n_mm=float(p.toughness*p.rule['face_area_mm2'][p.precracked_faces].sum())
        p.rebuild_contact();p.active_contact=np.zeros(p.n,bool);p.active_contact[p.seam[~p.released]]=True;p.set_free()
        p._accepted_predictor_history=[]
        if p.ipc.has_intersections(p.cmesh,p.rest[p.surface]):raise ValueError('Notch/blade initial intersections')
    p.initial_slit_halfwidth_mm=float(gap)
    return p


def sources():
    result=endpoint.sources();result['tools/run_precrack_entry.py']=digest(Path(__file__));return result


def finish(out,result):
    result.update(entry_control='pure precrack; no new face activation or damage',
        initial_slit_halfwidth_mm=result['config']['initial_slit_halfwidth_mm'],
        contact_model='unchanged native IPC barrier and endpoint friction, not hard-contact transfer',
        connected_length_measure='vertical reference depth; physical face area uses reference arc length',
        diagnostic_geometry=result['config']['initial_slit_halfwidth_mm']>0)
    endpoint.finish(out,result)


def main():
    base.problem=problem;base.sources=sources;base.finish=finish;base.main()
if __name__=='__main__':main()
