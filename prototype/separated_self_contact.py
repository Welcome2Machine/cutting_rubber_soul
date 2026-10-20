"""Explicit AL diagnostic valid ONLY before nonadjacent body contact.

Connected edge endpoints are not opposing physical surfaces. Improved-max
can repel them when their distance falls below dhat. This candidate removes
body barrier/friction only while all detected stencils are those endpoints.
Any other stencil rejects the trial. ALL original body CCD and intersection
checks remain inherited. This is not a general replacement self-contact law.
"""
import numpy as np
from scipy.sparse import csc_matrix
from .ipc_reference import LoadStepFailure

POLICY='separated-only-v1'


class SeparatedContactDomainError(LoadStepFailure):
    """A trial leaves this diagnostic's domain; no claim of actual intersection.

    A globalization line search may shorten this trial. An inadmissible current
    iterate must still stop. Other material/geometry errors are not this class.
    """


def certify_separated(ipc,mesh,x,dhat,adjacent):
    c=ipc.NormalCollisions();c.use_area_weighting=True
    c.collision_set_type=ipc.NormalCollisions.IMPROVED_MAX_APPROX
    c.build(mesh,x,dhat)
    other=sum(tuple(sorted((int(z.vertex0_id),int(z.vertex1_id)))) not in adjacent for z in c.vv_collisions)
    if other or c.ev_collisions or c.ee_collisions or c.fv_collisions:
        raise SeparatedContactDomainError('Separated-body diagnostic reached nonadjacent proximity; general contact law required outside this domain')
    return len(c.vv_collisions)


class SeparatedSelfContactMixin:
    def _self_terms(self,x,tangent):
        count=certify_separated(self.ipc,self.self_mesh,x[self.body_surface],self.dhat,self.separated_adjacency)
        self.last_separated_check={'adjacent_vv_excluded':count,'nonadjacent_contact':False}
        return 0.,np.zeros(len(self.self_dofs)),csc_matrix((self.total,self.total)) if tangent else None,0.

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),body_self_contact_policy=np.array(POLICY))

    def audit_state(self,state,record):
        if str(state.get('body_self_contact_policy',''))!=POLICY:
            raise ValueError('Separated-body policy missing/mismatched in checkpoint')
        return super().audit_state(state,record)


def enable_separated_self_contact(p):
    if isinstance(p,SeparatedSelfContactMixin):return p
    p.__class__=type('Separated'+type(p).__name__,(SeparatedSelfContactMixin,type(p)),{})
    p.separated_adjacency={tuple(sorted(e)) for e in np.asarray(p.self_mesh.edges)}
    p.contact_formulation+=';'+POLICY
    return p
