"""Opt-in continuous P1 multipliers on the original linear contact trace.

The positive trace shapes interpolate both pressure and shear. Their reference
area moments define nodal weak gap/slip constraints. On a fully active planar
P1 patch these involve a positive-definite consistent mass matrix, whereas
edge-P0 averaging cannot detect alternating signed nodal gaps. This is a new
contact discretization, not proof of pointwise contact or general inf-sup
stability. Original whole-edge geometry audits remain mandatory.
"""
import numpy as np
from scipy.sparse import diags
from .augmented_blade import AugmentedBladeProblem
from .al_globalization import WarmGlobalizedAugmentedBladeProblem


class ContinuousContactMixin:
    def _set_quadrature(self,edges,fractions,weights):
        super()._set_quadrature(edges,fractions,weights)
        nodes=np.unique(self.body_edges[self.qedges])
        # contact_edges is the base solver's algebraic-label interface. In this
        # explicit variant labels are reference NODE IDs, not physical edges.
        self.contact_edges=nodes
        self.edgeQ=self.Q[:,nodes].tocsr()
        self.edge_area=np.asarray(self.edgeQ.T@self.qweights).ravel()
        if np.any(self.edge_area<=0):raise ValueError('Nonpositive nodal contact area')
        self.average=diags(1/self.edge_area)@self.edgeQ.T@diags(self.qweights)

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),al_multiplier_basis=np.array('continuous-P1-reference-v1'),
                    al_multiplier_nodes=self.contact_edges)

    def audit_state(self,state,record):
        if str(state.get('al_multiplier_basis',''))!='continuous-P1-reference-v1':
            raise ValueError('Wrong contact multiplier basis')
        nodes=np.unique(self.body_edges[state['al_edges']])
        np.testing.assert_array_equal(state['al_multiplier_nodes'],nodes)
        return super().audit_state(state,record)


class ContinuousAugmentedBladeProblem(ContinuousContactMixin,AugmentedBladeProblem):
    """Original, unbatched assembly for independent P1-multiplier audits."""


class FastContinuousAugmentedBladeProblem(ContinuousContactMixin,WarmGlobalizedAugmentedBladeProblem):
    """P1 multiplier residual with the qualified optional algebraic backend."""


def enable_continuous_contact(problem,*,fast):
    if len(problem.multipliers):raise ValueError('Select basis before preparing any contact history')
    problem.__class__=FastContinuousAugmentedBladeProblem if fast else ContinuousAugmentedBladeProblem
    problem.contact_formulation='continuous-P1-AL-rigid-blade-v1'
    problem._set_quadrature(problem.qedges,problem.qfraction,problem.qweights)
    return problem
