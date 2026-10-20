"""Conforming P2 interior enrichment with the ORIGINAL piecewise-linear boundary.

Only interior edge midpoints are independent. On exterior edges the midpoint
displacement is the endpoint average, eliminated exactly in the shape gradients.
Thus IPC geometry, quadrature/area weights and friction history are unchanged.
This is a constrained P2 displacement space, not a mixed/locking-free element.
"""
from dataclasses import replace
import numpy as np
from .material import response
from .quadratic import quadratic_minimum


VERTICES = np.vstack((np.eye(3), [[.5,.5,0], [0,.5,.5], [.5,0,.5]]))


def triangle_rule(order):
    """Positive tensor Gauss rule under the Duffy map; weights sum to one."""
    if not isinstance(order, int) or not 3 <= order <= 10:
        raise ValueError('Triangle integration order must be an integer in [3,10]')
    z, w = np.polynomial.legendre.leggauss(order)
    z, w = (z+1)/2, w/2
    r, t = np.meshgrid(z, z, indexing='ij')
    s = (1-r)*t
    return np.column_stack(((1-r-s).ravel(), r.ravel(), s.ravel())), (2*w[:,None]*w[None,:]*(1-r)).ravel()


def composite_triangle_rule(order, levels=0):
    """Positive Gauss integration on 4**levels reference subtriangles.

    This refines integration only. Displacements and geometric positivity still
    use the original P2 map; every new point requires its own material history.
    """
    if not isinstance(levels,int) or not 0 <= levels <= 2:
        raise ValueError('Integration subdivision levels must be an integer in [0,2]')
    points,weights=triangle_rule(order)
    if levels==0:return points,weights
    cells=np.eye(3)[None]
    for _ in range(levels):
        a,b,c=cells.transpose(1,0,2);ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
        cells=np.stack((np.stack((a,ab,ca),axis=1),np.stack((ab,b,bc),axis=1),
                        np.stack((ca,bc,c),axis=1),np.stack((ab,bc,ca),axis=1)),axis=1).reshape(-1,3,3)
    return np.einsum('qa,eai->eqi',points,cells).reshape(-1,3),np.tile(weights/len(cells),len(cells))


class InteriorP2:
    def __init__(self, base, config, order=4, *, enrichment_radius=None, quadrature_refinement=0):
        self.config, self.order = config, order
        self.quadrature_refinement=quadrature_refinement
        tri = base.triangles
        edges, inverse, counts = np.unique(np.sort(tri[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),
                                           axis=0, return_inverse=True, return_counts=True)
        interior = counts == 2
        if enrichment_radius is not None:
            if not np.isfinite(enrichment_radius) or enrichment_radius <= 0:
                raise ValueError('Positive finite reference enrichment radius required')
            center = np.array([config.drop_s_mm, config.height_mm])
            interior &= np.linalg.norm(base.local_sn[edges].mean(axis=1)-center, axis=1) <= enrichment_radius
        if np.any(counts > 2):
            raise ValueError('Nonmanifold mesh')
        index = np.full(len(edges), -1, dtype=int)
        index[interior] = base.node_count + np.arange(np.count_nonzero(interior))
        mids = index[inverse].reshape(-1,3)
        self.boundary_slots = mids < 0
        # A zero-gradient boundary slot can share the first corner's DOF.
        self.elements = np.column_stack((tri, np.where(mids < 0, tri, mids)))
        self.base_nodes = base.node_count
        self.interior_edges = edges[interior]
        self.mesh = replace(base,
            reference_xy=np.vstack((base.reference_xy, base.reference_xy[self.interior_edges].mean(axis=1))),
            local_sn=np.vstack((base.local_sn, base.local_sn[self.interior_edges].mean(axis=1))),
            side_of_node=np.r_[base.side_of_node, np.zeros(len(self.interior_edges), int)])
        self.area, self.linear_grad, _ = base.bulk_data
        self.points, self.weights = composite_triangle_rule(order,quadrature_refinement)
        self.grad = self.gradients(self.points)
        self.vertex_grad = self.gradients(VERTICES)
        self.dofs = (2*self.elements[:,:,None]+np.arange(2)).reshape(-1,12)

    def gradients(self, l):
        g = self.linear_grad
        out = np.concatenate(((4*l[None,:,:,None]-1)*g[:,None],
            4*(l[None,:,[1,2,0],None]*g[:,None,[0,1,2]] +
               l[None,:,[0,1,2],None]*g[:,None,[1,2,0]])), axis=2)
        for k, (a,b) in enumerate(((0,1),(1,2),(2,0))):
            mask = self.boundary_slots[:,k]
            out[mask,:,a] += .5*out[mask,:,3+k]
            out[mask,:,b] += .5*out[mask,:,3+k]
            out[mask,:,3+k] = 0.
        return out

    def deformation(self, u, grad=None):
        return np.eye(2)+np.einsum('eai,eqaJ->eqiJ', np.asarray(u).reshape(-1,2)[self.elements],
                                  self.grad if grad is None else grad)

    def minimum_j(self, u):
        return float(quadratic_minimum(np.linalg.det(self.deformation(u,self.vertex_grad))).min())

    def bulk(self, u, *, with_tangent=True):
        minimum = self.minimum_j(u)
        if not np.isfinite(minimum) or minimum <= 0:
            raise FloatingPointError('P2 Jacobian nonpositive inside reference triangle')
        F = self.deformation(u)
        q = response(F.reshape(-1,2,2), self.config, tangent=with_tangent, diagnostics=False)
        volumes = self.area*self.config.width_mm
        nq = len(self.weights)
        r = np.einsum('e,q,eqiJ,eqaJ->eai', volumes, self.weights,
                      q['P'].reshape(-1,nq,2,2), self.grad, optimize=True).reshape(-1,12)
        k = None
        if with_tangent:
            C = q['tangent'].reshape(-1,nq,2,2,2,2)
            k = np.empty((len(volumes),12,12))
            # The default einsum planner retains a costly three-operand final
            # contraction. Contract C with one gradient first, then integrate
            # against the other. Blocks bound the intermediate to about 25 MB
            # at the default 16-point rule; the quadrature and tangent are exact
            # counterparts of the original contraction, up to roundoff.
            for start in range(0,len(volumes),4096):
                stop = min(start+4096,len(volumes))
                grad = self.grad[start:stop]
                z = np.einsum('eqiJkL,eqbL->eqiJbk',C[start:stop],grad,optimize=True)
                local = np.einsum('eqaJ,eqiJbk,q->eaibk',grad,z,self.weights,optimize=True)
                k[start:stop] = (local*volumes[start:stop,None,None,None,None]).reshape(-1,12,12)
        energy = np.einsum('e,q,eq->',volumes,self.weights,q['energy'].reshape(-1,nq))
        return r,k,self.dofs,float(energy),minimum

    def safe_step(self, u, direction):
        """Positive J over the WHOLE triangle and WHOLE straight time segment.

        J is quadratic in space and time. Its three temporal Bernstein
        coefficients are spatial quadratics. Positive exact spatial minima of
        all three certify the entire path, including hidden intermediate flips.
        This is sufficient in exact arithmetic, never just a Gauss-point
        check. The floating-point implementation is NOT an interval proof;
        PTC also checks the represented endpoint and rejects vanishing steps.
        """
        F = self.deformation(u,self.vertex_grad)
        D = self.deformation(direction,self.vertex_grad)-np.eye(2)
        c = np.linalg.det(F)
        if quadratic_minimum(c).min() <= 0:
            raise FloatingPointError('Invalid initial P2 map')
        b = F[:,:,0,0]*D[:,:,1,1]+D[:,:,0,0]*F[:,:,1,1]-F[:,:,0,1]*D[:,:,1,0]-D[:,:,0,1]*F[:,:,1,0]
        a = np.linalg.det(D)
        alpha = 1.
        for _ in range(60):
            if min(quadratic_minimum(c+.5*alpha*b).min(),
                   quadratic_minimum(c+alpha*b+alpha*alpha*a).min()) > 0:
                return alpha
            alpha *= .5
        raise FloatingPointError('No certified positive P2 step')

    def checkpoint_data(self):
        data=dict(p2_elements=self.elements, p2_boundary_slots=self.boundary_slots,
                  p2_base_nodes=self.base_nodes, p2_quadrature_order=self.order)
        if self.quadrature_refinement:
            data.update(p2_quadrature_refinement=self.quadrature_refinement,
                        p2_quadrature_points=self.points,p2_quadrature_weights=self.weights)
        return data
