"""Finite-strain 2D MINI (P1 + cubic bubble)/continuous P1 pressure.

u(X) = sum_i lambda_i u_i + 27 lambda_1 lambda_2 lambda_3 b_e.
The last mesh rows are INTERNAL COEFFICIENT SLOTS with zero reference values,
not material nodes/centroids. Boundary/contact connectivity contains vertices
only. The packed layout retains exact checkpoint/friction rollback semantics.
"""
from dataclasses import replace
import numpy as np
from .ipc_p2 import triangle_rule, VERTICES, InteriorP2
from .material import response
from .quadratic import quadratic_minimum


def refine_interior(base, center_sn, radius, levels=1):
    """Conforming edge refinement with the original boundary trace preserved.

    Only internal edges within the reference-radius window are bisected.
    Neighbouring triangles share midpoint IDs; green transition triangles avoid
    hanging nodes. This isolates interior accuracy from IPC contact sampling.
    It is not a complete contact-mesh convergence study.
    """
    if levels not in (0, 1, 2) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Interior diagnostic requires levels 0..2 and positive radius')
    mesh = base
    for _ in range(levels):
        tri = mesh.triangles
        edges = np.sort(np.vstack((tri[:, [0,1]], tri[:, [1,2]], tri[:, [2,0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        mid_sn = mesh.local_sn[edges].mean(axis=1)
        chosen = edges[(counts == 2) & (np.linalg.norm(mid_sn-center_sn, axis=1) <= radius)]
        ids = {tuple(edge): mesh.node_count+i for i,edge in enumerate(chosen)}
        children = []
        for a,b,c in tri:
            cuts = [(i,j,ids[tuple(sorted((i,j)))]) for i,j in ((a,b),(b,c),(c,a))
                    if tuple(sorted((i,j))) in ids]
            if len(cuts) == 0:
                children.append([a,b,c])
            elif len(cuts) == 1:
                i,j,m = cuts[0]; k = next(v for v in (a,b,c) if v not in (i,j))
                children.extend(([i,m,k],[m,j,k]))
            elif len(cuts) == 2:
                shared = next(v for v in cuts[0][:2] if v in cuts[1][:2])
                i = shared; j = next(v for v in cuts[0][:2] if v != i)
                k = next(v for v in cuts[1][:2] if v != i)
                m,n = cuts[0][2],cuts[1][2]
                children.extend(([i,m,n],[j,k,n],[j,n,m]))
            else:
                ab,bc,ca = [ids[tuple(sorted(e))] for e in ((a,b),(b,c),(c,a))]
                children.extend(([a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]))
        xy = np.vstack((mesh.reference_xy,mesh.reference_xy[chosen].mean(axis=1)))
        children = np.array(children,dtype=int)
        pts = xy[children]; v = pts[:,1]-pts[:,0]; w = pts[:,2]-pts[:,0]
        negative = v[:,0]*w[:,1]-v[:,1]*w[:,0] < 0
        children[negative] = children[negative][:,[0,2,1]]
        mesh = replace(mesh, reference_xy=xy,
            local_sn=np.vstack((mesh.local_sn,mesh.local_sn[chosen].mean(axis=1))),
            triangles=children,side_of_node=np.r_[mesh.side_of_node,np.zeros(len(chosen),int)])
    return mesh


def coarsen_interior(base, coarse, center_sn, keep_radius):
    """Retriangulate coarse interior with the exact original exterior vertices.

    Retain fine vertices in a reference tip patch, including the fixed bottom.
    This changes interior discretization and needs force/field qualification.
    No solution or friction history is transferred between these meshes.
    """
    from scipy.spatial import Delaunay
    from .model_config import FEMMesh
    if not np.isfinite(keep_radius) or keep_radius <= 0:
        raise ValueError('Retained tip radius must be positive')
    def boundary_nodes(mesh):
        tri = mesh.triangles
        edges, counts = np.unique(np.sort(np.vstack((tri[:, [0,1]], tri[:, [1,2]],
            tri[:, [2,0]])), axis=1), axis=0, return_counts=True)
        return np.unique(edges[counts == 1])
    original_boundary = boundary_nodes(base)
    candidates = np.setdiff1d(np.arange(coarse.node_count), boundary_nodes(coarse))
    distance = np.linalg.norm(base.local_sn-center_sn, axis=1)
    retained = np.union1d(original_boundary, np.flatnonzero(distance <= keep_radius))
    far = candidates[np.linalg.norm(coarse.local_sn[candidates]-center_sn, axis=1) > keep_radius]
    sn = np.vstack((base.local_sn[retained], coarse.local_sn[far]))
    xy = np.vstack((base.reference_xy[retained], coarse.reference_xy[far]))
    sn, unique = np.unique(sn, axis=0, return_index=True)
    xy = xy[unique]
    tri = Delaunay(sn).simplices.copy()
    # Qhull may choose the other diagonal of a cocircular grid cell after
    # distant vertices are removed. Restore original fine-patch connectivity
    # by convex-quad flips; do not perturb geometry or interpolate any field.
    lookup = {tuple(v):i for i,v in enumerate(sn)}
    protected = base.triangles[np.all(distance[base.triangles] <= keep_radius,axis=1)]
    wanted = np.array([[lookup[tuple(base.local_sn[v])] for v in cell] for cell in protected],dtype=int).reshape(-1,3)
    edges = {}; vertex_cells = [set() for _ in sn]
    def update(index, cell, add):
        for v in cell:
            if add:vertex_cells[v].add(index)
            else:vertex_cells[v].remove(index)
        for a,b in ((cell[0],cell[1]),(cell[1],cell[2]),(cell[2],cell[0])):
            key=tuple(sorted((a,b)))
            if add:edges.setdefault(key,set()).add(index)
            else:
                edges[key].remove(index)
                if not edges[key]:del edges[key]
    for i,cell in enumerate(tri):update(i,cell,True)
    desired = sorted({tuple(sorted((a,b))) for cell in wanted
                      for a,b in ((cell[0],cell[1]),(cell[1],cell[2]),(cell[2],cell[0]))})
    from itertools import combinations
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for a,b in desired:
        if (a,b) in edges:continue
        na={v for i in vertex_cells[a] for v in tri[i]}
        nb={v for i in vertex_cells[b] for v in tri[i]}
        for c,d in combinations(sorted(na & nb),2):
            pair=edges.get((c,d),set())
            if len(pair)!=2:continue
            i,j=sorted(pair)
            if set(tri[i]) | set(tri[j]) != {a,b,c,d}:continue
            ab=sn[b]-sn[a];cd=sn[d]-sn[c]
            if (cross(ab,sn[c]-sn[a])*cross(ab,sn[d]-sn[a])>=0 or
                    cross(cd,sn[a]-sn[c])*cross(cd,sn[b]-sn[c])>=0):continue
            update(i,tri[i],False);update(j,tri[j],False)
            tri[i]=[a,b,c];tri[j]=[b,a,d]
            update(i,tri[i],True);update(j,tri[j],True)
            break
        if (a,b) not in edges:
            raise ValueError('Unable to preserve original fine-patch edge during coarsening')
    actual={tuple(sorted(cell)) for cell in tri}
    if any(tuple(sorted(cell)) not in actual for cell in wanted):
        raise ValueError('Interior coarsening changed a protected fine-patch triangle')
    signed = np.linalg.det(np.stack((xy[tri[:,1]]-xy[tri[:,0]], xy[tri[:,2]]-xy[tri[:,0]]),axis=2))
    tri[signed < 0] = tri[signed < 0][:, [0,2,1]]
    mesh = FEMMesh(xy, sn, tri, np.empty((0,2),int), np.empty(0),
        np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(sn),int))
    if not np.array_equal(np.unique(mesh.local_sn[boundary_nodes(mesh)],axis=0),
                          np.unique(base.local_sn[original_boundary],axis=0)):
        raise ValueError('Interior coarsening changed the exterior trace')
    return mesh


class MiniElement:
    def __init__(self, base, config, order=4):
        self.config, self.order = config, order
        self.base_nodes = base.node_count
        ne = len(base.triangles)
        self.elements = np.column_stack((base.triangles, self.base_nodes+np.arange(ne)))
        self.mesh = replace(base,
            reference_xy=np.vstack((base.reference_xy, np.zeros((ne, 2)))),
            local_sn=np.vstack((base.local_sn, np.zeros((ne, 2)))),
            side_of_node=np.r_[base.side_of_node, np.zeros(ne, int)])
        self.area, self.linear_grad, _ = base.bulk_data
        self.points, self.weights = triangle_rule(order)
        self.grad = self.gradients(self.points)
        self.vertex_grad = self.gradients(VERTICES)
        self.dofs = (2*self.elements[:, :, None]+np.arange(2)).reshape(-1, 8)
        self.bubble_dofs = self.dofs[:, -2:]

    def gradients(self, l):
        bubble = 27*np.einsum('qa,eaJ->eqJ', l[:, [1, 2, 0]]*l[:, [2, 0, 1]], self.linear_grad)
        linear = np.broadcast_to(self.linear_grad[:, None], (len(self.area), len(l), 3, 2))
        return np.concatenate((linear, bubble[:, :, None]), axis=2)

    def deformation(self, u, grad=None):
        return np.eye(2)+np.einsum('eai,eqaJ->eqiJ', np.asarray(u).reshape(-1, 2)[self.elements],
                                  self.grad if grad is None else grad)

    def minimum_j(self, u):
        # F = F_linear + b (grad bubble)^T is a rank-one update. In 2D
        # det(F) is QUADRATIC in space, despite the cubic displacement.
        return float(quadratic_minimum(np.linalg.det(self.deformation(u, self.vertex_grad))).min())

    # The same space-time quadratic determinant certificate applies to MINI.
    safe_step = InteriorP2.safe_step

    def bulk(self, u, *, with_tangent=True):
        minimum = self.minimum_j(u)
        if not np.isfinite(minimum) or minimum <= 0:
            raise FloatingPointError('MINI Jacobian nonpositive inside reference triangle')
        F = self.deformation(u)
        q = response(F.reshape(-1, 2, 2), self.config, tangent=with_tangent, diagnostics=False)
        nq = len(self.weights)
        volumes = self.area*self.config.width_mm
        r = np.einsum('e,q,eqiJ,eqaJ->eai', volumes, self.weights,
                      q['P'].reshape(-1, nq, 2, 2), self.grad, optimize=True).reshape(-1, 8)
        k = None
        if with_tangent:
            C = q['tangent'].reshape(-1, nq, 2, 2, 2, 2)
            k = np.empty((len(volumes), 8, 8))
            for start in range(0, len(volumes), 2048):
                stop = min(start+2048, len(volumes))
                grad = self.grad[start:stop]
                z = np.einsum('eqiJkL,eqbL->eqiJbk', C[start:stop], grad, optimize=True)
                local = np.einsum('eqaJ,eqiJbk,q->eaibk', grad, z, self.weights, optimize=True)
                k[start:stop] = (local*volumes[start:stop, None, None, None, None]).reshape(-1, 8, 8)
        energy = np.einsum('e,q,eq->', volumes, self.weights, q['energy'].reshape(-1, nq))
        return r, k, self.dofs, float(energy), minimum

    def checkpoint_data(self):
        return dict(mini_elements=self.elements, mini_base_nodes=self.base_nodes,
                    mini_quadrature_order=self.order,
                    mini_layout='vertices; internal bubble coefficients; blade vertices')
