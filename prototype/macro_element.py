"""P1-iso-P2/P1 macro discretization with condensed lumped pressure mass.

Displacements are continuous P1 on four subtriangles; pressure is continuous
P1 on the parent triangulation. Only the p^2/K term is row-sum lumped.
The default adds no volume stabilization. An explicit diagnostic can penalize
unresolved volume fluctuations at shear scale; the homogeneous law is retained.
"""
import numpy as np
from scipy.sparse import coo_matrix
from .model_config import FEMMesh


def solve_pressure_blocks(local, metric_local, pressure, shift, rhs):
    """Solve the condensed PTC equation without explicitly filling its Schur block.

    Both exact tangent and metric contain B.T M_p^-1 B. Eliminating the
    auxiliary pressure below gives L + shift*L_metric + (1+shift)*B.T M_p^-1 B.
    Friction may make L nonsymmetric; no symmetric-indefinite solver is assumed.
    """
    from scipy.sparse import bmat, diags
    from scipy.sparse.linalg import splu
    B, mass = pressure
    if not np.isfinite(shift) or shift < 0 or np.any(mass <= 0):
        raise ValueError('Invalid pressure mass or PTC shift')
    L = local+shift*metric_local
    system = bmat([[L, B.T], [B, -diags(mass/(1+shift))]], format='csc')
    system.eliminate_zeros()
    factor = splu(system)
    answer = factor.solve(np.r_[rhs, np.zeros(len(mass))])
    du = answer[:len(rhs)]
    residual = L@du+(1+shift)*(B.T@((B@du)/mass))-rhs
    error = float(np.linalg.norm(residual))
    limit = 1e-8*max(1., float(np.linalg.norm(rhs)))
    if not np.isfinite(error) or error > limit:
        raise np.linalg.LinAlgError(f'Mixed linear residual {error:g} exceeds {limit:g}')
    return du, dict(method='sparse_pressure_blocks', matrix_nnz=system.nnz,
                    factor_nnz=factor.L.nnz+factor.U.nnz,
                    condensed_residual_n=error, residual_limit_n=limit)


def refine_macro_mesh(base,boundary):
    pairs=np.sort(np.concatenate((base.triangles[:,[0,1]],base.triangles[:,[1,2]],base.triangles[:,[2,0]])),axis=1)
    edges,inverse=np.unique(pairs,axis=0,return_inverse=True)
    mid=(inverse+base.node_count).reshape(3,-1).T
    a,b,c=base.triangles.T;ab,bc,ca=mid.T
    tri=np.stack((np.column_stack((a,ab,ca)),np.column_stack((ab,b,bc)),
                  np.column_stack((ca,bc,c)),np.column_stack((ab,bc,ca))),axis=1).reshape(-1,3)
    xy=np.vstack((base.reference_xy,base.reference_xy[edges].mean(axis=1)))
    sn=np.vstack((base.local_sn,base.local_sn[edges].mean(axis=1)))
    lookup={tuple(edge):i+base.node_count for i,edge in enumerate(edges)}
    mids=np.array([lookup[tuple(sorted(edge))] for edge in boundary])
    fine_edges=np.stack((np.column_stack((boundary[:,0],mids)),np.column_stack((mids,boundary[:,1]))),axis=1).reshape(-1,2)
    mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
    mesh.macro_triangles=base.triangles.copy()
    mesh.macro_node_count=base.node_count
    mesh.macro_boundary_edges=boundary.copy()
    mesh.macro_parents=np.repeat(np.arange(len(base.triangles)),4)
    return mesh,fine_edges


def pressure_incidence(mesh,volumes):
    parents=mesh.macro_triangles[mesh.macro_parents]
    x=mesh.reference_xy[parents]
    dm=np.stack((x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=2)
    centroid=mesh.reference_xy[mesh.triangles].mean(axis=1)
    rs=np.einsum('eij,ej->ei',np.linalg.inv(dm),centroid-x[:,0])
    shape=np.column_stack((1-rs.sum(axis=1),rs))
    if np.min(shape)<-1e-9:
        raise ValueError('Displacement subtriangle left its pressure parent')
    return coo_matrix(((volumes[:,None]*shape).ravel(),
        (parents.ravel(),np.repeat(np.arange(len(volumes)),3))),
        shape=(mesh.macro_node_count,len(volumes))).tocsr()


def boundary_coordinates(mesh,points):
    from scipy.spatial import cKDTree
    parent=mesh.reference_xy[mesh.macro_boundary_edges]
    _,near=cKDTree(parent.mean(axis=1)).query(points,k=min(8,len(parent)))
    near=np.asarray(near).reshape(len(points),-1)
    a,b=parent[near,0],parent[near,1];v=b-a
    fraction=np.clip(np.sum((points[:,None]-a)*v,axis=2)/np.sum(v*v,axis=2),0,1)
    error=np.sum((points[:,None]-a-fraction[:,:,None]*v)**2,axis=2)
    choice=error.argmin(axis=1);index=np.arange(len(points))
    if np.max(error[index,choice])>1e-16:raise ValueError('Cannot map refined boundary to multiplier parent')
    return near[index,choice],fraction[index,choice]


def checkpoint(mesh):
    if hasattr(mesh,'elements'):
        return {name:getattr(mesh,name) for name in ('elements','pressure_node_count','boundary_midpoints','plot_triangles','macro_boundary_edges')}
    if not hasattr(mesh,'macro_triangles'):return {}
    return {name:getattr(mesh,name) for name in ('macro_triangles','macro_node_count','macro_boundary_edges','macro_parents')}


def restore(mesh,archive):
    if 'elements' in archive:
        for name in ('elements','pressure_node_count','boundary_midpoints','plot_triangles','macro_boundary_edges'):
            setattr(mesh,name,archive[name].copy() if name!='pressure_node_count' else int(archive[name]))
    if 'macro_triangles' in archive:
        for name in ('macro_triangles','macro_node_count','macro_boundary_edges','macro_parents'):
            setattr(mesh,name,archive[name].copy() if name!='macro_node_count' else int(archive[name]))
