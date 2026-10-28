"""Zero-gap, exactly tied triangle topology for an extrinsic research branch.

No load solver or fracture criterion is supplied here. In particular coincident
new crack faces cannot be passed to the existing positive-distance IPC solver.
Local vertex fans are split only across released edges; unreleased material is
exactly conforming, not connected by a large penalty. Elements never remesh.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Partition:
    reference: np.ndarray
    triangles: np.ndarray
    original_vertex_ids: np.ndarray
    corner_to_node: np.ndarray
    released_edges: tuple

    def prolongation(self, dimensions=2):
        """Map connected DOFs to the element-corner representation."""
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError('Positive integer dimensions required')
        ids = (dimensions*self.corner_to_node[:, None]+np.arange(dimensions)).ravel()
        return coo_matrix((np.ones(len(ids)), (np.arange(len(ids)), ids)),
                          shape=(len(ids), dimensions*len(self.reference))).tocsr()


def partition(reference, triangles, released_edges=()):
    x = np.asarray(reference, float)
    raw = np.asarray(triangles)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('Finite 2D vertices and integer triangles required')
    tri = raw.astype(int, copy=True)
    if not len(tri) or np.any(tri < 0) or np.any(tri >= len(x)):
        raise ValueError('Invalid element vertices')
    area2 = np.linalg.det(np.stack((x[tri[:, 1]]-x[tri[:, 0]], x[tri[:, 2]]-x[tri[:, 0]]), axis=2))
    if np.any(area2 <= 0.):
        raise ValueError('Positive oriented nondegenerate triangles required')
    owners = {}
    for e, nodes in enumerate(tri):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(nodes[i]), int(nodes[j]))))
            owners.setdefault(key, []).append({int(nodes[i]): 3*e+i, int(nodes[j]): 3*e+j})
    if any(len(v) > 2 for v in owners.values()):
        raise ValueError('Nonmanifold edge')
    released = set()
    for edge in released_edges:
        if len(edge) != 2 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in edge):
            raise ValueError('Integer release edge required')
        key = tuple(sorted(map(int, edge)))
        if key not in owners or len(owners[key]) != 2:
            raise ValueError('Only existing internal edges can be released')
        released.add(key)
    parent = np.arange(tri.size)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for edge, adjacent in owners.items():
        if len(adjacent) == 2 and edge not in released:
            for vertex in edge:
                a, b = root(adjacent[0][vertex]), root(adjacent[1][vertex])
                parent[max(a, b)] = min(a, b)
    roots, inverse = np.unique([root(i) for i in range(tri.size)], return_inverse=True)
    original = tri.ravel()[roots]
    return Partition(x[original].copy(), inverse.reshape(-1, 3), original,
                     inverse, tuple(sorted(released)))


def transfer_release(old, new, positions, velocities):
    """Release-only transfer with identical corner positions and velocities.

    Bulk element internal variables keep their indices and must be copied
    unchanged by a future host. No history interpolation or nodal perturbation.
    """
    if not set(old.released_edges) <= set(new.released_edges):
        raise ValueError('Healing/merging is not a release transfer')
    if (old.corner_to_node.shape != new.corner_to_node.shape
            or not np.array_equal(old.reference[old.triangles], new.reference[new.triangles])):
        raise ValueError('Element reference identity changed')
    fields = []
    for field in (positions, velocities):
        a = np.asarray(field, float)
        if a.shape != old.reference.shape or not np.isfinite(a).all():
            raise ValueError('Finite nodal field required')
        corners = a[old.corner_to_node]
        out = np.empty_like(new.reference)
        for node in range(len(out)):
            values = corners[new.corner_to_node == node]
            if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                raise ValueError('New topology merges independently moving nodes')
            out[node] = values[0]
        fields.append(out)
    return tuple(fields)


def lumped_mass(topology, density, width):
    if not np.isfinite([density, width]).all() or min(density, width) <= 0:
        raise ValueError('Positive density and width required')
    v = topology.reference[topology.triangles]
    mass = .5*np.linalg.det(np.stack((v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]), axis=2))*density*width
    return np.bincount(topology.triangles.ravel(), weights=np.repeat(mass/3., 3),
                       minlength=len(topology.reference))
