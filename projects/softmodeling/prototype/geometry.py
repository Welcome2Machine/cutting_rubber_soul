"""Intact P1 specimen mesh and rigid rounded-blade geometry (mm)."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from .model_config import FEMMesh, local_to_lab, _rotation_vectors


@dataclass
class GeometryOptions:
    tip_radius_mm: float = 0.02
    refinement_mm: float = 0.15
    surface_spacing_mm: float = 0.0025
    mesh_grading: str = 'legacy'
    contact_layer_depth_mm: float = 0.
    reference_layout: str | None = None
    size_gradient: float = .4

    def validate(self, config):
        for name in ('tip_radius_mm', 'refinement_mm', 'surface_spacing_mm'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.mesh_grading not in ('legacy', 'graded'):
            raise ValueError('Unknown mesh grading')
        if not np.isfinite(self.size_gradient) or self.size_gradient <= 0:
            raise ValueError('Positive mesh size gradient required')
        depth = self.contact_layer_depth_mm
        if not np.isfinite(depth) or not 0 <= depth < config.height_mm or (depth and self.mesh_grading != 'graded'):
            raise ValueError('Contact layer requires a graded mesh and depth inside the specimen')
        if self.tip_radius_mm >= config.blade_thickness_mm / 2:
            raise ValueError('Tip radius must be smaller than half blade thickness')


def reference_mesh_digest(sn, triangles):
    """Identify the original material coordinates and ordered, oriented connectivity."""
    return hashlib.sha256(np.asarray(sn, dtype='<f8').tobytes() +
                          np.asarray(triangles, dtype='<i8').tobytes()).hexdigest()


def relocate_reference_nodes(sn, triangles, layout):
    """Apply an interior-only material-coordinate layout to its exact parent topology."""
    if layout.get('schema') != 'intact-reference-layout/1' or layout.get('base_sha256') != reference_mesh_digest(sn, triangles):
        raise ValueError('Reference layout does not match the parent mesh')
    indices = np.asarray(layout['node_ids'])
    coordinates = np.asarray(layout['local_sn_mm'], dtype=float)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices)
            or indices.min() < 0 or indices.max() >= len(sn) or len(np.unique(indices)) != len(indices)
            or coordinates.shape != (len(indices), 2) or not np.isfinite(coordinates).all()):
        raise ValueError('Invalid material node IDs or coordinates in reference layout')
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.isin(indices, unique[counts == 1]).any():
        raise ValueError('Reference layout must preserve all boundary nodes')
    relocated = sn.copy()
    relocated[indices] = coordinates
    points = relocated[triangles]
    edge1 = points[:, 1] - points[:, 0]
    edge2 = points[:, 2] - points[:, 0]
    if np.any(edge1[:, 0]*edge2[:, 1] - edge1[:, 1]*edge2[:, 0] <= 0):
        raise ValueError('Reference layout would invert or collapse an element')
    return relocated


def optimize_reference_layout(sn, triangles, deformed_samples, movable, step_mm):
    """Raise the worst signed mean ratio on each movable vertex star across supplied P1 maps.

    This offline operation changes reference nodes only. The material history for a
    simulation on the resulting layout must be integrated afresh from rest.
    """
    from matplotlib.tri import Triangulation
    sn = np.asarray(sn, dtype=float)
    triangles = np.asarray(triangles)
    samples = np.concatenate((sn[None], np.asarray(deformed_samples)), axis=0)
    movable = np.asarray(movable, dtype=int)
    if (not len(movable) or not np.isfinite(step_mm) or step_mm <= 0 or not np.isfinite(samples).all()
            or len(np.unique(movable)) != len(movable) or movable.min() < 0 or movable.max() >= len(sn)):
        raise ValueError('Invalid reference layout design inputs')
    finder = Triangulation(*sn.T, triangles).get_trifinder()
    corners = sn[triangles]
    inverse = np.linalg.inv((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    moving_set = set(movable.tolist())
    stars = {i: [] for i in movable}
    for tri in triangles:
        for i, node in enumerate(tri):
            if node in moving_set:
                stars[node].append((tri[(i+1) % 3], tri[(i+2) % 3]))
    colors = {}
    for node in movable:
        used = {colors[j] for pair in stars[node] for j in pair if j in colors}
        colors[node] = next(c for c in range(len(used)+1) if c not in used)
    groups = []
    for color in range(max(colors.values())+1):
        ids = np.array([i for i in movable if colors[i] == color])
        degree = max(len(stars[i]) for i in ids)
        opposite = np.array([stars[i] + [stars[i][-1]]*(degree-len(stars[i])) for i in ids])
        groups.append((ids, opposite[:, :, 0], opposite[:, :, 1]))
    moved = samples.copy()
    directions = np.stack((np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)), axis=1)

    def cross(a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

    # Preserve each sampled configuration's original worst shape, up to q=.45.
    # A mild reference anisotropy is intentional; rest has its own area and q bounds.
    floors = []
    for sample in samples[1:]:
        points = sample[triangles]
        ab, ac = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
        q = 2*np.sqrt(3)*cross(ab, ac) / np.sum(ab*ab + ac*ac + (ab-ac)**2, axis=1)
        if q.min() <= 0:
            raise ValueError('Design requires positive, oriented P1 maps')
        floors.append(min(.45, q.min())-1e-10)
    floors = np.asarray(floors)[:, None]

    for sweep, fraction in enumerate((.5, .5, .5, .5, .35, .35, .25, .25, .25, .125, .125, .0625, .0625)):
        for ids, left, right in (groups if sweep % 2 == 0 else reversed(groups)):
            b, c = moved[:, left], moved[:, right]
            length_sq = np.sum((b-c)**2, axis=-1)
            area0 = cross(sn[left]-sn[ids, None], sn[right]-sn[ids, None])

            def score(proposal):
                ab = b-proposal[:, :, None]
                ac = c-proposal[:, :, None]
                area = cross(ab, ac)
                quality = 2*np.sqrt(3)*area / (length_sq + np.sum(ab*ab + ac*ac, axis=-1))
                valid = np.all((area[0] >= .30*area0) & (area[0] <= 3*area0) & (quality[0] >= .35), axis=1)
                valid &= np.all(quality[1:].min(axis=2) >= floors, axis=0)
                return np.where(valid, quality.min(axis=(0, 2)), -np.inf)

            best = moved[:, ids].copy()
            scores = score(best)
            for direction in directions:
                query = moved[0, ids] + fraction*step_mm*direction
                element = finder(*query.T)
                weights = np.einsum('nij,nj->ni', inverse[element], query-corners[element, 0])
                vertices = samples[:, triangles[element]]
                proposal = vertices[:, :, 0] + np.einsum('fniq,ni->fnq',
                              vertices[:, :, 1:]-vertices[:, :, :1], weights)
                proposal[0] = query
                values = score(proposal)
                improve = (element >= 0) & (values > scores+1e-10)
                best[:, improve] = proposal[:, improve]
                scores[improve] = values[improve]
            moved[:, ids] = best
    ids = movable[np.any(moved[0, movable] != sn[movable], axis=1)]
    if not len(ids):
        raise ValueError('No admissible improvement of the reference layout')
    layout = dict(schema='intact-reference-layout/1', base_sha256=reference_mesh_digest(sn, triangles),
                  node_ids=ids.tolist(), local_sn_mm=moved[0, ids].tolist())
    relocate_reference_nodes(sn, triangles, layout)
    return layout


def intact_mesh(config, options, *, fine_halfwidth_mm=None):
    config.validate()
    options.validate(config)
    def grid(a, b, h):
        return np.linspace(a, b, int(np.ceil((b-a)/h))+1)
    s = grid(0, config.length_mm, config.mesh_size_mm)
    n = grid(0, config.height_mm, config.mesh_size_mm)
    lo, hi = max(0, config.drop_s_mm-4), min(config.length_mm, config.drop_s_mm+7)
    lower = max(0, config.height_mm-2.5)
    # Replace the closed patch, including its lower interface. Superposing
    # the two independent grids there creates almost coincident vertices and
    # triangles with angles below 0.03 degrees, even on a finer surface mesh.
    points = [(x,y) for x in s for y in n if not (lo <= x <= hi and y >= lower)]
    points += [(x,y) for x in grid(lo,hi,options.refinement_mm)
               for y in grid(lower,config.height_mm,options.refinement_mm)]
    # One surface grid, rather than superposed grids with tiny accidental edges.
    # Graded subsurface rows avoid attaching micrometre edges to very tall CSTs.
    h = options.surface_spacing_mm
    strip = min(options.refinement_mm,config.height_mm-lower)
    points = [(x,y) for x,y in points if not (lo <= x <= hi and y > config.height_mm-strip)]
    points += [(x,config.height_mm) for x in grid(lo,hi,h)]
    for factor in (2.,8.,32.):
        if factor*h < strip:
            points += [(x,config.height_mm-factor*h) for x in grid(lo,hi,factor*h)]
    points += [(config.drop_s_mm, config.height_mm)]
    if options.mesh_grading=='graded':
        # Quadtree corners with a Lipschitz size field remove the abrupt
        # fine-strip/coarse-grid junction at s=lo,hi. Start at the top so the
        # finest boundary intervals equal the requested surface spacing.
        base = h*2**max(0,int(np.floor(np.log2(config.mesh_size_mm/h))))
        xs = np.r_[np.arange(0,config.length_mm,base),config.length_mm]
        ys = np.r_[np.arange(config.height_mm,0,-base),0.][::-1]
        cells = [(a,b,c,d) for a,b in zip(xs[:-1],xs[1:]) for c,d in zip(ys[:-1],ys[1:])]
        points = []
        interior = []
        fine_halfwidth = fine_halfwidth_mm if fine_halfwidth_mm is not None else max(2*options.refinement_mm,8*options.tip_radius_mm)
        while cells:
            a,b,c,d = cells.pop()
            sx,ny = (a+b)/2,(c+d)/2
            lateral = max(lo-sx,0.,sx-hi)
            below = max(lower-ny,0.)
            fine_lateral = max(abs(sx-config.drop_s_mm)-fine_halfwidth,0.)
            distance = np.hypot(fine_lateral,config.height_mm-ny)
            target = min(config.mesh_size_mm,
                         options.refinement_mm+options.size_gradient*np.hypot(lateral,below),h+options.size_gradient*distance)
            if max(b-a,d-c)>target*(1+1e-10):
                cells.extend([(a,sx,c,ny),(sx,b,c,ny),(a,sx,ny,d),(sx,b,ny,d)])
            else:
                points.extend([(a,c),(b,c),(a,d),(b,d)])
                if options.contact_layer_depth_mm:
                    interior.append((a,b,c,d))
        # Split original leaves towards a subsurface rectangle. Keep every original
        # boundary point and introduce interior points only, including beneath the
        # first surface row. The surrounding graded mesh controls the far field.
        while interior:
            a,b,c,d = interior.pop()
            sx,ny = (a+b)/2,(c+d)/2
            outside = max(abs(sx-config.drop_s_mm)-fine_halfwidth/2, 0.)
            below = max(config.height_mm-ny-options.contact_layer_depth_mm, 0.)
            target = h/2 + options.size_gradient*np.hypot(outside, below)
            if max(b-a,d-c) <= target*(1+1e-10):
                continue
            points.extend((x,y) for x,y in ((sx,c),(sx,d),(a,ny),(b,ny),(sx,ny))
                          if 0 < x < config.length_mm and 0 < y < config.height_mm)
            interior.extend(((a,sx,c,ny),(sx,b,c,ny),(a,sx,ny,d),(sx,b,ny,d)))
    sn = np.unique(np.round(points, 10), axis=0)
    xy = local_to_lab(sn, config)
    tri = Delaunay(sn).simplices.copy()
    signed = np.linalg.det(np.stack((xy[tri[:,1]]-xy[tri[:,0]],xy[tri[:,2]]-xy[tri[:,0]]),axis=2))
    tri[signed < 0] = tri[signed < 0][:,[0,2,1]]
    if options.reference_layout is not None:
        layout = json.loads(Path(options.reference_layout).read_bytes())
        sn = relocate_reference_nodes(sn, tri, layout)
        xy = local_to_lab(sn, config)
    mesh = FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
                   np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(sn),int))
    edges = np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    unique, count = np.unique(edges,axis=0,return_counts=True)
    boundary = unique[count == 1]
    boundary = boundary[~np.all(np.isclose(sn[boundary,1],0),axis=1)]
    return mesh, boundary


def blade_boundary_vertices(config, options, depth):
    """Straight-boundary vertices and exact circular-nose tangent endpoints."""
    r = options.tip_radius_mm
    a = np.deg2rad(config.blade_tip_angle_deg/2)
    _, en = _rotation_vectors(config.inclination_deg)
    contact = local_to_lab(np.array([[config.drop_s_mm,config.height_mm]]),config)[0]
    center = contact + r*en - np.array([0.,depth])
    angles = np.array([-np.pi+a,-a])
    arc = center + r*np.column_stack((np.cos(angles),np.sin(angles)))
    w = config.blade_thickness_mm/2
    shoulder_z = center[1] - r/np.sin(a) + w/np.tan(a)
    top = center[1] + 2*(config.length_mm+config.height_mm)
    return np.vstack(([center[0]-w,top],[center[0]-w,shoulder_z],arc,
                      [center[0]+w,shoulder_z],[center[0]+w,top]))


def rounded_outline(config, options, depth, facets, shape):
    """Contact-ready control polygon. Lab x points downhill, lab z points up."""
    beta = np.deg2rad(config.blade_tip_angle_deg)
    left, right = {'double': (beta/2, beta/2), 'single_up': (beta, 0.),
                   'single_down': (0., beta)}[shape]
    es, en = _rotation_vectors(config.inclination_deg)
    # Surface-normal first touch, and the fixed lab-lowest pole, are actual vertices.
    contact_angle = -np.pi/2 - np.deg2rad(config.inclination_deg)
    canonical_left,canonical_right=(right,left) if shape=='single_down' else (left,right)
    lo,hi=-np.pi+canonical_left,-canonical_right
    reflected=-np.pi-contact_angle
    limits = np.unique([lo,hi,-np.pi/2]+[v for v in (contact_angle,reflected) if lo<v<hi])
    if contact_angle < -np.pi+left or contact_angle > -right:
        raise ValueError('First touch is outside rounded nose')
    lengths = np.diff(limits)
    counts = np.ones(len(lengths), dtype=int)
    for _ in range(facets-len(counts)):
        counts[np.argmax(lengths/counts)] += 1
    theta = np.r_[np.concatenate([np.linspace(a,b,k,endpoint=False)
        for a,b,k in zip(limits[:-1],limits[1:],counts)]),limits[-1]]
    if shape=='single_down':theta=-np.pi-theta[::-1]
    r = options.tip_radius_mm
    contact = local_to_lab(np.array([[config.drop_s_mm, config.height_mm]]),config)[0]
    center = contact + r*en - [0.,depth]
    arc = r*np.column_stack([np.cos(theta),np.sin(theta)])
    # At height h relative to center, each tangent has horizontal offset
    # r/cos(side) + h*tan(side). This also covers a vertical side exactly.
    shoulder = (config.blade_thickness_mm-r*(1/np.cos(left)+1/np.cos(right)))/(np.tan(left)+np.tan(right))
    if shoulder <= max(arc[0,1],arc[-1,1]):
        raise ValueError('Blade thickness cannot accommodate tangent flanks')
    xl = -r/np.cos(left)-shoulder*np.tan(left)
    xr = r/np.cos(right)+shoulder*np.tan(right)
    top = 2*(config.length_mm+config.height_mm)
    outline = center + np.vstack(([xl,top],[xl,shoulder],arc,[xr,shoulder],[xr,top]))
    pole = 2+int(np.argmin(arc[:,1]))
    metadata = dict(shape=shape,pole_local=pole,nose_first=2,nose_last=facets+2,
                    left_flank=[1,2],right_flank=[facets+2,facets+3],
                    left_angle_deg=float(np.rad2deg(left)),right_angle_deg=float(np.rad2deg(right)))
    # Check the entire closed outline against the original plane at nominal zero.
    separation = (outline + [0.,depth] - contact) @ en
    if abs(np.min(separation)) > 1e-10:
        raise ValueError('Nominal zero is inconsistent with whole-outline contact')
    return outline,metadata


def subdivide_blade(blade,nose_facets,spacing):
    """Split straight edges without moving any original vertex or nose facet.

    Original indices remain first, so the fixed pole and nose observers retain
    their identity. Added vertices are connected by explicit collision edges.
    Shanks use uniform spacing for 3 mm above each shoulder, then grade to 1 mm.
    The subdivision changes contact quadrature, never the polygonal shape.
    """
    blade=np.asarray(blade,dtype=float)
    count=nose_facets+5
    if blade.shape!=(count,2):raise ValueError('Canonical rounded blade required')
    if spacing is None:
        return blade,np.column_stack((np.arange(count),np.roll(np.arange(count),-1)))
    if not np.isfinite(spacing) or spacing<=0 or spacing>.05:
        raise ValueError('Blade spacing must be in (0, .05] mm')
    extra=[];edges=[]
    for i in range(count):
        j=(i+1)%count;length=float(np.linalg.norm(blade[j]-blade[i]))
        if i in (1,count-3):
            fractions=np.linspace(0.,1.,int(np.ceil(length/spacing))+1)[1:-1]
        elif i in (0,count-2):
            distance=0.;step=spacing;locations=[]
            while distance+step<length-1e-12:
                distance+=step;locations.append(distance/length)
                if distance>=3.:step=min(1.,1.5*step)
            fractions=np.asarray(locations)
            if i==0:fractions=1.-fractions[::-1]
        else:fractions=[]
        chain=[i]
        for fraction in fractions:
            chain.append(count+len(extra))
            extra.append(blade[i]+fraction*(blade[j]-blade[i]))
        chain.append(j);edges.extend(zip(chain[:-1],chain[1:]))
    return np.vstack((blade,np.asarray(extra).reshape(-1,2))),np.asarray(edges,dtype=int)


