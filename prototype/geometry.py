"""Intact P1 specimen mesh and rigid rounded-blade geometry (mm)."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay
from .model_config import FEMMesh, local_to_lab, _rotation_vectors


@dataclass
class GeometryOptions:
    tip_radius_mm: float = 0.02
    refinement_mm: float = 0.15
    surface_spacing_mm: float = 0.0025
    mesh_grading: str = 'legacy'

    def validate(self, config):
        for name in ('tip_radius_mm', 'refinement_mm', 'surface_spacing_mm'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.mesh_grading not in ('legacy', 'graded'):
            raise ValueError('Unknown mesh grading')
        if self.tip_radius_mm >= config.blade_thickness_mm / 2:
            raise ValueError('Tip radius must be smaller than half blade thickness')


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
        fine_halfwidth = fine_halfwidth_mm if fine_halfwidth_mm is not None else max(2*options.refinement_mm,8*options.tip_radius_mm)
        while cells:
            a,b,c,d = cells.pop()
            sx,ny = (a+b)/2,(c+d)/2
            lateral = max(lo-sx,0.,sx-hi)
            below = max(lower-ny,0.)
            fine_lateral = max(abs(sx-config.drop_s_mm)-fine_halfwidth,0.)
            distance = np.hypot(fine_lateral,config.height_mm-ny)
            target = min(config.mesh_size_mm,
                         options.refinement_mm+.4*np.hypot(lateral,below),h+.4*distance)
            if max(b-a,d-c)>target*(1+1e-10):
                cells.extend([(a,sx,c,ny),(sx,b,c,ny),(a,sx,ny,d),(sx,b,ny,d)])
            else:
                points.extend([(a,c),(b,c),(a,d),(b,d)])
    sn = np.unique(np.round(points, 10), axis=0)
    xy = local_to_lab(sn, config)
    tri = Delaunay(sn).simplices.copy()
    signed = np.linalg.det(np.stack((xy[tri[:,1]]-xy[tri[:,0]],xy[tri[:,2]]-xy[tri[:,0]]),axis=2))
    tri[signed < 0] = tri[signed < 0][:,[0,2,1]]
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


