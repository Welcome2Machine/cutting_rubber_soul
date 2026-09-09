"""Uncracked-body initiation screening, independent of the prescribed seam model.

No elements are deleted and no post-initiation crack propagation is claimed.
All dimensions follow FEMConfig (N, mm, s). See docs/initiation_model.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix, bmat, diags
from scipy.sparse.linalg import splu

from .model_config import FEMConfig, FEMMesh, local_to_lab, _rotation_vectors
from .material import bulk as _bulk_batch, response as material_response
from .fracture_criteria import stress_measures
from .contact import coulomb_return
from .contact_branches import branch_predictors


@dataclass
class InitiationOptions:
    tip_radius_mm: float = 0.02  # unmeasured sensitivity parameter, not blade drawing data
    refinement_mm: float = 0.15
    surface_spacing_mm: float = 0.0025
    averaging_radius_mm: float = 0.3  # diagnostic averaging, not calibrated material length
    tangential_stiffness_mpa_per_mm: float = 250.  # tangential augmentation, not material stiffness
    coulomb_formulation: str = 'alart_curnier'
    contact_discretization: str = 'nodal'
    mesh_grading: str = 'legacy'
    volumetric_discretization: str = 'element'
    volume_stabilization_shear: float = 0.

    def validate(self, config):
        for name,value in vars(self).items():
            if name in ('coulomb_formulation','contact_discretization','mesh_grading','volumetric_discretization','volume_stabilization_shear'):
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Initiation options must be finite and positive")
        if self.coulomb_formulation not in ('alart_curnier','jean_moreau'):
            raise ValueError('Unknown Coulomb projection formulation')
        if self.contact_discretization not in ('nodal','integral','macro','quadratic_nodes','quadratic_integral') or self.mesh_grading not in ('legacy','graded'):
            raise ValueError('Unknown contact discretization or mesh grading')
        if self.volumetric_discretization not in ('element','nodal_volume','macro_mixed','taylor_hood','quadratic_full'):
            raise ValueError('Unknown volumetric discretization')
        if not np.isfinite(self.volume_stabilization_shear) or self.volume_stabilization_shear<0:
            raise ValueError('Volume stabilization must be finite and nonnegative')
        if self.volume_stabilization_shear>config.bulk_modulus_n_mm2/config.shear_modulus_n_mm2:
            raise ValueError('Volume stabilization must not exceed K/G')
        if self.volume_stabilization_shear and self.volumetric_discretization!='nodal_volume':
            raise ValueError('Volume stabilization requires nodal_volume discretization')
        if self.contact_discretization=='macro' and self.volumetric_discretization not in ('macro_mixed','taylor_hood','quadratic_full'):
            raise ValueError('Macro contact requires the macro displacement mesh')
        if self.volumetric_discretization in ('taylor_hood','quadratic_full') and self.contact_discretization not in ('macro','quadratic_nodes','quadratic_integral'):
            raise ValueError('Quadratic displacement requires compatible boundary contact')
        if self.contact_discretization in ('quadratic_nodes','quadratic_integral') and self.volumetric_discretization not in ('taylor_hood','quadratic_full'):
            raise ValueError('Quadratic contact requires quadratic displacement elements')
        if self.tip_radius_mm >= config.blade_thickness_mm / 2:
            raise ValueError("Tip radius must be smaller than half blade thickness")


def intact_mesh(config, options, *, fine_halfwidth_mm=None):
    config.validate()
    options.validate(config)
    if options.volumetric_discretization=='macro_mixed':
        from .macro_element import refine_macro_mesh
        coarse_options=replace(options,volumetric_discretization='element',contact_discretization='integral',surface_spacing_mm=2*options.surface_spacing_mm,refinement_mm=2*options.refinement_mm)
        coarse_config=replace(config,mesh_size_mm=2*config.mesh_size_mm)
        return refine_macro_mesh(*intact_mesh(coarse_config,coarse_options,fine_halfwidth_mm=max(2*options.refinement_mm,8*options.tip_radius_mm)))
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


def _blade_boundary_vertices(config, options, depth):
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


def rounded_blade_distance(points, config, options, depth, *, with_tangent=False):
    """Exact circle/flank/shoulder distance; no faceted nose normals."""
    polygon = _blade_boundary_vertices(config,options,depth)
    starts = polygon[[0,1,-3,-2,-1]]
    ends = polygon[[1,2,-2,-1,0]]
    edges = ends-starts
    v = points[:,None,:]-starts
    t = np.clip(np.einsum('qei,ei->qe',v,edges)/(edges*edges).sum(axis=1),0,1)
    closest = starts+t[:,:,None]*edges
    a = np.deg2rad(config.blade_tip_angle_deg/2)
    r = options.tip_radius_mm
    _,en = _rotation_vectors(config.inclination_deg)
    center = local_to_lab(np.array([[config.drop_s_mm,config.height_mm]]),config)[0]+r*en-np.array([0.,depth])
    relative = points-center
    radial = relative/np.maximum(np.linalg.norm(relative,axis=1)[:,None],1e-14)
    angle = np.arctan2(relative[:,1],relative[:,0])
    on_arc = (angle >= -np.pi+a) & (angle <= -a)
    closest = np.concatenate((closest,(center+r*radial)[:,None,:]),axis=1)
    sq = ((points[:,None,:]-closest)**2).sum(axis=2)
    sq[~on_arc,-1] = np.inf
    index = sq.argmin(axis=1)
    distance = np.sqrt(sq[np.arange(len(points)),index])
    x = np.abs(relative[:,0])
    bottom = np.where(x <= r*np.cos(a),
                       -np.sqrt(np.maximum(0,r*r-x*x)),
                       -r/np.sin(a)+x/np.tan(a))
    inside = (x <= config.blade_thickness_mm/2) & (relative[:,1] >= bottom) & (points[:,1] <= polygon[0,1])
    sign = np.where(inside,1.,-1.)
    difference = points-closest[np.arange(len(points)),index]
    normal = -sign[:,None]*difference/np.maximum(distance[:,None],1e-14)
    zero = distance < 1e-12
    segment_normal = np.column_stack((edges[:,1],-edges[:,0]))/np.linalg.norm(edges,axis=1)[:,None]
    for k in range(5):
        # A normalized point-minus-projection loses significant digits as the
        # gap approaches zero. Use the exact surface normal on segment interiors.
        interior = (index == k) & (t[:,k] > 0) & (t[:,k] < 1)
        normal[interior | (zero & (index == k))] = segment_normal[k]
    normal[index == 5] = radial[index == 5]
    if with_tangent:
        dn = np.zeros((len(points),2,2))
        projector = np.eye(2)-normal[:,:,None]*normal[:,None,:]
        arc = index == 5
        dn[arc] = projector[arc]/np.maximum(np.linalg.norm(relative[arc],axis=1)[:,None,None],1e-14)
        segment = index < 5
        selected_t = t[np.arange(len(points))[segment],index[segment]]
        endpoint = np.zeros(len(points),bool)
        endpoint[segment] = (selected_t <= 0) | (selected_t >= 1)
        endpoint &= ~zero
        dn[endpoint] = -sign[endpoint,None,None]*projector[endpoint]/distance[endpoint,None,None]
        return sign*distance,normal,dn
    return sign*distance,normal


class ContactResolutionError(ValueError):
    """Spatial resolution is insufficient; time bisection cannot fix it."""
    def __init__(self,message,edges,*,elements=(),u=None,multipliers=None,depth=None):
        super().__init__(message)
        self.edges = np.asarray(edges,dtype=int)
        self.elements=np.asarray(elements,dtype=int)
        self.trial=None if u is None else (u.copy(),multipliers.copy(),depth)


class ContactSolveError(RuntimeError):
    def __init__(self,message,u,multipliers):
        super().__init__(message)
        self.u,self.multipliers = u.copy(),multipliers.copy()


def refine_resolution(problem,previous,state,error):
    kwargs={'trial':error.trial} if hasattr(problem,'_refine') else {}
    if len(error.elements):
        return problem.refine_bulk(previous,state,error.elements,**kwargs)
    return problem.refine_contact(previous,state,error.edges,**kwargs)



class IntactProblem:
    def __init__(self, config, options, strength_surface=None):
        self.config, self.options = config, options
        self.strength_surface = strength_surface
        self.mesh, edges = intact_mesh(config,options)
        self._rebuild_boundary(edges)

    def _rebuild_boundary(self,edges):
        self.edges = edges
        from .material import NodalVolume
        self.nodal_volume = NodalVolume(self.mesh,self.config,self.options.volume_stabilization_shear) if self.options.volumetric_discretization in ('nodal_volume','macro_mixed') else None
        multiplier_edges=self.mesh.macro_boundary_edges if self.options.contact_discretization=='macro' else edges
        self.contact_nodes = np.unique(multiplier_edges)
        lengths = np.linalg.norm(self.mesh.reference_xy[multiplier_edges[:,1]]-self.mesh.reference_xy[multiplier_edges[:,0]],axis=1)
        area = np.zeros(self.mesh.node_count)
        np.add.at(area,multiplier_edges.ravel(),np.repeat(lengths*self.config.width_mm/2,2))
        self.weights = area[self.contact_nodes]
        self.cdofs = 2*self.contact_nodes[:,None]+np.arange(2)
        from .surface_contact import SurfaceContact
        self.surface_contact = SurfaceContact(self) if self.options.contact_discretization in ('integral','macro') else None
        self.fixed = (2*self.mesh.bottom_nodes[:,None]+np.arange(2)).ravel()
        self.free = np.setdiff1d(np.arange(self.mesh.dof_count),self.fixed)
        centroids = self.mesh.reference_xy[self.mesh.triangles].mean(axis=1)
        tree = cKDTree(centroids)
        distances = tree.sparse_distance_matrix(tree,self.options.averaging_radius_mm,output_type='coo_matrix')
        areas = self.mesh.bulk_data[0]
        weights = areas[distances.col]*np.maximum(0,1-(distances.data/self.options.averaging_radius_mm)**2)**2
        average = coo_matrix((weights,(distances.row,distances.col)),shape=distances.shape).tocsr()
        self.average = average.multiply(1/np.asarray(average.sum(axis=1))).tocsr()
        # Connectivity remains fixed until refinement rebuilds this cache.
        dofs = self.mesh.bulk_data[2]
        bulk_shape = (len(dofs),6,6)
        contact_shape = (len(self.cdofs),2,2)
        self._assembly_rows = np.r_[np.broadcast_to(dofs[:,:,None],bulk_shape).ravel(),
                                    np.broadcast_to(self.cdofs[:,:,None],contact_shape).ravel()]
        self._assembly_cols = np.r_[np.broadcast_to(dofs[:,None,:],bulk_shape).ravel(),
                                    np.broadcast_to(self.cdofs[:,None,:],contact_shape).ravel()]

    def initial_contact_state(self):
        # Columns: normal multiplier (MPa), convected tangential shear (MPa).
        return np.zeros((len(self.contact_nodes),2))

    def contact(self,points,previous,depth,blade_increment,normal_multipliers,old_shear,*,formulation=None,roundoff_guard=True):
        c = self.config
        g,n,dn = rounded_blade_distance(points,c,self.options,depth,with_tangent=True)
        # Projection onto the blade is accurate only to coordinate roundoff.
        # Choose the zero-gap branch reproducibly within that arithmetic scale,
        # not within the much larger physical penetration/traction tolerances.
        # Keep the raw gap for diagnostics and final independent acceptance.
        gap_roundoff = 32*np.finfo(float).eps*np.maximum(
            max(1.,c.length_mm,c.height_mm,abs(depth)),np.max(np.abs(points),axis=1))
        projection_gap = np.where(np.abs(g)<=gap_roundoff,0.,g) if roundoff_guard else g
        pressure = np.maximum(0,normal_multipliers+c.contact_surface_penalty_n_mm3*projection_gap)
        active = pressure > 0
        formulation = formulation or self.options.coulomb_formulation
        if formulation not in ('alart_curnier','jean_moreau'):
            raise ValueError('Unknown Coulomb projection formulation')
        # Acary, Bremond & Huber (2018), equations 58 and 60: using the
        # independent normal multiplier in the friction capacity removes its
        # augmentation-gap coupling. Both formulations have the same exact
        # Signorini/Coulomb roots; neither changes the friction coefficient.
        independent_capacity = formulation == 'jean_moreau'
        capacity_pressure = np.maximum(0,normal_multipliers) if independent_capacity else pressure
        capacity_active = normal_multipliers > 0 if independent_capacity else active
        tangent = np.column_stack((-n[:,1],n[:,0]))
        dtangent = np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
        relative = points-previous-blade_increment
        slip = (relative*tangent).sum(axis=1)
        shear,ds,dp_shear,dissipation,stick = coulomb_return(slip,old_shear,capacity_pressure,
            c.friction_coefficient,self.options.tangential_stiffness_mpa_per_mm,
            c.adhesive_shear_strength_n_mm2)
        dg = -n
        dp = c.contact_surface_penalty_n_mm3*dg*active[:,None]
        dslip = tangent+np.einsum('qi,qij->qj',relative,dtangent)
        dshear = ds[:,None]*dslip
        if not independent_capacity:
            dshear += dp_shear[:,None]*dp
        force = self.weights[:,None]*(pressure[:,None]*n-shear[:,None]*tangent)
        jac = self.weights[:,None,None]*(n[:,:,None]*dp[:,None,:]+pressure[:,None,None]*dn
               -tangent[:,:,None]*dshear[:,None,:]-shear[:,None,None]*dtangent)
        dforce_dlambda = np.stack((self.weights[:,None]*(n*active[:,None]-dp_shear[:,None]*tangent*capacity_active[:,None]),
                                  -self.weights[:,None]*tangent*stick[:,None]),axis=2)
        projection_u = np.stack((dp,dshear),axis=1)
        projection_lambda = np.zeros((len(points),2,2))
        projection_lambda[:,0,0] = active.astype(float)-1
        projection_lambda[:,1,0] = dp_shear*capacity_active
        projection_lambda[:,1,1] = stick.astype(float)-1
        return dict(force=force,jac=jac,dforce_dlambda=dforce_dlambda,gap=g,normal=n,
                    gap_roundoff_mm=gap_roundoff,projection_gap=projection_gap,
                    normal_derivative=dn,
                    pressure=pressure,shear=shear,slip=slip,slip_derivative=dslip,stick=stick,
                    dissipation=self.weights*shear*slip,projection_u=projection_u,projection_lambda=projection_lambda)

    def assemble(self,u,previous,depth,old_depth,dt,multipliers,old_shear=None,*,mixed=False,formulation=None,with_tangent=True,roundoff_guard=True,pressure_blocks=False):
        if dt <= 0:
            raise ValueError('Physical time increment must be positive')
        mesh,c = self.mesh,self.config
        br,bk,bd,energy,minj = _bulk_batch(mesh,u,c,with_tangent=with_tangent)
        volume_k=None
        if self.nodal_volume is not None:
            if pressure_blocks and self.options.volumetric_discretization!='macro_mixed':
                raise ValueError('Pressure blocks require macro_mixed discretization')
            br,bk,volume_k,energy=self.nodal_volume.correct(u,br,bk,energy,condensed=not pressure_blocks)
        residual = np.zeros(mesh.dof_count)
        np.add.at(residual,bd.ravel(),br.ravel())
        points = (mesh.reference_xy+u.reshape(-1,2))[self.contact_nodes]
        old = (mesh.reference_xy+previous.reshape(-1,2))[self.contact_nodes]
        if old_shear is None:
            old_shear = np.zeros(len(self.weights))
        if self.surface_contact is not None:
            contact = self.surface_contact.evaluate(u,previous,depth,old_depth,np.column_stack((multipliers,old_shear)),
                with_tangent=with_tangent,formulation=formulation,roundoff_guard=roundoff_guard)
        else:
            contact = self.contact(points,old,depth,np.array([0.,old_depth-depth]),multipliers,old_shear,formulation=formulation,roundoff_guard=roundoff_guard)
            if mixed:
                # The mixed equilibrium equation uses the independent tractions.
                # Projection belongs to the constraint equations, not to a second
                # unilateral clipping of forces during Newton's trial iterates.
                n,dn = contact['normal'],contact['normal_derivative']
                tangent = np.column_stack((-n[:,1],n[:,0]))
                dtangent = np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
                contact['force'] = self.weights[:,None]*(multipliers[:,None]*n-old_shear[:,None]*tangent)
                contact['jac'] = self.weights[:,None,None]*(multipliers[:,None,None]*dn-old_shear[:,None,None]*dtangent)
                contact['dforce_dlambda'] = self.weights[:,None,None]*np.stack((n,-tangent),axis=2)
        if self.surface_contact is not None:
            residual-=contact['global_force']
        else:
            np.add.at(residual,self.cdofs.ravel(),-contact['force'].ravel())
        matrix = None
        if with_tangent:
            if pressure_blocks:
                contact['pressure_coupling']=self.nodal_volume.pressure_coupling
                contact['pressure_mass']=self.nodal_volume.pressure_mass
            ck = -contact['jac'] if self.surface_contact is None else np.zeros((len(self.contact_nodes),2,2))
            matrix = coo_matrix((np.r_[bk.ravel(),ck.ravel()],(self._assembly_rows,self._assembly_cols)),shape=(mesh.dof_count,mesh.dof_count)).tocsr()
            if self.surface_contact is not None:
                matrix = matrix+contact['global_K']
            if volume_k is not None:
                matrix = matrix+volume_k
        force = contact['force']
        d = dict(reaction_fz_n=float(-force[:,1].sum()),reaction_fx_n=float(-force[:,0].sum()),
                 min_j=minj,bulk_energy_n_mm=energy,max_pressure_mpa=float(contact['pressure'].max(initial=0)),
                 max_pressure_node=int(self.contact_nodes[contact['pressure'].argmax()]),
                 max_incremental_displacement_mm=float(np.linalg.norm((u-previous).reshape(-1,2),axis=1).max(initial=0)),
                 max_penetration_mm=float(contact['gap'].max(initial=0)),
                 stick_nodes=int(contact['stick'].sum()),
                 slip_nodes=int(((contact['pressure']>0)&~contact['stick']).sum()),
                 friction_dissipation_n_mm=float(contact['dissipation'].sum()),
                 global_force_balance_error_n=float(np.linalg.norm(residual[self.fixed].reshape(-1,2).sum(axis=0)+force.sum(axis=0))))
        return residual,matrix,contact,d

    def edge_penetration(self,u,depth,*,with_edges=False):
        """Audit between collocation nodes, including the nearest circle point."""
        xy = self.mesh.reference_xy+u.reshape(-1,2)
        a,b = xy[self.edges[:,0]],xy[self.edges[:,1]]
        # Include the segment's exact closest point to the circular nose centre.
        _,en = _rotation_vectors(self.config.inclination_deg)
        center = local_to_lab(np.array([[self.config.drop_s_mm,self.config.height_mm]]),self.config)[0]
        center += self.options.tip_radius_mm*en-np.array([0.,depth])
        v = b-a
        nearest = np.clip(((center-a)*v).sum(axis=1)/(v*v).sum(axis=1),0,1)
        fractions = np.column_stack((np.tile(np.linspace(0,1,17),(len(a),1)),nearest))
        points = a[:,None,:]+fractions[:,:,None]*v[:,None,:]
        gaps = rounded_blade_distance(points.reshape(-1,2),self.config,self.options,depth)[0]
        edge_gaps = gaps.reshape(len(a),-1).max(axis=1)
        maximum = float(edge_gaps.max(initial=0))
        if with_edges:
            return maximum,np.flatnonzero(edge_gaps > self.config.contact_penetration_tolerance_mm)
        return maximum

    def refine_contact(self,previous,old_state,edge_indices,*,internal_edges=None):
        """Bisect boundary CSTs while preserving the previous displacement field.

        Refinement is geometric, never element deletion or crack insertion.
        Nodal tractions are interpolated as warm starts; the next solve must
        re-equilibrate them. No new physical time/displacement is committed.
        """
        history = {int(node):old_state[i].copy() for i,node in enumerate(self.contact_nodes)}
        multiplier_edges=None
        if self.options.contact_discretization=='macro':
            from .macro_element import boundary_coordinates
            mapping,_=boundary_coordinates(self.mesh,self.mesh.reference_xy[self.edges].mean(axis=1))
            chosen=np.unique(mapping[np.unique(edge_indices)])
            edge_indices=np.flatnonzero(np.isin(mapping,chosen))
            multiplier_edges=self.mesh.macro_boundary_edges.tolist()
            tree=cKDTree(self.mesh.reference_xy)
            for index in chosen:
                a,b=multiplier_edges[index]
                distance,mid=tree.query(self.mesh.reference_xy[[a,b]].mean(axis=0))
                if distance>1e-9:raise ValueError('Multiplier midpoint is absent from displacement mesh')
                history[int(mid)]=(history[int(a)]+history[int(b)])/2
                multiplier_edges[index]=[a,int(mid)]
                multiplier_edges.append([int(mid),b])
        edges_to_split=(self.edges[np.unique(np.asarray(edge_indices,dtype=int))].copy()
            if internal_edges is None else np.asarray(internal_edges,dtype=int))
        xy = self.mesh.reference_xy.tolist()
        sn = self.mesh.local_sn.tolist()
        displacement = previous.reshape(-1,2).tolist()
        triangles = self.mesh.triangles.tolist()
        from .macro_element import checkpoint
        macro=checkpoint(self.mesh)
        parents=self.mesh.macro_parents.tolist() if macro else None
        edges = self.edges.tolist()
        def split(a,b):
            if np.linalg.norm(np.array(xy[a])-xy[b]) < self.options.surface_spacing_mm/32:
                raise ValueError('Contact refinement limit reached; further spatial convergence study required')
            containing = [i for i,tri in enumerate(triangles) if a in tri and b in tri]
            if not containing:
                return
            if self.options.mesh_grading=='graded':
                # Longest-edge propagation preserves a conforming mesh and
                # refines the interior as the contact patch moves. All inserted
                # values interpolate the SAME accepted affine displacement.
                length = np.linalg.norm(np.array(xy[a])-xy[b])
                longer = []
                for i in containing:
                    tri=triangles[i]
                    for j in range(3):
                        e,f=tri[j],tri[(j+1)%3]
                        size=np.linalg.norm(np.array(xy[e])-xy[f])
                        if size>length*(1+1e-10):longer.append((size,e,f))
                if longer:
                    _,e,f=max(longer)
                    split(e,f)
                    split(a,b)
                    return
            mid = len(xy)
            xy.append(((np.array(xy[a])+xy[b])/2).tolist())
            sn.append(((np.array(sn[a])+sn[b])/2).tolist())
            displacement.append(((np.array(displacement[a])+displacement[b])/2).tolist())
            for i in containing:
                opposite = next(node for node in triangles[i] if node not in (a,b))
                triangles[i] = [int(a),mid,opposite]
                triangles.append([mid,int(b),opposite])
                if parents is not None:parents.append(parents[i])
            ei = next((j for j,edge in enumerate(edges) if set(edge)=={a,b}),None)
            if ei is not None:
                if self.options.contact_discretization!='macro':
                    history[mid] = (history[int(a)]+history[int(b)])/2
                edges[ei] = [int(a),mid]
                edges.append([mid,int(b)])
        for a,b in edges_to_split:
            split(a,b)
        xy,sn,triangles = np.asarray(xy),np.asarray(sn),np.asarray(triangles,dtype=int)
        signed = np.linalg.det(np.stack((xy[triangles[:,1]]-xy[triangles[:,0]],xy[triangles[:,2]]-xy[triangles[:,0]]),axis=2))
        triangles[signed < 0] = triangles[signed < 0][:,[0,2,1]]
        self.mesh = FEMMesh(xy,sn,triangles,np.empty((0,2),int),np.empty(0),
            np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
        if macro:
            for name,value in macro.items():setattr(self.mesh,name,value)
            self.mesh.macro_parents=np.asarray(parents,dtype=int)
            if multiplier_edges is not None:self.mesh.macro_boundary_edges=np.asarray(multiplier_edges,dtype=int)
        self._rebuild_boundary(np.asarray(edges,dtype=int))
        return np.asarray(displacement).ravel(),np.array([history[int(node)] for node in self.contact_nodes])

    def _mixed_step(self,previous,old_depth,depth,dt,old_state,initial=None,*,initial_multipliers=None):
        if old_state.shape != (len(self.contact_nodes),2):
            raise ValueError('Contact state must contain unique-node normal and shear history; old quadrature checkpoints are incompatible')
        u = previous.copy() if initial is None else initial.copy()
        lam = (old_state if initial_multipliers is None else initial_multipliers).copy()
        if lam.shape != old_state.shape or not np.all(np.isfinite(lam)):
            raise ValueError('Invalid contact multiplier predictor')
        c = self.config
        recent = []
        regularized_steps = 0
        damping = 1e-3
        def evaluate(state,pressure,*,with_tangent=True):
            r,k,contact,d = self.assemble(state,previous,depth,old_depth,dt,pressure[:,0],pressure[:,1],mixed=True,with_tangent=with_tangent,pressure_blocks=self.options.volumetric_discretization in ('macro_mixed','taylor_hood'))
            f = np.column_stack((contact['pressure'],contact['shear']))-pressure
            if self.surface_contact is not None:
                # Globalization must see the errors used by acceptance. Area
                # weighting suppresses a traction defect at a refined node,
                # although the same MPa tolerance still applies at that node.
                merit = float(np.hypot(np.linalg.norm(r[self.free])/c.newton_tolerance_n,
                                       np.linalg.norm(f)/c.contact_multiplier_tolerance_n_mm2))
            else:
                merit = float(np.linalg.norm(np.r_[r[self.free],(self.weights[:,None]*f).ravel()]))
            return r,k,contact,d,f,merit
        for iteration in range(c.max_contact_iterations):
            r,k,contact,d,f,merit = evaluate(u,lam)
            norm = float(np.linalg.norm(r[self.free]))
            change = float(np.max(np.abs(f)))
            print(f"  contact={iteration+1} depth={depth:.8g} residual={norm:.3g} projection={change:.3g} stick={d['stick_nodes']} slip={d['slip_nodes']}",flush=True)
            physically_converged = False
            admissible = lam.copy()
            # Test the corrected candidate, not prerequisites on the uncorrected
            # trial. Forces are linear in tractions at fixed displacement, so
            # this exact cheap force check avoids needless full reassemblies.
            # It never replaces the independent original-equation checks below.
            admissible[:,0] = np.maximum(admissible[:,0],0.)
            bound = c.friction_coefficient*admissible[:,0]
            admissible[:,1] = np.clip(admissible[:,1],-bound,bound)
            correction=admissible-lam
            if self.surface_contact is not None:
                candidate_r=r+contact['global_B']@correction.ravel()
            else:
                n=contact['normal'];t=np.column_stack((-n[:,1],n[:,0]))
                candidate_r=r.copy()
                delta_force=self.weights[:,None]*(correction[:,0,None]*n-correction[:,1,None]*t)
                np.add.at(candidate_r,self.cdofs.ravel(),-delta_force.ravel())
            if np.linalg.norm(candidate_r[self.free]) <= c.newton_tolerance_n:
                actual_r,_,actual_contact,actual_d = self.assemble(u,previous,depth,old_depth,dt,admissible[:,0],admissible[:,1],formulation='alart_curnier',with_tangent=False,roundoff_guard=False)
                actual_norm = float(np.linalg.norm(actual_r[self.free]))
                actual_change = float(np.max(np.abs(np.column_stack((actual_contact['pressure'],actual_contact['shear']))-admissible)))
                physically_converged = actual_norm <= c.newton_tolerance_n and actual_change <= c.contact_multiplier_tolerance_n_mm2
                if physically_converged:
                    lam = admissible
            if physically_converged:
                norm,contact,d,change = actual_norm,actual_contact,actual_d,actual_change
                edge_gap,bad_edges = self.edge_penetration(u,depth,with_edges=True)
                if edge_gap > c.contact_penetration_tolerance_mm:
                    raise ContactResolutionError(f'Boundary mesh under-resolves contact at depth={depth:.8g}: edge penetration={edge_gap:.6g} mm exceeds {c.contact_penetration_tolerance_mm:g}; refine surface spacing',bad_edges,u=u,multipliers=lam,depth=depth)
                dissipation_tolerance = float(np.sum(self.weights*np.abs(contact['shear']))*c.contact_multiplier_tolerance_n_mm2/self.options.tangential_stiffness_mpa_per_mm+1e-12)
                if d['friction_dissipation_n_mm'] < -dissipation_tolerance:
                    raise RuntimeError('Negative Coulomb dissipation beyond projection tolerance')
                d['friction_dissipation_tolerance_n_mm'] = dissipation_tolerance
                d.update(residual_n=norm,contact_multiplier_change_mpa=change,
                         max_edge_penetration_mm=edge_gap,
                         contact_regularized_steps=regularized_steps,
                         contact_branch_predictors=0,contact_branch_combinations=0,
                         contact_solver=f'{self.options.contact_discretization} Coulomb {self.options.coulomb_formulation} Newton')
                return u,lam,d
            recent.append(merit)
            stalled = False
            if len(recent) > 8:
                recent.pop(0)
                if merit > recent[0]*.99:
                    stalled = True
            active = contact['pressure'] > 0
            selected = np.flatnonzero(active | np.any(lam != 0,axis=1))
            if self.surface_contact is not None:
                chosen = (2*selected[:,None]+np.arange(2)).ravel()
                B = contact['global_B'][self.free][:,chosen]
                C = contact['global_C'][chosen][:,self.free]
                D = contact['global_D'][chosen][:,chosen]
            else:
                mdofs = (2*np.arange(len(selected))[:,None]+np.arange(2))
                shape = (len(selected),2,2)
                rows_u = np.broadcast_to(self.cdofs[selected,:,None],shape).ravel()
                cols_m = np.broadcast_to(mdofs[:,None,:],shape).ravel()
                rows_m = np.broadcast_to(mdofs[:,:,None],shape).ravel()
                cols_u = np.broadcast_to(self.cdofs[selected,None,:],shape).ravel()
                B = coo_matrix((-contact['dforce_dlambda'][selected].ravel(),(rows_u,cols_m)),
                               shape=(len(u),2*len(selected))).tocsc()[self.free]
                C = coo_matrix((contact['projection_u'][selected].ravel(),(rows_m,cols_u)),
                               shape=(2*len(selected),len(u))).tocsr()[:,self.free]
                D = coo_matrix((contact['projection_lambda'][selected].ravel(),(rows_m,cols_m)),
                               shape=(2*len(selected),2*len(selected))).tocsc()
            W = diags(np.repeat(self.weights[selected],2))
            jac = bmat([[k[self.free][:,self.free],B],[W@C,W@D]],format='csc')
            rhs = np.r_[r[self.free],(self.weights[selected,None]*f[selected]).ravel()]
            # Keep pressure increments as sparse auxiliary unknowns in the
            # LINEAR solve. Their exact Schur complement is the condensed
            # tangent; nonlinear pressure is still evaluated from U. Avoiding
            # premature condensation substantially reduces factorization fill.
            linear_jac,linear_rhs=jac,rhs
            if 'pressure_coupling' in contact:
                H=contact['pressure_coupling'][:,self.free]
                M=contact['pressure_mass_matrix'] if 'pressure_mass_matrix' in contact else diags(contact['pressure_mass'])
                linear_jac=bmat([[k[self.free][:,self.free],B,H.T],
                                 [W@C,W@D,None],[H,None,-M]],format='csc')
                linear_rhs=np.r_[rhs,np.zeros(H.shape[0])]
            def backtrack(delta):
                for cut in range(c.max_line_search_iterations):
                    alpha = .5**cut
                    trial = u.copy(); trial[self.free] += alpha*delta[:len(self.free)]
                    trial_lam = lam.copy(); trial_lam[selected] += alpha*delta[len(self.free):len(self.free)+2*len(selected)].reshape(-1,2)
                    # Intermediate multipliers need not be admissible; clipping
                    # would invalidate the linearized residual direction.
                    trial_lam[np.abs(trial_lam) < 1e-12] = 0
                    try:
                        trial_merit=evaluate(trial,trial_lam,with_tangent=False)[-1]
                    except FloatingPointError:
                        continue
                    if np.isfinite(trial_merit) and trial_merit < merit*(1-1e-4*alpha):
                        return trial,trial_lam
                return None
            candidate = None if stalled else backtrack(splu(linear_jac).solve(-linear_rhs))
            if candidate is None and 'pressure_mass_matrix' in contact and regularized_steps < 24:
                # Exact condensed Jacobian ACTION, including consistent pressure
                # mass inversion. Matrix-free least squares avoids a dense Schur
                # matrix and does not minimize an inconsistent local tangent.
                from scipy.sparse.linalg import LinearOperator,lsmr
                K=k[self.free][:,self.free];factor=splu(M)
                n=len(self.free);nc=2*len(selected)
                row_scale=np.r_[np.full(n,1/c.newton_tolerance_n),np.full(nc,1/c.contact_multiplier_tolerance_n_mm2)]
                approximate=bmat([[K+H.T@diags(1/M.diagonal())@H,B],[C,D]],format='csr')
                approximate=diags(row_scale)@approximate
                scale=1/np.maximum(np.sqrt(np.asarray(approximate.power(2).sum(axis=0))).ravel(),1e-20)
                def product(v):
                    v=scale*v;du,dl=v[:n],v[n:]
                    return row_scale*np.r_[K@du+H.T@factor.solve(H@du)+B@dl,C@du+D@dl]
                def transpose(v):
                    v=row_scale*v;ru,rc=v[:n],v[n:]
                    return scale*np.r_[K.T@ru+H.T@factor.solve(H@ru,trans='T')+C.T@rc,B.T@ru+D.T@rc]
                operator=LinearOperator((n+nc,n+nc),matvec=product,rmatvec=transpose)
                target=-row_scale*np.r_[r[self.free],f[selected].ravel()]
                for retry in range(6):
                    least=lsmr(operator,target,damp=np.sqrt(damping),atol=1e-6,btol=1e-6,maxiter=500)
                    candidate=backtrack(scale*least[0])
                    if candidate is not None:
                        regularized_steps+=1;recent.clear()
                        print(f'CONTACT MATRIX-FREE DAMPING step={regularized_steps} lambda={damping:g} lsmr_iterations={least[2]}',flush=True)
                        damping=max(1e-12,damping/3);break
                    damping*=10
            if candidate is None and self.surface_contact is not None and regularized_steps < 24 and 'pressure_mass_matrix' not in contact:
                # Damped least-squares correction of the ORIGINAL residual.
                # Normalize equations by acceptance tolerances and columns by
                # their Jacobian norm. Damping changes the search direction,
                # never the material, contact law or physical acceptance test.
                row_scale=np.r_[np.full(len(self.free),1/c.newton_tolerance_n),
                    1/(np.repeat(self.weights[selected],2)*c.contact_multiplier_tolerance_n_mm2)]
                if 'pressure_coupling' in contact:
                    condensed=k[self.free][:,self.free]+H.T@diags(1/contact['pressure_mass'])@H
                    jac=bmat([[condensed,B],[W@C,W@D]],format='csc')
                normalized=diags(row_scale)@jac
                scale=1/np.maximum(np.sqrt(np.asarray(normalized.power(2).sum(axis=0))).ravel(),1e-20)
                scaled=normalized@diags(scale)
                normal=(scaled.T@scaled).tocsc();gradient=scaled.T@(row_scale*rhs)
                for retry in range(8):
                    delta=scale*splu(normal+diags(np.full(len(scale),damping))).solve(-gradient)
                    candidate=backtrack(delta)
                    if candidate is not None:
                        regularized_steps+=1
                        print(f'CONTACT DAMPING step={regularized_steps} lambda={damping:g}',flush=True)
                        damping=max(1e-12,damping/3)
                        recent.clear()
                        break
                    damping*=10
            accepted = candidate is not None
            if accepted:
                u,lam = candidate
            if not accepted:
                raise ContactSolveError(f'Contact line search failed depth={depth:.8g}, residual={norm:.6g} N, projection={change:.6g} MPa',u,lam)
        # The final iteration may have accepted a line-search update. Report
        # the saved trial, rather than the residual before that last update.
        r,_,_,_,f,_=evaluate(u,lam,with_tangent=False)
        norm=float(np.linalg.norm(r[self.free]));change=float(np.max(np.abs(f)))
        raise ContactSolveError(f'Contact iteration limit at depth={depth:.8g}, residual={norm:.6g} N, projection={change:.6g} MPa, min_normal={lam[:,0].min():.6g} MPa',u,lam)

    def step(self,previous,old_depth,depth,dt,old_state,*,predictor_failures=()):
        if self.options.contact_discretization=='macro':
            # Resolve blade curvature BEFORE attempting a saddle-point solve.
            # A coarse multiplier support may straddle rapidly turning normals
            # even when its displacement edges look adequate. The circle
            # sagitta r*(1-cos(theta/2)) provides a geometric error estimate.
            from .macro_element import boundary_coordinates
            parent=self.mesh.macro_boundary_edges
            xy=self.mesh.reference_xy+previous.reshape(-1,2)
            ends=xy[parent]
            gap,normal=rounded_blade_distance(ends.reshape(-1,2),self.config,self.options,depth)
            gap=gap.reshape(-1,2);normal=normal.reshape(-1,2,2)
            sagitta=self.options.tip_radius_mm*(1-np.clip(np.linalg.norm(normal.sum(axis=1),axis=1)/2,0,1))
            near=gap.max(axis=1)>-self.options.tip_radius_mm
            bad=np.flatnonzero(near & (sagitta>.5*self.config.contact_penetration_tolerance_mm))
            if len(bad):
                mapping,_=boundary_coordinates(self.mesh,self.mesh.reference_xy[self.edges].mean(axis=1))
                raise ContactResolutionError('Multiplier supports under-resolve blade curvature before Newton',np.flatnonzero(np.isin(mapping,bad)))
        try:
            return self._mixed_step(previous,old_depth,depth,dt,old_state)
        except ContactSolveError as exc:
            failures=list(predictor_failures)+[exc]
        def rank(failure):
            r,_,q,_=self.assemble(failure.u,previous,depth,old_depth,dt,
                failure.multipliers[:,0],failure.multipliers[:,1],with_tangent=False,roundoff_guard=False)
            force=np.linalg.norm(r[self.free])/self.config.newton_tolerance_n
            defect=np.linalg.norm(np.column_stack((q['pressure'],q['shear']))-failure.multipliers)/self.config.contact_multiplier_tolerance_n_mm2
            return float(np.hypot(force,defect))
        if self.options.volumetric_discretization=='quadratic_full':
            best=min(failures,key=rank)
            try:
                seed,warm=self.feasible_contact_predictor(previous,old_depth,depth,best.u,best.multipliers)
                result=self._mixed_step(previous,old_depth,depth,dt,old_state,initial=seed,initial_multipliers=warm)
                result[2]['contact_solver']+=' (active-surface predictor)'
                return result
            except ContactSolveError as exc:failures.append(exc)
            except FloatingPointError:pass
        # Retry geometry without changing the physical slip reference. Failed
        # secant/refinement/restart iterates remain available for branch search.
        print(f'RETRY {failures[-1]}; closest-surface displacement predictor',flush=True)
        try:
            seed=self.contact_predictor(previous,depth)
            result=self._mixed_step(previous,old_depth,depth,dt,old_state,initial=seed)
            result[2]['contact_solver']+=' (closest-surface predictor)'
            return result
        except ContactSolveError as exc:failures.append(exc)
        except FloatingPointError:pass
        # A local Newton branch can miss a feasible opening/sliding combination.
        # Condense the tangent mechanics and search bounded affine contact
        # branches. Each candidate must still solve the ORIGINAL nonlinear
        # problem, including raw-gap acceptance and the edge/dissipation audits.
        failure=min(failures,key=rank)
        if self.surface_contact is not None and self.options.volumetric_discretization!='quadratic_full':
            raise failure
        attempted=0
        for seed,warm,examined in branch_predictors(self,previous,old_depth,depth,dt,
                                                    failure.u,failure.multipliers):
            attempted+=1
            try:
                result=self._mixed_step(previous,old_depth,depth,dt,old_state,
                                        initial=seed,initial_multipliers=warm)
                result[2].update(contact_branch_predictors=attempted,contact_branch_combinations=examined)
                result[2]['contact_solver']+=' (contact branch predictor)'
                return result
            except ContactSolveError as exc:
                print(f'BRANCH RETRY {exc}',flush=True)
            except FloatingPointError as exc:
                # An invalid auxiliary seed is not the physical time step.
                # Reject it, keep the last valid failure state, and try another.
                print(f'BRANCH REJECT invalid deformation: {exc}',flush=True)
        raise failure

    def contact_predictor(self,previous,depth):
        seed = previous.copy()
        nodes=np.unique(self.edges)
        points = (self.mesh.reference_xy+previous.reshape(-1,2))[nodes]
        gap,normal = rounded_blade_distance(points,self.config,self.options,depth)
        seed.reshape(-1,2)[nodes] += np.maximum(gap,0)[:,None]*normal
        seed[self.fixed] = previous[self.fixed]
        return seed

    def fields(self,u):
        mesh,c = self.mesh,self.config
        F2 = np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[mesh.triangles],mesh.bulk_data[1])
        F = np.tile(np.eye(3),(len(F2),1,1)); F[:,:2,:2] = F2  # axes x,z,y
        J = np.linalg.det(F)
        B = F@F.transpose(0,2,1)
        stretches2,directions = np.linalg.eigh(B)
        stretch = np.sqrt(stretches2)
        cauchy = material_response(F2,c,tangent=False)['cauchy']
        volume_pressure=c.bulk_modulus_n_mm2*(J-1)
        if self.nodal_volume is not None:
            projected_pressure=self.nodal_volume.pressure(J)
            cauchy += (projected_pressure-volume_pressure)[:,None,None]*np.eye(3)
            volume_pressure=projected_pressure
        principal = np.einsum('eai,eab,ebi->ei',directions,cauchy,directions)
        # Principal nominal traction: equals engineering stress in homogeneous
        # uniaxial tension, unlike Cauchy stress. Catalogue strength is nominal.
        nominal = np.maximum(0,J[:,None]*principal/stretch)
        maxindex = nominal.argmax(axis=1)
        nominal_max = nominal[np.arange(len(nominal)),maxindex]
        first_piola,second_piola = stress_measures(F,cauchy)
        fields = dict(cauchy_principal_mpa=principal,principal_stretches=stretch,
                    volumetric_pressure_mpa=volume_pressure,
                    deformation_gradient_xzy=F,cauchy_stress_xzy_mpa=cauchy,
                    first_piola_stress_xzy_mpa=first_piola,second_piola_stress_xzy_mpa=second_piola,
                    cauchy_mean_tensile_stress_mpa=principal.mean(axis=1),
                    nominal_tensile_mpa=nominal_max,averaged_nominal_tensile_mpa=self.average@nominal_max,
                    tensile_direction_xzy=directions[np.arange(len(nominal)),:,maxindex],
                    reference_centroid_xz_mm=mesh.reference_xy[mesh.triangles].mean(axis=1),
                    current_centroid_xz_mm=(mesh.reference_xy+u.reshape(-1,2))[mesh.triangles].mean(axis=1),
                    reference_centroid_sn_mm=mesh.local_sn[mesh.triangles].mean(axis=1),J=J)
        if self.strength_surface is not None:
            index = self.strength_surface.index(second_piola)
            fields['multiaxial_strength_index'] = index
            fields['averaged_multiaxial_strength_index'] = self.average@index
        return fields


def make_problem(config,options,strength_surface=None):
    if options.volumetric_discretization in ('taylor_hood','quadratic_full'):
        from .quadratic import QuadraticProblem
        return QuadraticProblem(config,options,strength_surface)
    return IntactProblem(config,options,strength_surface)
