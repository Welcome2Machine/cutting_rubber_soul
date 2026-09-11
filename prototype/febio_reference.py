"""Independent FEBio finite-strain plane-strain contact reference.

Coordinates are (laboratory x, laboratory vertical, out-of-plane width).
This is an intact-material comparison, not a fracture implementation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace
import xml.etree.ElementTree as ET
import numpy as np
from run_support import write_json,source_hashes
from .model_config import FEMConfig,local_to_lab,_rotation_vectors
from .initiation import InitiationOptions,_blade_boundary_vertices,rounded_blade_distance


def tag(parent,element,value=None,**attrs):
    item=ET.SubElement(parent,element,{k:str(v) for k,v in attrs.items()})
    if value is not None:item.text=str(value)
    return item


def graded_axis(length,spacing,small,lo,hi):
    """Fine contact window with geometric grading to the background mesh."""
    lo=max(0.,lo);hi=min(length,hi)
    axis=list(np.linspace(lo,hi,max(1,int(np.ceil((hi-lo)/small)))+1))
    for sign,origin,limit in ((-1,lo,0.),(1,hi,length)):
        value=origin;step=small
        while sign*(limit-value)>1e-12:
            value+=sign*min(step,sign*(limit-value));axis.append(value)
            step=min(spacing,1.35*step)
    return np.unique(axis)


def export(path,config,options,*,case='patch',spacing=.25,depth=None,linear_solver='superlu_mt',surface_spacing=None,
           contact_variant='upstream',friction_penalty=None,mesh_depth=None,threads=2,
           contact_padding=None,blade_arc_segments=256,speed_profile='reference'):
    if speed_profile not in ('reference','balanced'):raise ValueError('Invalid speed profile')
    balanced=speed_profile=='balanced'
    if balanced and case=='oblique':
        config=replace(config,contact_penetration_tolerance_mm=2e-4)
        if surface_spacing is None:surface_spacing=.0025
    config.validate();options.validate(config)
    if type(threads) is not int or threads<1:raise ValueError('Threads must be a positive integer')
    if type(blade_arc_segments) is not int or blade_arc_segments<8:raise ValueError('Blade arc requires at least eight segments')
    if contact_padding is not None and (not np.isfinite(contact_padding) or contact_padding<0):
        raise ValueError('Contact padding must be finite and nonnegative')
    if case not in ('patch','oblique') or not np.isfinite(spacing) or spacing<=0:raise ValueError('Invalid reference case')
    patch=case=='patch';end=(.3 if patch else config.max_indentation_mm) if depth is None else depth
    if not 0<end<=(.5 if patch else config.max_indentation_mm):raise ValueError('Invalid reference loading depth')
    if contact_variant not in ('upstream','audit','corrected'):raise ValueError('Unknown contact variant')
    if contact_variant!='upstream' and (config.contact_surface_penalty_n_mm3!=250. or config.contact_multiplier_tolerance_n_mm2!=.001):
        raise ValueError('Isolated contact patch currently audits only rho=250 N/mm^3 and tolerance=.001 MPa')
    if mesh_depth is not None and (patch or not end<=mesh_depth<=config.max_indentation_mm):raise ValueError('Invalid mesh depth')
    grid_end=end if mesh_depth is None else mesh_depth
    if friction_penalty is None:friction_penalty=config.contact_surface_penalty_n_mm3 if contact_variant=='corrected' else 1.
    if not np.isfinite(friction_penalty) or friction_penalty<=0:raise ValueError('Invalid friction penalty')
    length,height=(2.,1.) if patch else (config.length_mm,config.height_mm)
    xs=np.linspace(0,length,int(np.ceil(length/spacing))+1)
    ys=np.linspace(0,height,int(np.ceil(height/spacing))+1)
    fine_window=None;padding=None
    if surface_spacing is not None:
        if patch or not np.isfinite(surface_spacing) or not 0<surface_spacing<=spacing:
            raise ValueError('Surface grading requires oblique case and 0 < surface spacing <= background spacing')
        es,_=_rotation_vectors(config.inclination_deg)
        # Laboratory downward motion projected into MATERIAL coordinates.
        # For +60 degrees, es=(cos(theta),-sin(theta)), hence ds is POSITIVE.
        travel=float(np.dot(np.array([0.,-grid_end]),es))
        # Extra material may enter contact through deformation. This buffer
        # is a mesh heuristic, not a kinematic bound; keep the live audit.
        padding=max(.1,.25*grid_end) if contact_padding is None else contact_padding
        margin=3*options.tip_radius_mm+.5*config.blade_thickness_mm*abs(es[0])+padding
        fine_window=[max(0.,config.drop_s_mm+min(0.,travel)-margin),
                     min(length,config.drop_s_mm+max(0.,travel)+margin)]
        xs=graded_axis(length,spacing,surface_spacing,*fine_window)
        ys=graded_axis(height,spacing,surface_spacing,height-surface_spacing,height)
    sn=np.array([(x,y) for y in ys for x in xs]);nx=len(xs);ny=len(ys);n=len(sn)
    xy=sn if patch else local_to_lab(sn,config)
    nodes=np.vstack((np.column_stack((xy,np.zeros(n))),np.column_stack((xy,np.full(n,config.width_mm)))))
    quads=np.array([[j*nx+i,j*nx+i+1,(j+1)*nx+i+1,(j+1)*nx+i] for j in range(ny-1) for i in range(nx-1)])
    elements=np.column_stack((quads,quads+n))+1
    # CCW in-plane boundary; tangent cross thickness points outward.
    outline=np.r_[np.arange(nx),np.arange(2*nx-1,n,nx),np.arange(n-2,n-nx-1,-1),np.arange(n-2*nx,0,-nx)]
    surface=np.array([[a,b,b+n,a+n] for a,b in zip(outline,np.roll(outline,-1))])+1
    if patch:
        surface=surface[np.all(np.isclose(nodes[surface-1,1],height),axis=1)]
    else:
        surface=surface[~np.all(np.isclose(sn[(surface[:,:2]-1)%n,1],0),axis=1)]
    if patch:
        cutter=np.array([[-1.,height],[3.,height],[3.,height+.5],[-1.,height+.5]])
        arc_error=0.
    else:
        polygon=_blade_boundary_vertices(config,options,0.)
        a=np.deg2rad(config.blade_tip_angle_deg/2);r=options.tip_radius_mm
        _,en=_rotation_vectors(config.inclination_deg)
        center=local_to_lab(np.array([[config.drop_s_mm,height]]),config)[0]+r*en
        # Inscribed faceting with a recorded geometric error below gap tolerance.
        angle=np.linspace(-np.pi+a,-a,blade_arc_segments+1)
        arc=center+r*np.column_stack((np.cos(angle),np.sin(angle)))
        cutter=np.vstack((polygon[:2],arc,polygon[-2:]))
        arc_error=float(r*(1-np.cos(np.diff(angle).max()/2)))
    m=len(cutter);offset=len(nodes)
    center=cutter.mean(axis=0);tool2=np.vstack((cutter,center));m2=m+1
    tool_nodes=np.vstack((np.column_stack((tool2,np.zeros(m2))),np.column_stack((tool2,np.full(m2,config.width_mm)))))
    tool_tri=np.array([[m,i,(i+1)%m] for i in range(m)])
    tool_elements=np.column_stack((tool_tri,tool_tri+m2))+offset+1
    tool_surface=np.array([[i,(i+1)%m,(i+1)%m+m2,i+m2] for i in range(m)])+offset+1
    # Only the exposed cutting face can meet the specimen. The back of the
    # finite rigid cap must not act as a second obstacle through its interior.
    tool_surface=tool_surface[:1] if patch else tool_surface[:-1]
    nodes=np.vstack((nodes,tool_nodes))
    root=ET.Element('febio_spec',version='4.0');tag(root,'Module',type='solid')
    control=tag(root,'Control');tag(control,'analysis','STATIC')
    increment=.01 if patch else .0025
    steps=max(10 if patch else 1,int(np.ceil(end/increment)))
    tag(control,'time_steps',steps);tag(control,'step_size',end/config.speed_mm_s/steps)
    # Relative displacement convergence is not an absolute penetration bound.
    # 1e-8 drove required corrections below coordinate roundoff in the graded
    # patch. Keep force/contact checks separate and audit geometry explicitly.
    solver=tag(control,'solver',type='solid');tag(solver,'dtol',1e-4 if balanced else 1e-5)
    tag(solver,'etol',1e-6 if balanced else 1e-8);tag(solver,'rtol',1e-8)
    # FEBio compares the SQUARED force norm to min_residual. A 1e-6 N
    # force floor can stop Newton before small contact patches have sufficiently
    # accurate traction for the requested augmentation tolerance.
    tag(solver,'min_residual',1e-16 if balanced else 1e-18);tag(solver,'max_refs',80);tag(solver,'reform_each_time_step',1)
    tag(solver,'symmetric_stiffness',0)
    qn=tag(solver,'qn_method',type='Broyden');tag(qn,'max_ups',0)
    if balanced and not patch:tag(control,'plot_level','PLOT_FINAL')
    ls=tag(solver,'linear_solver',type=linear_solver)
    if linear_solver=='superlu_mt':tag(ls,'nprocs',threads)
    if not patch:
        timer=tag(control,'time_stepper',type='default');tag(timer,'dtmin',1e-6)
        tag(timer,'dtmax',end/config.speed_mm_s/steps);tag(timer,'max_retries',12)
        # The previous opt_iter=20 controller counted 77--177 iterations
        # including augmentations, shrinking every SUCCESSFUL step despite
        # zero retries. Use the prescribed maximum after success; failed
        # attempts still go through FEBio's unchanged Retry/cutback path.
        tag(timer,'dtforce',1)
    material=tag(root,'Material');rubber=tag(material,'material',id=1,name='rubber',type='Mooney-Rivlin')
    tag(rubber,'c1',config.shear_modulus_n_mm2/2);tag(rubber,'c2',0)
    tag(rubber,'k',config.bulk_modulus_n_mm2);tag(rubber,'pressure_model',2)
    rigid=tag(material,'material',id=2,name='cutter',type='rigid body');tag(rigid,'density',1.)
    mesh=tag(root,'Mesh');nd=tag(mesh,'Nodes',name='all')
    for i,x in enumerate(nodes,1):tag(nd,'node',','.join(format(v,'.16g') for v in x),id=i)
    for name,kind,array,start in [('rubber','hex8',elements,1),('cutter','penta6',tool_elements,len(elements)+1)]:
        es=tag(mesh,'Elements',type=kind,name=name)
        for i,conn in enumerate(array,start):tag(es,'elem',','.join(map(str,conn)),id=i)
    bottom=np.r_[np.arange(nx),np.arange(nx)+n]+1
    sets={'rubber_nodes':np.arange(1,2*n+1),'bottom':bottom,'front':np.arange(1,n+1)}
    for name,indices in sets.items():tag(mesh,'NodeSet',','.join(map(str,indices)),name=name)
    tag(mesh,'ElementSet',','.join(map(str,range(1,len(elements)+1))),name='rubber_elements')
    for name,array in [('rubber_surface',surface),('cutter_surface',tool_surface)]:
        sf=tag(mesh,'Surface',name=name)
        for i,conn in enumerate(array,1):tag(sf,'quad4',','.join(map(str,conn)),id=i)
    pair=tag(mesh,'SurfacePair',name='contact_pair');tag(pair,'primary','rubber_surface');tag(pair,'secondary','cutter_surface')
    domains=tag(root,'MeshDomains');tag(domains,'SolidDomain',name='rubber',mat='rubber',three_field='true');tag(domains,'SolidDomain',name='cutter',mat='cutter')
    boundary=tag(root,'Boundary')
    def fix(name,axes):
        bc=tag(boundary,'bc',type='zero displacement',node_set=name)
        for axis in 'xyz':tag(bc,axis+'_dof',int(axis in axes))
    fix('rubber_nodes','xz' if patch else 'z');fix('bottom','y' if patch else 'xy')
    # Equality constraints remove all variation through the width. Fixing only
    # out-of-plane displacement would not impose strict plane strain.
    for i in range(n):
        if i<nx:continue
        for axis in ('y',) if patch else ('x','y'):
            bc=tag(boundary,'bc',type='linear constraint');tag(bc,'node',i+n+1);tag(bc,'dof',axis)
            child=tag(bc,'child_dof');tag(child,'node',i+1);tag(child,'dof',axis);tag(child,'value',1.)
    rigid_bc=tag(root,'Rigid');fixed=tag(rigid_bc,'rigid_bc',type='rigid_fixed');tag(fixed,'rb',2)
    for axis in ('Rx','Ry','Rz','Ru','Rv','Rw'):tag(fixed,axis+'_dof',int(axis not in (('Rx','Ry') if patch else ('Ry',))))
    for axis,value in ([('y',-end),('x',1.)] if patch else [('y',-end)]):
        prescribed=tag(rigid_bc,'rigid_bc',type='rigid_displacement');tag(prescribed,'rb',2);tag(prescribed,'dof',axis);tag(prescribed,'value',value,lc=1)
    contact=tag(tag(root,'Contact'),'contact',type='sliding-node-on-facet',surface_pair='contact_pair')
    for name,value in dict(laugon=1,tolerance=1e-5 if balanced else 1e-6,gaptol=1e-6,penalty=10,auto_penalty=1,
        update_penalty=1,two_pass=0,search_radius=2.,node_reloc=0,fric_penalty=friction_penalty,
        fric_coeff=config.friction_coefficient,minaug=1,maxaug=50).items():tag(contact,name,value)
    if contact_variant!='upstream':
        tag(contact,'contact_audit',1)
        tag(contact,'consistent_tangent',int(contact_variant=='corrected'))
        tag(contact,'strict_augmentation',int(contact_variant=='corrected'))
    lc=tag(tag(root,'LoadData'),'load_controller',id=1,type='loadcurve');tag(lc,'interpolate','LINEAR')
    points=tag(lc,'points');tag(points,'pt','0,0');tag(points,'pt',f'{end/config.speed_mm_s},1')
    output=tag(root,'Output');plot=tag(output,'plotfile',type='febio')
    for name in ('displacement','stress','relative volume','contact pressure','contact gap','reaction forces'):tag(plot,'var',type=name)
    log=tag(output,'logfile')
    tag(log,'node_data',data='x;y;z;ux;uy;uz;Rx;Ry;Rz',file='nodes.txt',node_set='rubber_nodes')
    # The generic nodal-contact face traction record contains normal traction
    # only. Use rigid actuator reactions to audit friction, not that record.
    if not balanced or patch:
        tag(log,'face_data',data='contact gap;contact pressure',file='contact.txt',surface='rubber_surface')
        tag(log,'element_data',data='sx;sy;sz;sxy;J',file='elements.txt',elem_set='rubber_elements')
    # FEBio indexes this record by MATERIAL id, not rigid-body ordinal.
    tag(log,'rigid_body_data','2',data='x;y;z;Fx;Fy;Fz',file='cutter.txt')
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return dict(case=case,config=config.to_dict(),options=vars(options),target_depth_mm=end,threads=threads,
        speed_profile=speed_profile,target_increment_mm=end/steps,max_broyden_updates=0,
        output_policy='every accepted node and tool state; plot only final; no duplicate face/element text' if balanced and not patch else 'full fields at every accepted step',
        plane_strain='all uz=0 and corresponding front/back ux,uy equal',
        coordinates='FEBio x,y,z = laboratory x,vertical,out-of-plane',
        material='C1=G/2,C2=0,pressure_model=2: G/2*(I1bar-3)+K/2*(J-1)^2',
        body_elements=len(elements),body_nodes=2*n,spacing_mm=spacing,blade_faceting_error_mm=arc_error,
        surface_spacing_mm=surface_spacing,
        contact_fine_window_s_mm=fine_window,contact_padding_mm=padding,
        blade_arc_segments=0 if patch else blade_arc_segments,
        mesh_target_depth_mm=grid_end,contact_variant=contact_variant,friction_penalty_n_mm3=friction_penalty,
        step_control='fixed patch steps' if patch else 'dtforce=1 after convergence; failed-step retry remains enabled',
        bulk_discretization='hex8 three-field: element-averaged volumetric pressure, pointwise deviatoric stress; differs from mainline P2 full integration',
        contact_formulation='FEBio sliding-node-on-facet, augmented Lagrangian, Coulomb',
        patch_horizontal_displacement_mm=1. if patch else None,
        scope='independent intact contact reference; no fracture or validated nucleation')


def records(path):
    """Read FEBio text records without relying on Windows newline conventions."""
    step=None;rows=[];time_value=None
    with Path(path).open(encoding='utf-8',errors='strict') as stream:
        for line in stream:
            line=line.strip()
            if not line:continue
            if line.startswith('*Step'):
                if step is not None:yield step,time_value,np.asarray(rows)
                step=int(line.split('=')[1]);rows=[];time_value=None
            elif line.startswith('*Time'):time_value=float(line.split('=')[1])
            elif not line.startswith('*'):
                values=np.fromstring(line,sep=' ')
                if not np.isfinite(values).all():raise ValueError('Non-finite FEBio output')
                rows.append(values)
    if step is not None:yield step,time_value,np.asarray(rows)


def contact_log_audit(text):
    """Separate accepted augmentations from failed attempts and their retries."""
    accepted=[];index=0;local=None;tangents=[]
    for line in text.splitlines(keepends=True):
        if not line.endswith('\n'):continue  # timed-out writer may leave a partial record
        match=re.search(r'augmentation #\s*(\d+)',line)
        if match:index=int(match[1])
        if line.startswith('CONTACT_AUDIT '):
            local={k:float(v) for k,v in re.findall(r'(\w+)=([^ ]+)',line)}
        if line.startswith('CONTACT_TANGENT '):
            tangents.append({k:float(v) for k,v in re.findall(r'(\w+)=([^ ]+)',line)})
        match=re.search(r'converged at time\s*:\s*([\d.eE+-]+)',line)
        if match:
            time_value=float(match[1]);row=dict(time_s=time_value,augmentation_index=index)
            # The convergence message uses six significant digits; the audit
            # retains 12. An absolute 1e-6 test incorrectly drops long-path data.
            if local is not None and np.isclose(local['t'],time_value,rtol=5.1e-6,atol=1e-12):
                row.update(time_s=local['t'],normal_projection_mpa=local['dn'],tangential_projection_mpa=local['dt'],
                           max_slip_mm=local['slip'])
            accepted.append(row);local=None;index=0
        if 'Retrying time step' in line:local=None;index=0
    return dict(accepted_steps=accepted,tangent_checks=tangents,
                max_accepted_augmentation_index=max((s['augmentation_index'] for s in accepted),default=0),
                projection_scale_n_mm3=250.,scope='actual nodal contact/history; excludes whole-boundary geometry and bulk residual')


def solver_log_text(folder):
    """Use one complete stream, never concatenate duplicated solver events.

    FEBio may flush redirected stdout before its own log. Killing the owned
    process after a geometry violation must not lose the last accepted event.
    """
    candidates=[]
    for name in ('model.log','stdout.log'):
        path=Path(folder)/name
        if path.exists():
            text=path.read_text(errors='replace')
            # Exclude a line interrupted by process termination.
            text=text[:text.rfind('\n')+1]
            audit=contact_log_audit(text)
            candidates.append((len(audit['accepted_steps']),len(audit['tangent_checks']),len(text),name,text))
    if not candidates:return '',None
    selected=max(candidates,key=lambda item:item[:3])
    return selected[4],selected[3]


def validate_patch(folder,metadata):
    """Analytic constrained compression + full sliding, checked at every step.

    This homogeneous patch has F=diag(1,1-depth,1). The actuator forces,
    sigma_yy and geometric gap have independent closed-form targets. This is
    not an oblique mesh-convergence or crack-initiation certificate.
    """
    folder=Path(folder);c=metadata['config'];depth=metadata['target_depth_mm']
    streams=[list(records(folder/name)) for name in ('cutter.txt','elements.txt','nodes.txt','contact.txt')]
    signature=[(s,t) for s,t,_ in streams[0]]
    if len(signature)<2 or any([(s,t) for s,t,_ in stream]!=signature for stream in streams):
        raise ValueError('Missing or mismatched FEBio record steps')
    force_error=stress_error=volume_error=penetration=plane_error=cone_error=0.
    min_j=float('inf');dissipation=0.;previous_x=0.;previous_fx=0.;motion_error=0.
    ref=ET.parse(folder/'model.feb').getroot()
    nodes=np.array([np.fromstring(v.text,sep=',') for v in ref.findall('Mesh/Nodes/node')])
    initial=nodes[:metadata['body_nodes']]
    final_force=None
    for tool,solid,body,contact in zip(*streams):
        _,t,rigid=tool;d=t*c['speed_mm_s'];j=1-d
        if j<=0 or rigid.shape!=(1,7):raise ValueError('Invalid patch geometry or rigid record')
        shear=c['shear_modulus_n_mm2'];bulk=FEMConfig(**c).bulk_modulus_n_mm2
        sy=shear*j**(-5/3)*(j*j-(2+j*j)/3)+bulk*(j-1)
        expected_fy=sy*2*c['width_mm'];expected_fx=-c['friction_coefficient']*expected_fy
        fx,fy=rigid[0,4:6];final_force=[float(fx),float(fy)]
        force_error=max(force_error,abs(fx-expected_fx),abs(fy-expected_fy))
        cone_error=max(cone_error,abs(fx)-c['friction_coefficient']*abs(fy))
        stress_error=max(stress_error,float(np.max(abs(solid[2][:,2]-sy))))
        stress_error=max(stress_error,float(np.max(abs(contact[2][:,2]+sy))))
        volume_error=max(volume_error,float(np.max(abs(solid[2][:,5]-j))))
        min_j=min(min_j,float(np.min(solid[2][:,5])))
        data=body[2]
        if data.shape!=(len(initial),10) or not np.array_equal(data[:,0],np.arange(1,len(initial)+1)):
            raise ValueError('Incomplete patch nodal output')
        expected=initial.copy();expected[:,1]*=j
        plane_error=max(plane_error,float(np.max(abs(data[:,1:4]-expected))))
        # All top edges are linear and the platen is flat. Endpoint maxima
        # therefore certify the entire boundary here, unlike face averages.
        penetration=max(penetration,float(np.max(data[np.isclose(initial[:,1],1),2]-(1-d))))
        x=rigid[0,1]-1.
        motion_error=max(motion_error,abs(x-d/depth*metadata['patch_horizontal_displacement_mm']),
            abs(rigid[0,2]-(1.25-d)))
        dissipation+=.5*(fx+previous_fx)*(x-previous_x)
        previous_x=x;previous_fx=fx
    reached=abs(signature[-1][1]*c['speed_mm_s']-depth)<1e-10
    result=dict(all_passed=bool(reached and force_error<=.001 and stress_error<=1e-4 and volume_error<=1e-5
        and penetration<=c['contact_penetration_tolerance_mm'] and plane_error<=1e-5
        and cone_error<=.001 and min_j>0 and dissipation>=0 and motion_error<1e-9),
        reached_target=reached,steps=len(signature)-1,max_force_error_n=force_error,
        max_sigma_yy_error_mpa=stress_error,max_J_error=volume_error,min_J=min_j,
        whole_boundary_penetration_mm=penetration,max_affine_displacement_error_mm=plane_error,
        max_force_cone_excess_n=cone_error,friction_work_n_mm=float(dissipation),
        max_rigid_motion_error_mm=float(motion_error),
        final_actuator_force_xy_n=final_force,scope='analytic finite compression/sliding patch only')
    write_json(folder/'validation.json',result);return result


def boundary_penetration_bounds(a,b,config,options,depth,accuracy=1e-6):
    """Certified signed-distance maximum on straight edges (1-Lipschitz)."""
    ga=rounded_blade_distance(a,config,options,depth)[0]
    gb=rounded_blade_distance(b,config,options,depth)[0]
    lower=max(0.,float(max(ga.max(),gb.max())))
    settled_upper=0.
    for _ in range(40):
        upper=(ga+gb+np.linalg.norm(b-a,axis=1))/2
        refine=upper>lower+accuracy
        settled_upper=max(settled_upper,float(upper[~refine].max(initial=0)))
        if not np.any(refine):return lower,max(lower,settled_upper)
        a=a[refine];b=b[refine];ga=ga[refine];gb=gb[refine]
        middle=(a+b)/2;gm=rounded_blade_distance(middle,config,options,depth)[0]
        lower=max(lower,float(gm.max()))
        a,b=np.concatenate((a,middle)),np.concatenate((middle,b))
        ga,gb=np.r_[ga,gm],np.r_[gm,gb]
    raise ValueError('Boundary penetration certificate did not resolve')


def plane_strain_hex_forces(reference,current,quads,config):
    """Independent integration of the exported Hex8 three-field material.

    Front/back equality reduces the extrusion exactly to four in-plane Gauss
    points with the physical width as weight. Volumetric pressure uses mean J;
    the deviatoric stress uses each actual F. No FEBio reaction data is used.
    """
    signs=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    xi=signs/np.sqrt(3)
    dn=np.empty((4,4,2))
    dn[:,:,0]=.25*signs[None,:,0]*(1+xi[:,None,1]*signs[None,:,1])
    dn[:,:,1]=.25*signs[None,:,1]*(1+xi[:,None,0]*signs[None,:,0])
    ref_jac=np.einsum('eai,qaj->eqij',reference[quads],dn)
    weight=np.linalg.det(ref_jac)*config.width_mm
    if np.any(weight<=0):raise ValueError('Invalid reference cell orientation')
    gradient=np.einsum('qai,eqij->eqaj',dn,np.linalg.inv(ref_jac))
    f=np.einsum('eai,eqaj->eqij',current[quads],gradient)
    j=np.linalg.det(f)
    if np.any(j<=0):raise ValueError('Nonpositive volume in force reconstruction')
    mean_j=np.sum(j*weight,axis=1)/weight.sum(axis=1)
    pressure=config.bulk_modulus_n_mm2*(mean_j-1)
    factor=config.shear_modulus_n_mm2*j**(-2/3)
    i1=np.sum(f*f,axis=(2,3))+1
    p=factor[:,:,None,None]*f+(pressure[:,None]*j-factor*i1/3)[:,:,None,None]*np.linalg.inv(f).swapaxes(-1,-2)
    element_force=np.einsum('eqij,eqaj,eq->eai',p,gradient,weight)
    forces=np.zeros_like(reference,dtype=float)
    np.add.at(forces,quads.reshape(-1),element_force.reshape(-1,2))
    energy=float(np.sum(weight*(config.shear_modulus_n_mm2/2*(j**(-2/3)*i1-3)))
        +np.sum(weight.sum(axis=1)*config.bulk_modulus_n_mm2/2*(mean_j-1)**2))
    return forces,energy


def audit_oblique_state(root,metadata,record,tool_record,cache=None):
    """Audit one synchronized accepted state using the original constitutive law."""
    step,time_value,data=record
    if cache is None:cache={}
    def cached(name,build):
        if name not in cache:cache[name]=build()
        return cache[name]
    c=FEMConfig(**metadata['config']);o=InitiationOptions(**metadata['options'])
    count=metadata['body_nodes']
    if (not np.isfinite(time_value) or time_value<0 or not np.all(np.isfinite(data))
            or tool_record.shape!=(1,7) or tool_record[0,0]!=2
            or not np.all(np.isfinite(tool_record))):
        raise ValueError('Invalid accepted state or rigid reaction')
    ref=cached('reference',lambda:np.array([np.fromstring(v.text,sep=',') for v in root.findall('Mesh/Nodes/node')]))
    count=metadata['body_nodes'];half=count//2
    if data.shape!=(count,10) or not np.array_equal(data[:,0],np.arange(1,count+1)):
        raise ValueError('Last accepted nodal record is incomplete')
    # Default FEBio text records have 12 significant digits. At coordinates
    # O(10 mm), subtracting rounded current coordinates across 0.0025 mm
    # cells creates a false force residual. Small displacements retain many
    # more absolute digits; reconstruct from the full-precision input instead.
    xy=ref[:count,:2]+data[:,4:6];z=ref[:count,2]+data[:,6]
    edges=cached('edges',lambda:np.array([np.fromstring(v.text,sep=',',dtype=int)[:2]-1
        for v in root.findall("Mesh/Surface[@name='rubber_surface']/quad4")]))
    depth=time_value*c.speed_mm_s
    lower,upper=boundary_penetration_bounds(xy[edges[:,0]],xy[edges[:,1]],c,o,depth)
    ids=cached('quads',lambda:np.array([np.fromstring(v.text,sep=',',dtype=int)[:4]-1
        for v in root.findall("Mesh/Elements[@name='rubber']/elem")]))
    # Q1 planar Jacobian is affine in natural coordinates; its exact minimum
    # on a quadrilateral occurs at a corner. Reference cells are rectangles.
    ratios=[]
    for corner in range(4):
        next_id=(corner+1)%4;prev=(corner-1)%4
        a=xy[ids[:,next_id]]-xy[ids[:,corner]];b=xy[ids[:,prev]]-xy[ids[:,corner]]
        ar=ref[ids[:,next_id],:2]-ref[ids[:,corner],:2];br=ref[ids[:,prev],:2]-ref[ids[:,corner],:2]
        ratios.append((a[:,0]*b[:,1]-a[:,1]*b[:,0])/(ar[:,0]*br[:,1]-ar[:,1]*br[:,0]))
    min_j=float(np.min(ratios));plane_error=float(max(np.max(abs(xy[:half]-xy[half:])),np.max(abs(z-ref[:count,2]))))
    forces,energy=plane_strain_hex_forces(ref[:half,:2],xy[:half],ids,c)
    bottom=cached('bottom',lambda:np.fromstring(root.find("Mesh/NodeSet[@name='bottom']").text,sep=',',dtype=int)-1)
    bottom=bottom[bottom<half]
    # Side edges include clamped bottom corners; their reactions belong to the
    # fixture, not to blade contact on the free boundary.
    boundary=np.setdiff1d(np.unique(edges),bottom)
    interior=np.setdiff1d(np.arange(half),np.union1d(boundary,bottom))
    interior_residual=float(np.linalg.norm(forces[interior]))
    # FEBio rigid reactions are actuator loads, same sign as the net load on
    # the rubber. Sum the independent material forces on its free boundary.
    force_balance=float(np.linalg.norm(forces[boundary].sum(axis=0)-tool_record[0,4:6]))
    necessary_force_checks=interior_residual<=c.newton_tolerance_n and force_balance<=c.newton_tolerance_n
    result=dict(step=step,time_s=time_value,last_depth_mm=depth,compressive_fz_n=float(-tool_record[0,5]),whole_boundary_penetration_lower_mm=lower,
        whole_boundary_penetration_upper_mm=upper,min_corner_J=min_j,plane_strain_error_mm=plane_error,
        geometry_passed=bool(upper<=c.contact_penetration_tolerance_mm and min_j>0 and plane_error<=1e-8),
        interior_force_residual_n=interior_residual,independent_tool_force_error_n=force_balance,
        bulk_energy_n_mm=energy,necessary_force_checks_passed=bool(necessary_force_checks),
        mainline_acceptance_verified=False,scope='accepted-state geometry and necessary force checks; full local friction equilibrium and initiation unverified')
    return result


class RecordTail:
    """Read complete FEBio records without mistaking a buffered tail for a state."""
    def __init__(self,path,rows,width):
        self.path=Path(path);self.rows=rows;self.width=width
        self.offset=0;self.partial=b'';self.step=None;self.time=None;self.lines=[]
        self.emitted=False;self.previous=-1

    def poll(self,final=False):
        if not self.path.exists():
            if final:raise ValueError(f'Missing history: {self.path.name}')
            return []
        with self.path.open('rb') as stream:
            stream.seek(self.offset);chunk=stream.read();self.offset=stream.tell()
        lines=(self.partial+chunk).split(b'\n');self.partial=lines.pop();result=[]
        for raw in lines:
            line=raw.decode('utf-8').strip()
            if not line:continue
            if line.startswith('*Step'):
                if self.step is not None and not self.emitted:raise ValueError('Truncated accepted record')
                self.step=int(line.split('=')[1])
                if self.step!=self.previous+1:raise ValueError('Nonconsecutive accepted steps')
                self.previous=self.step;self.time=None;self.lines=[];self.emitted=False
            elif line.startswith('*Time'):
                self.time=float(line.split('=')[1])
            elif not line.startswith('*'):
                if self.step is None or self.time is None or self.emitted:raise ValueError('Unexpected history row')
                # Validate each row; otherwise a short and long row can cancel.
                tokens=line.split()
                if len(tokens)!=self.width:raise ValueError('Invalid history row width')
                self.lines.append(line)
                if len(self.lines)==self.rows:
                    data=np.array([float(v) for row in self.lines for v in row.split()]).reshape(self.rows,self.width)
                    if not np.all(np.isfinite(data)) or not np.isfinite(self.time):raise ValueError('Nonfinite history')
                    result.append((self.step,self.time,data));self.emitted=True;self.lines=[]
        if final and (self.partial.strip() or self.step is None or not self.emitted):
            raise ValueError(f'Incomplete final history: {self.path.name}')
        return result


class ObliquePathAudit:
    """Stream synchronized states and retain failures anywhere along the path."""
    def __init__(self,folder,metadata):
        self.folder=Path(folder);self.metadata=metadata
        self.root=ET.parse(self.folder/'model.feb').getroot()
        self.tails=[RecordTail(self.folder/'nodes.txt',metadata['body_nodes'],10),
            RecordTail(self.folder/'cutter.txt',1,7)]
        self.pending=[{},{}];self.rows=[];self.summary=None;self.cache={}

    def poll(self,final=False):
        for tail,pending in zip(self.tails,self.pending):
            for record in tail.poll(final):pending[record[0]]=record
        new=[]
        for step in sorted(self.pending[0].keys() & self.pending[1].keys()):
            node=self.pending[0].pop(step);tool=self.pending[1].pop(step)
            if abs(node[1]-tool[1])>1e-10:raise ValueError('Nodal and rigid histories have different times')
            if self.rows and node[1]<=self.rows[-1]['time_s']:raise ValueError('Nonincreasing accepted time')
            audit_started=time.perf_counter()
            try:row=audit_oblique_state(self.root,self.metadata,node,tool[2],cache=self.cache)
            except Exception as error:
                write_json(self.folder/'audit_failure.json',dict(step=step,time_s=node[1],
                    depth_mm=node[1]*self.metadata['config']['speed_mm_s'],
                    error=f'{type(error).__name__}: {error}'))
                raise
            row['audit_seconds']=time.perf_counter()-audit_started
            self.rows.append(row);new.append(row)
            path=self.folder/'path_audit.csv'
            with path.open('w' if len(self.rows)==1 else 'a',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row))
                if len(self.rows)==1:writer.writeheader()
                writer.writerow(row)
            self.summary=dict(row,audited_states=len(self.rows),audited_loaded_steps=sum(r['time_s']>0 for r in self.rows),
                geometry_passed=all(r['geometry_passed'] for r in self.rows),
                necessary_force_checks_passed=all(r['necessary_force_checks_passed'] for r in self.rows),
                path_max_penetration_upper_mm=max(r['whole_boundary_penetration_upper_mm'] for r in self.rows),
                path_min_corner_J=min(r['min_corner_J'] for r in self.rows),
                first_failed_depth_mm=next((r['last_depth_mm'] for r in self.rows
                    if not r['geometry_passed'] or not r['necessary_force_checks_passed']),None))
            write_json(self.folder/'geometry_audit.json',self.summary)
        if final and (any(self.pending) or not self.rows):raise ValueError('Unmatched or empty accepted histories')
        return new

    def verify_coverage(self,accepted):
        times=[r['time_s'] for r in self.rows if r['time_s']>0]
        expected=[r['time_s'] for r in accepted]
        # FEBio convergence messages print fewer digits than nodal records.
        if len(times)!=len(expected) or any(not np.isclose(t,e,
                rtol=5.1e-9 if 'normal_projection_mpa' in row else 5.1e-6,atol=1e-9)
                for t,e,row in zip(times,expected,accepted)):
            raise ValueError('Geometry history does not cover every accepted solver step')


def audit_oblique_geometry(folder,metadata):
    audit=ObliquePathAudit(folder,metadata);audit.poll(final=True)
    return audit.summary


def run_guarded(executable,out,env,timeout,audit,status):
    """The worker monitors its own child; the interactive agent need not wait."""
    started=time.monotonic()
    with (out/'stdout.log').open('wb') as log:
        process=subprocess.Popen([str(executable),'-i','model.feb','-dump','restart.dmp','-nosplash'],
            cwd=out,stdout=log,stderr=subprocess.STDOUT,env=env)
        status['solver_pid']=process.pid;write_json(out/'status.json',status)
        try:
            while True:
                ended=process.poll() is not None
                if audit is not None:
                    for row in audit.poll(final=ended):
                        print('AUDIT '+json.dumps(row),flush=True)
                        status.update(last_depth_mm=row['last_depth_mm'],audited_states=len(audit.rows),
                            audit_seconds_total=sum(r['audit_seconds'] for r in audit.rows))
                        write_json(out/'status.json',status)
                    if audit.summary and (not audit.summary['geometry_passed'] or
                            not audit.summary['necessary_force_checks_passed']):
                        raise RuntimeError(f"Independent path audit failed at {audit.summary['first_failed_depth_mm']} mm")
                if ended:return process.returncode
                if timeout is not None and time.monotonic()-started>timeout:raise TimeoutError('Solver wall-clock limit exceeded')
                time.sleep(.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


def launch_detached(args,out):
    if not args.execute:raise ValueError('--detach requires --execute')
    (args.executable or default_executable(args.contact_variant)).resolve(strict=True)
    command=[sys.executable,'-u','-m','prototype.febio_reference','--output',str(out/'case')]
    for key,value in vars(args).items():
        if key in ('detach','output') or value is None:continue
        if isinstance(value,bool):
            if value:command.append('--'+key.replace('_','-'))
        else:command.extend(['--'+key.replace('_','-'),str(value)])
    kwargs={}
    if os.name=='nt':
        info=subprocess.STARTUPINFO();info.dwFlags|=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
        kwargs.update(startupinfo=info,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    else:kwargs['start_new_session']=True
    with (out/'driver.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,**kwargs)
    result=dict(state='launched',pid=process.pid,command=command,log=str(out/'driver.log'),
        status=str(out/'case'/'status.json'),completed=False)
    write_json(out/'launch.json',result);print(json.dumps(result))


def default_executable(variant):
    return Path(__file__).resolve().parents[1]/'.runtime'/('febio' if variant=='upstream' else 'febio_audit')/'bin'/('febio4.exe' if os.name=='nt' else 'febio4')


def solver_environment(threads,executable):
    if type(threads) is not int or threads<1:raise ValueError('Threads must be a positive integer')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if os.name!='nt':
        lib=str(executable.parent.parent/'lib')
        env['LD_LIBRARY_PATH']=lib+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('patch','oblique'),default='patch')
    parser.add_argument('--speed-profile',choices=('reference','balanced'),default='reference')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,default=.25)
    parser.add_argument('--surface-spacing',type=float,help='Graded oblique contact window, mm; requires geometry audit')
    parser.add_argument('--contact-padding',type=float,help='Additional material-coordinate mesh buffer; default max(0.1, 0.25*mesh depth) mm')
    parser.add_argument('--blade-arc-segments',type=int,default=256,help='Circular blade discretization; radius and wedge angle unchanged')
    parser.add_argument('--depth',type=float)
    parser.add_argument('--mesh-depth',type=float,help='Keep the same mesh window when truncating a reference path')
    parser.add_argument('--contact-variant',choices=('upstream','audit','corrected'),default='upstream')
    parser.add_argument('--friction-penalty',type=float,help='Diagnostic numerical tangential penalty, N/mm^3')
    parser.add_argument('--solver',choices=('superlu_mt','LU'),default='superlu_mt')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--detach',action='store_true',help='Launch a hidden local worker with persistent logs')
    parser.add_argument('--timeout',type=float,help='Wall-clock cap for a bounded diagnostic, seconds')
    parser.add_argument('--executable',type=Path)
    parser.add_argument('--threads',type=int,default=2,help='OpenMP and SuperLU_MT threads; match the allocated CPUs')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    if args.detach:
        launch_detached(args,out);return
    metadata=export(out/'model.feb',FEMConfig.from_json('prototype/initiation_config.json'),InitiationOptions(),
        case=args.case,spacing=args.spacing,depth=args.depth,linear_solver=args.solver,surface_spacing=args.surface_spacing,
        contact_variant=args.contact_variant,friction_penalty=args.friction_penalty,mesh_depth=args.mesh_depth,threads=args.threads,
        contact_padding=args.contact_padding,blade_arc_segments=args.blade_arc_segments,speed_profile=args.speed_profile)
    packages=('prototype','mechanics') if args.contact_variant=='upstream' else ('prototype','mechanics','tools')
    metadata['source_sha256']=source_hashes(packages,snapshot=out/'source_snapshot');write_json(out/'config.json',metadata)
    status=dict(completed=False,state='exported',experimental_validation=False,
        mainline_acceptance_verified=False,pid=os.getpid());write_json(out/'status.json',status)
    if args.execute:
        started=time.monotonic()
        path_audit=None
        try:
            executable=(args.executable or default_executable(args.contact_variant)).resolve(strict=True)
            env=solver_environment(args.threads,executable)
            metadata['execution_environment']=dict(platform=platform.platform(),machine=platform.machine(),
                processor=platform.processor(),python=platform.python_version(),
                variables={k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
            libraries=list(executable.parent.glob('*.dll'))+list(executable.parent.glob('*.so*'))
            libraries+=list((executable.parent.parent/'lib').glob('*.so*'))
            metadata['solver_binary_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [executable,*libraries] if p.is_file()}
            manifest=executable.parent.parent/'manifest.json'
            if manifest.exists():metadata['solver_build']=json.loads(manifest.read_text())
            write_json(out/'config.json',metadata)
            status.update(state='running',executable=str(executable));write_json(out/'status.json',status)
            if args.case=='oblique':path_audit=ObliquePathAudit(out,metadata)
            returncode=run_guarded(executable,out,env,args.timeout,path_audit,status)
            text,status['audit_log_source']=solver_log_text(out)
            normal='N O R M A L   T E R M I N A T I O N' in text and returncode==0
            status.update(numerically_completed=normal,completed=normal,
                state='numerically_completed' if normal else 'failed',returncode=returncode)
            # Upstream forces augmentation convergence at maxaug. Do not
            # mistake this budget escape for proof of satisfying constraints.
            contact_audit=contact_log_audit(text)
            # Printed augmentation index is naug+1; maxaug=50 forces acceptance
            # at printed index 51, not at the legitimately converged index 50.
            if normal and contact_audit['max_accepted_augmentation_index']>=51:
                status.update(completed=False,state='augmentation_limit_unverified')
            if normal and args.case=='patch':
                validation=validate_patch(out,metadata)
                passed=validation['all_passed'] and status['completed']
                status.update(completed=passed,patch_verified=passed,
                    state='patch_verified' if passed else 'validation_failed')
        except Exception as error:
            status.update(completed=False,state='failed',error=f'{type(error).__name__}: {error}')
        finally:
            text,status['audit_log_source']=solver_log_text(out)
            contact_audit=contact_log_audit(text)
            status['max_accepted_augmentation_index']=contact_audit['max_accepted_augmentation_index']
            if args.contact_variant!='upstream':
                write_json(out/'contact_audit.json',contact_audit)
                accepted=contact_audit['accepted_steps'];tol=metadata['config']['contact_multiplier_tolerance_n_mm2']
                passed=bool(accepted) and all(s.get('normal_projection_mpa',float('inf'))<=tol and
                    s.get('tangential_projection_mpa',float('inf'))<=tol for s in accepted)
                status['accepted_nodal_contact_checks_passed']=passed
                if not passed:status.update(completed=False,state='contact_validation_failed')
            status['max_augmentation_index']=max(
                (int(v) for v in re.findall(r'augmentation #\s*(\d+)',text)),default=0)
            status['failed_step_retries']=len(re.findall(r'Retrying time step',text,re.IGNORECASE))
            status['matrix_reformations']=text.count('===== reforming stiffness matrix:')
            iterations=[int(v) for v in re.findall(r'number of iterations\s*:\s*(\d+)',text)]
            status['mean_accepted_iterations']=float(np.mean(iterations)) if iterations else None
            archive=out/'restart.dmp'
            status['restart_archive_written']=archive.exists() and archive.stat().st_size>0
            status['restart_roundtrip_verified']=False
            if args.case=='oblique' and (out/'nodes.txt').exists():
                try:
                    if path_audit is None:raise ValueError('Path auditor was not initialized')
                    path_audit.poll(final=True)
                    path_audit.verify_coverage(contact_audit['accepted_steps'])
                    audit=path_audit.summary
                    status['last_depth_mm']=audit['last_depth_mm'];status['geometry_passed']=audit['geometry_passed']
                    if not audit['geometry_passed']:
                        status.update(completed=False,state='geometry_validation_failed')
                    status['necessary_force_checks_passed']=audit['necessary_force_checks_passed']
                    if not audit['necessary_force_checks_passed']:
                        status.update(completed=False,state='force_validation_failed')
                except Exception as error:
                    status.update(completed=False,audit_error=f'{type(error).__name__}: {error}')
            status['wall_seconds']=time.monotonic()-started;write_json(out/'status.json',status)
    print(json.dumps(status))
    if args.execute and not status['completed']:raise SystemExit(1)


if __name__=='__main__':main()
