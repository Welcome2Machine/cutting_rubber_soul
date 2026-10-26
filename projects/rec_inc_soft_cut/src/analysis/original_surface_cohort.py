"""Fixed identities originating on the pre-cut inclined free surface.

Contact is a later, separately evaluated image observation. Proximity to the
blade can never confer original-surface membership on an interior point.
"""
import numpy as np


def _labels_at(labels,xy):
    q=np.asarray(xy);ij=np.rint(np.nan_to_num(q,nan=-10000)).astype(int)
    h,w=labels.shape;inside=np.isfinite(q).all(-1)&(ij[...,0]>=0)&(ij[...,0]<w)&(ij[...,1]>=0)&(ij[...,1]<h)
    out=np.full(q.shape[:-1],-1,np.int16)
    out[inside]=labels[ij[...,1][inside],ij[...,0][inside]]
    return out


def original_surface_stations(labels,center,tangent,contact_x,*,half_span=180.,pitch=3.):
    """Find the original air/material interface before any cutting occurs.

    Two near-surface rows (1.5/4.5 px inward) and two corresponding interior
    rows (13.5/16.5 px inward) share each reference surface station. Actual
    mathematical boundary points are not presented as observed texture nodes.
    """
    c=np.asarray(center,float);t=np.asarray(tangent,float);t/=np.linalg.norm(t)
    if t[0]<0:t=-t
    n=np.array([-t[1],t[0]])
    y,x=np.nonzero(labels==1)
    if not len(x):raise ValueError('No original material')
    if np.median((np.column_stack((x,y))-c)@n)<0:n=-n
    s0=(contact_x-c[0])/t[0]
    stations=[];addresses=[];normals=[];rejected=[]
    probe=np.arange(-16.,16.01,.5)
    h,w=labels.shape
    for s in np.arange(s0-half_span,s0+half_span+.01,pitch):
        nominal=c+s*t;ray=nominal+probe[:,None]*n
        values=_labels_at(labels,ray)
        transitions=[j for j in range(2,len(probe)-4) if np.all(values[j-2:j]==0) and np.all(values[j:j+4]==1)]
        if not transitions:rejected.append(float(s));continue
        j=min(transitions,key=lambda j:abs(probe[j]))
        boundary=nominal+(probe[j]-.25)*n
        nodes=boundary+np.array([1.5,4.5,13.5,16.5])[:,None]*n
        if (not np.all(_labels_at(labels,nodes)==1) or np.any(nodes[:,0]<8)
                or np.any(nodes[:,0]>w-8) or np.any(nodes[:,1]<48) or np.any(nodes[:,1]>h-8)):
            rejected.append(float(s));continue
        stations.append(boundary);addresses.append(s);normals.append(n)
    boundary=np.asarray(stations,np.float32).reshape(-1,2)
    if len(boundary)<8:raise ValueError('Insufficient original sloping free surface; no interior fallback')
    nodes=boundary[:,None,:]+np.array([1.5,4.5,13.5,16.5],np.float32)[None,:,None]*n.astype(np.float32)
    return dict(boundary_xy=boundary,node_xy=nodes.astype(np.float32),surface_s_px=np.asarray(addresses,np.float32),
        inward_normal=n.astype(np.float32),tangent=t.astype(np.float32),
        layer_offsets_px=np.array([1.5,4.5,13.5,16.5],np.float32),
        station_pitch_px=np.array(pitch),contact_s_px=np.array(s0),rejected_s_px=np.asarray(rejected,np.float32))
