"""Read microscope silhouettes from PPTX and estimate coarse blade geometry.

No FE loading, force data, or microscopic tip-radius inference. Width/angle
become mechanical cross-section parameters only with a confirmed viewing plane.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def largest_component(mask):
    seen=np.zeros_like(mask,dtype=bool);best=[]
    for y,x in zip(*np.nonzero(mask)):
        if seen[y,x]:continue
        seen[y,x]=True;stack=[(y,x)];component=[]
        while stack:
            a,b=stack.pop();component.append((a,b))
            for da in (-1,0,1):
                for db in (-1,0,1):
                    u,v=a+da,b+db
                    if (0<=u<mask.shape[0] and 0<=v<mask.shape[1]
                            and mask[u,v] and not seen[u,v]):
                        seen[u,v]=True;stack.append((u,v))
        if len(component)>len(best):best=component
    out=np.zeros_like(mask,dtype=bool)
    if best:
        ys,xs=np.array(best).T;out[ys,xs]=True
    return out


def scale_bar(rgb, length_um):
    """Calibrate red horizontal bar by its full pixel-cell width."""
    red=(rgb[:,:,0]>.7)&(rgb[:,:,1]<.25)&(rgb[:,:,2]<.25)
    red[:int(.7*len(rgb))]=False
    mask=largest_component(red)
    ys,xs=np.nonzero(mask)
    if len(xs)<10:raise ValueError('No unambiguous red scale bar')
    width=int(xs.max()-xs.min()+1)
    if width<10 or ys.max()-ys.min()>5:raise ValueError('Scale bar must be horizontal')
    return dict(length_um=length_um,span_pixels=width,x_first=int(xs.min()),x_last=int(xs.max()),
                um_per_pixel=length_um/width,endpoint_sensitivity_um_per_pixel=[length_um/(width+2),length_um/(width-2)])


def silhouette(rgb, roi, threshold):
    gray=rgb[:,:,:3].mean(axis=2)
    x0,y0,x1,y1=roi;mask=np.zeros_like(gray,dtype=bool)
    mask[y0:y1,x0:x1]=gray[y0:y1,x0:x1]>threshold
    mask=largest_component(mask);xs=[];upper=[];lower=[]
    for x in range(x0,x1):
        ys=np.flatnonzero(mask[:,x])
        if not len(ys):continue
        a,b=ys[0],ys[-1]
        if a==0 or b==len(gray)-1:continue
        # Linear gray-level crossings retain antialiasing without a blur filter.
        top=a-1+(threshold-gray[a-1,x])/(gray[a,x]-gray[a-1,x])
        bottom=b+(gray[b,x]-threshold)/(gray[b,x]-gray[b+1,x])
        xs.append(x);upper.append(top);lower.append(bottom)
    if len(xs)<80:raise ValueError('Insufficient visible blade silhouette')
    return np.array(xs,dtype=float),np.array(upper),np.array(lower)


def robust_fit(design, values):
    weights=np.ones(len(values))
    for _ in range(8):
        root=np.sqrt(weights)
        coef=np.linalg.lstsq(design*root[:,None],values*root,rcond=None)[0]
        residual=values-design@coef
        weights=np.minimum(1.,.65/np.maximum(np.abs(residual),1e-12))
    loss=np.where(abs(residual)<=.65,.5*residual**2,.65*(abs(residual)-.325)).sum()
    return coef,float(loss),float(np.sqrt(np.mean(residual**2)))


def fit_face(x,y):
    best=None
    for shoulder in np.arange(x.min()+25,x.max()-45,.5):
        design=np.column_stack((np.ones(len(x)),x,np.minimum(x-shoulder,0.)))
        coef,loss,rms=robust_fit(design,y)
        if best is None or loss<best['loss']:
            b,m,change=coef
            best=dict(shoulder_px=float(shoulder),plate_intercept=float(b),plate_slope=float(m),
                flank_intercept=float(b-change*shoulder),flank_slope=float(m+change),loss=loss,rms_px=rms)
    if best is None:raise ValueError('Insufficient flank and shank extent')
    return best


def fit_outline(x,top,bottom,mm_per_pixel):
    # First ~50 um contain too few pixels for a resolved nose fit.
    keep=x>=x.min()+5
    a,b=fit_face(x[keep],top[keep]),fit_face(x[keep],bottom[keep])
    denom=a['flank_slope']-b['flank_slope']
    if denom>=0:raise ValueError('Expected a left-pointing blade with opening flanks')
    tip=(b['flank_intercept']-a['flank_intercept'])/denom
    slope=.5*(a['plate_slope']+b['plate_slope']);axis=np.arctan(slope)
    if abs(np.rad2deg(axis))>5:
        raise ValueError('Rotate/review the image first; this extractor expects a nearly horizontal blade')
    xa=float(np.quantile(x,.85));thickness=((b['plate_intercept']+b['plate_slope']*xa)
        -(a['plate_intercept']+a['plate_slope']*xa))/np.sqrt(1+slope*slope)*mm_per_pixel
    angles=[float(np.rad2deg(axis-np.arctan(a['flank_slope']))),float(np.rad2deg(np.arctan(b['flank_slope'])-axis))]
    widths={}
    for distance in (.1,.25,.5,.75):
        xx=tip+distance/mm_per_pixel*np.cos(axis)
        widths[str(distance)]=float(np.interp(xx,x,bottom-top)*np.cos(axis)*mm_per_pixel) if x[0]<=xx<=x[-1] else None
    return dict(thickness_mm=float(thickness),included_angle_deg=sum(angles),face_angles_deg=angles,
        flank_angle_asymmetry_deg=abs(angles[0]-angles[1]),axis_image_angle_deg=float(np.rad2deg(axis)),
        bevel_lengths_mm=[float((a['shoulder_px']-tip)*mm_per_pixel/np.cos(axis)),float((b['shoulder_px']-tip)*mm_per_pixel/np.cos(axis))],
        virtual_sharp_apex_px=float(tip),detected_front_x_px=float(x.min()),face_fits=[a,b],
        widths_at_distance_from_virtual_apex_mm=widths)


def analyze(rgb, spec):
    scale=scale_bar(rgb,spec['scale_bar_um']);all_fits=[];traces=[]
    for threshold in spec['gray_thresholds']:
        x,top,bottom=silhouette(rgb,spec['roi_xyxy'],threshold)
        for um_per_px in (scale['um_per_pixel'],*scale['endpoint_sensitivity_um_per_pixel']):
            all_fits.append(fit_outline(x,top,bottom,um_per_px/1000))
        traces.append((x,top,bottom))
    middle=len(traces)//2;x,top,bottom=traces[middle]
    estimate=fit_outline(x,top,bottom,scale['um_per_pixel']/1000)
    sensitivity={key:[float(min(v[key] for v in all_fits)),float(max(v[key] for v in all_fits))]
        for key in ('thickness_mm','included_angle_deg','flank_angle_asymmetry_deg')}
    # Conservative raster localization bounds, not repeated-specimen statistics.
    sensitivity['thickness_mm'][0]-=2*scale['um_per_pixel']/1000
    sensitivity['thickness_mm'][1]+=2*scale['um_per_pixel']/1000
    span=min((f['shoulder_px']-estimate['virtual_sharp_apex_px']) for f in estimate['face_fits'])
    angle_bound=2*np.rad2deg(np.arctan2(2.,span))
    sensitivity['included_angle_deg'][0]-=angle_bound;sensitivity['included_angle_deg'][1]+=angle_bound
    sensitivity['flank_angle_asymmetry_deg'][0]=max(0.,sensitivity['flank_angle_asymmetry_deg'][0]-angle_bound)
    sensitivity['flank_angle_asymmetry_deg'][1]+=angle_bound
    return dict(scale=scale,image_size_px=[rgb.shape[1],rgb.shape[0]],estimate=estimate,
        sensitivity_envelope=sensitivity,envelope_kind='threshold + scale endpoints +/-1px + contour localization +/-1px; NOT statistical CI',
        tip_radius_mm=None,tip_radius_resolved=False,roughness_resolved=False,
        outline_image_px=dict(x=x.tolist(),upper_y=top.tolist(),lower_y=bottom.tolist()),
        qualification=dict(side_view_confirmed=spec['view_plane']=='normal_to_cutting_edge',
            isotropic_pixel_assumption=True,microscopic_edge_qualified=False,physical_prediction_qualified=False))


def extract(source,manifest_path,output):
    import matplotlib.image as mpimg
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    source=Path(source);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=manifest['source_sha256']:raise ValueError('Measurement source differs; review a new manifest')
    result=dict(schema='softcut-blade-microscopy/1',source_name=source.name,source_sha256=digest,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),experimental_force_read=False,loading_started=False,blades=[])
    with zipfile.ZipFile(source) as z:
        for spec in manifest['blades']:
            data=z.read(spec['image_part']);rgb=mpimg.imread(io.BytesIO(data),format='png')[:,:,:3]
            result['blades'].append(dict(identity=spec,embedded_image_sha256=hashlib.sha256(data).hexdigest(),**analyze(rgb,spec)))
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous extraction output')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


def parameter_catalog(result):
    """Small reviewed configuration; pixels and instrument source stay local."""
    profiles=[]
    for blade in result['blades']:
        estimate=blade['estimate']
        profiles.append(dict(
            identity=blade['identity'], embedded_image_sha256=blade['embedded_image_sha256'],
            apparent_profile={key:estimate[key] for key in (
                'thickness_mm','included_angle_deg','face_angles_deg',
                'flank_angle_asymmetry_deg','bevel_lengths_mm',
                'widths_at_distance_from_virtual_apex_mm')},
            scale=blade['scale'], sensitivity_envelope=blade['sensitivity_envelope'],
            envelope_kind=blade['envelope_kind'], qualification=blade['qualification'],
            tip_radius_mm=None, tip_radius_resolved=False, roughness_resolved=False,
            fe_mapping=dict(shape='symmetric_wedge',
                status='conditional approximation; not a resolved microscope edge',
                thickness_mm=estimate['thickness_mm'],
                included_angle_deg=estimate['included_angle_deg'],
                tip_radius_mm=None, requires_explicit_tip_radius=True,
                requires_confirmed_view_plane=True,
                requires_explicit_symmetric_approximation=True)))
    return dict(schema='softcut-blade-shape-catalog/1',
        source_name=result['source_name'],source_sha256=result['source_sha256'],
        tool_sha256=result['tool_sha256'],experimental_force_read=False,
        loading_started=False, profiles=profiles)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--catalog-output',type=Path);args=parser.parse_args()
    if args.catalog_output and args.catalog_output.exists():
        raise FileExistsError('Preserve previous parameter catalog')
    q=extract(args.source,args.manifest,args.output)
    if args.catalog_output:
        args.catalog_output.parent.mkdir(parents=True,exist_ok=True)
        args.catalog_output.write_text(json.dumps(parameter_catalog(q),indent=2,ensure_ascii=False),encoding='utf-8')
    for b in q['blades']:
        print(json.dumps(dict(id=b['identity']['id'],estimate=b['estimate'],envelope=b['sensitivity_envelope']),ensure_ascii=False))


if __name__=='__main__':main()
