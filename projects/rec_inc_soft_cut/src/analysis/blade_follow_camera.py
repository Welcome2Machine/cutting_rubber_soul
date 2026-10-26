"""Display-only, fixed-zoom camera independent of material support/selection."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter


def blade_follow_camera(tips,*,shaft_x,geometry_supported=None,frame_size=(1024,768),crop_size=(256,192),top=40):
    points=np.asarray(tips,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<1:raise ValueError('Tip coordinates must be T by 2')
    good=np.isfinite(points[:,1])
    if geometry_supported is not None:
        observed=np.asarray(geometry_supported,bool)
        if observed.shape!=good.shape:raise ValueError('Tip support must address every frame')
        good&=observed
    if not good.any() or not np.isfinite(shaft_x):raise ValueError('Camera requires a finite blade guide')
    t=np.arange(len(points));guide=np.interp(t,t[good],points[good,1])
    # These filters affect the viewport only. Tracking and depth axes still
    # read the original saved coordinates. Smooth at source cadence, before
    # accelerated display sampling; isolated segmentation errors cannot steer
    # the camera, nor can point appearance/loss change its target.
    guide=median_filter(guide,size=9,mode='nearest')
    guide=gaussian_filter1d(guide,sigma=3.5,mode='nearest',truncate=3.)
    width,height=frame_size;cw,ch=crop_size
    if not (0<cw<=width and 0<ch<=height-top):raise ValueError('Invalid camera viewport')
    origins=np.empty((len(points),2),float)
    origins[:,0]=np.clip(shaft_x-cw/2,0,width-cw)
    origins[:,1]=np.clip(guide-.58*ch,top,height-ch)
    return dict(origin_xy=origins,guide_tip_y=guide,guide_supported=good,
        crop_size=np.asarray(crop_size),mode=np.array('fixed_x_smoothed_tip_y_subpixel_v1'))


def crop_for_display(frame,origin,crop_size=(256,192),output_size=(1152,864)):
    origin=np.asarray(origin,float);scale=np.asarray(output_size)/np.asarray(crop_size)
    matrix=np.array([[scale[0],0,(-origin[0]+.5)*scale[0]-.5],
                     [0,scale[1],(-origin[1]+.5)*scale[1]-.5]],np.float64)
    return cv2.warpAffine(frame,matrix,output_size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def point_in_display(point,origin,crop_size=(256,192),output_size=(1152,864)):
    return (np.asarray(point)-origin+.5)*(np.asarray(output_size)/np.asarray(crop_size))-.5
