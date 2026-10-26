"""Register a fixed blade-edge template; never switch endpoint detectors."""
import cv2
import numpy as np
from src.tracking.fixed_grid import FixedGridProcessor,FixedGridConfig,grid_0915_config


def blade_feature(gray):
    im=cv2.GaussianBlur(gray.astype(np.float32),(3,3),.7)
    gx=cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    return gx


class RigidBladeReference:
    def __init__(self,frame,tip,shaft_width,*,grid_enabled=True):
        self.grid=FixedGridProcessor(grid_0915_config() if grid_enabled else FixedGridConfig(1024,768))
        self.tip=np.asarray(tip,float)
        gray=self.grid(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        x,y=np.rint(tip).astype(int)
        self.half=int(np.clip(shaft_width*1.7,24,64))
        self.top=max(38,y-115);self.bottom=y-5
        if self.bottom-self.top<45:raise ValueError('Reference blade too short')
        self.left=x-self.half;self.right=x+self.half+1
        feature=blade_feature(gray)
        self.template=feature[self.top:self.bottom,self.left:self.right].copy()
        magnitude=abs(self.template)
        mask=magnitude>np.maximum(12,np.max(magnitude,axis=1,keepdims=True)*.25)
        self.mask=cv2.dilate(mask.astype(np.uint8),np.ones((3,5),np.uint8))
        if self.mask.sum()<150:raise ValueError('Insufficient reference edges')

    def locate(self,frame,guide):
        gray=self.grid(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY));feature=blade_feature(gray)
        expected=np.asarray(guide)-self.tip
        sx,sy=np.rint(expected).astype(int)
        x0=max(0,self.left+sx-9);x1=min(gray.shape[1],self.right+sx+10)
        y0=max(0,self.top+sy-32);y1=min(gray.shape[0],self.bottom+sy+33)
        roi=feature[y0:y1,x0:x1]
        if roi.shape[0]<self.template.shape[0] or roi.shape[1]<self.template.shape[1]:return np.full(2,np.nan),0.,np.inf
        scores=cv2.matchTemplate(roi,self.template,cv2.TM_CCORR_NORMED,mask=self.mask)
        scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        iy,ix=np.unravel_index(np.argmax(scores),scores.shape)
        def offset(axis):
            i=ix if axis==1 else iy;n=scores.shape[axis]
            if not 0<i<n-1:return 0.
            a,b,c=scores[iy,ix-1:ix+2] if axis==1 else scores[iy-1:iy+2,ix]
            denominator=a-2*b+c
            return float(np.clip(.5*(a-c)/denominator,-.5,.5)) if denominator< -1e-8 else 0.
        xy=self.tip+np.array([x0+ix+offset(1)-self.left,y0+iy+offset(0)-self.top])
        # Independent upper/lower edge bands reveal non-rigid or ambiguous matches.
        candidates=[]
        h=len(self.template);mid=h//2
        for a,b in [(0,mid),(mid,h)]:
            part=cv2.matchTemplate(roi[a:roi.shape[0]-(h-b)],self.template[a:b],cv2.TM_CCORR_NORMED,mask=self.mask[a:b])
            part=np.nan_to_num(part,nan=-1,posinf=-1,neginf=-1)
            py,px=np.unravel_index(np.argmax(part),part.shape);candidates.append([px,py])
        agreement=float(np.linalg.norm(np.diff(candidates,axis=0)))
        return xy,float(scores[iy,ix]),agreement
