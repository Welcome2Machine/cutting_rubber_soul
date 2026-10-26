"""Bounded material-only illumination gain; never averages image textures."""
from dataclasses import dataclass,asdict
import cv2
import numpy as np
from .fixed_grid import FixedGridProcessor,FixedGridConfig,grid_0915_config


@dataclass(frozen=True)
class ShadowConfig:
    target_gray: float = 90.
    max_gain: float = 2.2
    illumination_sigma_px: float = 20.
    temporal_alpha: float = .3
    edge_feather_px: float = 3.
    downsample: int = 4

    def __post_init__(self):
        if not (0<self.target_gray<200 and 1<=self.max_gain<=4
                and self.illumination_sigma_px>=8 and 0<self.temporal_alpha<=1
                and self.edge_feather_px>0 and self.downsample>=1):
            raise ValueError('Invalid bounded illumination settings')


class MaterialShadowNormalizer:
    """Sequential preprocessing, baked into a lossless video before tracking.

    Temporal smoothing applies only to a low-frequency gain field. Every output
    texture comes from that same input frame; positions and masks are untouched.
    A new instance is required at frame zero. State is never shared with tracking
    overlap windows or resumed from an unrelated frame.
    """
    def __init__(self,config=None,*,grid_enabled=True):
        self.config=config or ShadowConfig()
        self.grid=FixedGridProcessor(grid_0915_config() if grid_enabled else FixedGridConfig(1024,768))
        self.exclude_grid=cv2.dilate(self.grid.mask().astype(np.uint8),np.ones((5,5),np.uint8))>0
        self.previous_log_gain=None
        self.frame_index=0

    def metadata(self):
        return dict(algorithm='masked_bounded_illumination_gain_v1',**asdict(self.config),
            grid_preprocessing=self.grid.metadata(),initialization='first-frame gain, no startup ramp',
            temporal_filter='EMA of log illumination gain only; no image averaging',
            reference_target='fixed gray level for entire sequence',
            estimation_exclusions='air, blade with 5-pixel margin, overlay grid, bright glints, near-black pixels',
            material_edge='gain feathered inside current material mask',reconstruction=False)

    def process(self,frame,labels):
        if frame.shape!=(768,1024,3) or labels.shape!=(768,1024):raise ValueError('Native geometry required')
        image=self.grid(frame);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        material=(labels==1).astype(np.uint8)
        blade=cv2.dilate((labels==2).astype(np.uint8),np.ones((11,11),np.uint8))>0
        weight=((material>0)&~blade&~self.exclude_grid&(gray>=10)&(gray<190)).astype(np.float32)
        factor=self.config.downsample;size=(1024//factor,768//factor);sigma=self.config.illumination_sigma_px/factor
        denominator=cv2.GaussianBlur(cv2.resize(weight,size,interpolation=cv2.INTER_AREA),(0,0),sigma)
        numerator=cv2.GaussianBlur(cv2.resize(gray*weight,size,interpolation=cv2.INTER_AREA),(0,0),sigma)
        illumination=np.full_like(numerator,self.config.target_gray)
        np.divide(numerator,denominator,out=illumination,where=denominator>.01)
        desired=np.clip(self.config.target_gray/np.maximum(illumination,10),1,self.config.max_gain)
        # Deep black is not recoverable detail. Fade enhancement out below the
        # useful illumination range instead of presenting amplified noise as texture.
        desired=1+(desired-1)*np.clip((illumination-8)/16,0,1)
        current=np.log(desired)
        if self.previous_log_gain is None:self.previous_log_gain=current.copy()
        else:
            a=self.config.temporal_alpha
            self.previous_log_gain=(1-a)*self.previous_log_gain+a*current
        smooth_gain=cv2.resize(np.exp(self.previous_log_gain),(1024,768),interpolation=cv2.INTER_LINEAR)
        feather=np.minimum(cv2.distanceTransform(material,cv2.DIST_L2,3)/self.config.edge_feather_px,1)
        highlights=np.clip((240-gray)/100,0,1)
        gain=1+(smooth_gain-1)*feather*highlights
        corrected=np.rint(np.clip(image.astype(np.float32)*gain[...,None],0,255)).astype(np.uint8)
        self.frame_index+=1
        return corrected,gain
