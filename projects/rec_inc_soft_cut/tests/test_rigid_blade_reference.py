import cv2
import numpy as np
from src.analysis.rigid_blade_reference import RigidBladeReference


def test_fixed_blade_template_tracks_translation_with_brightness_change():
    frame=np.full((768,1024,3),18,np.uint8)
    cv2.fillPoly(frame,[np.array([[98,0],[122,0],[122,100],[110,180],[98,100]])],(210,210,210))
    frame=cv2.GaussianBlur(frame,(5,5),1)
    reference=RigidBladeReference(frame,np.array([110.,180.]),24)
    moved=cv2.warpAffine(frame,np.array([[1,0,3.3],[0,1,11.4]],np.float32),(1024,768))
    moved=np.clip(moved.astype(float)*.7+8,0,255).astype(np.uint8)
    tip,score,agreement=reference.locate(moved,np.array([113.,194.]))
    np.testing.assert_allclose(tip,[113.3,191.4],atol=.65)
    assert score>.95 and agreement<2
