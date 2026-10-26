import cv2
import numpy as np
from src.tracking.shadow_normalization import MaterialShadowNormalizer,ShadowConfig


def test_bright_blade_excluded_from_gain_and_nonmaterial_preserved():
    image=np.full((768,1024,3),40,np.uint8);labels=np.ones((768,1024),np.uint8)
    image[:,400:440]=255;labels[:,400:440]=2;labels[:40]=255
    processor=MaterialShadowNormalizer();output,gain=processor.process(image,labels)
    clean=processor.grid(image)
    np.testing.assert_array_equal(output[labels!=1],clean[labels!=1])
    np.testing.assert_allclose(gain[200:220,370:380],2.2,atol=1e-5)
    assert gain.min()>=1 and gain.max()<=2.20001


def test_stationary_scene_has_no_initial_brightness_ramp_or_texture_motion():
    rng=np.random.default_rng(91)
    gray=rng.integers(25,65,(768,1024),dtype=np.uint8)
    image=np.repeat(gray[...,None],3,2);labels=np.ones_like(gray)
    processor=MaterialShadowNormalizer();first,_=processor.process(image,labels)
    for _ in range(5):
        result,_=processor.process(image,labels)
        np.testing.assert_array_equal(result,first)


def test_illumination_disparity_reduced_without_erasing_fine_texture():
    x=np.arange(1024);texture=1+.14*np.sin(2*np.pi*x/8)
    light=np.where(x<512,60.,30.)
    gray=np.tile(np.rint(texture*light).astype(np.uint8),(768,1))
    image=np.repeat(gray[...,None],3,2);labels=np.ones_like(gray)
    result,_=MaterialShadowNormalizer().process(image,labels)
    a=result[200:240,200:240,0];b=result[200:240,720:760,0]
    assert abs(float(a.mean())-float(b.mean()))<abs(float(gray[200:240,200:240].mean())-float(gray[200:240,720:760].mean()))
    assert a.std()>3 and b.std()>3
    assert result.max()<255


def test_black_and_highlights_are_not_boosted():
    labels=np.ones((768,1024),np.uint8)
    for level in (0,245):
        image=np.full((768,1024,3),level,np.uint8)
        result,gain=MaterialShadowNormalizer().process(image,labels)
        np.testing.assert_array_equal(result,image)
        np.testing.assert_allclose(gain,1)
