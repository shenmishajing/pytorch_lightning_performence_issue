# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import CutOut as _CutOut
from mmdet.datasets.pipelines import MixUp as _MixUp
from mmdet.datasets.pipelines import Mosaic as _Mosaic
from mmdet.datasets.pipelines import Resize as _Resize


@PIPELINES.register_module(force = True)
class Resize(_Resize):
    def __init__(self, img_scale = None, *args, **kwargs):
        if isinstance(img_scale, list):
            if not mmcv.is_list_of(img_scale, tuple):
                img_scale = [tuple(img_scale)]
        super().__init__(img_scale = img_scale, *args, **kwargs)


@PIPELINES.register_module(force = True)
class CutOut(_CutOut):
    def __init__(self, n_holes, *args, **kwargs):
        if isinstance(n_holes, list):
            n_holes = tuple(n_holes)
        super().__init__(n_holes = n_holes, *args, **kwargs)


@PIPELINES.register_module(force = True)
class Mosaic(_Mosaic):
    def __init__(self, img_scale = (640, 640), *args, **kwargs):
        if isinstance(img_scale, list):
            img_scale = tuple(img_scale)
        super().__init__(img_scale = img_scale, *args, **kwargs)


@PIPELINES.register_module(force = True)
class MixUp(_MixUp):
    def __init__(self, img_scale = (640, 640), *args, **kwargs):
        if isinstance(img_scale, list):
            img_scale = tuple(img_scale)
        super().__init__(img_scale = img_scale, *args, **kwargs)
