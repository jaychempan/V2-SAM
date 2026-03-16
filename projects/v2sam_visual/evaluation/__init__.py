# Copyright (c) OpenMMLab. All rights reserved.
# from .metrics import CityscapesMetric, DepthMetric, IoUMetric

# __all__ = ['IoUMetric', 'CityscapesMetric', 'DepthMetric']

from .seg_metric import SegMetric
__all__ = ['SegMetric']