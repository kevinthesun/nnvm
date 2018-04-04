"""Definition of object detection ops"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern
from .tensor import _fschedule_broadcast

# multibox_prior
@reg.register_schedule("multibox_prior")
def schedule_multibox_prior(_, outs, target):
    """Schedule definition of multibox_prior"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_prior(outs)

@reg.register_compute("multibox_prior")
def compute_conv2d(attrs, inputs, _):
    """Compute definition of conv2d"""
    sizes = attrs.get_float_tuple('sizes')
    ratios = attrs.get_float_tuple('ratios')
    steps = attrs.get_float_tuple('steps')
    offsets = attrs.get_float_tuple('offsets')
    clip = attrs.get_bool('clip')

    return topi.vision.ssd.multibox_prior(inputs[0], sizes, ratios,
                                          steps, offsets, clip)

reg.register_pattern("multibox_prior", OpPattern.OPAQUE)

# multibox_detection
@reg.register_schedule("multibox_detection")
def schedule_multibox_detection(_, outs, target):
    """Schedule definition of multibox_detection"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_detection(outs)

@reg.register_compute("multibox_detection")
def compute_multibox_detection(attrs, inputs, _):
    """Compute definition of multibox_detection"""
    clip = attrs.get_bool('clip')
    threshold = attrs.get_float('threshold')
    nms_threshold = attrs.get_float('nms_threshold')
    force_suppress = attrs.get_bool('force_suppress')
    variance = attrs.get_float_tuple('variances')
    nms_topk = attrs.get_int('nms_topk')

    return topi.vision.ssd.multibox_detection(inputs[0], inputs[1], inputs[2],
                                              clip, threshold, nms_threshold,
                                              force_suppress, variance, nms_topk)

reg.register_pattern("multibox_detection", OpPattern.OPAQUE)
