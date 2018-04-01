from __future__ import absolute_import as _abs

from . import avx512_conv_common, avx512_conv_1x1

from .avx512_conv_common import AVX512ConvCommonFwd
from .avx512_conv_1x1 import AVX512Conv1x1Fwd

import nnvm
import nnvm.symbol as sym
from nnvm.top import registry as reg

import tvm
import topi
from topi.nn.conv2d import conv2d, _get_schedule
from topi.util import get_const_tuple, get_const_int
from topi.nn.conv2d import Workload
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride
from topi import nn
from topi import tag

fp32_vec_len = 16

_WORKLOADS = [
    # SSD VGG16 512 * 512 65-89
    Workload('float32', 'float32', 512, 512, 3, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 512, 512, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 256, 256, 64, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 256, 256, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 128, 128, 128, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 128, 128, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 256, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 1024, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 1, 1, 1024, 20, 1, 1, 0, 0, 1, 1),
"""
    Workload('float32', 'float32', 512, 512, 3, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 512, 512, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 256, 256, 64, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 256, 256, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 128, 128, 128, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 128, 128, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 256, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 84, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 512, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 1024, 3, 3, 6, 6, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 16, 16, 512, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 16, 16, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 16, 16, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 8, 8, 256, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 8, 8, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 4, 4, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 2, 2, 128, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 64, 64, 512, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 1024, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 16, 16, 512, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 8, 8, 256, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 32, 32, 512, 1024, 3, 3, 1, 1, 1, 1),
"""
]

_SCHEDULES = [
    # SSD VGG16
    AVX512ConvCommonFwd(ic_bn=3, oc_bn=16, reg_n=64, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=128, oh_factor=1, ow_factor=2),
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=20, oh_factor=1, ow_factor=1),
"""
    AVX512ConvCommonFwd(ic_bn=3, oc_bn=8, reg_n=64, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=1, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=1, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=1, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=14, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=32, reg_n=7, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=64, oh_factor=2, ow_factor=2),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=14, reg_n=32, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=9, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=64, oh_factor=1, ow_factor=4),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=4, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=64, oh_factor=2, ow_factor=4),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=64, reg_n=1, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=16, reg_n=32, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=8, oc_bn=8, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=8, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=6, reg_n=8, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
"""
]


_SCH_TO_DECL_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._declaration_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._declaration_conv
}

_SCH_TO_SCH_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._schedule_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._schedule_conv
}

def get_workload(data, kernel, stride, padding, out_dtype):
    """ Get the workload structure. """
    if len(data.shape) == 4:
        return _get_workload(data, kernel, stride, padding, out_dtype)
    n, c_c, h, w, c_b = [x.value for x in data.shape]
    original_data = tvm.placeholder((n, c_c * c_b, h, w))
    return _get_workload(original_data, kernel, stride, padding, out_dtype)

@_get_schedule.register("cpu", override=True)
def _get_schedule_conv(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


def _declare_conv2d(data, kernel, num_filter, kernel_size, stride, padding, bias=None, out_dtype="float32"):
    assert data.shape[0].value == 1, "only support batch size=1 convolution on avx"
    if len(data.shape) == 5:
        n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
        ic = ic_chunk * ic_block
    else:
        n, ic, h, w = [x.value for x in data.shape]
    oc = num_filter
    kh, kw = kernel_size
    wkl = get_workload(data, tvm.placeholder((oc, ic, kh, kw), dtype=out_dtype), stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    DOPAD = (padding[0] != 0 or padding[1] != 0)
    # Pack data if necessary
    shape = (data.shape[0], ic // sch.ic_bn, h + 2 * padding[0], w + 2 * padding[1], sch.ic_bn)
    if len(data.shape) == 4:
        if DOPAD:
            data_pad = nn.pad(data, (0, 0, padding[0], padding[1]), name="data_pad")
        else:
            data_pad = data
        data_vec = tvm.compute(shape, lambda n, C, h, w, c:
                               data_pad[n, C * sch.ic_bn + c, h, w], tag='conv2d_data_pack')
        print("Workload %s needs pack" % str(wkl))
    else:
        if DOPAD:
            data_pad = nn.pad(data, (0, 0, padding[0], padding[1], 0), name="data_pad")
        else:
            data_pad = data
        if ic_block != sch.ic_bn:
            data_vec = tvm.compute(shape, lambda n, C, h, w, c:
                                   data_pad[n, (C * sch.ic_bn + c) // ic_block, h, w, (C * sch.ic_bn + c) % ic_block],
                                   tag='conv2d_data_pack')
            print("Workload %s needs pack" % str(wkl))
        else:
            data_vec = data_pad

    out =  _SCH_TO_DECL_FUNC[type(sch)](data_vec, wkl, sch, kernel)

    if bias is not None:
        expand_axis = 1
        bias = topi.expand_dims(bias, axis=expand_axis, num_newaxis=2)
        out = topi.broadcast_add(out, bias)
    return out


@reg.register_compute("conv2d", level=20)
def compute_conv2d(attrs, inputs, _):
    """Compute definition of conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    kernel_size = attrs.get_int_tuple("kernel_size")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]

    bias = inputs[2] if attrs.get_bool("use_bias") else None

    out = _declare_conv2d(inputs[0], inputs[1], channels, kernel_size, strides, padding, bias, "float32")

    return out


def schedule_conv2d_nChwc(padding, strides, num_filter, kernel_size, use_bias, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d' in op.tag:
            output = op.output(0)
            conv_out = output
            data_vec = conv_out.op.input_tensors[0]
            DOPAD = padding[0] != 0 or padding[1] != 0
            data_pad = data_vec.op.input_tensors[0] if hasattr(data_vec.op, 'tag') and "conv2d_data_pack" in data_vec.op.tag else data_vec
            data = data_pad.op.input_tensors[0] if DOPAD else data_pad

            if len(data.shape) == 5:
                n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
                ic = ic_chunk * ic_block
            else:
                n, ic, h, w = [x.value for x in data.shape]
            original_data = tvm.placeholder((n, ic, h, w), dtype=output.dtype)

            oc = num_filter
            kh, kw = kernel_size
            original_kernel = tvm.placeholder((oc, ic, kh, kw), dtype=output.dtype)

            wkl = _get_workload(original_data, original_kernel, strides, padding, output.dtype)
            sch = _get_schedule(wkl)
            _SCH_TO_SCH_FUNC[type(sch)](s, data_pad, data_vec, wkl, sch, conv_out, op.output(0), outs[0])
   

    traverse(outs[0].op)
    return s


@reg.register_schedule("conv2d", level=20)
def schedule_conv2d(attrs, outs, target):
    with tvm.target.create(target):
        padding = attrs.get_int_tuple("padding")
        strides = attrs.get_int_tuple("strides")
        channels = attrs.get_int("channels")
        kernel_size = attrs.get_int_tuple("kernel_size")
        use_bias = attrs.get_bool("use_bias")
        return schedule_conv2d_nChwc(padding, strides, channels, kernel_size, use_bias, outs)


@reg.register_infershape("conv2d", level=20)
def infer_shape_conv2d(attrs, in_shapes, p_in_shapes, p_out_shapes):
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    kernel_size = attrs.get_int_tuple("kernel_size")
    channels = attrs.get_int("channels")
    out_dtype = "float32"

    batch_size = in_shapes[0][0].value
    in_channels = in_shapes[0][1].value if len(in_shapes[0]) == 4 else in_shapes[0][1].value * in_shapes[0][4].value
    in_height = in_shapes[0][2].value
    in_width = in_shapes[0][3].value
    wkl = _get_workload(tvm.placeholder((batch_size, in_channels, in_height, in_width), dtype=out_dtype),
                        tvm.placeholder((channels, in_channels, kernel_size[0], kernel_size[1]), dtype=out_dtype),
                        strides, padding, out_dtype)
    sch = _get_schedule_conv(wkl)

    out_height = (in_height + 2 * padding[0] - kernel_size[0]) // strides[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_size[1]) // strides[1] + 1

    out_shape = (batch_size, channels // sch.oc_bn, out_height, out_width, sch.oc_bn)
    kernel_shape = (channels // sch.oc_bn, in_channels // sch.ic_bn, kernel_size[0], kernel_size[1], sch.ic_bn, sch.oc_bn) if kernel_size[0] != 1 or kernel_size[1] != 1 else (channels // sch.oc_bn, in_channels // sch.ic_bn, sch.ic_bn, sch.oc_bn, kernel_size[0], kernel_size[1])
    bias_shape = (channels // sch.oc_bn, sch.oc_bn)

    if attrs.get_bool("use_bias"):
        reg.assign_shape(p_in_shapes, 2, True, bias_shape)
    reg.assign_shape(p_in_shapes, 1, True, kernel_shape)
    reg.assign_shape(p_out_shapes, 0, False, out_shape)

    return True

