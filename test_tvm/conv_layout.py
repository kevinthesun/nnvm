from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from topi import nn, tag
from nnvm.top.tensor import _fschedule_broadcast
from nnvm.top import registry as reg


def schedule_pool(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
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

        if 'pad' in op.tag or 'pool' in op.tag:
            C = op.output(0)
            if len(C.op.axis) ==5:
                batch, c_c, h, w, c_b = C.op.axis
                #s[C].reorder(batch, c_c, c_b, h, w)
                fused = s[C].fuse(batch, c_c, h)
            else:
                batch, c, h, w = C.op.axis
                s[C].reorder(batch, c, h, w)
                fused = s[C].fuse(batch, c)
            s[C].parallel(fused)

    traverse(outs[0].op)
    return s


@reg.register_compute("max_pool2d", level=20)
def compute_max_pool_2d_nChwc(attrs, inputs, _):
    kernel = attrs.get_int_tuple("pool_size")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    ceil_mode = attrs.get_bool("ceil_mode")
    layout = attrs["layout"]

    if len(inputs[0].shape) == 4:
        print("pooling pack")
        return nn.pool(inputs[0], kernel,
                       strides, padding, "max", ceil_mode, layout)

    batch, channel_chunk, height, width, channel_block = inputs[0].shape
    if padding[0] > 0 or padding[1] > 0:
        pad_data = nn.pad(inputs[0], (0, 0, padding[0], padding[1], 0),
                          pad_value=tvm.min_value(inputs[0].dtype), name="data_pad")
    else:
        pad_data = inputs[0]

    out_height = nn.util.simplify((height - kernel[0] + 2 * padding[0]) // strides[0] + 1)
    out_width = nn.util.simplify((width - kernel[1] + 2 * padding[1]) // strides[1] + 1)
    dheight = tvm.reduce_axis((0, kernel[0]))
    dwidth = tvm.reduce_axis((0, kernel[1]))
    oshape = (batch, channel_chunk, out_height, out_width, channel_block)
    return tvm.compute(oshape,
                       lambda n, c_c, h, w, c_b: tvm.max(pad_data[n, c_c, h * strides[0] + dheight,
                                                         w * strides[1] + dwidth, c_b],
                       axis=[dheight, dwidth]), tag="pool_max")


@reg.register_schedule("max_pool2d", level=20)
def schedule_max_pool_2d_nChwc(attrs, outs, target):
    with tvm.target.create(target):
        return schedule_pool(outs)


@reg.register_compute("avg_pool2d", level=20)
def compute_avg_pool_2d_nChwc(attrs, inputs, _):
    kernel = attrs.get_int_tuple("pool_size")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    ceil_mode = attrs.get_bool("ceil_mode")
    layout = attrs["layout"]

    if len(inputs[0].shape) == 4:
        return nn.pool(inputs[0], kernel,
                       strides, padding, "max", ceil_mode, layout)

    batch, channel_chunk, height, width, channel_block = inputs[0].shape
    if padding[0] > 0 or padding[1] > 0:
        pad_data = nn.pad(inputs[0], (0, 0, padding[0], padding[1], 0),
                          pad_value=tvm.min_value(inputs[0].dtype), name="data_pad")
    else:
        pad_data = inputs[0]

    out_height = nn.util.simplify((height - kernel[0] + 2 * padding[0]) // strides[0] + 1)
    out_width = nn.util.simplify((width - kernel[1] + 2 * padding[1]) // strides[1] + 1)
    dheight = tvm.reduce_axis((0, kernel[0]))
    dwidth = tvm.reduce_axis((0, kernel[1]))
    oshape = (batch, channel_chunk, out_height, out_width, channel_block)
    tsum =  tvm.compute(oshape,
                       lambda n, c_c, h, w, c_b:
                       tvm.sum(pad_data[n, c_c, h * strides[0] + dheight,
                                        w * strides[1] + dwidth, c_b], axis=[dheight, dwidth]),
                       tag="pool_avg")
    return tvm.compute(oshape, \
                       lambda n, c_c, h, w, c_b: \
                           tsum[n, c_c, h, w, c_b] / (kernel[0] * kernel[1]), \
                       tag=tag.ELEMWISE)


@reg.register_schedule("avg_pool2d", level=20)
def schedule_avg_pool_2d_nChwc(attrs, outs, target):
    with tvm.target.create(target):
        return schedule_pool(outs)


@reg.register_compute("global_max_pool2d", level=20)
def compute_global_max_pool2d(attrs, inputs, _):
    if len(inputs[0].shape) == 4:
        return nn.global_pool(inputs[0])
    batch, channel_chunk, height, width, channel_block = inputs[0].shape
    dheight = tvm.reduce_axis((0, height))
    dwidth = tvm.reduce_axis((0, width))
    return tvm.compute((batch, channel_chunk, 1, 1, channel_block), lambda n, c_c, h, w, c_b: \
                        tvm.max(inputs[0][n, c_c, dheight, dwidth, c_b], axis=[dheight, dwidth]), \
                        tag="global_pool_max")


@reg.register_schedule("global_max_pool2d", level=20)
def schedule_global_max_pool_2d_nChwc(attrs, outs, target):
    with tvm.target.create(target):
        return schedule_pool(outs)


@reg.register_compute("global_avg_pool2d", level=20)
def compute_global_avg_pool2d(attrs, inputs, _):
    if len(inputs[0].shape) == 4:
        return nn.global_pool(inputs[0])
    batch, channel_chunk, height, width, channel_block = inputs[0].shape
    dheight = tvm.reduce_axis((0, height))
    dwidth = tvm.reduce_axis((0, width))
    oshape = (batch, channel_chunk, 1, 1,channel_block)
    tsum =  tvm.compute(oshape,
                        lambda n, c_c, h, w, c_b:
                        tvm.sum(inputs[0][n, c_c, h, w, c_b], axis=[dheight, dwidth]),
                        tag="pool_avg")
    return tvm.compute(oshape, \
                       lambda n, c_c, h, w, c_b: \
                           tsum[n, c_c, h, w, c_b] / (height * width), \
                       tag=tag.ELEMWISE)


@reg.register_schedule("global_avg_pool2d", level=20)
def schedule_global_avg_pool_2d_nChwc(attrs, outs, target):
    with tvm.target.create(target):
        return schedule_pool(outs)

