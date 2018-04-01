# pylint: disable=invalid-name,unused-variable,invalid-name
"""1x1 Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from topi.util import get_const_tuple
from topi.nn.conv2d import _get_schedule, _get_workload
from topi.nn.util import infer_pad, infer_stride
from topi.nn.pad import pad

AVX512Conv1x1Fwd = namedtuple('AVX512Conv1x1Fwd', ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor'])

def _declaration_conv(data_vec, wkl, sch, kernel):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data_vec.shape)
    oc_c, _, _, oc_b, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    in_height -= 2 * HPAD
    in_width -= 2 * WPAD
    in_channel = in_channel_block * in_channel_chunk
    num_filter = oc_c * oc_b

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn] *
                kernel[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                axis=[ic]), name='conv2d', tag='conv2d')
    return conv


def _schedule_conv(s, data_pad, data_vec, wkl, sch, conv_out, output, last):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 or WPAD != 0)

    A0, A1 = data_pad, data_vec

    # schedule data
    if hasattr(s[A1].op, 'tag') and  "conv2d_data_pack" in s[A1].op.tag:
        if DOPAD:
            s[A0].compute_inline()
        batch, ic_chunk, ih, iw, ic_block = s[A1].op.axis
        parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
        s[A1].parallel(parallel_axis)
    elif DOPAD:
        n, c_c, h, w, c_b = s[A0].op.axis
        parallel_axis = s[A0].fuse(n, c_c, h)
        s[A0].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=sch.ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(oc_chunk, oh_outer)
    s[CC].compute_at(s[C], parallel_axis)
    if C == O:
        s[C].parallel(parallel_axis)

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].fuse(oc_chunk, oh_outer)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis

        # oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
        oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)

        s[O].parallel(parallel_axis)

    return s
