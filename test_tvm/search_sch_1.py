import nnvm.compiler
import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import subprocess
import traceback

from schedule_pack.avx512_conv_common import AVX512ConvCommonFwd
from schedule_pack.avx512_conv_1x1 import AVX512Conv1x1Fwd
from schedule_pack.avx512_conv_fwd import AVX512ConvCommonFwd, AVX512Conv1x1Fwd, _WORKLOADS, _get_schedule
from schedule_pack.avx512_conv_fwd import *
from topi.nn.conv2d import _get_workload
from multiprocessing import Process, Queue
from test_conv import end2end_benchmark

global sch
run_times = 50

_SCH_TO_DECL_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._declaration_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._declaration_conv
}

_SCH_TO_SCH_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._schedule_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._schedule_conv
}

@_get_schedule.register("cpu", override=True)
def _get_schedule_conv(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    global sch
    return sch


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
            data_pad = data_vec.op.input_tensors[0] if "conv2d_data_pack" in data_vec.op.tag else data_vec
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
            sch = _get_schedule_conv(wkl)
            _SCH_TO_SCH_FUNC[type(sch)](s, data_pad, data_vec, wkl, sch, conv_out, op.output(0), outs[0])


    traverse(outs[0].op)
    return s

@reg.register_schedule("conv2d", level=30)
def schedule_conv2d(attrs, outs, target):
    with tvm.target.create(target):
        padding = attrs.get_int_tuple("padding")
        strides = attrs.get_int_tuple("strides")
        channels = attrs.get_int("channels")
        kernel_size = attrs.get_int_tuple("kernel_size")
        use_bias = attrs.get_bool("use_bias")
        return schedule_conv2d_nChwc(padding, strides, channels, kernel_size, use_bias, outs)


@reg.register_infershape("conv2d", level=30)
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
    kernel_shape = (channels // sch.oc_bn, in_channels // sch.ic_bn, kernel_size[0], kernel_size[1], sch.ic_bn, sch.oc_bn)
    bias_shape = (channels // sch.oc_bn, sch.oc_bn)

    if attrs.get_bool("use_bias"):
        reg.assign_shape(p_in_shapes, 2, True, bias_shape)
    reg.assign_shape(p_in_shapes, 1, True, kernel_shape)
    reg.assign_shape(p_out_shapes, 0, False, out_shape)

    return True

def get_factor(i):
    rtv = []
    for j in range(1, i + 1):
        if i % j == 0:
            rtv.append(j)
    return rtv
        
ic_bn = [32]
out_sch = []
for i in range(1, 2):
    workload = _WORKLOADS[i]
    isize = workload[2]
    ic = workload[4]
    oc = workload[5]
    k = workload[6]
    p = workload[8]
    s = workload[10]
    osize = (isize - k + 2 * p)/s + 1

    oc_bn = [16, 32, 64]#get_factor(oc)#[64, 32, 16, 8, 4, 2, 1]
    #oc_bn.remove(1)

    reg_bn = get_factor(osize)
    reg_bn_l = [1, 2] if osize > 1 else [1]

    unroll = [True, False]

    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"

    min_time = 9999
    current_sch = ""
    for ic_b in ic_bn:
        for oc_b in oc_bn:
            for reg_b in reg_bn:
                if k != 1:
                    for ul in unroll:
                        global sch
                        sch = AVX512ConvCommonFwd(*[ic_b, oc_b, reg_b, ul])
                        try:
                            current_time = end2end_benchmark("conv", target, 1, i, sch, 1)
                            if current_time < min_time:
                                min_time = current_time
                                current_sch = sch
                        except Exception as e:
                            traceback.print_exc()
                else:
                    for reg_l in reg_bn_l:
                        global sch
                        sch = AVX512Conv1x1Fwd(*[ic_b, oc_b, reg_l, reg_b])
                        try:
                            current_time = end2end_benchmark("conv", target, 1, i, sch, 1)
                            if current_time < min_time:
                                min_time = current_time
                                current_sch = sch
                        except Exception as e:
                            traceback.print_exc()
    out_sch.append(current_sch)
    ic_bn = [current_sch.oc_bn]

for sch in out_sch:
    print(sch)
    #sorted_result = sorted(result, key=lambda x: x[1])
