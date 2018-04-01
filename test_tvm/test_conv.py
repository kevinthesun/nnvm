import nnvm.compiler
import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
import argparse

from tvm.contrib import graph_runtime
#from topi.nn.conv2d import _get_workload, _get_schedule
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from topi.nn.conv2d import _get_workload
from schedule_pack.avx512_conv_fwd import AVX512ConvCommonFwd, AVX512Conv1x1Fwd, _WORKLOADS, _get_schedule
from schedule_pack.avx512_conv_fwd import *
#from conv_layout import *


parser = argparse.ArgumentParser(description='Benchmark convolution workload.')
parser.add_argument('--workload_idx', type=int, required=True,
                    help="Workload index")
parser.add_argument('--unit', type=int, default=0,
                    help="Whether 1x1")
parser.add_argument('--ic_bn', type=int, required=True,
                    help="ic_bn")
parser.add_argument('--oc_bn', type=int, required=True,
                    help="oc_bn")
parser.add_argument('--reg_l', type=int, default=1,
                    help="reg_l")
parser.add_argument('--reg_r', type=int, required=True,
                    help="reg_r")
parser.add_argument('--unroll', type=int, default=1,
                    help="unroll")

args = parser.parse_args()
if args.unit > 0:
    input_sch = AVX512Conv1x1Fwd(*[args.ic_bn, args.oc_bn, args.reg_l, args.reg_r])
else:
    input_sch = AVX512ConvCommonFwd(*[args.ic_bn, args.oc_bn, args.reg_r, args.unroll > 0])

#sch = AVX512ConvCommonFwd(ic_bn=64, oc_bn=16, reg_n=16, unroll_kw=True)

@_get_schedule.register("cpu", override=True)
def _get_schedule_conv(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    return input_sch

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
    kernel_shape = (channels // sch.oc_bn, in_channels // sch.ic_bn, kernel_size[0], kernel_size[1], sch.ic_bn, sch.oc_bn) if kernel_size[0] != 1 or kernel_size[1] != 1 else (channels // sch.oc_bn, in_channels // sch.ic_bn, sch.ic_bn, sch.oc_bn, kernel_size[0], kernel_size[1])
    bias_shape = (channels // sch.oc_bn, sch.oc_bn)

    if attrs.get_bool("use_bias"):
        reg.assign_shape(p_in_shapes, 2, True, bias_shape)
    reg.assign_shape(p_in_shapes, 1, True, kernel_shape)
    reg.assign_shape(p_out_shapes, 0, False, out_shape)

    return True

def end2end_benchmark(model, target, batch_size, workload_idx, sch, run_times, test_mkl=True):
    #print(workload)
    workload = _WORKLOADS[workload_idx]
    print(workload)
    ic_bn = sch.ic_bn
    oc_bn = sch.oc_bn
    num_classes = 100
    image_shape = (workload.in_filter // ic_bn, workload.height, workload.width, ic_bn)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, workload.out_filter // oc_bn,
                 (workload.height + 2 * workload.hpad - workload.hkernel) // workload.hstride + 1,
                 (workload.width + 2 * workload.wpad - workload.wkernel) // workload.wstride + 1,
                 oc_bn)

    ctx = tvm.cpu()
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    origin_data = np.transpose(data_array, (0, 1, 4, 2, 3))
    origin_data = np.reshape(origin_data, (batch_size, workload.in_filter, workload.height, workload.width))
    mx_data = mx.nd.array(origin_data)
    block = gluon.nn.Conv2D(workload.out_filter, kernel_size=(workload.hkernel, workload.wkernel),
                            padding=(workload.hpad, workload.wpad), strides=(workload.hstride, workload.wstride),
                            in_channels=workload.in_filter, weight_initializer='normal', use_bias=True)
    block.initialize(ctx=mx.cpu())

    net, params = nnvm.frontend.from_mxnet(block)
    for key, value in params.items():
        if "conv" in key:
            if "weight" in key:
                oc, ic, kh, kw = value.asnumpy().shape
                tmp = np.reshape(value.asnumpy(), (oc // oc_bn, oc_bn, ic // ic_bn, ic_bn, kh, kw))
                params[key] = tvm.nd.array(np.transpose(tmp, (0, 2, 4, 5, 3, 1))) if args.unit == 0 else tvm.nd.array(np.transpose(tmp, (0, 2, 3, 1, 4, 5)))
            elif "bias" in key:
                oc, = value.asnumpy().shape
                params[key] = tvm.nd.array(np.reshape(value.asnumpy(), (oc // oc_bn, oc_bn)))

    opt_level = 3
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    module.set_input('data', input_data)

    s = time.time()
    for _ in range(run_times):
        module.run()
    tvm_time = time.time() - s
    #print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, tvm_time))
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))
    _, _, oh, ow, _ = tvm_out.asnumpy().shape
    np_tvm_out = np.transpose(tvm_out.asnumpy(), (0, 1, 4, 2, 3))
    np_tvm_out = np.reshape(np_tvm_out, (batch_size, workload.out_filter, oh, ow))
    #print(tvm_out.shape)


    mxnet_out = block(mx_data)
    #print("MKL %s inference time for batch size of %d: %f" % (model, batch_size, mkl_time))
    try:
        np.testing.assert_array_almost_equal(np_tvm_out, mxnet_out.asnumpy(), decimal=2)
    except:
        print("Error!")
        return
    print("Workload #%d | schedule: %s | time: %f" % (workload_idx, str(sch), tvm_time))

    return tvm_time

end2end_benchmark("conv", "llvm -mcpu=skylake-avx512", 1, args.workload_idx, input_sch, 100)

"""
if __name__ == "__main__":

    run_times = 50
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    tvm_dense = 0
    tvm_mobilenet = 0
    mkl_dense = 0
    mkl_mobilenet = 0
    td = end2end_benchmark('conv', target, batch_size, args.workload_idx, run_times)

    #print("tvm conv: %f, mkl conv: %f"
    #      % (tvm_dense/(run_times * batch_size), mkl_dense/(run_times * batch_size)))
"""
