import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
import nnvm
import json
import argparse

from tvm.contrib import graph_runtime
from topi.nn.conv2d import Workload
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from symbol.symbol_factory import get_symbol
from schedule_pack.avx512_conv_fwd import _WORKLOADS

_WORKLOADS1 = [
"""
    # workloads of resnet18_v1 on imagenet 12 0-11
    Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    # workloads of resnet34_v1 on imagenet, no extra workload required
    # workloads of resnet50_v1 on imagenet 14 12-25
    Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
    # workloads of resnet101_v1 on imagenet, no extra workload required
    # workloads of resnet152_v1 on imagenet, no extra workload required
    # workloads of resnet18_v2 on imagenet, no extra workload required
    # workloads of resnet34_v2 on imagenet, no extra workload required
    # SSD resnet50_v2 224*224 118-137
    Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 2048, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 7, 7, 2048, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 4, 4, 512, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 4, 4, 512, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 2, 2, 256, 24, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 2, 2, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 1, 1, 256, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 1, 1, 256, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 1, 1, 128, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 1, 1, 128, 16, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 1024, 84, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 7, 7, 2048, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 4, 4, 512, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 2, 2, 256, 126, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 1, 1, 256, 84, 3, 3, 1, 1, 1, 1),
"""
]


num_classes = 1000
batch_size = 1
image_shape = (3, 512, 512)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_classes)

def get_conv2d_workload(model, in_dtype='float32', out_dtype='float32'):
    #_, arg_params, aux_params = mx.model.load_checkpoint('model/ssd_resnet50_512', 0)
    sym = get_symbol('vgg16_reduced', 512, num_classes=20)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(data_shapes=[('data', data_shape)])
    #mod.set_params(arg_params, aux_params, allow_extra=True, allow_missing=True)
    mod.init_params()

    #print(arg_params)
    net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])
    g = nnvm.graph.create(net)
    g = nnvm.compiler.graph_attr.set_shape_inputs(g, {'data': data_shape})
    g = g.apply("InferShape")
    g_dict = json.loads(g.json())
    print(g.json())
    node_list = g_dict["nodes"]
    shape_list = g_dict['attrs']['shape'][1]
    node_map = g_dict["node_row_ptr"]
    workload_list = []
    workload_set = set()
    for workload in _WORKLOADS:
        workload_set.add(workload)

    for node in node_list:
        if node['op'] != 'conv2d':
            continue
        attrs = node["attrs"]
        if int(attrs["groups"]) != 1:
            continue
        input_index = node["inputs"][0][0]
        input_shape = shape_list[node_map[input_index]]
        if attrs["layout"] == "NCHW":
            height, width, in_filter = input_shape[2], input_shape[3], input_shape[1]
        else:
            height, width, in_filter = input_shape[1], input_shape[2], input_shape[3]
        out_filter = attrs["channels"]
        hkernel, wkernel = (attrs["kernel_size"])[1:-1].split(',')
        hpad, wpad = (attrs["padding"])[1:-1].split(',')
        hstride, wstride = (attrs["strides"])[1:-1].split(',')

        workload = Workload(*[in_dtype, out_dtype, height, width, in_filter, int(out_filter),
                              int(hkernel), int(wkernel), int(hpad), int(wpad), int(hstride), int(wstride)])
        #print(workload)
        if workload not in workload_set:
            workload_set.add(workload)
            workload_list.append(workload)

    return workload_list


if __name__ == "__main__":
    model = "ssd"
    workload_list = get_conv2d_workload(model)
    for workload in workload_list:
       print('    Workload(\'%s\', \'%s\', %d, %d, %d, %d, %d, %d, %d, %d, %d, %d),' % (
           workload.in_dtype, workload.out_dtype, workload.height, workload.width,
           workload.in_filter, workload.out_filter, workload.hkernel, workload.wkernel,
           workload.hpad, workload.wpad, workload.hstride, workload.wstride))
