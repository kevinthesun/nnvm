import mxnet as mx
import numpy as np
import nnvm
import tvm
import time
import json

from collections import namedtuple
from symbol.symbol_factory import get_symbol
from topi.nn.conv2d import Workload
from mxnet import gluon
from tvm.contrib import graph_runtime
from conv_layout import *
from schedule_pack.avx512_conv_fwd import *
from schedule_pack.avx512_conv_fwd import _get_schedule_conv, _WORKLOADS, _SCHEDULES

def weight_prepack(net, params):
    in_dtype='float32'
    out_dtype='float32'
    g = nnvm.graph.create(net)
    g = nnvm.compiler.graph_attr.set_shape_inputs(g, {'data': data_shape})
    g = g.apply("InferShape")
    g_dict = json.loads(g.json())
    node_list = g_dict["nodes"]
    shape_list = g_dict['attrs']['shape'][1]
    node_map = g_dict["node_row_ptr"]
    for node in node_list:
        if node['op'] != 'conv2d':
            continue
        attrs = node["attrs"]
        if int(attrs["groups"]) != 1:
            continue
        input_index = node["inputs"][0][0]
        kernel_index = node["inputs"][1][0]
        bias_index = node["inputs"][2][0] if len(node["inputs"]) > 2 else None
        input_shape = shape_list[node_map[input_index]]
        height, width, in_filter = input_shape[2], input_shape[3], input_shape[1] * input_shape[4] if len(input_shape) == 5 else input_shape[1]
        out_filter = attrs["channels"]
        hkernel, wkernel = (attrs["kernel_size"])[1:-1].split(',')
        hpad, wpad = (attrs["padding"])[1:-1].split(',')
        hstride, wstride = (attrs["strides"])[1:-1].split(',')

        workload = Workload(*[in_dtype, out_dtype, height, width, in_filter, int(out_filter),
                              int(hkernel), int(wkernel), int(hpad), int(wpad), int(hstride), int(wstride)])
        sch = _get_schedule_conv(workload)
        ic_bn = sch.ic_bn
        oc_bn = sch.oc_bn
        kernel_name = node_list[kernel_index]['name']
        oc, ic, kh, kw = params[kernel_name].asnumpy().shape
        tmp = np.reshape(params[kernel_name].asnumpy(), (oc // oc_bn, oc_bn, ic // ic_bn, ic_bn, kh, kw))
        params[kernel_name] = tvm.nd.array(np.transpose(tmp, (0, 2, 4, 5, 3, 1))) if kh != 1 or kw != 1 else tvm.nd.array(np.transpose(tmp, (0, 2, 3, 1, 4, 5)))
        if bias_index is not None:
            bias_name = node_list[bias_index]['name']
            oc, = params[bias_name].asnumpy().shape
            params[bias_name] = tvm.nd.array(np.reshape(params[bias_name].asnumpy(), (oc // oc_bn, oc_bn)))

    for node in node_list:
        if node['op'] != 'batch_norm' or node['name'] == "batch_norm0":
            continue
        gamma_index = node["inputs"][1][0]
        beta_index = node["inputs"][2][0]
        mm_index = node["inputs"][3][0]
        mv_index = node["inputs"][4][0]

        gamma_name = node_list[gamma_index]['name']
        beta_name = node_list[beta_index]['name']
        mm_name = node_list[mm_index]['name']
        mv_name = node_list[mv_index]['name']

        _, _, _, _, oc_bn = input_shape = shape_list[node_map[node["inputs"][0][0]]]
        oc, = params[gamma_name].asnumpy().shape
        for key in [gamma_name, beta_name, mm_name, mv_name]:
            params[key] = tvm.nd.array(np.reshape(params[key].asnumpy(), (oc // oc_bn, oc_bn)))


Batch = namedtuple('Batch', ['data'])

run_times = 100

target = "llvm -mcpu=skylake-avx512"
batch_size = 1
image_shape = (3, 512, 512)
data_shape = (batch_size,) + image_shape


_, arg_params, aux_params = mx.model.load_checkpoint('model/ssd_resnet50_512', 0)
sym = get_symbol('resnet50', 512, num_classes=20)

mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', data_shape)])
mod.set_params(arg_params, aux_params)
#mod.init_params()

#print(arg_params)
net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])
weight_prepack(net, params)

ctx = tvm.cpu()
opt_level = 3
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**params)

data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
tvm_data = tvm.nd.array(data_array, ctx=ctx)
module.set_input('data', tvm_data)

# Warm up
for _ in range(100):
    module.run()

s = time.time()
for _ in range(run_times):
    module.run()
tvm_time = time.time() - s

mx_data = mx.nd.array(data_array)
np.testing.assert_array_almost_equal(tvm_data.asnumpy(), mx_data.asnumpy(), decimal=6)
mod.forward(Batch(data=[mx_data]), is_train=False)
mx_out = mod.get_outputs()[0]
print(mx_out.shape)
#tvm_out = module.get_output(0, out=tvm.nd.empty((batch_size, 6132, 6)))
#_, _, oh, ow, _ = tvm_out.asnumpy().shape
#np_tvm_out = np.transpose(tvm_out.asnumpy(), (0, 1, 4, 2, 3))
#np_tvm_out = np.reshape(np_tvm_out, (batch_size, -1, oh, ow))
#np_tvm_out = tvm_out.asnumpy()

#np.testing.assert_array_almost_equal(np_tvm_out, mx_out.asnumpy(), decimal=3)

print("TVM %s inference time for batch size of %d: %f" % ('ssd_resnet50', batch_size, tvm_time))
