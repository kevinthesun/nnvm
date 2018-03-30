import nnvm.compiler
import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
import topi

from tvm.contrib import graph_runtime
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from collections import namedtuple
from conv_layout import *


def end2end_benchmark():
    #print(workload)
    num_classes = 1000
    image_shape = (64, 512, 512)
    data_shape = (batch_size,) + image_shape
    #out_shape = (batch_size, workload.out_filter,
    #             (workload.height + 2 * workload.hpad - workload.hkernel) // workload.hstride + 1,
    #             (workload.width + 2 * workload.wpad - workload.wkernel) // workload.wstride + 1)

    ctx = tvm.cpu()
    model = "pool"
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    mx_data = mx.nd.array(data_array)

    data = mx.sym.Variable("data")
    sym = mx.sym.Pooling(data, kernel=(2,2), stride=(2,2), pool_type="max")
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 64, 512, 512))],
             label_shapes=mod._label_shapes)
    mod.init_params()


    net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])
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
    tvm_out = module.get_output(0, out=tvm.nd.empty((1, 64, 256, 256)))
    #print(tvm_out.shape)
    print(tvm_time)

if __name__ == "__main__":

    run_times = 100
    batch_size = 1
    target = "llvm -mcpu=core-avx2"
    tvm_dense = 0
    tvm_mobilenet = 0
    mkl_dense = 0
    mkl_mobilenet = 0
    end2end_benchmark()


