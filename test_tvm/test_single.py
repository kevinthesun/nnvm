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
#from conv_layout import *
from schedule_pack.avx512_conv_fwd import *

Batch = namedtuple('Batch', ['data'])

def end2end_benchmark():
    #print(workload)
    num_classes = 1000
    image_shape = (2, 128, 128, 32)
    data_shape = (batch_size,) + image_shape
    #out_shape = (batch_size, workload.out_filter,
    #             (workload.height + 2 * workload.hpad - workload.hkernel) // workload.hstride + 1,
    #             (workload.width + 2 * workload.wpad - workload.wkernel) // workload.wstride + 1)

    ctx = tvm.cpu()
    model = "conv"
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    data_array = np.transpose(data_array, (0, 1, 4, 2, 3))
    data_array = np.reshape(data_array,(1, 64, 128, 128))
    mx_data = mx.nd.array(data_array)

    """
    data_array = np.random.uniform(0, 255, size=(1, 64, 512, 512)).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    mx_data = mx.nd.array(data_array)"""

    data = mx.sym.Variable("data")
    sym = mx.sym.Convolution(data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2", no_bias=False)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 64, 128, 128))],
             label_shapes=mod._label_shapes)
    mod.init_params()


    net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])
    ic_bn = 32
    oc_bn = 32
    for key, value in params.items():
        if "conv" in key:
            if "weight" in key:
                oc, ic, kh, kw = value.asnumpy().shape
                tmp = np.reshape(value.asnumpy(), (oc // oc_bn, oc_bn, ic // ic_bn, ic_bn, kh, kw))
                params[key] = tvm.nd.array(np.transpose(tmp, (0, 2, 4, 5, 3, 1)))
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
    #tvm_out = module.get_output(0, out=tvm.nd.empty((1, 1, 512, 512, 64)))
    #np_tvm_out = np.transpose(tvm_out.asnumpy(), (0, 1, 4, 2, 3))
    #np_tvm_out = np.reshape(np_tvm_out, (1, 64, 512, 512))
    #print(tvm_out.shape)
    print(tvm_time)

    s = time.time()
    for _ in range(1):
        mod.forward(Batch([mx_data]))
        for output in mod.get_outputs():
            output.wait_to_read()
    mkl_time = time.time() - s
    print(mkl_time)
    #np.testing.assert_array_almost_equal(np_tvm_out, mod.get_outputs()[0].asnumpy(), decimal=3)


if __name__ == "__main__":

    run_times = 100
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"
    tvm_dense = 0
    tvm_mobilenet = 0
    mkl_dense = 0
    mkl_mobilenet = 0
    end2end_benchmark()
