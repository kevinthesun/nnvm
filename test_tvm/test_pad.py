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

if __name__ == '__main__':
    data = tvm.placeholder((1, 1, 512, 512, 3), name='data')
    out = topi.nn.pad(data, (0, 0, 3, 3, 0), name="data_pad")
    s = tvm.create_schedule(out)

    ctx = tvm.cpu(0)
    f = tvm.build(s, [data, out], "llvm -mcpu=core-avx2")

    aa = tvm.nd.array(np.random.uniform(size=(1, 1, 512, 512, 3)).astype(data.dtype), ctx)
    bb = tvm.nd.array(np.zeros((1, 1, 518, 518, 3)).astype("float32"), ctx)

    for _ in range(1000):
        f(aa, bb)

