import mxnet as mx
import numpy as np
import time

from collections import namedtuple
from symbol.symbol_factory import get_symbol

Batch = namedtuple('Batch', ['data'])

run_times = 100
batch_size = 1
image_shape = (3, 512, 512)
data_shape = (batch_size,) + image_shape

data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
mx_data = mx.nd.array(data_array)

_, arg_params, aux_params = mx.model.load_checkpoint('model/ssd_512', 0)

sym = get_symbol('vgg16_reduced', 512, num_classes=20)
#sym.save("ssd-symbol.json")

mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(data_shapes=[('data', data_shape)])
#mod.set_params(arg_params, aux_params, allow_extra=True, allow_missing=True)
mod.init_params()

# Warmup
data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
mx_data = mx.nd.array(data_array)
for _ in range(100):
    mod.forward(Batch(data=[mx_data]), is_train=False)
    for output in mod.get_outputs():
        output.wait_to_read()


mkl_time = 0
for _ in range(run_times):
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    mx_data = mx.nd.array(data_array)
    s = time.time()
    mod.forward(Batch(data=[mx_data]), is_train=False)
    for output in mod.get_outputs():
        output.wait_to_read()
    mkl_time += time.time() - s
print(mod.get_outputs()[0].shape)
print("MKL %s inference time for batch size of %d: %f" % ('ssd_resnet50', batch_size, mkl_time))
