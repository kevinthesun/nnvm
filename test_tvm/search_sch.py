import nnvm.compiler
import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
import argparse
import subprocess

from tvm.contrib import graph_runtime
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from multiprocessing import Process, Queue
from schedule_pack.avx512_conv_fwd import AVX512ConvCommonFwd, AVX512Conv1x1Fwd, _WORKLOADS

def get_factor(i):
    rtv = []
    for j in range(1, i + 1):
        if i % j == 0:
            rtv.append(j)
    return rtv
        

for i in range(20, 34):
    workload = _WORKLOADS[i]
    isize = workload[2]
    ic = workload[4]
    oc = workload[5]
    k = workload[6]
    p = workload[8]
    s = workload[10]
    osize = (isize - k + 2 * p)/s + 1

    ic_bn = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    oc_bn = get_factor(oc)#[64, 32, 16, 8, 4, 2, 1]

    if ic < 4:
        ic_bn = [ic]
    else:
        temp = []
        for x in ic_bn:
            if ic >= x:
                temp.append(x)
        ic_bn = temp

    """
    if oc < 4:
        oc_bn = [oc]
    else:
        temp = []
        for x in oc_bn:
            if oc // x >= 2:
                temp.append(x)
        oc_bn = temp
    """

    reg_bn = get_factor(osize)
    reg_bn_l = [1, 2] if osize > 1 else [1]
    #reg_candidate = [32, 16, 8, 4, 2, 1]
    #for item in reg_candidate:
    #    if len(reg_bn) >= 3:
    #        break
    #    if osize >= item:
    #        reg_bn.append(item)

    unroll = [True, False]

    run_times = 100
    batch_size = 1
    target = "llvm -mcpu=skylake-avx512"

    for ic_b in ic_bn:
        for oc_b in oc_bn:
            for reg_b in reg_bn:
                if k != 1:
                    for ul in unroll:
                        sch = AVX512ConvCommonFwd(*[ic_b, oc_b, reg_b, ul])
                        try:
                            cmd = "TVM_NUM_THREADS=18 python test_conv.py --workload_idx %d --ic_bn %d --oc_bn %d --reg_r %d --unroll %d" % (i, ic_b, oc_b, reg_b, 1 if ul else 0)
                            process = subprocess.Popen(cmd, shell=True)
                            process.wait()
                        except Exception as e:
                            print("Mismatch!")
                else:
                    for reg_l in reg_bn_l:
                        sch = AVX512Conv1x1Fwd(*[ic_b, oc_b, reg_l, reg_b])
                        try:
                            cmd = "TVM_NUM_THREADS=18 python test_conv.py --workload_idx %d --unit %d --ic_bn %d --oc_bn %d --reg_l %d --reg_r %d" % (i, 1, ic_b, oc_b, reg_l, reg_b)
                            process = subprocess.Popen(cmd, shell=True)
                            process.wait()
                        except Exception as e:
                            print("Mismatch!")
    #sorted_result = sorted(result, key=lambda x: x[1])
