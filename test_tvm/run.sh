#!/bin/bash
for i in {1..18}
do
    #KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$i python test_mkldnn.py
    TVM_NUM_THREADS=$i python test_tvm.py
done

