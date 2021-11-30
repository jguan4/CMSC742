# Copyright 2021 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
import numpy as np
from python.constructs.loops import Loops
from python.constructs.output import Output
from python.constructs.optimizer import Adam
from python.constructs.metrics import Metrics
from python.constructs.loggable import Loggable
from python.constructs.ema_weights import EMAWeights
from python.constructs.loss import MeanSquaredError
from python.constructs.loss import CategoricalCrossEntropy
from python.constructs.loss import MarginPlusMeanSquaredError
from python.constructs.loss import CategoricalCrossEntropyPlusMeanSquaredError
from python.constructs.learning_rate import ManualExponentialDecay
from python.input.MNIST_input_pipeline import MNIST
from python.input.cifar10_input_pipeline import Cifar10
from python.input.cifar100_input_pipeline import Cifar100
from python.input.smallNORB_input_pipeline import smallNORB
from python.models.SmallImageBranchingMerging import SmallImageBranchingMerging
from python.models.Sketch import Sketch
from python.models.HCS import HCS
from python.models.CP_Decomp import CP_decomp
import tensorflow as tf
import pandas as pd


def find_best_top1_name(log_dir):
    files = [f for f in os.listdir(log_dir) if 'best' in f]
    best_name = files[0].split('.')[0]
    best_file = log_dir+'/'+best_name
    return best_file

def go(run_name, data_dir, log_dir, input_pipeline, merge_strategy, loss_type,
       use_hvcs=True, hvc_type=1, hvc_dims=None, total_convolutions=None,
       branches_after=None, batch_size = 120, realization = 10, kernel_reconstruct_method = 'sk', params = None):

    weights_file = find_best_top1_name(log_dir)
    colnames = ["realization{0}".format(r) for r in range(realization)]
    colnames.insert(0,'num_of_var')
    colnames.insert(0,'factor')
    colnames.insert(0,'l')
    colnames.insert(0,'k')

    if reconstruct_method == 'sk':
        sk_l = params['sk_l']
        sk_k = params['sk_k']
        sk_factor = params['sk_factor']
        facstr = 'True' if sk_factor else 'False'
        vals = np.full((len(sk_l)*len(sk_k)+1, 4+realization), None)
    elif reconstruct_method == 'hcs':
        hcs_l = params['hcs_l']
        hcs_k = params['hcs_k']
        hcs_factor = params['hcs_factor']
        facstr = 'True' if hcs_factor else 'False'
        vals = np.full((len(hcs_l)*len(hcs_k)+1, 4+realization), None)
    elif reconstruct_method == 'cp':
        cp_l = params['cp_l']
        cp_k = params['cp_k']
        cp_factor = params['cp_factor']
        facstr = 'True' if cp_factor else 'False'
        realization = 1
        if cp_factor:
            vals = np.full((len(cp_l)+1, 3+realization), None)
        else:
            vals = np.full((len(cp_k)+1, 3+realization), None)
        colnames = ["realization{0}".format(r) for r in range(realization)]
        colnames.insert(0,'num_of_var')
        colnames.insert(0,'factor')
        colnames.insert(0,'comp_dim')

    if input_pipeline == 3:
        in_pipe = Cifar10(data_dir, False, 0)
    elif input_pipeline == 4:
        in_pipe = Cifar100(data_dir, False, 0)
    elif input_pipeline == 5:
        in_pipe = smallNORB(data_dir, False, 48, 32)
    else:
        in_pipe = MNIST(data_dir, False, 1)

    out = Output(log_dir, run_name, None, None)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("Building model...")

        model = SmallImageBranchingMerging(in_pipe.get_class_count(),
                    in_pipe.get_image_size(), in_pipe.get_image_channels(),
                    merge_strategy, use_hvcs, hvc_type, 32, 16, hvc_dims,
                    total_convolutions, branches_after, True)

        if loss_type == 2:
            loss = MeanSquaredError()
        elif loss_type == 3:
            loss = MarginPlusMeanSquaredError()
        elif loss_type == 4:
            loss = CategoricalCrossEntropyPlusMeanSquaredError()
        else:
            loss = CategoricalCrossEntropy()

        lr          = ManualExponentialDecay(0.001, 0.98, 1e-7)
        optimizer   = Adam(lr)
        metrics     = Metrics(True, False)
        ema_weights = EMAWeights(0.999, model.get_all_trainable_variables())
        loops       = Loops(in_pipe, out, strategy, model, optimizer,
                        lr, loss, metrics, ema_weights, batch_size)

        out.log_method_info(Loggable.get_this_method_info())
        out.log_loggables([out, in_pipe, model,
            lr, optimizer, loss, metrics, ema_weights, loops])
        
   
        print("Restoring weights file: {}".format(weights_file))
        ckpt = tf.train.Checkpoint(
                vars=model.get_all_savable_variables())
        ckpt.restore(weights_file).expect_partial()

        kernels = model.get_conv_kernels()
        caps = model.get_cap_weights()
        loss,top1,top5 = loops._validate_test()
        vals[0][0] = 0
        vals[0][1] = 0
        if reconstruct_method == 'cp':
            vals[0][3] = top1.numpy()
        else:
            vals[0][4] = top1.numpy()
        num_k = 0
        for i in range(len(kernels)):
            ki = kernels[i]
            num_k += tf.size(ki).numpy()
        num_c = 0
        for i in range(len(caps)):
            num_c += tf.size(caps[i]).numpy()
        if reconstruct_method == 'cp':
            vals[0][2] = num_k + num_c
        else:
            vals[0][3] = num_k + num_c

        counter = 1
        if reconstruct_method == 'sk':
            for k in sk_k:
                for l in sk_l:
                    vals[counter][0] = k
                    vals[counter][1] = l
                    vals[counter][2] = sk_factor
                    print("Start k = {0}, l = {1}:".format(k,l))
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()

                        sketch = Sketch(l,k,sk_factor)
                        c_sketch = Sketch(l, k, sk_factor)
                        c_sketch.start_compressing(caps)
                        sketch.start_compressing(kernels)
                        if r == 0:
                            vals[counter][2] = sketch.var_num

                        sk_kernels = sketch.reconstruct_kernels()
                        model.load_conv_kernels(sk_kernels)
                        sk_caps = c_sketch.reconstruct_kernels()
                        model.load_cap_weights(sk_caps)
                        loss,top1,top5 = loops._validate_test()
                        vals[counter][r+4] = top1.numpy()

                    counter +=1
        elif reconstruct_method == 'hcs':
            for k in hcs_k:
                for l in hcs_l:
                    vals[counter][0] = k
                    vals[counter][1] = l
                    vals[counter][2] = hcs_factor
                    print("Start k = {0}, l = {1}:".format(k,l))
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()

                        hcs = HCS(l,k,hcs_factor)
                        hcs.start_compressing(kernels)
                        if r == 0:
                            vals[counter][2] = hcs.var_num

                        hcs_kernels = hcs.reconstruct_kernels()
                        model.load_conv_kernels(hcs_kernels)
                        loss,top1,top5 = loops._validate_test()
                        vals[counter][r+4] = top1.numpy()

                    counter +=1

        elif reconstruct_method == 'cp':
            if cp_factor:
                for l in cp_l:
                    vals[counter][0] = l
                    vals[counter][1] = cp_factor
                    print("Start l = {0}:".format(l))
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        CPD = CP_decomp(cp_factor, l)
                        CPD.decomp_kernels(kernels)
                        cp_kernels = CPD.reconstruct_kernels()
                        if r == 0:
                            vals[counter][2] = CPD.var_num
                        model.load_conv_kernels(cp_kernels)
                        loss, top1, top5 = loops._validate_test()
                        vals[counter][r + 3] = top1.numpy()

                    counter += 1
            else:
                counter = 1
                for k in cp_k:
                    vals[counter][0] = k
                    vals[counter][1] = cp_factor
                    print("Start k = {0}:".format(k))
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        CPD = CP_decomp(cp_factor, k)
                        CPD.decomp_kernels(kernels)
                        cp_kernels = CPD.reconstruct_kernels()
                        if r == 0:
                            vals[counter][2] = CPD.var_num
                        model.load_conv_kernels(cp_kernels)
                        loss, top1, top5 = loops._validate_test()
                        vals[counter][r + 3] = top1.numpy()

                    counter += 1

    df = pd.DataFrame(vals, columns=colnames)
    df.to_csv("{0}/{1}/test_factor{2}.csv".format(log_dir,run_name,facstr))

################################################################################
if __name__ == "__main__":

    # do not need to change
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"../Datasets/mnist_data")
    # common parameters
    p.add_argument("--input_pipeline", default=1, type=int)
    p.add_argument("--merge_strategy", default=0, type=float)
    p.add_argument("--use_hvcs", default=True, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--hvc_dims", default=[64, 112, 160], type=int)
    p.add_argument("--total_convolutions", default=9, type=int)
    p.add_argument("--branches_after", default=[2, 5, 8])
    p.add_argument("--loss_type", default=1, type=int)
    a = p.parse_args()

    # choose merge strategy, options: 0, 1, 2
    merge_strategy = 0
    # choose reconstruction strategy, options 'sk' (sketch), 'hcs' (higher-order count sketch), 'cp' (CP decomposition)
    reconstruct_method = 'sk'
    # number of realizations to average over, used for 'sk' and 'hcs'
    realization_num = 1

    log_dir = "../logs_ms{0}/".format(merge_strategy)
    sub_dir = os.listdir(log_dir)[0]
    log_dir += sub_dir 
    run_name = 'test_' + reconstruct_method

    if reconstruct_method == 'sk':
        params = {'sk_l':[2], 'sk_k':[2], 'sk_factor':True}
    elif reconstruct_method == 'hcs':
        params = {'hcs_l':[2], 'hcs_k':[2], 'hcs_factor':True}
    elif reconstruct_method == 'cp':
        params = {'cp_l':[2], 'cp_k':[2], 'cp_factor': True}


    go(run_name = run_name, data_dir=a.data_dir, log_dir=log_dir,
       input_pipeline=a.input_pipeline, merge_strategy=merge_strategy,
       use_hvcs=a.use_hvcs, hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions, branches_after=a.branches_after, loss_type = a.loss_type, realization = realization_num, kernel_reconstruct_method = reconstruct_method, params = params)
