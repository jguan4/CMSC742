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
from python.models.Tucker import Tucker
import tensorflow as tf
import pandas as pd


def find_best_top1_name(log_dir):
    files = [f for f in os.listdir(log_dir) if 'best' in f]
    best_name = files[0].split('.')[0]
    best_file = log_dir+'/'+best_name
    return best_file

def create_col_names(kernel_reconstruct_method, cap_reconstruct_method, realization):
    if kernel_reconstruct_method == 'cp' or kernel_reconstruct_method == 'tucker':
        kercolnames = ["kernel_reconstruct_method", "comp_dim", "factor","num_of_var"]
        kervar_ind = 3
    elif kernel_reconstruct_method == 'none':
        kercolnames = ["kernel_reconstruct_method","num_of_var"]
        kervar_ind = 1
    else:
        kercolnames = ["kernel_reconstruct_method", "k", "l", "factor","num_of_var"]
        kervar_ind = 4

    if cap_reconstruct_method == 'cp' or cap_reconstruct_method == 'tucker':
        capcolnames = ["cap_reconstruct_method", "comp_dim", "factor","num_of_var"]
        capvar_ind = kervar_ind + 4
    elif cap_reconstruct_method == 'none':
        capcolnames = ["cap_reconstruct_method","num_of_var"]
        capvar_ind = kervar_ind + 2
    else:
        capcolnames = ["cap_reconstruct_method", "k", "l", "factor","num_of_var"]
        capvar_ind = kervar_ind + 5

    reacolnames = ["realization{0}".format(r) for r in range(realization)]
    colnames = kercolnames + capcolnames  + reacolnames
    headingnum = len(colnames)
    rea0ind = capvar_ind + 1
    return colnames,headingnum,kervar_ind, capvar_ind, rea0ind

def sk_recon(model, kernels,l,k,factor,toggle):
    sketch = Sketch(l,k,factor)
    sketch.start_compressing(kernels)
    sk_kernels = sketch.reconstruct_kernels()
    var_num = sketch.var_num
    if toggle == 'conv':
        model.load_conv_kernels(sk_kernels)
    elif toggle == 'cap':
        model.load_cap_weights(sk_kernels)
    return var_num

def hcs_recon(model, kernels,l,k,factor,toggle):
    hcs = HCS(l,k,factor)
    hcs.start_compressing(kernels)
    hcs_kernels = hcs.reconstruct_kernels()
    var_num = hcs.var_num
    if toggle == 'conv':
        model.load_conv_kernels(hcs_kernels)
    elif toggle == 'cap':
        model.load_cap_weights(hcs_kernels)
    return var_num

def cp_recon(model, kernels,compdim,factor,toggle):
    CPD = CP_decomp(factor, compdim)
    CPD.decomp_kernels(kernels)
    cp_kernels = CPD.reconstruct_kernels()
    var_num = CPD.var_num
    if toggle == 'conv':
        model.load_conv_kernels(cp_kernels)
    elif toggle == 'cap':
        model.load_cap_weights(cp_kernels)
    return var_num

def tk_recon(model, kernels,compdim,factor,toggle):
    TK = Tucker(factor, compdim)
    TK.decomp_kernels(kernels)
    tk_kernels = TK.reconstruct_kernels()
    var_num = TK.var_num
    if toggle == 'conv':
        model.load_conv_kernels(tk_kernels)
    elif toggle == 'cap':
        model.load_cap_weights(tk_kernels)
    return var_num


def go(run_name, data_dir, log_dir, input_pipeline, merge_strategy, loss_type,
       use_hvcs=True, hvc_type=1, hvc_dims=None, total_convolutions=None,
       branches_after=None, batch_size = 120, realization = 10, kernel_reconstruct_method = 'none', kernel_params = None, cap_reconstruct_method = 'none', cap_params = None):

    weights_file = find_best_top1_name(log_dir)

    if kernel_reconstruct_method == 'sk':
        ker_sk_l = kernel_params['sk_l']
        ker_sk_k = kernel_params['sk_k']
        ker_sk_factor = kernel_params['sk_factor']
        ker_facstr = 'True' if ker_sk_factor else 'False'
        ker_realization = realization
        ker_rownum = len(ker_sk_l)*len(ker_sk_k)

    elif kernel_reconstruct_method == 'hcs':
        ker_hcs_l = kernel_params['hcs_l']
        ker_hcs_k = kernel_params['hcs_k']
        ker_hcs_factor = kernel_params['hcs_factor']
        ker_facstr = 'True' if ker_hcs_factor else 'False'
        ker_realization = realization
        ker_rownum = len(ker_hcs_l)*len(ker_hcs_k)

    elif kernel_reconstruct_method == 'cp':
        ker_cp_factor = kernel_params['cp_factor']
        if ker_cp_factor:
            ker_cpdim = kernel_params['cp_l']
        else:
            ker_cpdim = kernel_params['cp_k']
        ker_facstr = 'True' if ker_cp_factor else 'False'
        ker_realization = 1
        ker_rownum = len(ker_cpdim)

    elif kernel_reconstruct_method == 'tucker':
        ker_tk_factor = kernel_params['tk_factor']
        if ker_tk_factor:
            ker_tkdim = kernel_params['tk_l']
        else:
            ker_tkdim = kernel_params['tk_k']
        ker_facstr = 'True' if ker_tk_factor else 'False'
        ker_realization = 1
        ker_rownum = len(ker_tkdim)

    else:
        ker_realization = 1
        ker_facstr = 'False'
        ker_rownum = 1

    if cap_reconstruct_method == 'sk':
        cap_sk_l = cap_params['sk_l']
        cap_sk_k = cap_params['sk_k']
        cap_sk_factor = cap_params['sk_factor']
        cap_facstr = 'True' if cap_sk_factor else 'False'
        cap_realization = realization
        cap_rownum = len(cap_sk_l)*len(cap_sk_k)

    elif cap_reconstruct_method == 'hcs':
        cap_hcs_l = cap_params['hcs_l']
        cap_hcs_k = cap_params['hcs_k']
        cap_hcs_factor = cap_params['hcs_factor']
        cap_facstr = 'True' if cap_hcs_factor else 'False'
        cap_realization = realization
        cap_rownum = len(cap_hcs_l)*len(cap_hcs_k)

    elif cap_reconstruct_method == 'cp':
        cap_cp_factor = cap_params['cp_factor']
        if cap_cp_factor:
            cap_cpdim = cap_params['cp_l']
        else:
            cap_cpdim = cap_params['cp_k']
        cap_facstr = 'True' if cap_cp_factor else 'False'
        cap_realization = 1
        cap_rownum = len(cap_cpdim)

    elif cap_reconstruct_method == 'tucker':
        cap_tk_factor = cap_params['tk_factor']
        if cap_tk_factor:
            cap_tkdim = cap_params['tk_l']
        else:
            cap_tkdim = cap_params['tk_k']
        cap_facstr = 'True' if cap_tk_factor else 'False'
        cap_realization = 1
        cap_rownum = len(cap_tkdim)

    else:
        cap_facstr = 'False'
        cap_realization = 1
        cap_rownum = 1
           
    realization = np.maximum(cap_realization,ker_realization)
    colnames, headingnum, kervar_ind, capvar_ind, rea0ind = create_col_names(kernel_reconstruct_method, cap_reconstruct_method, realization)

    vals = np.full((ker_rownum*cap_rownum+1, headingnum), None)

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
        vals[0][rea0ind] = top1.numpy()
        num_k = 0
        for i in range(len(kernels)):
            ki = kernels[i]
            num_k += tf.size(ki).numpy()
        vals[0][kervar_ind] = num_k
        num_c = 0
        for i in range(len(caps)):
            num_c += tf.size(caps[i]).numpy()
        vals[0][capvar_ind] = num_c

        counter = 1
        infos = []
        if kernel_reconstruct_method == 'sk':
            for k in ker_sk_k:
                for l in ker_sk_l:
                    if cap_reconstruct_method == 'sk':
                        for ck in cap_sk_k:
                            for cl in cap_sk_l:
                                infos.append(kernel_reconstruct_method)
                                infos.append(k)
                                infos.append(l)
                                infos.append(ker_facstr)
                                infos.append(cap_reconstruct_method)
                                infos.append(ck)
                                infos.append(cl)
                                infos.append(cap_facstr)
                                # print("Start k = {0}, l = {1}:".format(k,l))

                                for r in range(realization):
                                    ckpt.restore(weights_file).expect_partial()
                                    ker_varnum = sk_recon(model, kernels,l,k,ker_sk_factor,'conv')
                                    cap_varnum = sk_recon(model, caps, cl, ck,cap_sk_factor,'cap')
                                    if r == 0:
                                        infos.insert(kervar_ind,ker_varnum)
                                        infos.append(cap_varnum)
                                    loss,top1,top5 = loops._validate_test()
                                    infos.append(top1.numpy())
                                vals[counter] = infos
                                counter +=1
                                infos = []

                    elif cap_reconstruct_method == 'hcs':
                        for ck in cap_hcs_k:
                            for cl in cap_hcs_l:
                                infos.append(kernel_reconstruct_method)
                                infos.append(k)
                                infos.append(l)
                                infos.append(ker_facstr)
                                infos.append(cap_reconstruct_method)
                                infos.append(ck)
                                infos.append(cl)
                                infos.append(cap_facstr)
                                for r in range(realization):
                                    ckpt.restore(weights_file).expect_partial()
                                    ker_varnum = sk_recon(model, kernels,l,k,ker_sk_factor,'conv')
                                    cap_varnum = hcs_recon(model, caps,cl,ck,cap_hcs_factor,'cap')
                                    if r == 0:
                                        infos.insert(kervar_ind,ker_varnum)
                                        infos.append(cap_varnum)
                                    loss,top1,top5 = loops._validate_test()
                                    infos.append(top1.numpy())
                        
                                vals[counter] = infos
                                counter +=1
                                infos = []

                    elif cap_reconstruct_method == 'cp':
                        for ccompdim in cap_cpdim:
                            infos.append(kernel_reconstruct_method)
                            infos.append(k)
                            infos.append(l)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ccompdim)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = sk_recon(model, kernels,l,k,ker_sk_factor,'conv')
                                cap_varnum = cp_recon(model,caps, ccompdim, cap_cp_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())

                            vals[counter] = infos
                            counter +=1
                            infos = []

                    elif cap_reconstruct_method == 'tucker':
                        for ccompdim in cap_tkdim:
                            infos.append(kernel_reconstruct_method)
                            infos.append(k)
                            infos.append(l)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ccompdim)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = sk_recon(model, kernels,l,k,ker_sk_factor,'conv')
                                cap_varnum = tk_recon(model,caps, ccompdim, cap_tk_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())

                            vals[counter] = infos
                            counter +=1
                            infos = []

                    else:
                        infos.append(kernel_reconstruct_method)
                        infos.append(k)
                        infos.append(l)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(num_c)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = sk_recon(model, kernels,l,k,ker_sk_factor,'conv')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())
                        
                        vals[counter] = infos
                        counter +=1
                        infos = []

        elif kernel_reconstruct_method == 'hcs':
            for k in ker_hcs_k:
                for l in ker_hcs_l:
                    # print("Start k = {0}, l = {1}:".format(k,l))
                    if cap_reconstruct_method == 'sk':
                        for ck in cap_sk_k:
                            for cl in cap_sk_l:
                                infos.append(kernel_reconstruct_method)
                                infos.append(k)
                                infos.append(l)
                                infos.append(ker_facstr)
                                infos.append(cap_reconstruct_method)
                                infos.append(ck)
                                infos.append(cl)
                                infos.append(cap_facstr)
                                # print("Start k = {0}, l = {1}:".format(k,l))

                                for r in range(realization):
                                    ckpt.restore(weights_file).expect_partial()
                                    ker_varnum = hcs_recon(model, kernels,l,k,ker_hcs_factor,'conv')
                                    cap_varnum = sk_recon(model, caps, cl, ck,cap_sk_factor,'cap')
                                    if r == 0:
                                        infos.insert(kervar_ind,ker_varnum)
                                        infos.append(cap_varnum)
                                    loss,top1,top5 = loops._validate_test()
                                    infos.append(top1.numpy())
                                vals[counter] = infos
                                counter +=1
                                infos = []

                    elif cap_reconstruct_method == 'hcs':
                        for ck in cap_hcs_k:
                            for cl in cap_hcs_l:
                                infos.append(kernel_reconstruct_method)
                                infos.append(k)
                                infos.append(l)
                                infos.append(ker_facstr)
                                infos.append(cap_reconstruct_method)
                                infos.append(ck)
                                infos.append(cl)
                                infos.append(cap_facstr)
                                for r in range(realization):
                                    ckpt.restore(weights_file).expect_partial()
                                    ker_varnum = hcs_recon(model, kernels,l,k,ker_hcs_factor,'conv')
                                    cap_varnum = hcs_recon(model, caps,cl,ck,cap_hcs_factor,'cap')
                                    if r == 0:
                                        infos.insert(kervar_ind,ker_varnum)
                                        infos.append(cap_varnum)
                                    loss,top1,top5 = loops._validate_test()
                                    infos.append(top1.numpy())
                        
                                vals[counter] = infos
                                counter +=1
                                infos = []

                    elif cap_reconstruct_method == 'cp':
                        for ccompdim in cap_cpdim:
                            infos.append(kernel_reconstruct_method)
                            infos.append(k)
                            infos.append(l)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ccompdim)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = hcs_recon(model, kernels,l,k,ker_hcs_factor,'conv')
                                cap_varnum = cp_recon(model,caps, ccompdim, cap_cp_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())

                            vals[counter] = infos
                            counter +=1
                            infos = []

                    elif cap_reconstruct_method == 'tucker':
                        for ccompdim in cap_tkdim:
                            infos.append(kernel_reconstruct_method)
                            infos.append(k)
                            infos.append(l)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ccompdim)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = hcs_recon(model, kernels,l,k,ker_hcs_factor,'conv')
                                cap_varnum = tk_recon(model,caps, ccompdim, cap_tk_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())

                            vals[counter] = infos
                            counter +=1
                            infos = []

                    else:
                        infos.append(kernel_reconstruct_method)
                        infos.append(k)
                        infos.append(l)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(num_c)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = hcs_recon(model, kernels,l,k,ker_hcs_factor,'conv')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())
                        
                        vals[counter] = infos
                        counter +=1
                        infos = []
                    
        elif kernel_reconstruct_method == 'cp':
            for compdim in ker_cpdim:  
                if cap_reconstruct_method == 'sk':
                    for ck in cap_sk_k:
                        for cl in cap_sk_l:
                            infos.append(kernel_reconstruct_method)
                            infos.append(compdim)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ck)
                            infos.append(cl)
                            infos.append(cap_facstr)
                            # print("Start k = {0}, l = {1}:".format(k,l))

                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = cp_recon(model, kernels,compdim,ker_cp_factor,'conv')
                                cap_varnum = sk_recon(model, caps, cl, ck,cap_sk_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())
                            vals[counter] = infos
                            counter +=1
                            infos = []

                elif cap_reconstruct_method == 'hcs':
                    for ck in cap_hcs_k:
                        for cl in cap_hcs_l:
                            infos.append(kernel_reconstruct_method)
                            infos.append(compdim)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ck)
                            infos.append(cl)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = cp_recon(model, kernels,compdim,ker_cp_factor,'conv')
                                cap_varnum = hcs_recon(model, caps,cl,ck,cap_hcs_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())
                    
                            vals[counter] = infos
                            counter +=1
                            infos = []

                elif cap_reconstruct_method == 'cp':
                    for ccompdim in cap_cpdim:
                        infos.append(kernel_reconstruct_method)
                        infos.append(compdim)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(ccompdim)
                        infos.append(cap_facstr)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = cp_recon(model, kernels,compdim,ker_cp_factor,'conv')
                            cap_varnum = cp_recon(model,caps, ccompdim, cap_cp_factor,'cap')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())

                        vals[counter] = infos
                        counter +=1
                        infos = []

                elif cap_reconstruct_method == 'tucker':
                    for ccompdim in cap_tkdim:
                        infos.append(kernel_reconstruct_method)
                        infos.append(compdim)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(ccompdim)
                        infos.append(cap_facstr)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = cp_recon(model, kernels,compdim,ker_cp_factor,'conv')
                            cap_varnum = tk_recon(model,caps, ccompdim, cap_tk_factor,'cap')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())

                        vals[counter] = infos
                        counter +=1
                        infos = []

                else:
                    infos.append(kernel_reconstruct_method)
                    infos.append(compdim)
                    infos.append(ker_facstr)
                    infos.append(cap_reconstruct_method)
                    infos.append(num_c)
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        ker_varnum = cp_recon(model, kernels,compdim,ker_cp_factor,'conv')
                        if r == 0:
                            infos.insert(kervar_ind,ker_varnum)
                        loss,top1,top5 = loops._validate_test()
                        infos.append(top1.numpy())
                    
                    vals[counter] = infos
                    counter +=1
                    infos = []

        elif kernel_reconstruct_method == 'tucker':
            for compdim in ker_tkdim:
                if cap_reconstruct_method == 'sk':
                    for ck in cap_sk_k:
                        for cl in cap_sk_l:
                            infos.append(kernel_reconstruct_method)
                            infos.append(compdim)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ck)
                            infos.append(cl)
                            infos.append(cap_facstr)
                            # print("Start k = {0}, l = {1}:".format(k,l))

                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = tk_recon(model, kernels,compdim,ker_tk_factor,'conv')
                                cap_varnum = sk_recon(model, caps, cl, ck,cap_sk_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())
                            vals[counter] = infos
                            counter +=1
                            infos = []

                elif cap_reconstruct_method == 'hcs':
                    for ck in cap_hcs_k:
                        for cl in cap_hcs_l:
                            infos.append(kernel_reconstruct_method)
                            infos.append(compdim)
                            infos.append(ker_facstr)
                            infos.append(cap_reconstruct_method)
                            infos.append(ck)
                            infos.append(cl)
                            infos.append(cap_facstr)
                            for r in range(realization):
                                ckpt.restore(weights_file).expect_partial()
                                ker_varnum = tk_recon(model, kernels,compdim,ker_tk_factor,'conv')
                                cap_varnum = hcs_recon(model, caps,cl,ck,cap_hcs_factor,'cap')
                                if r == 0:
                                    infos.insert(kervar_ind,ker_varnum)
                                    infos.append(cap_varnum)
                                loss,top1,top5 = loops._validate_test()
                                infos.append(top1.numpy())
                    
                            vals[counter] = infos
                            counter +=1
                            infos = []

                elif cap_reconstruct_method == 'cp':
                    for ccompdim in cap_cpdim:
                        infos.append(kernel_reconstruct_method)
                        infos.append(compdim)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(ccompdim)
                        infos.append(cap_facstr)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = tk_recon(model, kernels,compdim,ker_tk_factor,'conv')
                            cap_varnum = cp_recon(model,caps, ccompdim, cap_cp_factor,'cap')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())

                        vals[counter] = infos
                        counter +=1
                        infos = []

                elif cap_reconstruct_method == 'tucker':
                    for ccompdim in cap_tkdim:
                        infos.append(kernel_reconstruct_method)
                        infos.append(compdim)
                        infos.append(ker_facstr)
                        infos.append(cap_reconstruct_method)
                        infos.append(ccompdim)
                        infos.append(cap_facstr)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            ker_varnum = tk_recon(model, kernels,compdim,ker_tk_factor,'conv')
                            cap_varnum = tk_recon(model,caps, ccompdim, cap_tk_factor,'cap')
                            if r == 0:
                                infos.insert(kervar_ind,ker_varnum)
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())

                        vals[counter] = infos
                        counter +=1
                        infos = []

                else:
                    infos.append(kernel_reconstruct_method)
                    infos.append(compdim)
                    infos.append(ker_facstr)
                    infos.append(cap_reconstruct_method)
                    infos.append(num_c)
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        ker_varnum = tk_recon(model, kernels,compdim,ker_tk_factor,'conv')
                        if r == 0:
                            infos.insert(kervar_ind,ker_varnum)
                        loss,top1,top5 = loops._validate_test()
                        infos.append(top1.numpy())
                    
                    vals[counter] = infos
                    counter +=1
                    infos = []

        else:
            if cap_reconstruct_method == 'sk':
                for ck in cap_sk_k:
                    for cl in cap_sk_l:
                        infos.append(kernel_reconstruct_method)
                        infos.append(num_k)
                        infos.append(cap_reconstruct_method)
                        infos.append(ck)
                        infos.append(cl)
                        infos.append(cap_facstr)
                        # print("Start k = {0}, l = {1}:".format(k,l))

                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            cap_varnum = sk_recon(model, caps, cl, ck,cap_sk_factor,'cap')
                            if r == 0:
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())
                        vals[counter] = infos
                        counter +=1
                        infos = []

            elif cap_reconstruct_method == 'hcs':
                for ck in cap_hcs_k:
                    for cl in cap_hcs_l:
                        infos.append(kernel_reconstruct_method)
                        infos.append(num_k)
                        infos.append(cap_reconstruct_method)
                        infos.append(ck)
                        infos.append(cl)
                        infos.append(cap_facstr)
                        for r in range(realization):
                            ckpt.restore(weights_file).expect_partial()
                            cap_varnum = hcs_recon(model, caps,cl,ck,cap_hcs_factor,'cap')
                            if r == 0:
                                infos.append(cap_varnum)
                            loss,top1,top5 = loops._validate_test()
                            infos.append(top1.numpy())
                
                        vals[counter] = infos
                        counter +=1
                        infos = []

            elif cap_reconstruct_method == 'cp':
                for ccompdim in cap_cpdim:
                    infos.append(kernel_reconstruct_method)
                    infos.append(num_k)
                    infos.append(cap_reconstruct_method)
                    infos.append(ccompdim)
                    infos.append(cap_facstr)
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        cap_varnum = cp_recon(model,caps, ccompdim, cap_cp_factor,'cap')
                        if r == 0:
                            infos.append(cap_varnum)
                        loss,top1,top5 = loops._validate_test()
                        infos.append(top1.numpy())

                    vals[counter] = infos
                    counter +=1
                    infos = []

            elif cap_reconstruct_method == 'tucker':
                for ccompdim in cap_tkdim:
                    infos.append(kernel_reconstruct_method)
                    infos.append(num_k)
                    infos.append(cap_reconstruct_method)
                    infos.append(ccompdim)
                    infos.append(cap_facstr)
                    for r in range(realization):
                        ckpt.restore(weights_file).expect_partial()
                        cap_varnum = tk_recon(model,caps, ccompdim, cap_tk_factor,'cap')
                        if r == 0:
                            infos.append(cap_varnum)
                        loss,top1,top5 = loops._validate_test()
                        infos.append(top1.numpy())

                    vals[counter] = infos
                    counter +=1
                    infos = []     

            else:
                infos.append(kernel_reconstruct_method)
                infos.append(num_k)
                infos.append(cap_reconstruct_method)
                infos.append(num_c)
                for r in range(realization):
                    ckpt.restore(weights_file).expect_partial()
                    loss,top1,top5 = loops._validate_test()
                    infos.append(top1.numpy())
                
                vals[counter] = infos
                counter +=1
                infos = []
            

    df = pd.DataFrame(vals, columns=colnames)
    df.to_csv("{0}/{1}/test_kerfactor{2}_capfactor{3}.csv".format(log_dir,run_name,ker_facstr, cap_facstr))

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
    merge_strategy = 1
    # choose reconstruction strategy, options 'sk' (sketch), 'hcs' (higher-order count sketch), 'cp' (CP decomposition), 'tucker' (tucker_decomposition), none' (no reconstruction)
    kernel_reconstruct_method = 'sk'
    cap_reconstruct_method = 'hcs'
    # number of realizations to average over, used for 'sk' and 'hcs'
    realization_num = 15

    log_dir = "../logs_ms{0}/".format(merge_strategy)
    sub_dir = os.listdir(log_dir)[0]
    log_dir += sub_dir 
    run_name = 'test_' + kernel_reconstruct_method +"_"+cap_reconstruct_method

    if kernel_reconstruct_method == 'sk':
        kernel_params = {'sk_l':[2,3], 'sk_k':[2,3], 'sk_factor':False}
    elif kernel_reconstruct_method == 'hcs':
        kernel_params = {'hcs_l':[2,3], 'hcs_k':[2,3], 'hcs_factor':True}
    elif kernel_reconstruct_method == 'cp':
        kernel_params = {'cp_l':[2,3], 'cp_k':[2,3], 'cp_factor': True}
    elif kernel_reconstruct_method == 'tucker':
        kernel_params = {'tk_l':[2,3], 'tk_k':[2,3], 'tk_factor': False}
    elif kernel_reconstruct_method == 'none':
        kernel_params = None

    if cap_reconstruct_method == 'sk':
        cap_params = {'sk_l':[2,3], 'sk_k':[2,3], 'sk_factor':True}
    elif cap_reconstruct_method == 'hcs':
        cap_params = {'hcs_l':[2,3], 'hcs_k':[2,3], 'hcs_factor':True}
    elif cap_reconstruct_method == 'cp':
        cap_params = {'cp_l':[2,3], 'cp_k':[2,3], 'cp_factor': True}
    elif cap_reconstruct_method == 'tucker':
        cap_params = {'tk_l':[2,3], 'tk_k':[2,3], 'tk_factor': False}
    elif cap_reconstruct_method == 'none':
        cap_params = None

    go(run_name = run_name, data_dir=a.data_dir, log_dir=log_dir,
       input_pipeline=a.input_pipeline, merge_strategy=merge_strategy,
       use_hvcs=a.use_hvcs, hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions, branches_after=a.branches_after, loss_type = a.loss_type, realization = realization_num, kernel_reconstruct_method = kernel_reconstruct_method, kernel_params = kernel_params, cap_reconstruct_method = cap_reconstruct_method, cap_params = cap_params)
