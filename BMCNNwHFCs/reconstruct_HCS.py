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
from python.models.HCS import HCS
import tensorflow as tf
import pandas as pd


def make_compdim(compdims):
    ccompdims = []
    for c in range(len(compdims)):
        compdim = compdims[c]
        ccompdim = []
        for i in range(len(compdim)):
            [d1,d2] = compdim[i]
            ccompdim.append([1,1,d1,d2])
        ccompdims.append(ccompdim)
    return ccompdims


def go(run_name, data_dir, log_dir, output_file, input_pipeline, merge_strategy, loss_type, compdims, 
       use_hvcs=True, hvc_type=1, hvc_dims=None, total_convolutions=None,
       branches_after=None, batch_size = 120, realization = 10):

    files = []
    for dirname, _, filenames in os.walk(log_dir):
        file = list(set([os.path.join(dirname,
            os.path.splitext(fn)[0]) for fn in filenames]))
        if len(file) > 0:
            files.append(file[0])
    files = ['../logs_ms0/20211103121518/best_top1-135']
    # print(files)
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
        
        compdims = make_compdim(compdims)
        vals = np.full((len(compdims)+1, 2+realization), None)
        colnames = ["realization{0}".format(r) for r in range(realization)]
        colnames.insert(0,'num_of_var')
        colnames.insert(0,'comp_dim')

        for weights_file in files:
            print("Restoring weights file: {}".format(weights_file))
            ckpt = tf.train.Checkpoint(
                    vars=model.get_all_savable_variables())
            ckpt.restore(weights_file).expect_partial()
            weights = model.get_all_trainable_variables()
            # for i in range(len(weights)):
            #     print(weights[i].name, weights[i].shape)
            #     input()

            kernels = model.get_conv_kernels()
            weights = model.get_cap_weights()

            loss,top1,top5 = loops._validate_test()
            vals[0][0] = 0
            vals[0][2] = top1.numpy()
            num_k = 0
            for i in range(len(kernels)):
                ki = kernels[i]
                num_k += tf.size(ki).numpy()
            vals[0][1] = num_k


            # print(model.get_all_trainable_variables())
            # input()
            # branch_weights.append(model.branch_weights.variable.numpy())
            counter = 1
            for compdim in compdims:
                vals[counter][0] = compdim

                print("Start compdim = {0}:".format(compdim))
                for r in range(realization):
                    ckpt.restore(weights_file).expect_partial()

                    HCSo = HCS(compdim)
                    HCSo.generate_HCSs(kernels)
                    if r == 0:
                        vals[counter][2] = HCSo.var_num

                    HCS_kernels = HCSo.reconstruct_kernels()
                    model.load_conv_kernels(HCS_kernels)
                    loss,top1,top5 = loops._validate_test()
                    vals[counter][r+2] = top1.numpy()

                counter +=1

            # branch_weights.append(model.get_all_trainable_variables())

    
    df = pd.DataFrame(vals, columns=colnames)
    df.to_csv("{0}/{1}/HCS_test.csv".format(log_dir,run_name))
    # print("Saving final branch weights...")
    # # (False Positive)
    # # noinspection PyTypeChecker
    # np.savetxt(output_file, np.array(branch_weights), delimiter=',', fmt='%0f')
    # print("Finished.")


################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default=r"test")
    p.add_argument("--data_dir", default=r"../Datasets/mnist_data")
    p.add_argument("--log_dir", default=r"../logs_ms0/20211103121518")
    p.add_argument("--output_file",
        default=r"../logs_ms0/20211103121518/final_branch_weights.txt")
    p.add_argument("--input_pipeline", default=1, type=int)
    p.add_argument("--merge_strategy", default=0, type=float)
    p.add_argument("--use_hvcs", default=True, type=bool)
    p.add_argument("--hvc_type", default=2, type=int)
    p.add_argument("--hvc_dims", default=[64, 112, 160], type=int)
    p.add_argument("--total_convolutions", default=9, type=int)
    p.add_argument("--branches_after", default=[2, 5, 8])
    p.add_argument("--loss_type", default=1, type=int)
    p.add_argument("--compdims", default=[[[1,16],[16,48],[48,64],[64,80],[80,96],[96,112],[112,128],[128,144],[144,160]]])
    p.add_argument("--realization", default=10, type=int)
    a = p.parse_args()

    # go(run_name=datetime.now().strftime("%Y%m%d%H%M%S"), end_epoch=300,
    #    data_dir=r"../../../Datasets/mnist_data", input_pipeline=1,
    #    log_dir="../logs_ms0", batch_size=120, merge_strategy=0, loss_type=1,
    #    use_hvcs=True, hvc_type=2, hvc_dims=[64, 112, 160],
    #    use_augmentation=True, augmentation_type=1, total_convolutions=9,
    #    branches_after=[2, 5, 8], reconstruct_from_hvcs=True)

    go(run_name = a.run_name, data_dir=a.data_dir, log_dir=a.log_dir, output_file=a.output_file,
       input_pipeline=a.input_pipeline, merge_strategy=a.merge_strategy,
       use_hvcs=a.use_hvcs, hvc_type=a.hvc_type, hvc_dims=a.hvc_dims,
       total_convolutions=a.total_convolutions, branches_after=a.branches_after,loss_type=a.loss_type,realization = a.realization, compdims = a.compdims)
