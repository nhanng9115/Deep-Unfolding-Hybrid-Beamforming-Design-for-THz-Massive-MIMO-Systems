# -* coding: utf-8 -*-
'''
@File；Global_Vars.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-15 15:45
Configure global variables for DNN_mmWave_torch.py
'''
import os
import torch
import math
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Parameters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# System parameters
Nt = 128
Nr = 2
Nrf = Nr
Ns = Nrf
N = 2 * Nt * Nrf
Ncl = 1  ## number of clusters
Nray = 4 ## number of rays in each cluster
GHz = 1e+9
K = 128
Num_layers = 4#math.ceil(math.log2(Nt))

Bandwidth = 30 * GHz  # system bandwidth
fc = 300 * GHz   # carrier frequency
Array_Type = 'UPA'
init_scheme = 2  # 0 for random initialization or 2 fro OMP initialization
Sub_Connected = False
# Sub_Structure_Type = 'fixed'
Sub_Structure_Type = 'dyn'

train_data_name = 'train_set.hdf5'
GenNum_Batch_tr = 50  # used for generating training data
Gen_Batch_size_tr = 10  # used for generating training data
training_set_size = GenNum_Batch_tr * Gen_Batch_size_tr

test_data_name = 'test_set.hdf5'
GenNum_Batch_te = 10  # used for generating testing data
Gen_Batch_size_te = 10  # used for generating testing data
testing_set_size = GenNum_Batch_te * Gen_Batch_size_te
test_batch_size = 10

# Training parameters
Seed_train = 1
Seed_test = 101

training_set_size_truncated = 100#int(Num_layers*100)
train_batch_size = 20
Ntrain_batch_total = 500  # total number of training batches without considering the iteratively generated batches
Ntrain_Batch_perEpoch = training_set_size_truncated // train_batch_size
Ntrain_Epoch = math.ceil(Ntrain_batch_total/Ntrain_Batch_perEpoch)
# Ntrain_Batch = min(training_set_size_truncated // train_batch_size, 100)
# Ntrain_Batch = 2

training_method = 'unsupervised'
Iterative_Training = True
Iterations_train = 3

Increamental_Learning = False
Black_box = False
Wideband_Net = False
Keep_Bias = False
Residule_NN = False
Loss_coef = 1

SUM_LOSS = 1
Weight_decay = 0  # add L2 regularizer to weight, the penalty is larger with high Weight_decay
start_learning_rate = 1e-4
Log_interval = 10  # interval for print loss
set_Lr_decay = False
Lr_min = 1e-5
Lr_keep_steps = 5
Lr_decay_factor = 0.95


Iterations_test = 10


# To save trained model
dataset_file = "./trained_model/" + str(Nt) + "x" + str(Nr) + "x" + str(Nrf) + "x" + str(K) + "x" + str(Num_layers) + "/"
directory_model = dataset_file
dat_file_name = directory_model + "data.mat"
model_file_name = directory_model + "trained_model.pth"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)



Cuda_set = 1  # whether to use GPU
# 检测GPU
MainDevice = torch.device("cuda:0" if torch.cuda.is_available() and Cuda_set else "cpu")
NumGPU = torch.cuda.device_count()
print(MainDevice, flush=True)
use_gpu = torch.cuda.is_available() and Cuda_set
if use_gpu:
    print('using GPU for training:\n',flush=True)
    print('cuda.is_available:', torch.cuda.is_available(),flush=True)
    print('cuda.device_count:', torch.cuda.device_count(),flush=True)
    print('cuda.device_name:', torch.cuda.get_device_name(0),flush=True)
else:
    print('Using CPU for training:\n',flush=True)


pass

