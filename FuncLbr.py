# -* coding: utf-8 -*-
'''
@File；FuncLbr.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-16 15:15
 This is the self-defined function/module library
'''
import numpy as np
import torch

from Global_Vars import *
import math
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def array_dimension(Nt):
    '''
    :param Nt: number of total antennas
    :return: the configuration of UPA that minimizes beam squint effect
    '''
    n = math.ceil(Nt ** 0.5)
    for i in range(n+1,1,-1):
        if Nt%i==0:
            Nth = i
            Ntv = int(Nt/i)
            break
    return Nth, Ntv

def pulase_filter(t, Ts, beta):
    '''
    Raised cosine filter
    :param t: time slot
    :param Ts: sampling frequency
    :param beta: roll-off factor
    :return: filtered value
    '''
    if abs(t-Ts/2/beta)/abs(t) <1e-4 or abs(t+Ts/2/beta)/abs(t)<1e-4:
        p = np.pi/4 * np.sinc(1/2/beta)
    else:
        p = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts)/(1-(2*beta*t/Ts)**2)
    return p


def array_response(Nh,Nv, Angle_H, Angle_V, f,fc, array_type = 'UPA', AtDs=0.5):
    '''
    This function defines a steering vector for a Nh*Nv uniform planar array (UPA).
    See paper 'Dynamic Hybrid Beamforming With Low-Resolution PSs for Wideband mmWave MIMO-OFDM Systems'
    :param Nh: number of antennas in horizontal direction
    :param Nv: number of antennas in vertical direction
    :param fc: carrier frequency
    :param f: actual frequency
    :param AtDs: normalized antenna spacing distance, set to 0.5 by default
    :return: steering a vector at frequency f with azimuth and elevation angles
    '''
    N = int(Nh*Nv)
    Np = Angle_H.shape[0]
    AtDs_h = AtDs
    AtDs_v = AtDs
    array_matrix = np.zeros([N,Np], dtype=np.complex_)
    if array_type == 'ULA':
        spatial_h = np.sin(Angle_H)
        factor_h = np.array(range(N))
        for n in range(Np):
            array_matrix[:, n] = 1/np.sqrt(N)*np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])

    else:
        # Nh, Nv = array_dimension(N)
        spatial_h = np.sin(Angle_H) * np.sin(Angle_V)
        spatial_v = np.cos(Angle_V)
        factor_h = np.array(range(Nh))
        factor_v = np.array(range(Nv))
        for n in range(Np):
            steering_vector_h = 1/np.sqrt(Nh) * np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])
            steering_vector_v = 1/np.sqrt(Nv) * np.exp(1j*2*np.pi* AtDs_v * factor_v*f/fc*spatial_v[n])
            array_matrix[:,n] = np.kron(steering_vector_h, steering_vector_v)
    ccc = 1
    return array_matrix


def OMP(Fopt, At):
    Frf = []
    # print(np.shape(Frf))
    Fbb = np.zeros([Nrf, Ns], dtype='complex_')
    Fres = Fopt
    for n in range(Nrf):
        Pu = np.matmul(At.conj().T, Fres)
        best_idx = np.argmax(np.sum(np.abs(Pu), 1))
        best_at = np.asmatrix(np.sqrt(Nt)*At[:, best_idx]).T

        if n == 0:
            Frf = best_at
            Frf_pinv = np.linalg.pinv(np.asmatrix(Frf))
        else:
            Frf = np.append(Frf, best_at, axis=1)
            Frf_pinv = np.linalg.pinv(Frf)

        Fbb = np.matmul(Frf_pinv, Fopt)
        # print(np.shape(Frf))
        # print(np.shape(Fbb))
        Fres = (Fopt - np.matmul(np.asmatrix(Frf), Fbb)) / (np.linalg.norm(Fopt - np.matmul(Frf, Fbb), 'fro'))
    return Frf, Fbb


def channel_model(Nt, Nr, Pulse_Filter = True, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    Np = Ncl * Nray
    gamma = np.sqrt(Nt * Nr / Np)  # normalization factor
    sigma = 1  # according to the normalization condition of the H
    Ntv = 4
    Nth = Nt // Ntv

    Nrh = 2
    Nrv = Nr // Nrh

    beta = 1
    Ts = 1/Bandwidth
    Delay_taps = int(K/4)
    angle_sigma = 10 / 180 * np.pi  # standard deviation of the angles in azimuth and elevation both of Rx and Tx

    AoH_all = np.zeros([2, Np])  # azimuth angle at Tx and Rx
    AoV_all = np.zeros([2, Np])  # elevation angle at Tx and Rx

    for cc in range(Ncl):
        AoH = np.random.uniform(0, 2, 2) * np.pi
        AoV = np.random.uniform(-0.5, 0.5, 2) * np.pi

        AoH_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoH_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(0, 2, Nray) * np.pi
        AoV_all[0, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi
        AoV_all[1, cc * Nray:(cc + 1) * Nray] = np.random.uniform(-0.5, 0.5, Nray) * np.pi

        # med = np.random.laplace(AoD_m[0], angle_sigma, Nray)

        # AoH_all[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoH[0], angle_sigma, Nray)
        # AoH_all[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoH[1], angle_sigma, Nray)
        # AoV_all[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoV[0], angle_sigma, Nray)
        # AoV_all[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoV[1], angle_sigma, Nray)

    # alpha = np.sqrt(sigma / 2) * (
    #         np.random.normal(0, 1, size=[Np, K]) + 1j * np.random.normal(0, 1, size=[Np, K]))
    alpha = np.sqrt(sigma / 2) * (
            np.random.normal(0, 1, size=[Np, ]) + 1j * np.random.normal(0, 1, size=[Np, ]))
    Delay = np.random.uniform(0, Delay_taps, size=Np) * Ts
    # AoH_all = np.random.uniform(-1, 1, size=[2, Np]) * np.pi
    # AoV_all = np.random.uniform(-0.5, 0.5, size=[2, Np]) * np.pi
    Coef_matrix = np.zeros([Np, K], dtype='complex_')
    H_all = np.zeros([Nr, Nt, K], dtype='complex_')
    At_all = np.zeros([Nt, Np, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    for k in range(K):
        # fk = 2
        fk = fc + bandwidth * (2 * k - K + 1) / (2 * K)
        At = array_response(Nth, Ntv, AoH_all[0, :], AoV_all[0, :], fk, fc, array_type=Array_Type)
        Ar = array_response(Nrh, Nrv, AoH_all[1, :], AoV_all[1, :], fk, fc, array_type=Array_Type)

        # AhA_t=np.matmul(At.conj().T, At)
        # AhA_r = np.matmul(Ar.conj().T, Ar)

        At_all[:, :, k] = At
        for n in range(Np):
            if Pulse_Filter:
                med = 0
                for d in range(Delay_taps):
                    med += pulase_filter(d * Ts - Delay[n], Ts, beta) * np.exp(-1j * 2 * np.pi * k * d / K)
                Coef_matrix[n, k] = med
            else:
                Coef_matrix[n, k] = np.exp(-1j * 2 * np.pi * Delay[n] * fk)
        gain = gamma * Coef_matrix[:, k] * alpha#[:, k]
        H_all[:, :, k] = np.matmul(np.matmul(Ar, np.diag(gain)), At.conj().T)
        # power_H = np.linalg.norm(H_all[:, :, k],'fro') ** 2 / (Nr * Nt)
        # print(f'channel power is {power_H}')

        ccc = 1
    return H_all, At_all


def masking_dyn(H, sub_connected=False, sub_structure_type="fixed"):
    batch_size, Nrf, Nt, K = H.shape
    N = 2 * Nt * Nrf
    bin_mask_mat = np.ones([batch_size, Nt, Nrf], dtype='int_') + 1j * np.ones([batch_size, Nt, Nrf], dtype='int_')
    bin_mask_vec_real = np.zeros([batch_size, N])

    for ii in range(batch_size):
        if sub_connected:
            if sub_structure_type == "fixed":
                bin_mask_mat[ii, Nt // 2:Nt, 0] = 0
                bin_mask_mat[ii, 0:Nt // 2, 1] = 0
            else:  # dynamic
                # choose best channel
                power_H = np.zeros([K], dtype='float')
                for k in range(K):
                    power_H[k] = np.linalg.norm(H[ii,:, :, k])

                k_max = np.argmax(power_H)
                Hmax = H[ii, :, :, k]
                # print(Hmax)
                D = np.abs(Hmax.T)
                # print(np.shape(D))
                bin_mask_mat_k = np.ones([Nt, Nrf], dtype='int_') + 1j * np.ones([Nt, Nrf], dtype='int_')
                for m in range(Nt // Nrf):
                    for n in range(Nrf):
                        m_min = np.argmin(D[:, n], axis=0)
                        bin_mask_mat_k[m_min, n] = 0
                        D[m_min, :] = 1000
                # print(bin_mask_mat_k)

                bin_mask_mat[ii, :, :] = bin_mask_mat_k

            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
        # print(bin_mask_mat[ii, :, :])

        else:
            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
    return bin_mask_vec_real


def normalize(FRF,sub_connected=False, sub_structure_type="fixed"):
    Nt, Nrf = FRF.shape
    if sub_connected:
        if sub_structure_type == "fixed":
            FRF[0:Nt // 2, 0] = FRF[0:Nt // 2, 0] / np.abs(FRF[0:Nt // 2, 0])
            FRF[Nt // 2:, 1] = FRF[Nt // 2:, 1] / np.abs(FRF[Nt // 2:, 1])

        else:
            for tt in range(Nt):
                for nn in range(Nrf):
                    if np.abs(FRF[tt, nn]) > 0.0001:
                        FRF[tt, nn] = FRF[tt, nn] / np.abs(FRF[tt, nn])
    else:
        FRF = FRF / np.abs(FRF)
    ccc=1
    return FRF


def gen_data_wideband(Nt, Nr, Nrf, Ns, batch_size=1,
                      Sub_Connected=False,
                      Sub_Structure_Type='fixed',
                      Pulse_Filter=False,
                      fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning

    batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab

    for ii in range(batch_size):
        if init_scheme == 0:
            FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
            # FRF = normalize(FRF, Nt, Nrf, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
            FRF_vec = FRF.flatten('F')
            batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)


        H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
        batch_H[ii, :, :, :] = H_ii
        batch_At[ii, :, :, :] = At_ii

        for k in range(K):
            At = At_ii[:, :, k]
            U, S, VH = np.linalg.svd(H_ii[:, :, k])
            V = VH.T.conj()
            Fopt = V[:, 0:Ns]   # np.sqrt(Ns) *
            Wopt = U[:, 0:Ns]

            ## construct training data
            ztilde = Fopt.flatten('F')
            z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
            # z_vector = np.matrix(z)
            if init_scheme == 0: # random FRF, FBB = LS solution
                FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')
            else: # obtain FRF and FBB based on OMP for all frequencies ==> better
                FRF, FBB = OMP(Fopt, At)

            # FBB = np.matmul(np.linalg.pinv(FRF), Fopt)

            Btilde = np.kron(FBB.T, np.identity(Nt))
            B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
            B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
            B = np.concatenate((B1, B2), axis=0)
            # print(np.shape(B))

            # new for array response
            AtH = At.conj().T
            Atilde = np.kron(np.identity(Nrf), AtH)
            A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
            A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
            A = np.concatenate((A1, A2), axis=0)
            # print(np.shape(A))

            # Assign data to the ii-th batch
            # err = z_vector.dot(B) -np.matmul(B.T, z)
            # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

            batch_Bz[ii, :, k] = np.matmul(B.T, z)
            batch_BB[ii, :, :, k] = np.matmul(B.T, B)
            batch_z[ii, :, k] = z
            batch_B[ii, :, :, k] = B.T
            batch_Fopt[ii, :, :, k] = Fopt
            batch_Wopt[ii, :, :, k] = Wopt
            batch_Fbb[ii, :, :, k] = FBB
            batch_AA[ii, :, :, k] = np.matmul(A.T, A)

    # dis_sum = 0
    # for k in range(K):
    #     med = np.matmul(np.expand_dims(batch_X,1), batch_B[:,:,:, k]).squeeze()
    #     diff = batch_z[ :, :, k] - med
    #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
    # print(f'{ii} error:{dis_sum}')
    # ccc = 1
    return batch_Bz, batch_BB, batch_X, batch_z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At


def gen_data_large(Nt, Nr, Nrf, Ns, Num_batch,batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth, Pulse_Filter=False, data='taining'):
    def append_data(data_set, num_data, new_data):
        dims = list(data_set.shape)
        num_sp = dims[0] + num_data
        dims_new = list(dims[1:])
        dims_new.insert(0, num_sp)
        data_set.resize(tuple(dims_new))
        data_set[dims[0]:num_sp] = new_data
        return data_set
    # Channel setup
    channel_type = 'geometry'
    # data to get
    data_name = train_data_name
    if data == 'testing':
        data_name = test_data_name
    data_path = dataset_file + data_name
    hf = h5py.File(data_path, 'a')
    batch_Bz_set = hf.get('batch_Bz')
    batch_BB_set = hf.get('batch_BB')
    batch_X_set = hf.get('batch_X')
    batch_Z_set = hf.get('batch_Z')
    batch_B_set = hf.get('batch_B')
    batch_H_real_set = hf.get('batch_H_real')
    batch_H_imag_set = hf.get('batch_H_imag')
    batch_Fopt_real_set = hf.get('batch_Fopt_real')
    batch_Fopt_imag_set = hf.get('batch_Fopt_imag')
    if data == 'testing':
        batch_Wopt_real_set = hf.get('batch_Wopt_real')
        batch_Wopt_imag_set = hf.get('batch_Wopt_imag')
        batch_Fbb_real_set = hf.get('batch_Fbb_real')
        batch_Fbb_imag_set = hf.get('batch_Fbb_imag')
        batch_At_real_set = hf.get('batch_At_real')
        batch_At_imag_set = hf.get('batch_At_imag')


    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning
    batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N, K], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab


    for n in range(Num_batch):
        print(f'Generating {n}th batch data', flush=True)
        for ii in range(batch_size):
            if init_scheme == 0:
                FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
                FRF_vec = FRF.flatten('F')
                batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)

            # generate channel matrix
            if channel_type == 'Rician':
                Hii = 1 / np.sqrt(2) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))
                batch_H[ii, :, :, :] = Hii
            else:
                H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
                batch_H[ii, :, :, :] = H_ii
                batch_At[ii, :, :, :] = At_ii
                for k in range(K):
                    At = At_ii[:, :, k]
                    U, S, VH = np.linalg.svd(H_ii[:, :, k])
                    V = VH.T.conj()
                    Fopt = V[:, 0:Ns]  # np.sqrt(Ns) *
                    Wopt = U[:, 0:Ns]

                    ## construct training data
                    ztilde = Fopt.flatten('F')
                    z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
                    # z_vector = np.matrix(z)

                    if init_scheme == 0:  # random FRF, FBB = LS solution
                        FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                        FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')
                    else:  # obtain FRF and FBB based on OMP for all frequencies ==> better
                        FRF, FBB = OMP(Fopt, At)

                    Btilde = np.kron(FBB.T, np.identity(Nt))
                    B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                    B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                    B = np.concatenate((B1, B2), axis=0)
                    # print(np.shape(B))

                    # new for array response
                    AtH = At.conj().T
                    Atilde = np.kron(np.identity(Nrf), AtH)
                    A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
                    A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
                    A = np.concatenate((A1, A2), axis=0)
                    # print(np.shape(A))

                    # Assign data to the ii-th batch
                    # err = z_vector.dot(B) -np.matmul(B.T, z)
                    # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

                    batch_Bz[ii, :, k] = np.matmul(B.T, z)
                    batch_BB[ii, :, :, k] = np.matmul(B.T, B)
                    batch_z[ii, :, k] = z
                    batch_B[ii, :, :, k] = B.T
                    batch_Fopt[ii, :, :, k] = Fopt
                    batch_Wopt[ii, :, :, k] = Wopt
                    batch_Fbb[ii, :, :, k] = FBB
                    batch_AA[ii, :, :, k] = np.matmul(A.T, A)

            # Hgap = np.linalg.norm(H,ord='fro')/np.sqrt(Nt*Nr)
            # print(f'HQ is: {Hgap:.4f}')
            # Compute optimal digital precoder


        batch_Bz_set = append_data(batch_Bz_set, batch_size, batch_Bz)  # add new data into set
        batch_BB_set = append_data(batch_BB_set, batch_size, batch_BB)
        batch_X_set = append_data(batch_X_set, batch_size, batch_X)
        batch_Z_set = append_data(batch_Z_set, batch_size, batch_z)
        batch_B_set = append_data(batch_B_set, batch_size, batch_B)

        batch_H_real_set = append_data(batch_H_real_set, batch_size, batch_H.real)
        batch_H_imag_set = append_data(batch_H_imag_set, batch_size, batch_H.imag)

        batch_Fopt_real_set = append_data(batch_Fopt_real_set, batch_size, batch_Fopt.real)
        batch_Fopt_imag_set = append_data(batch_Fopt_imag_set, batch_size, batch_Fopt.imag)
        if data == 'testing':
            batch_Wopt_real_set = append_data(batch_Wopt_real_set, batch_size, batch_Wopt.real)
            batch_Wopt_imag_set = append_data(batch_Wopt_imag_set, batch_size, batch_Wopt.imag)

            batch_Fbb_real_set = append_data(batch_Fbb_real_set, batch_size, batch_Fbb.real)
            batch_Fbb_imag_set = append_data(batch_Fbb_imag_set, batch_size, batch_Fbb.imag)

            batch_At_real_set = append_data(batch_At_real_set, batch_size, batch_At.real)
            batch_At_imag_set = append_data(batch_At_imag_set, batch_size, batch_At.imag)



    ccc = 1


def exponentially_decay_lr(lr_ini, lr_lb, decay_factor, learning_steps, decay_steps, staircase=1):
    '''
    The latex formular is given as
        $\alpha = \max(\alpha_0 \beta^{\left \lfloor \frac{t}{{\Delta t}^I}\right \rfloor},\alpha_e)$

    :param lr_ini(\alpha_0): initial learning rate
    :param lr_lb(\alpha_e): learning rate lower bound
    :param decay_factor(\beta): decay factor of learning rate
    :param learning_steps(t): number of learning steps
    :param decay_steps(\Delta t): the number of steps that the learning rate keeps the same
    :param staircase(I): whether the staircase decrease of learning rate is adopted. 1 indicates True by default. If it is
    False, then the decay_steps doesn't function anymore.
    :return: decayed learning rate (\alpha)
    '''
    import math
    if staircase:
        med_steps = decay_steps
    else:
        med_steps = 1
    lr_decayed = lr_ini*decay_factor**(math.floor(learning_steps/med_steps))
    lr = max(lr_decayed,lr_lb)
    return lr




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        The architecture of ScNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class VectorLinear(nn.Module):
    def __init__(self, N, keep_bias=True):
        super(VectorLinear, self).__init__()
        self.keep_bias = keep_bias
        # print(f'mask is {self.mask}')
        self.weight = nn.Parameter(torch.randn([1, N]))  # initialize weight
        # print(f'0 weight is {self.weight}')
        if self.keep_bias:
            self.bias = nn.Parameter(torch.randn([1, N]))  # initialize bias
        # print(f'0 bias is {self.bias}')
        self.reset_parameters()  # self-defined initialization

    def forward(self, input):
        if self.keep_bias:
            return input*self.weight + self.bias
        else:
            return input * self.weight

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)
            else:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)


class FcNet(nn.Module):
    def __init__(self, dim, num_subcarriers, Loss_scalar=1, training_method='unsupervised', device=MainDevice):
        super(FcNet, self).__init__()
        self.in_dim = dim*(dim+1)
        self.training_method = training_method
        self.device = device
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers

        self.layer1 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.in_dim, dim ** 2),
            nn.BatchNorm1d(dim ** 2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(dim ** 2, dim ** 2),
            nn.BatchNorm1d(dim ** 2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer_end = nn.Sequential(
            nn.Linear(dim ** 2, dim),
            nn.BatchNorm1d(dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(dim, dim)
        )


    def forward(self, input, x, z, B, Mask):
        # x_est = torch.zeros_like(x, requires_grad=True)
        # t = torch.tensor(data=[0.5], device=self.device0)
        x_tmp = self.layer1(input)
        x_tmp = self.layer2(x_tmp)
        # x_tmp = self.layer3(x_tmp)
        x_tmp = self.layer_end(x_tmp)
        x_tmp = x_tmp*Mask
        # x_est = torch.tanh(x_tmp)
        x_est = -1 + torch.nn.functional.relu(x_tmp + 0.5) / 0.5 - torch.nn.functional.relu(
            x_tmp - 0.5) / 0.5
        if self.training_method == 'supervised':

            dis = torch.mean(torch.square(x - x_est))

        else:
            dis_sum = 0
            for k in range(self.num_subcarriers):
                diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                dis_sum += torch.mean(torch.square(diff))

        LOSS = self.scalar * dis_sum

        return x_est, LOSS



# class Cnn_Net(nn.Module):
#     def __init__(self, dim, num_subcarriers, Loss_scalar=10, residule=False, training_method='unsupervised', device=MainDevice):
#         super(Cnn_Net, self).__init__()
#         self.in_dim = dim
#         self.device = device
#         self.training_method = training_method
#         self.Rsdl = residule
#         self.scalar = Loss_scalar
#         self.num_subcarriers = num_subcarriers
#         sqrt_dim = int(np.sqrt(dim))
#
#         self.conv2d = nn.Sequential(         # input shape (1, N, N)
#             nn.Conv2d(
#                 in_channels=1,              # input height
#                 out_channels=2,       # n_filters
#                 kernel_size=3,              # filter size
#                 stride=1,                   # filter movement/step
#                 padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
#             ),                              # output shape (sqrt_dim, N, N)
#             nn.BatchNorm2d(2),
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (2, 128, 128)
#
#             nn.Conv2d(2, 4, 3, 1, 1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (4, 64, 64)
#
#             nn.Conv2d(4, 6, 3, 1, 1),
#             nn.BatchNorm2d(6),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (8, 32, 32)
#
#             nn.Conv2d(6, 8, 3, 1, 1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 16, 16)
#
#             nn.Conv2d(8, 12, 3, 1, 1),
#             nn.BatchNorm2d(12),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 8, 8)
#
#             nn.Conv2d(12, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 4, 4)
#             #
#             nn.Conv2d(16, 24, 3, 1, 1),
#             nn.BatchNorm2d(24),
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (24, 2, 2)
#         )
#
#         self.conv1d = nn.Sequential(         # input shape (1, N)
#             nn.Conv1d(1, 2, 3, 1, 1),  # output shape (sqrt_dim, N, N)
#             nn.BatchNorm1d(2),
#             nn.ReLU(),                      # activation
#             nn.MaxPool1d(kernel_size=2),    # choose max value in 2x2 area, output shape (2, 128)
#
#             nn.Conv1d(2, 4, 3, 1, 1),
#             nn.BatchNorm1d(4),
#             nn.ReLU(),  # activation
#             nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (4, 64)
#
#             nn.Conv1d(4, 6, 3, 1, 1),
#             nn.BatchNorm1d(6),
#             nn.ReLU(),  # activation
#             nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (8, 32)
#
#             nn.Conv1d(6, 8, 3, 1, 1),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),  # activation
#             nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 16)
#
#             nn.Conv1d(8, 12, 3, 1, 1),
#             nn.BatchNorm1d(12),
#             nn.ReLU(),  # activation
#             nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 8)
#
#             nn.Conv1d(12, 16, 3, 1, 1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),  # activation
#             nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 4)
#         )
#         self.flatten_dim = 160
#
#         self.Fc2D = nn.Sequential(
#             nn.Linear(160, 160),
#             nn.BatchNorm1d(160, momentum=0.2),
#             nn.PReLU(),
#             # nn.ReLU(),
#             # nn.Dropout(p=0.2),
#
#             nn.Linear(160, 200),
#             nn.BatchNorm1d(200, momentum=0.2),
#             nn.PReLU(),
#             # nn.ReLU(),
#             # nn.Dropout(p=0.2),
#         )
#
#
#         self.layer_end = nn.Linear(200, self.in_dim)
#
#
#
#     def forward(self, BB, Bz, x, z, B):
#         # x_est = torch.zeros_like(x, requires_grad=True)
#         # t = torch.tensor(data=[0.5], device=self.device0)
#         BB_med = BB.unsqueeze(1)
#         Bz_med = Bz.unsqueeze(1)
#         x_tmp1 = self.conv2d(BB.unsqueeze(1))
#         x_tmp1 = x_tmp1.view(x_tmp1.size(0), -1)
#         x_tmp2 = self.conv1d(Bz.unsqueeze(1))
#         x_tmp2 = x_tmp2.view(x_tmp2.size(0), -1)
#
#         x_temp = torch.cat((x_tmp1, x_tmp2), 1)
#
#         x_tmp = self.Fc2D(x_temp)
#         x_tmp = self.layer_end(x_tmp)
#
#         # x_est = torch.tanh(x_tmp)
#         x_est = -1 + torch.nn.functional.relu(x_tmp + 0.5) / 0.5 - torch.nn.functional.relu(
#             x_tmp - 0.5) / 0.5
#         if self.training_method == 'supervised':
#
#             dis = torch.mean(torch.square(x - x_est))
#
#         else:
#             dis_sum = 0
#             for k in range(self.num_subcarriers):
#                 diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
#                 dis_sum += torch.mean(torch.square(diff))
#
#         LOSS = self.scalar * dis_sum
#
#         return x_est, LOSS


class Cnn_Net(nn.Module):
    def __init__(self, dim, num_subcarriers, Loss_scalar=10, residule=False, training_method='unsupervised', device=MainDevice):
        super(Cnn_Net, self).__init__()
        self.in_dim = dim
        self.device = device
        self.training_method = training_method
        self.Rsdl = residule
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers
        sqrt_dim = int(np.sqrt(dim))

        self.conv2d = nn.Sequential(         # input shape (1, N, N)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=sqrt_dim,       # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (sqrt_dim, N, N)
            nn.BatchNorm2d(sqrt_dim),
            nn.ReLU(),                      # activation
            # nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (sqrt_dim, N, N)

            nn.Conv2d(sqrt_dim, 2 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm2d(2 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (2*sqrt_dim, N/2, N/2)

            nn.Conv2d(2 * sqrt_dim, 4 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm2d(4 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (4 * sqrt_dim, N/4, N/4)
        )

        self.conv1d = nn.Sequential(         # input shape (1, N)
            nn.Conv1d(1, sqrt_dim, 3, 1, 1),  # output shape (sqrt_dim, N, N)
            nn.BatchNorm1d(sqrt_dim),
            nn.ReLU(),                      # activation
            # nn.MaxPool1d(kernel_size=2),    # choose max value in 2x2 area, output shape (sqrt_dim, N)

            nn.Conv1d(sqrt_dim, 2 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm1d(2 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (2*sqrt_dim, N/2)

            nn.Conv1d(2 * sqrt_dim, 4 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm1d(4 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (4 * sqrt_dim, N/4)
        )
        self.flatten_dim = 4 * sqrt_dim * (self.in_dim/4 + (self.in_dim/4) ** 2)

        self.Fc2D = nn.Sequential(
            nn.Linear(int(self.flatten_dim), int(self.in_dim**2)),
            nn.BatchNorm1d(self.in_dim**2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(self.in_dim**2, self.in_dim),
            nn.BatchNorm1d(self.in_dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
        )


        self.layer_end = nn.Linear(self.in_dim, self.in_dim)



    def forward(self, BB, Bz, x, z, B):
        # x_est = torch.zeros_like(x, requires_grad=True)
        # t = torch.tensor(data=[0.5], device=self.device0)
        BB_med = BB.unsqueeze(1)
        Bz_med = Bz.unsqueeze(1)
        x_tmp1 = self.conv2d(BB.unsqueeze(1))
        x_tmp1 = x_tmp1.view(x_tmp1.size(0), -1)
        x_tmp2 = self.conv1d(Bz.unsqueeze(1))
        x_tmp2 = x_tmp2.view(x_tmp2.size(0), -1)

        x_temp = torch.cat((x_tmp1, x_tmp2), 1)

        x_tmp = self.Fc2D(x_temp)
        x_tmp = self.layer_end(x_tmp)

        # x_est = torch.tanh(x_tmp)
        x_est = -1 + torch.nn.functional.relu(x_tmp + 0.5) / 0.5 - torch.nn.functional.relu(
            x_tmp - 0.5) / 0.5
        if self.training_method == 'supervised':

            dis = torch.mean(torch.square(x - x_est))

        else:
            dis_sum = 0
            for k in range(self.num_subcarriers):
                diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                dis_sum += torch.mean(torch.square(diff))

        LOSS = self.scalar * dis_sum

        return x_est, LOSS


class ScNet(nn.Module):
    def __init__(self, in_dim, num_subcarriers, num_layer, Loss_scalar=1, IL=False, Keep_Bias=True, BN = True, Sub_Connected=False, training_method='unsupervised'):
        super(ScNet, self).__init__()
        self.in_dim = in_dim
        self.training_method = training_method
        self.dobn = BN
        self.IL = IL
        self.Sub_Connected = Sub_Connected
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers1K = nn.ModuleList()
        self.layers2K = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bnsK = nn.ModuleList()
        self.num_layer = num_layer
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers
        # self.t = torch.tensor(data=[0.5])

        for i in range(num_layer):  # define all layers

            # self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers2.append(VectorLinear(N, keep_bias=Keep_Bias))
            for k in range(self.num_subcarriers):
                self.layers1K.append(VectorLinear(N, keep_bias=Keep_Bias))
                self.layers2K.append(VectorLinear(N, keep_bias=Keep_Bias))
                if self.dobn:
                    # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                    # setattr(self, 'bn_layers%i'%i, bn_layer)
                    self.bnsK.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))

            if self.dobn:
                # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                # setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bns.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))


    def forward(self, BB, zB, x, z, B, Mask):
        # batch_size = zB.size()[0]
        LOSS = []
        x_est = torch.zeros_like(x, requires_grad=True)

        for l in range(self.num_layer):
            batch_Bz_sum = 0
            batch_BB_sum = 0
            for k in range(self.num_subcarriers):
                index = l * self.num_subcarriers + k
                batch_Bz_sum = batch_Bz_sum + zB[:, :, k]
                batch_BB_sum = batch_BB_sum + BB[:, :, :, k]
                if self.IL:
                    aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
                    out = self.layers1K[index](aux_term) + self.layers2K[index](x_est)
                    # out = self.layers1K[index](aux_term + x_est)
                    if self.dobn:
                        x_est = self.bnsK[index](out)
                    x_est = x_est * Mask
                    x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                        x_est - 0.5) / 0.5
            if not self.IL:
                aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
                if self.Sub_Connected:
                    out = self.layers1[l](aux_term * Mask) + self.layers2[l](x_est)
                else:
                    out = self.layers1[l](aux_term) + self.layers2[l](x_est)
                # out = self.layers1[l](aux_term + x_est)
                if self.dobn:
                    x_est = self.bns[l](out)
                x_est = x_est * Mask
                # if l<self.num_layer-1:
                #     x_est = torch.nn.functional.relu(x_est)
                #     # x_est = torch.nn.functional.leaky_relu(x_est)
                # else:
                #     # x_est = -1 + tf.nn.relu(x_tmp + t) / tf.abs(t) - tf.nn.relu(x_tmp - t) / tf.abs(t)
                #     # x_est = -1 + torch.nn.functional.relu(x_est + t) / torch.abs(t) - torch.nn.functional.relu(x_est - t) / torch.abs(t)
                #     x_est = torch.tanh(x_est)
                x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                    x_est - 0.5) / 0.5

            if self.training_method == 'supervised':

                dis = torch.mean(torch.square(x - x_est))

            else:
                dis_sum = 0
                for k in range(self.num_subcarriers):
                    diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                    dis_sum += torch.mean(torch.square(diff))

            LOSS.append(self.scalar*np.log(l+1) * dis_sum)

        return x_est, LOSS


class ScNet_Wideband(nn.Module):
    def __init__(self, in_dim, num_subcarriers, num_layer, Loss_scalar=10, Residule=False, Keep_Bias=False, BN = True, training_method='unsupervised', device=MainDevice):
        super(ScNet_Wideband, self).__init__()
        self.in_dim = in_dim
        self.training_method = training_method
        self.device = device
        self.Rsdl = Residule
        self.dobn = BN
        self.layers_x = nn.ModuleList()
        self.layers_KL = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layer = num_layer
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers

        for i in range(num_layer):  # define all layers
            # layer = VectorLinear(N, keep_bias=Keep_Bias)
            self.layers_x.append(VectorLinear(N, keep_bias=Keep_Bias))
            # setattr(self, 'layer_x_%i' % i, layer)
            for k in range(num_subcarriers):
                # layerk = VectorLinear(N, keep_bias=Keep_Bias)
                # layer_id = str(i)+str(k)
                # setattr(self, 'layer_bzx_'+ layer_id, layerk)  ## another method is to use nn.ModuleList
                self.layers_KL.append(VectorLinear(N, keep_bias=Keep_Bias))

            if self.dobn:
                # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                # setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bns.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))


    def forward(self, BB, zB, x, z, B):
        # batch_size = zB.size()[0]
        LOSS = []
        x_est = torch.randn_like(x, requires_grad=True)

        for l in range(self.num_layer):
            out_x = self.layers_x[l](x_est)
            # Bzx_sum = torch.zeros_like(x, device=self.device)
            for k in range(self.num_subcarriers):
                index = l*self.num_subcarriers + k
                aux_term = torch.bmm(x_est.unsqueeze(1), BB[:, :, :, k]).squeeze() - zB[:, :, k]
                out_x += self.layers_KL[index](aux_term)

            x_est = out_x
            # xxx = x_est @ BB
            # err = torch.bmm(x_est.unsqueeze(1), BB) - torch.matmul(x_est.unsqueeze(1), BB)
            # print(f'BN layers:{self.bns[l]}')
            if self.dobn:
                x_est = self.bns[l](x_est)

            if l<self.num_layer-1:
                x_est = torch.nn.functional.relu(x_est)
                # x_est = torch.nn.functional.leaky_relu(x_est)
            else:
                # x_est = torch.tanh(x_est)
                x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                    x_est - 0.5) / 0.5
            if self.training_method == 'supervised':

                dis = torch.mean(torch.square(x - x_est))

            else:
                dis_sum = 0
                for k in range(self.num_subcarriers):
                    diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                    dis_sum += torch.mean(torch.square(diff))

            LOSS.append(self.scalar*np.log(l+1) * dis_sum)

        return x_est, LOSS


class SaveParameters():
    def __init__(self, directory_model, para_file_name='Logs_Info.txt'):
        self.dir_para_file = os.path.join(directory_model, para_file_name)

        pass

    def first_write(self):
        file_para = open(self.dir_para_file, 'w')
        file_para.write('System parameters:\n')
        file_para.write('Nt= ' + str(Nt) + '\n')
        file_para.write('Nr= ' + str(Nr) + '\n')
        file_para.write('Nrf= ' + str(Nrf) + '\n')
        file_para.write('Ns= ' + str(Ns) + '\n')
        file_para.write('K= ' + str(K) + '\n')
        file_para.write('Ncl= ' + str(Ncl) + '\n')
        file_para.write('Nray= ' + str(Nray) + '\n')
        file_para.write('Array type= ' + str(Array_Type) + '\n')
        file_para.write('Sub_Connected= ' + str(Sub_Connected) + '\n')
        file_para.write('Sub_Structure_Type= ' + str(Sub_Structure_Type) + '\n\n')

        file_para.write('Training setup:\n')
        file_para.write('Training method:' + str(training_method) + '\n')
        file_para.write('Device= ' + str(MainDevice) + '\n')
        file_para.write('Use GPU= ' + str(bool(use_gpu)) + '\n')

        file_para.write('Black_box Net= ' + str(Black_box) + '\n')
        file_para.write('Keep_Bias= ' + str(Keep_Bias) + '\n')
        file_para.write('Wideband Net= ' + str(Wideband_Net) + '\n')
        file_para.write('Residule= ' + str(Residule_NN) + '\n')
        file_para.write('Init_scheme= ' + str(init_scheme) + '\n')
        file_para.write('Iterative_Training= ' + str(Iterative_Training) + '\n')
        file_para.write('Iterations_training= ' + str(Iterations_train) + '\n')
        file_para.write('Increamental_Learning= ' + str(Increamental_Learning) + '\n\n')

        file_para.write('Loss_coef= ' + str(Loss_coef) + '\n')
        file_para.write('Number of layers = ' + str(Num_layers) + '\n')
        file_para.write('Training seed = ' + str(Seed_train) + '\n')
        file_para.write('Testing seed = ' + str(Seed_test) + '\n')
        file_para.write('Traning Epoches= ' + str(Ntrain_Epoch) + '\n')
        file_para.write('Traning dataset size= ' + str(training_set_size_truncated) + '\n')
        file_para.write('Traning batch size= ' + str(train_batch_size) + '\n')
        file_para.write('Total number of training batches= ' + str(Ntrain_batch_total) + '\n')
        file_para.write('Number of training steps per Epoch= ' + str(Ntrain_Batch_perEpoch) + '\n\n')

        file_para.write('Optimizer info:\n')
        file_para.write('Optimizer type: Adam \n')
        file_para.write('Decayed learning rate: ' + str(bool(set_Lr_decay)) + '\n')
        file_para.write('Start learning rate= ' + str(start_learning_rate) + '\n')
        file_para.write('Decay factor= ' + str(Lr_decay_factor) + '\n')
        file_para.write('Learning keep steps= ' + str(Lr_keep_steps) + '\n')
        file_para.write('Learning rate lower bound= ' + str(Lr_min) + '\n')
        file_para.write('Weight decay= ' + str(Weight_decay) + '\n\n')
        # file_para.write('This learning is to improve the accuracy of phase, keep other NN fixed\n')

        file_para.write('Testing setup: \n')
        file_para.write('Testing batch size: ' + str(test_batch_size) + '\n')
        file_para.write('Iterations_testing: ' + str(Iterations_test) + '\n')

        # file_para.write('The training starts at the time= ' + str(time_now_start) + '\n')
        # file_para.write('The training ends at the time= ' + str(time_now_end) + '\n')
        # file_para.write('Training time cost =' + str(time_cost) + 'h\n\n')

        file_para.close()

    def add_logs(self, str):

        file_para = open(self.dir_para_file, 'a')
        file_para.write('\n')
        file_para.write(str + '\n')
        file_para.close()
        pass



class Data_Fetch():
    def __init__(self, file_dir, file_name, batch_size, training_set_size, training_set_size_truncated=training_set_size, data_str='training'):
        self.data_path = file_dir + file_name
        self.batch_size = batch_size
        self.data_str = data_str
        self.len = training_set_size+1
        self.len_truncated = training_set_size_truncated +1
        self.reset()

    def reset(self):
        self.pointer = np.random.randint(self.len_truncated)  # initialize the start position
        self.start_idx = self.pointer

    def get_item(self):
        data_all = h5py.File(self.data_path, 'r')

        self.end_idx = self.start_idx + self.batch_size
        if self.end_idx <= self.len_truncated-1:

            Bz = data_all['batch_Bz'][self.start_idx:self.end_idx, :, :]
            BB = data_all['batch_BB'][self.start_idx:self.end_idx, :, :, :]
            X = data_all['batch_X'][self.start_idx:self.end_idx, :]
            Z = data_all['batch_Z'][self.start_idx:self.end_idx, :, :]
            B = data_all['batch_B'][self.start_idx:self.end_idx, :, :, :]
            batch_H_real = data_all['batch_H_real'][self.start_idx:self.end_idx, :, :, :]
            batch_H_imag = data_all['batch_H_imag'][self.start_idx:self.end_idx, :, :, :]
            batch_Fopt_real = data_all['batch_Fopt_real'][self.start_idx:self.end_idx, :, :, :]
            batch_Fopt_imag = data_all['batch_Fopt_imag'][self.start_idx:self.end_idx, :, :, :]


            if self.data_str== 'testing':
                batch_Wopt_real = data_all['batch_Wopt_real'][self.start_idx:self.end_idx, :, :, :]
                batch_Wopt_imag = data_all['batch_Wopt_imag'][self.start_idx:self.end_idx, :, :, :]
                batch_Fbb_real = data_all['batch_Fbb_real'][self.start_idx:self.end_idx, :, :, :]
                batch_Fbb_imag = data_all['batch_Fbb_imag'][self.start_idx:self.end_idx, :, :, :]
                batch_At_real = data_all['batch_At_real'][self.start_idx:self.end_idx, :, :, :]
                batch_At_imag = data_all['batch_At_imag'][self.start_idx:self.end_idx, :, :, :]

                batch_Wopt = batch_Wopt_real + 1j * batch_Wopt_imag
                batch_Fbb = batch_Fbb_real + 1j * batch_Fbb_imag
                batch_At = batch_At_real + 1j * batch_At_imag

            data_all.close()
            self.start_idx = self.end_idx

        else:
            remain_num = self.end_idx - self.len_truncated

            Bz1 = data_all['batch_Bz'][self.start_idx:self.len_truncated, :, :]
            BB1 = data_all['batch_BB'][self.start_idx:self.len_truncated, :, :, :]
            X1 = data_all['batch_X'][self.start_idx:self.len_truncated, :]
            Z1 = data_all['batch_Z'][self.start_idx:self.len_truncated, :, :]
            B1 = data_all['batch_B'][self.start_idx:self.len_truncated, :, :, :]
            batch_H_real1 = data_all['batch_H_real'][self.start_idx:self.len_truncated, :, :, :]
            batch_H_imag1 = data_all['batch_H_imag'][self.start_idx:self.len_truncated, :, :, :]
            batch_Fopt_real1 = data_all['batch_Fopt_real'][self.start_idx:self.len_truncated, :, :, :]
            batch_Fopt_imag1 = data_all['batch_Fopt_imag'][self.start_idx:self.len_truncated, :, :, :]

            Bz2 = data_all['batch_Bz'][:remain_num, :, :]
            BB2 = data_all['batch_BB'][:remain_num, :, :, :]
            X2 = data_all['batch_X'][:remain_num, :]
            Z2 = data_all['batch_Z'][:remain_num, :, :]
            B2 = data_all['batch_B'][:remain_num, :, :, :]
            batch_H_real2 = data_all['batch_H_real'][:remain_num, :, :, :]
            batch_H_imag2 = data_all['batch_H_imag'][:remain_num, :, :, :]
            batch_Fopt_real2 = data_all['batch_Fopt_real'][:remain_num, :, :, :]
            batch_Fopt_imag2 = data_all['batch_Fopt_imag'][:remain_num, :, :, :]

            Bz = np.concatenate((Bz1, Bz2), axis=0)
            BB = np.concatenate((BB1, BB2), axis=0)
            X = np.concatenate((X1, X2), axis=0)
            Z = np.concatenate((Z1, Z2), axis=0)
            B = np.concatenate((B1, B2), axis=0)
            batch_H_real = np.concatenate((batch_H_real1, batch_H_real2), axis=0)
            batch_H_imag = np.concatenate((batch_H_imag1, batch_H_imag2), axis=0)
            batch_Fopt_real = np.concatenate((batch_Fopt_real1, batch_Fopt_real2), axis=0)
            batch_Fopt_imag = np.concatenate((batch_Fopt_imag1, batch_Fopt_imag2), axis=0)


            data_all.close()
            self.start_idx = remain_num

        batch_H = batch_H_real + 1j * batch_H_imag
        batch_Fopt = batch_Fopt_real + 1j * batch_Fopt_imag
        if self.data_str == 'testing':
            return Bz, BB, X, Z, B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_At
        else:
            return Bz, BB, X, Z, B, batch_H, batch_Fopt



if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def generate_training_data():
        # training_set_size = 70
        print('----------------------training data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth) # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z,
                    'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    }

        train_data_path = dataset_file + train_data_name
        file_handle = h5py.File(train_data_path, 'w')
        # for name in data_all.keys():
        #     file_handle.attrs[name]=data_all[name]
        # file_handle.close()
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            # print(f'dshp shape:{dshp}, dims shape:{dims}')
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
        # hf = h5py.File(train_data_path, 'r')
        # print('----------------------training data-------------------------')
        # for key in hf.keys():
        #     print(key, hf[key])


    def generate_testing_data(Pulse_Filter=False):
        # testing_set_size = 30
        print('----------------------testing data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_AA, batch_At = gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)  # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z, 'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    'batch_Wopt_real': batch_Wopt.real,
                    'batch_Wopt_imag': batch_Wopt.imag,
                    'batch_Fbb_real': batch_Fbb.real,
                    'batch_Fbb_imag': batch_Fbb.imag,
                    'batch_At_real': batch_At.real,
                    'batch_At_imag':  batch_At.imag
                    }

        # data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z, 'batch_B': batch_B,
        #             'batch_H_real': batch_H.real, 'batch_H_imag': batch_H.imag,
        #             'batch_Fopt_real': batch_Fopt.real, 'batch_Fopt_imag': batch_Fopt.imag,
        #             'batch_Wopt_real': batch_Wopt.real, 'batch_Wopt_imag':batch_Wopt.imag,
        #             'batch_Fbb_real': batch_Fbb.real, 'batch_Fbb_imag': batch_Fbb.imag,
        #             'batch_At_real': batch_At.real,'batch_At_imag': batch_At.imag}
        # print(f'H:{batch_Hb}')
        # test_data_name = 'test_set.hdf5'
        test_data_path = dataset_file + test_data_name
        file_handle = h5py.File(test_data_path, 'w')
        # for name in data_all.keys():
        #     file_handle.attrs[name]=data_all[name]
        # file_handle.close()
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
            # file_handle.create_dataset(name, data=data_all[name], chunks=True, compression='gzip',
            #                            compression_opts=9)
        # print(name)

        hf = h5py.File(test_data_path, 'r')
        print('----------------------testing data-------------------------')
        for key in hf.keys():
            print(key, hf[key])


    generate_testing_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_te, batch_size=Gen_Batch_size_te, fc=fc, Ncl=Ncl, Nray=Nray,
                   bandwidth=Bandwidth, data='testing')

    generate_training_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_tr, batch_size=Gen_Batch_size_tr, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)

    # train_data_path = dataset_file + train_data_name
    # hf = h5py.File(train_data_path, 'r')
    # print('----------------------training data-------------------------')
    # for key in hf.keys():
    #     print(key, hf[key])

    test_data_path = dataset_file + test_data_name
    hf = h5py.File(test_data_path, 'r')
    print('----------------------testing data-------------------------')
    for key in hf.keys():
        print(key, hf[key])
    # for idx, data in enumerate(batch_Bz):
    #     print(f'idx={idx},data is : {data}')

    # print(hf.attrs.keys() )
    # s1 = hf.attrs['batch_Bz']
    # print(f'batch_Bz is : {s1.dtpye}')
    ccc=1
    pass
    def draw_lrfunc():
        lr_ini = 0.001
        lr_lb = 1e-4
        decay_factor = 0.8
        decay_steps = 10
        staircase = 1
        num_learning_steps = 400
        Lr_all = []
        for step in range(num_learning_steps):
            lr = exponentially_decay_lr(lr_ini, lr_lb, decay_factor=decay_factor, learning_steps=step,
                                        decay_steps=decay_steps, staircase=staircase)
            Lr_all.append(lr)

        plt.figure(dpi=100)
        plt.plot(Lr_all, label=r'$\psi(x)$')

        plt.legend(loc='center right')
        # plt.xticks(x)
        plt.xlabel('steps')
        plt.ylabel('lr value')
        plt.grid(True)
        plt.show()
        # print(batch_Z)
