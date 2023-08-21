# -*- oding:utf-8 -*-
'''
# @File: THz_Huge.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2022-11-17 5:23 PM
'''
# -* coding: utf-8 -*-
import torch

'''
@File: DNN_mmWave_large.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-24 18:04

This codes target for training large-sized systems. The training and testing data are firstly generated and stored into
disk. In the training and testing phases, the required data are fetched from the disk.
'''

import logging
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io as spio
from FuncLbr import *
from Global_Vars import *

train_ManNet = 0

# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable INFO\WARNING prompt
#logging.disable(logging.WARNING)  # forbidden all log info

save_paras = SaveParameters(directory_model, para_file_name='Logs_Info.txt')
save_paras.first_write()
time_now_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings starts at the time:', time_now_start,flush=True)  # 当前时间
save_paras.add_logs('The training starts at the time= ' + str(time_now_start))
t_start = time.time()


# train the model
def train_model(BB_beamformer = 'LS'):
    torch.manual_seed(Seed_train)
    np.random.seed(Seed_train)
    myModel.train()  # training mode
    print('start training',flush=True)
    Lr_list = []
    Loss_cache = []

    batch_count = 0
    for epoch in range(Ntrain_Epoch):
        dataloader_tr.reset()
        print('-----------------------------------------------')
        for batch_idx in range(Ntrain_Batch_perEpoch):
            batch_count += 1
            batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt = dataloader_tr.get_item()
            batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
            batch_Mask = torch.from_numpy(batch_Mask).float()
            batch_Bz = torch.from_numpy(batch_Bz).float()
            batch_BB = torch.from_numpy(batch_BB).float()
            batch_X = torch.from_numpy(batch_X).float()
            batch_Z = torch.from_numpy(batch_Z).float()
            batch_B = torch.from_numpy(batch_B).float()

            batch_Bz_sum = 0
            batch_BB_sum = 0
            for k in range(K):
                batch_Bz_sum += batch_Bz[:, :, k]
                batch_BB_sum += batch_BB[:, :, :, k]

            BB_sum_vec = torch.reshape(batch_BB_sum, [-1, N**2])
            FcNet_input = torch.cat((batch_Bz_sum, BB_sum_vec), axis=1)

            # for s in range(train_batch_size):
            #     BBs_vec = BB_sum_vec[s,:]
            #     stmp = batch_BB_sum[s,:,:]
            #     err = BBs_vec - torch.flatten(stmp)
            #     print(f'err is {err}')

            # dis_sum = 0
            # for k in range(K):
            #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
            #     diff = batch_Z[:, :, k] - med
            #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
            # print(f'{batch_idx} error:{dis_sum}')


            if Black_box:
                x_est, loss = myModel(FcNet_input.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice), batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                # x_est, loss = myModel(batch_BB_sum.to(MainDevice), batch_Bz_sum.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                #                       batch_B.to(MainDevice))
            else:

                if Wideband_Net:
                    s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice),
                                               batch_Z.to(MainDevice), batch_B.to(MainDevice))
                else:
                    s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                                               batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                if SUM_LOSS==1:
                    loss = sum(loss_list)
                else:
                    loss = loss_list[-1]

            Loss_cache.append(loss.item())

            if set_Lr_decay:
                for g in optimizer.param_groups:
                    g['lr'] = exponentially_decay_lr(lr_ini=start_learning_rate, lr_lb=Lr_min, decay_factor=Lr_decay_factor,
                                                     learning_steps=batch_count, decay_steps=Lr_keep_steps, staircase=1)
            loss.requires_grad_(True)
            torch.cuda.empty_cache()
            optimizer.zero_grad()  # zero gradient
            loss.backward()  # backpropagation
            optimizer.step()  # update training prapameters
            torch.cuda.empty_cache()
            Lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            # for name, params in myModule.named_parameters():
            #     if 'layer_axz_0.weight' in name:
            #         print(f'epoch {epoch} after update: name {name}, params {params}')
            if (batch_idx) % Log_interval == 0:
                len_loss = len(Loss_cache)
                if len_loss > 2 * Log_interval:
                    avr_loss = np.mean(Loss_cache[len_loss-Log_interval:])  # 取倒数Log_interval个loss做平均
                    print(f'Epoch:{epoch}, batch_id:{batch_idx}, learning rate: {Lr_list[-1]:.5f}, average loss:{avr_loss:.6f}',flush=True)

            if not Black_box and Iterative_Training:  # start iterative training
                s_hat = s_hat.detach().cpu().numpy()
                batch_Bz = batch_Bz.numpy()
                batch_BB = batch_BB.numpy()
                batch_X = batch_X.numpy()
                batch_Z = batch_Z.numpy()
                batch_B = batch_B.numpy()

                for jj in range(Iterations_train):
                    # s_dim = s_hat.shape[0]
                    # 1. Update input to the network: only update data related to Frf, not change the channels
                    for ii in range(s_hat.shape[0]):
                        ff = s_hat[ii, :]  # prepare testing data
                        FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
                        ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                        FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
                        FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)

                        for k in range(K):
                            Fopt_ii = batch_Fopt[ii, :, :, k]  # recall optimal fully digital precoder
                            Hii = batch_H[ii, :, :, k]
                            Uo, So, VoH = np.linalg.svd(Hii)

                            # solution to Fbb
                            if BB_beamformer == 'LS':  # LS
                                FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder
                            else:  # equal-weight water-filling
                                Heff = np.matmul(Hii, FRF)
                                Q = np.matmul(FRF.conj().T, FRF)
                                Qrank = np.linalg.matrix_rank(Q, tol=1e-4)
                                Uq, S, UqH = np.linalg.svd(Q)
                                Uqnew = Uq[:, 0:Qrank]
                                # print(f'S:{S}')
                                # Snew = S[0:Qrank]
                                Snew = 1 / (np.sqrt(S[0:Qrank]))
                                Qinvsqrt = np.dot(Uqnew * Snew, Uqnew.conj().T)
                                # term1 = Qnew @ Qnew
                                # term2 = term1 @ Q

                                # err= np.linalg.norm(term2-np.eye(Nrf))
                                # print(f'err:{err}')

                                U, S, VH = np.linalg.svd(Heff * Qinvsqrt)

                                V = VH.T.conj()
                                FBB = np.matmul(Qinvsqrt, V[:, 0:Ns])
                            FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')

                            Btilde = np.kron(FBB.T, np.identity(Nt))

                            # convert to real values
                            z_ii = batch_Z[ii, :, k]
                            B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                            B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                            B_ii = np.concatenate((B1, B2), axis=0)

                            # B1 = B_ii.T
                            batch_Bz[ii, :, k] = np.matmul(B_ii.T, z_ii)  # update values
                            batch_BB[ii, :, :, k] = np.matmul(B_ii.T, B_ii)
                            batch_B[ii, :, :, k] = B_ii.T


                    # Update training data

                    batch_Bz = torch.from_numpy(batch_Bz)
                    batch_BB = torch.from_numpy(batch_BB)
                    batch_X = torch.from_numpy(batch_X)
                    batch_Z = torch.from_numpy(batch_Z)
                    batch_B = torch.from_numpy(batch_B)

                    if Wideband_Net:
                        s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice),
                                                   batch_Z.to(MainDevice), batch_B.to(MainDevice))
                    else:
                        batch_Bz_sum = 0
                        batch_BB_sum = 0
                        for k in range(K):
                            batch_Bz_sum += batch_Bz[:, :, k]
                            batch_BB_sum += batch_BB[:, :, :, k]
                        s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                                                   batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                    if SUM_LOSS == 1:
                        loss = sum(loss_list)
                    else:
                        loss = loss_list[-1]

                    torch.cuda.empty_cache()
                    optimizer.zero_grad()  # zero gradient
                    loss.backward()  # backpropagation
                    optimizer.step()  # update training prapameters
                    torch.cuda.empty_cache()


                    s_hat = s_hat.detach().cpu().numpy()
                    batch_X = batch_X.numpy()
                    batch_Z = batch_Z.numpy()
                    batch_B = batch_B.numpy()
                    batch_Bz = batch_Bz.numpy()
                    batch_BB = batch_BB.numpy()


            if batch_idx >= Ntrain_Batch_perEpoch:
                break


    return Loss_cache, Lr_list


def tst_model(BB_beamformer ='LS'):
    torch.manual_seed(Seed_test)
    np.random.seed(Seed_test)
    save_paras.add_logs('\n Test:')
    myModel.eval()  # testing mode
    myModel.to('cpu')  # test on CPU
    dataloader_te.start_idx = 0
    f_all = []

    for batch_idx in range(1):
        print(f'batch_id:{batch_idx}')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_At = dataloader_te.get_item()

        batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
        batch_Mask = torch.from_numpy(batch_Mask).float()
        batch_Bz = torch.from_numpy(batch_Bz).float()
        batch_BB = torch.from_numpy(batch_BB).float()
        batch_X = torch.from_numpy(batch_X).float()
        batch_Z = torch.from_numpy(batch_Z).float()
        batch_B = torch.from_numpy(batch_B).float()



        # batch_Bz = data['batch_Bz'].float()
        # batch_BB = data['batch_BB'].float()
        # batch_X = data['batch_X'].float()
        # batch_Z = data['batch_Z'].float()
        # batch_B = data['batch_B'].float()
        # batch_H = data['batch_H'].float()
        # batch_Fopt = data['batch_Fopt'].float()
        # batch_Wopt = data['batch_Wopt'].float()
        # batch_Fbb = data['batch_Fbb'].float()
        # At = data['batch_At'].float()

        # dis_sum = 0
        # for k in range(K):
        #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
        #     diff = batch_Z[:, :, k] - med
        #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
        # print(f'{batch_idx} error:{dis_sum}')

        batch_Bz_sum = 0
        batch_BB_sum = 0
        for k in range(K):
            batch_Bz_sum += batch_Bz[:, :, k]
            batch_BB_sum += batch_BB[:, :, :, k]

        BB_sum_vec = torch.reshape(batch_BB_sum, [-1, N ** 2])
        FcNet_input = torch.cat((batch_Bz_sum, BB_sum_vec), axis=1)

        if Black_box:
            s_hat, loss = myModel(FcNet_input, batch_X, batch_Z, batch_B,batch_Mask)
            # s_hat, loss = myModel(batch_BB_sum, batch_Bz_sum, batch_X,batch_Z, batch_B)
            s_hat = s_hat.detach().numpy()
        else:

            if Wideband_Net:
                s_hat, loss = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B)
            else:
                s_hat, loss = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B, batch_Mask)




            # batch_Bz_sum = torch.from_numpy(batch_Bz_sum).float()
            # batch_BB_sum = torch.from_numpy(batch_BB_sum).float()

            # batch_Fopt = torch.from_numpy(batch_Fopt).float()

            s_hat = s_hat.detach().numpy()
            batch_Bz = batch_Bz.numpy()
            batch_BB = batch_BB.numpy()
            batch_X = batch_X.numpy()
            batch_Z = batch_Z.numpy()
            batch_B = batch_B.numpy()

            # batch_H = batch_H.numpy()
            # batch_Fopt = batch_Fopt.numpy()
            # batch_Wopt = batch_Wopt.numpy()
            # batch_Fbb = batch_Fbb.numpy()
            # At = At.numpy()

            # batch_Fopt_real = batch_Fopt[:, 0, :, :, :]
            # batch_Fopt_imag = batch_Fopt[:, 1, :, :, :]
            # batch_Wopt_real = batch_Wopt[:, 0, :, :, :]
            # batch_Wopt_imag = batch_Wopt[:, 1, :, :, :]

            # batch_H = batch_H[:, 0, :, :, :] + 1j * batch_H[:, 1, :, :, :]
            # batch_Fopt = batch_Fopt[:, 0, :, :, :] + 1j * batch_Fopt[:, 1, :, :, :]
            # batch_Wopt = batch_Wopt[:, 0, :, :, :] + 1j * batch_Wopt[:, 1, :, :, :]
            # batch_Fbb = batch_Fbb[:, 0, :, :, :] + 1j * batch_Fbb[:, 1, :, :, :]
            # At = At[:, 0, :, :, :] + 1j * At[:, 1, :, :, :]

            # At = np.transpose(At, (1, 2, 0))

            for jj in range(Iterations_test):
                # 1. Update input to the network: only update data related to Frf, not change the channels
                for ii in range(test_batch_size):
                    ff = s_hat[ii, :]# prepare testing data
                    FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
                    ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                    FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
                    FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
                    FRF_vec = FRF.flatten('F')
                    batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)
                    for k in range(K):
                        Fopt_ii = batch_Fopt[ii, :, :, k]  # recall optimal fully digital precoder
                        Hii = batch_H[ii, :, :, k]
                        Uo, So, VoH = np.linalg.svd(Hii)
                        Wopt = Uo[:, 0:Ns]
                        # solution to Fbb
                        if BB_beamformer == 'LS':  # LS
                            FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder
                        else:  # equal-weight water-filling
                            Heff = np.matmul(Hii, FRF)
                            Q = np.matmul(FRF.conj().T, FRF)
                            Qrank = np.linalg.matrix_rank(Q, tol=1e-4)
                            Uq, S, UqH = np.linalg.svd(Q)
                            Uqnew = Uq[:, 0:Qrank]
                            # print(f'S:{S}')
                            # Snew = S[0:Qrank]
                            Snew = 1 / (np.sqrt(S[0:Qrank]))
                            Qinvsqrt = np.dot(Uqnew * Snew, Uqnew.conj().T)
                            # term1 = Qnew @ Qnew
                            # term2 = term1 @ Q

                            # err= np.linalg.norm(term2-np.eye(Nrf))
                            # print(f'err:{err}')

                            U, S, VH = np.linalg.svd(Heff * Qinvsqrt)

                            V = VH.T.conj()
                            FBB = np.matmul(Qinvsqrt, V[:, 0:Ns])
                        FBB = np.sqrt(Ns) * FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')

                        Btilde = np.kron(FBB.T, np.identity(Nt))

                        # convert to real values
                        z_ii = batch_Z[ii, :, k]
                        B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                        B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                        B_ii = np.concatenate((B1, B2), axis=0)

                        # B1 = B_ii.T
                        batch_Bz[ii, :, k] = np.matmul(B_ii.T, z_ii)  # update values
                        batch_BB[ii, :, :, k] = np.matmul(B_ii.T, B_ii)
                        batch_B[ii, :, :, k] = B_ii.T
                        batch_Fbb[ii, :, :, k] = FBB

                # Update training data

                batch_Bz = torch.from_numpy(batch_Bz)
                batch_BB = torch.from_numpy(batch_BB)
                batch_X = torch.from_numpy(batch_X)
                batch_Z = torch.from_numpy(batch_Z)
                batch_B = torch.from_numpy(batch_B)

                # batch_Bz = torch.from_numpy(batch_Bz).float()
                # batch_BB = torch.from_numpy(batch_BB).float()
                # batch_X = torch.from_numpy(batch_X).float()
                # batch_Z = torch.from_numpy(batch_Z).float()
                # batch_B = torch.from_numpy(batch_B).float()

                # dis_sum = 0
                # for k in range(K):
                #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
                #     diff = batch_Z[:, :, k] - med
                #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
                # print(f'{jj} error:{dis_sum}')

                if Wideband_Net:
                    s_hat, loss_list = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B)
                else:
                    batch_Bz_sum = 0
                    batch_BB_sum = 0
                    for k in range(K):
                        batch_Bz_sum += batch_Bz[:, :, k]
                        batch_BB_sum += batch_BB[:, :, :, k]
                    s_hat, loss_list = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B, batch_Mask)

                if SUM_LOSS:
                    loss = sum(loss_list)
                else:
                    loss = loss_list[-1]
                s_hat = s_hat.detach().numpy()
                f_all.append(s_hat)
                batch_X = batch_X.numpy()
                batch_Z = batch_Z.numpy()
                batch_B = batch_B.numpy()
                batch_Bz = batch_Bz.numpy()
                batch_BB = batch_BB.numpy()
                print(f'Iteration:{jj}, loss:{loss:.4f}')

                save_paras.add_logs(' Iteration= ' + str(jj)+', loss=' +str(loss.item()))
    return s_hat, f_all, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, batch_At

if __name__== '__main__':
# load data
    dataloader_tr = Data_Fetch(file_dir=dataset_file,
                               file_name=train_data_name,
                               batch_size=train_batch_size,
                               training_set_size=training_set_size,
                               training_set_size_truncated=training_set_size_truncated,
                               data_str='training')
    dataloader_te = Data_Fetch(file_dir=dataset_file,
                               file_name=test_data_name,
                               batch_size=test_batch_size,
                               training_set_size=testing_set_size,
                               data_str='testing')

    # define the network
    if Black_box:
        with torch.no_grad():
            myModel = FcNet(N, K, Loss_scalar=Loss_coef, training_method='unsupervised')
            # myModel = Cnn_Net(N, K, Loss_scalar=Loss_coef, residule=Residule_NN, training_method='unsupervised', device=MainDevice)
    else:
        if Wideband_Net:
            myModel = ScNet_Wideband(N, K, Num_layers, Loss_coef, Residule=Residule_NN, Keep_Bias=Keep_Bias, BN=True)
        else:
            myModel = ScNet(N, K, Num_layers, Loss_coef, IL=Increamental_Learning, Keep_Bias=Keep_Bias, BN=True, Sub_Connected=Sub_Connected)

    myModel.to(MainDevice)

    # print(torch.cuda.memory_allocated(device=MainDevice), flush=True)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()
    #     print('Realsing cache',flush=True)
    # torch.cuda.memory_allocated(device=MainDevice)
    #
    # print('preparing to load model into GPU',flush=True)
    #
    # print(torch.cuda.memory_allocated(device=MainDevice), flush=True)
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()
    # print('After loading model into GPU', flush=True)
    # for name, params in myModel.named_parameters():
    #     print(f'e name {name}, params device {params.device}')
    optimizer = torch.optim.Adam(myModel.parameters(), lr=start_learning_rate, weight_decay=Weight_decay)
    Loss_cache, Lr_list = train_model()

    checkpoint = {'model_state_dict': myModel.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, model_file_name)  # save model

    time_now_end = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
    print('The trainings ends at the time:', time_now_end,flush=True)  # 当前时间
    t_end = time.time()
    time_cost = (t_end - t_start)/3600
    print(f'---------End training------time cost: {time_cost:.4f}h',flush=True)
    save_paras.add_logs('The training ends at the time= ' + str(time_now_end))
    save_paras.add_logs('Training time cost =' + str(time_cost))

    # --------------------draw figure----------------------------
    fig, axs = plt.subplots(ncols=2, nrows=1)

    ax = axs.flatten()

    ax[0].plot(np.arange(len(Loss_cache )) , Loss_cache)

    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('loss value')
    ax[0].grid(True)

    ax[1].plot(np.arange(len(Lr_list)), Lr_list)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('learning rate')
    ax[1].grid(True)

    fig.tight_layout()
    fig_name = 'loss_lr-Epoch.png'
    fig_path = directory_model + '/' + fig_name
    plt.savefig(fig_path) # save figure
    plt.show()
    # plt.plot(Loss_cache, label='loss')
    # plt.legend()
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.show()
    var_dict = {'loss': Loss_cache, 'Lr': Lr_list}
    fullpath = directory_model + '/' + 'training_record.mat'
    spio.savemat(fullpath, var_dict)

    print('-----------------------------Start Test---------------------------------',flush=True)
    s_hat, f_all, batch_H, batch_Fopt, batch_Wopt, batch_Fbb, At = tst_model()
    spio.savemat(dat_file_name,
                 {"H": batch_H, "Fopt": batch_Fopt, "Wopt": batch_Wopt, "Fbb": batch_Fbb, "f": s_hat, 'At': At,
                  'f_all': f_all})

    print('-----------------------------Test Finished---------------------------------')




