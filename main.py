import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import lib.utils as utils
import lib.datasets as dset

from sub_eval import elbo_decomposition_ivvae_catY
from sub_function import *

import scipy.io as sio

parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('--expNum', default=1, type=int, help='experiment number')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset name', choices=['shapes', 'mnist','fashMni'])

parser.add_argument('--modelNum', default=1, type=int, help='1 for mnist, fashMni; 2 for shapes')
parser.add_argument('--latent-dim_z', default=10, type=int, help='size of z dimension')
parser.add_argument('--n_class', default=10, type=int, help='the number of classes = size of y dim')
parser.add_argument('--ngf', default=32, type=int, help='number of encoder filters in first conv layer')
parser.add_argument('--fc_h_dim', default=256, type=int, help='number of hidden neurons in fc layer')
parser.add_argument('--imSz', default=32, type=int, help='image size')
parser.add_argument('--imCh', default=1, type=int, help='number of image channels')

parser.add_argument('--tcvae_flag', default=1, type=int, help='1 for using TC loss; 0 for otherwise')
parser.add_argument('--indepLs_flag', default=1, type=int, help='1 for using vector independence loss; 0 for otherwise')
parser.add_argument('--useSepaUnit', default=1, type=int, help='1 for using separate TC; 2 for using collective TC')

parser.add_argument('--beta_z', default=0, type=float, help='loss weight for KL term | for models with separateTC')
parser.add_argument('--alpha_z', default=0, type=float, help='loss weight for DataLatentMI | for models with separateTC')
parser.add_argument('--gamma_z', default=0, type=float, help='loss weight for DimWiseKL | for models with separateTC')
parser.add_argument('--beta_y', default=0, type=float, help='loss weight for KL term | for models with separateTC')
parser.add_argument('--alpha_y', default=0, type=float, help='loss weight for DataLatentMI | for models with separateTC')
parser.add_argument('--gamma_y', default=0, type=float, help='loss weight for DimWiseKL | for models with separateTC')

parser.add_argument('--beta_h', default=0, type=float, help='loss weight for KL term | for models with collectiveTC')
parser.add_argument('--alpha_h', default=0, type=float, help='loss weight for DataLatentMI | for models with collectiveTC')
parser.add_argument('--gamma_h_z', default=0, type=float, help='loss weight for DimWiseKL | for models with collectiveTC')
parser.add_argument('--gamma_h_y', default=0, type=float, help='loss weight for DimWiseKL | for models with collectiveTC')

parser.add_argument('--lamb_indep', default=4, type=float, help='loss weight for vector independence')

parser.add_argument('--lamb_cls', default=32, type=float, help='loss weight for cls supervision')
parser.add_argument('--temperature', default=.5, type=float, help='gumbel softmax temperature')

parser.add_argument('--lamb_recon', default=1, type=float, help='loss weight for reconstruction')
parser.add_argument('--lamb_obj_sup', default=1, type=float, help='loss weight for supervised training samples')
parser.add_argument('--lamb_obj_unsup', default=1, type=float, help='loss weight for unsupervised training samples')

parser.add_argument('--gpu', type=int, default=1, help='gpu number')
parser.add_argument('--plot_ex', default=1, type=int, help='1 for plotting gen-recon exImgs')
parser.add_argument('--nPlotData', default=12, type=int, help='number of samples for exImgs')

parser.add_argument('--batchSize', default=1024, type=int, help='batch size for each of labeled and unlabeled sets at training')
parser.add_argument('--batchSize_eval', default=1000, type=int, help='batch size at evaluation')
parser.add_argument('--learnRate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--adam_beta1', default=0.9, type=float, help='momentum term of adam')

parser.add_argument('--nEpoch', default=1, type=int, help='number of training epochs')
parser.add_argument('--continueFlag', default=0, type=int, help='1 to resume previous exp, 0 for otherwise')

parser.add_argument('--rngNum', default=12, type=int, help='rng seed for weight initialization and batch sampling')
parser.add_argument('--dataSeed', default=1, type=int, help='rng seed for selecting labeled data')
parser.add_argument('--labels_per_class', default=100, type=int, help='number of labeled samples per class')

parser.add_argument('--unsupLearn', default=1, type=int, help='1 for unsupervised setup; 2 for semi-supervised setup')

parser.add_argument('--result', default='result', type=str, help='path for saving result')


args = parser.parse_args()

######################################################################
check_args(args)
torch.cuda.set_device(args.gpu)
if not os.path.exists(args.result): os.makedirs(args.result)

######################################################################
# Compute fixed random noises for generation examples
torch.manual_seed(1234)
zs_gen = torch.randn(args.nPlotData, args.latent_dim_z)

ys_gen = torch.rand(args.nPlotData - args.n_class, args.n_class)
ys_gen = ys_gen / ys_gen.sum(-1, keepdim=True)
ys_gen = torch.cat((ys_gen, torch.eye(args.n_class)), 0)


######################################################################
# Define datasets and dataloaders
if args.dataset == 'fashMni' or args.dataset == 'mnist':
    train_set = dset.mnist_fashMni_im32(dataset=args.dataset, data_flag='train')
    valid_set = dset.mnist_fashMni_im32(dataset=args.dataset, data_flag='valid')
    test_set = dset.mnist_fashMni_im32(dataset=args.dataset, data_flag='test')

elif args.dataset == 'shapes':
    train_set = dset.shapes_im64(data_flag='train')
    valid_set = dset.shapes_im64(data_flag='valid')
    test_set = dset.shapes_im64(data_flag='test')

valid_loader_eval = DataLoader(valid_set, batch_size=args.batchSize_eval, num_workers=1, shuffle=False, pin_memory=True)
test_loader_eval = DataLoader(test_set, batch_size=args.batchSize_eval, num_workers=1, shuffle=False, pin_memory=True)

# Compute indices of labeled training samples with evenly-distributed classes
args.indices_path_gen = args.result + '/'+ args.dataset +'_indices'
if not os.path.exists(args.indices_path_gen): os.makedirs(args.indices_path_gen)
args.indices_path = os.path.join(args.indices_path_gen, 'ssl%d_rng%d' % (args.labels_per_class, args.dataSeed))

indices_ssl_sup = make_labelDataIdx(args, train_set)

dataset_size_unsup = len(train_set)
nBatchTotal_unsup = int( math.ceil(dataset_size_unsup / float(args.batchSize)) )

if args.unsupLearn != 1: # for semi-supervised learning
    dataset_size_sup = indices_ssl_sup.size(0)
    nBatchTotal_sup = int(math.ceil(dataset_size_sup / float(args.batchSize)))
    if dataset_size_sup != args.labels_per_class * args.n_class:
        raise ValueError('*** n labeledSamples: wrong')


######################################################################
# Set random seed for weight initialization and batch sampling
if args.rngNum != 0:
    torch.manual_seed(args.rngNum)
elif args.rngNum == 0:
    torch.manual_seed(1234)

######################################################################
# Set path for saving results
if args.unsupLearn != 1: # for semi-supervised learning
    args.save_gen = args.result + '/'+ args.dataset +'_ssl'
else: # for unsupervised learning
    args.save_gen = args.result + '/'+ args.dataset +'_unSup'

if not os.path.exists(args.save_gen): os.makedirs(args.save_gen)


args.save = ('exp%d_tc%d_ipLs%d_m%d_r%d_dr%ds%d') % (
    args.expNum, args.tcvae_flag, args.indepLs_flag,
    args.modelNum, args.rngNum, args.dataSeed, args.labels_per_class)

if args.useSepaUnit == 2: # '1 for using separate TC; 2 for using collective TC'
    args.save = 'zyUt_'+args.save

args.save = os.path.join(args.save_gen, args.save)
if not os.path.exists(args.save): os.makedirs(args.save)

args.save_fig = os.path.join(args.save, 'fig')
if not os.path.exists(args.save_fig): os.makedirs(args.save_fig)

###################################################################
# Set model and optimizer

args.tcvae = False
if args.tcvae_flag == 1: args.tcvae = True

args.indepLs = False
if args.indepLs_flag == 1: args.indepLs = True

vae = VAE_idpVec_catY(z_dim=args.latent_dim_z, t_dim = args.n_class, use_cuda=True,
                    tcvae=args.tcvae, indepLs=args.indepLs, useSepaUnit=args.useSepaUnit,
                    modelNum=args.modelNum, ngf=args.ngf, h_dim=args.fc_h_dim, imCh=args.imCh)

optimizer = optim.Adam(vae.parameters(), lr=args.learnRate, betas=(args.adam_beta1, 0.999))


###########################################################################
if args.continueFlag == 0: # at zero th epoch, setup for the loss record and model

    loss_train = init_lossRecord('loss_train', args)
    eval_valid = init_lossRecord('eval_dict', args)

    vae.encoder.apply(weights_init)
    vae.decoder.apply(weights_init)
    vae.eval()

    count_epoch = 0
    iteration = 0

else: # at non-zero th epoch, load the previous loss record and model
    count_epoch = args.nEpoch
    checkpt_filename = os.path.join(args.save, 'model_eph%d.pth' % count_epoch)

    while os.path.exists(checkpt_filename) == False:
        count_epoch -= 1
        checkpt_filename = os.path.join(args.save, 'model_eph%d.pth' % count_epoch)
    print('*** curEpoch = %d' % count_epoch)

    checkpt = torch.load(checkpt_filename)
    vae.load_state_dict(checkpt['state_dict'])

    loss_train = torch.load( os.path.join(args.save, 'lossTrain_eph%d.pth' % count_epoch) )
    eval_valid = torch.load( os.path.join(args.save, 'evalValid_eph%d.pth' % count_epoch) )
    iteration = loss_train['iteration']


##################################################################################################
# Evaluation at zero-th epoch

if count_epoch == 0:
    print('============= Eval Valid @ Eph0')
    vae.eval()
    set_lossWeight(args, vae)
    optimizer.zero_grad()
    torch.set_grad_enabled(False)

    time_eval = time.time()
    eval_valid_indi = elbo_decomposition_ivvae_catY(vae, valid_loader_eval)
    time_eval = time.time() - time_eval

    eval_valid = update_eval(eval_valid, eval_valid_indi, 0.0, time_eval, count_epoch, iteration)


    if args.plot_ex == 1:
        plot_exImg(args, vae, zs_gen, ys_gen, count_epoch, trainSet=train_set, testSet=test_set)

##################################################################################################

print('============= Start Training')

count_epoch += 1
iteration += 1

while count_epoch <= args.nEpoch:

    time_train = time.time()
    count_batch = 0

    shuffle_unsup = torch.randperm(dataset_size_unsup).long()
    if args.unsupLearn != 1: # labeled samples for semi-supervised learning
        shuffle_sup = torch.randperm(dataset_size_sup).long()
        count_sup = 1

    for count_unsup in range(1, nBatchTotal_unsup+1):
        vae.train()
        set_lossWeight(args, vae)
        optimizer.zero_grad()
        torch.set_grad_enabled(True)

        s_idx_unsup = 1 + (count_unsup - 1) * args.batchSize 
        e_idx_unsup = min(count_unsup * args.batchSize, dataset_size_unsup)
        n_unsup = e_idx_unsup - s_idx_unsup + 1
        idxSet_unsup = shuffle_unsup[s_idx_unsup - 1: e_idx_unsup]
        
        x_unsup = train_set.imgs[idxSet_unsup]
        x_unsup = x_unsup.cuda(async=True)

        obj_unsup, elbo_unsup, loss_indi_unsup = vae.elbo_modi(x=x_unsup, t=None, dataset_size=dataset_size_unsup)
        if utils.isnan(obj_unsup).any(): raise ValueError('NaN spotted in objective_unsup')

        obj = args.lamb_obj_unsup * (obj_unsup.mean())
        loss_train = update_lossTrain(loss_train=loss_train, loss_indi_unsup=loss_indi_unsup)
        
        if args.unsupLearn != 1: # labeled samples for semi-supervised learning
            s_idx_sup = 1 + (count_sup - 1) * args.batchSize 
            e_idx_sup = min(count_sup * args.batchSize, dataset_size_sup)
            n_sup = e_idx_sup - s_idx_sup + 1

            idxSet_sup = indices_ssl_sup[shuffle_sup[s_idx_sup - 1: e_idx_sup]]  

            x_sup = train_set.imgs[idxSet_sup]
            t_sup = train_set.labels[idxSet_sup]
            x_sup = x_sup.cuda(async=True)
            t_sup = t_sup.cuda(async=True)

            obj_sup, elbo_sup, loss_indi_sup = vae.elbo_modi(x=x_sup, t=t_sup, dataset_size=dataset_size_sup)
            if utils.isnan(obj_sup).any(): raise ValueError('NaN spotted in objective_sup')

            obj += args.lamb_obj_sup * (obj_sup.mean())
            loss_train = update_lossTrain(loss_train=loss_train, loss_indi_sup=loss_indi_sup)
            

        obj.mul(-1).backward()
        optimizer.step()

        loss_train['epoch'].append(count_epoch)
        loss_train['iteration'] = iteration

        ########################################################
        if args.unsupLearn != 1:
            print('[eph %d/%d %d/%d] U modi_elbo %.2f | L modi_elbo %.2f acc %.1f' %
                  (count_epoch, args.nEpoch, count_batch, nBatchTotal_unsup,
                   loss_indi_unsup['elbo_modi'], loss_indi_sup['elbo_modi'], loss_indi_sup['acc_cls']))
            if count_sup != nBatchTotal_sup:
                count_sup = count_sup + 1
            elif count_sup == nBatchTotal_sup:
                count_sup = 1
                shuffle_sup = torch.randperm(dataset_size_sup).long()
        else:
            print('[eph %d/%d %d/%d] U modi_elbo %.2f' %
                  (count_epoch, args.nEpoch, count_batch, nBatchTotal_unsup, loss_indi_unsup['elbo_modi']))

        count_batch += 1
        iteration += 1

    ############################################################################################
    ## end of one epoch -> evaluation on valid data
    time_train = time.time() - time_train

    vae.eval()
    optimizer.zero_grad()
    torch.set_grad_enabled(False)

    print('*** eval valid')
    time_eval = time.time()
    eval_valid_indi = elbo_decomposition_ivvae_catY(vae, valid_loader_eval)
    time_eval = time.time() - time_eval
    eval_valid = update_eval(eval_valid, eval_valid_indi, time_train, time_eval, count_epoch, iteration)


    ############################################################################################
    time_etc = time.time()

    if args.plot_ex == 1:
        plot_exImg(args, vae, zs_gen, ys_gen, count_epoch, trainSet=train_set, testSet=test_set)

    if count_epoch == 1 or ( eval_valid['elbo_modi'][-1] > max(eval_valid['elbo_modi'][1:-1]) ):
        print('*** save minValLoss = maxValModiElbo @ eph %d' % count_epoch)
        save_model_loss('minValLs', count_epoch, loss_train, eval_valid, vae, args)

    if args.unsupLearn != 1:
        if count_epoch == 1 or ( eval_valid['acc_cls'][-1] > max(eval_valid['acc_cls'][1:-1]) ):
            print('*** save maxClsAcc @ eph %d' % count_epoch)
            save_model_loss('maxClsAcc', count_epoch, loss_train, eval_valid, vae, args)

    print('*** save curEpoch loss & model | delete prevEpoch')
    save_model_loss('countEph', count_epoch, loss_train, eval_valid, vae, args)


    time_etc = time.time() - time_etc
    print('\n[eph %d/%d] time %.1f sec = tr %.1f + eval %.1f + etc %.1f ' % (
        count_epoch, args.nEpoch, (time_train+time_eval+time_etc), time_train, time_eval, time_etc))
    time_remain = (args.nEpoch - count_epoch) * (time_train+time_eval+time_etc)
    print('[eph %d/%d] remaining time %.1f sec ~ %.1f min ~ %.1f hour' %(
        count_epoch, args.nEpoch, time_remain, (time_remain/60), (time_remain/3600) ))
    print('=========================================================')

    count_epoch += 1


print('*** training done @ eph %d' % args.nEpoch)

#################################################################################################

print('*** Final Test Eval: at Minimum Valid Total Loss')

temp_eval = torch.load(os.path.join(args.save, 'evalValid_minValLs.pth'))
eval_epoch = temp_eval['epoch']
eval_iter = temp_eval['iteration']

if not os.path.isfile(os.path.join(args.save, 'evalTest_minVal_eph%d.pth' % eval_epoch)):
    del vae, eval_valid

    eval_valid = init_lossRecord('eval_dict', args)
    eval_test = init_lossRecord('eval_dict', args)

    vae = VAE_idpVec_catY(z_dim=args.latent_dim_z, t_dim=args.n_class, use_cuda=True,
                          tcvae=args.tcvae, indepLs=args.indepLs, useSepaUnit=args.useSepaUnit,
                          modelNum=args.modelNum, ngf=args.ngf, h_dim=args.fc_h_dim, imCh=args.imCh)

    print('   Load model @ eph %d' % eval_epoch)
    checkpt_filename = os.path.join(args.save, 'model_minValLs.pth')
    checkpt = torch.load(checkpt_filename)
    vae.load_state_dict(checkpt['state_dict'])

    vae.eval()
    set_lossWeight(args, vae)
    torch.set_grad_enabled(False)

    print('\n\n Eval valid data')
    eval_valid_indi = elbo_decomposition_ivvae_catY(vae, valid_loader_eval)
    eval_valid = update_eval(eval_valid, eval_valid_indi, 0.0, 0.0, eval_epoch, eval_iter)

    filename = os.path.join(args.save, 'evalValid_minVal_eph%d' % eval_epoch)
    torch.save(eval_valid, filename + '.pth')
    sio.savemat(filename+'.mat', eval_valid)

    print('\n\n Eval test data')
    eval_test_indi = elbo_decomposition_ivvae_catY(vae, test_loader_eval)
    eval_test = update_eval(eval_test, eval_test_indi, 0.0, 0.0, eval_epoch, eval_iter)

    filename = os.path.join(args.save, 'evalTest_minVal_eph%d')
    torch.save(eval_test, filename + '.pth')
    sio.savemat(filename+'.mat', eval_test)


else:
    print('No Need to Compute: evalTest_minVal_eph%d.pth Exists!' % eval_epoch)

