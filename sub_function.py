import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torchvision.utils import save_image

import lib.dist as dist

from sub_network import *
import scipy.io as sio

from operator import __or__
from functools import reduce

import numpy as np

def check_args(args):
    if args.useSepaUnit == 1:
        if args.beta_h != 0 or args.alpha_h != 0 or \
                args.gamma_h_z != 0 or args.gamma_h_y != 0:
            raise ValueError('* separate TC: set beta_alpha_gamma_h = 0')

        print('* separate TC')

    elif args.useSepaUnit == 2:
        if args.beta_z != 0 or args.alpha_z != 0 or args.gamma_z != 0 or \
                args.beta_y != 0 or args.alpha_y != 0 or args.gamma_y != 0:
            raise ValueError('* collecTC: set beta_alpha_gamma_z_y = 0')

        print('* collective TC')


    if args.dataset == 'fashMni' or args.dataset == 'mnist':
        if args.modelNum != 1 or args.imCh != 1 or args.n_class != 10 or args.imSz != 32:
            raise ValueError('*** mnist or fashMni: Check args')

    elif args.dataset == 'shapes':
        if args.modelNum != 2 or args.imCh != 1 or args.n_class != 3 or args.imSz != 64:
            raise ValueError('*** shapes: Check args')



def init_lossRecord(out_type, args):
    tmp_dict = {'logpx': [], 'logpz': [], 'logqz': [], 'logqz_condx': [], 'logqz_prodmarginals': [],
                  'logpy': [], 'logqy': [], 'logqy_condx': [], 'logqy_prodmarginals': [], 'logqzy': [],
                  'elbo_modi': [], 'elbo': [], 'loss_cls': [], 'acc_cls': []}

    if out_type == 'loss_train':
        loss_train = {'epoch': [], 'iteration': []}
        for k in tmp_dict.keys():
            loss_train['u_'+k] = [] # loss record for unlabeled data
            if args.unsupLearn != 1: # loss record for labeled data in semi-supervised setup
                loss_train['s_'+k] = []
            else: # no use of labeled data in unsupervised setup
                loss_train['s_'+k] = [0.0]

        return loss_train

    elif out_type == 'eval_dict':
        eval_dict = {'information_z': [], 'dependence_z': [], 'dimwise_kl_z': [], 'analytical_cond_kl_z': [],
                      'information_y': [], 'dependence_y': [], 'dimwise_kl_y': [], 'analytical_cond_kl_y': [],
                      'dependence_zy': [], 'time_train': [], 'time_eval': [], 'epoch': [], 'iteration': []}
        for k in tmp_dict.keys():
            eval_dict[k] = []

        for i in range(args.n_class):
            eval_dict['nCorr_c' + str(i)] = []

        return eval_dict


def make_labelDataIdx(args, train_set):
    if not os.path.isfile(args.indices_path + '.pth'):
        print('MAKE labeled data indices: ssl_n%d_rng%d' % (args.labels_per_class, args.dataSeed))

        torch.manual_seed(args.dataSeed)

        temp_labels = train_set.labels.numpy()
        (indices_ssl_sup,) = np.where(reduce(__or__, [temp_labels == i for i in np.arange(args.n_class)]))
        np.random.shuffle(indices_ssl_sup)
        indices_ssl_sup = np.hstack(
            [list(filter(lambda idx: temp_labels[idx] == i, indices_ssl_sup))[:args.labels_per_class] for i in
             range(args.n_class)])

        indices_ssl = {'indices_ssl_sup': indices_ssl_sup, 'labels_per_class': args.labels_per_class,
                       'n_class': args.n_class, 'rngNum': args.dataSeed}
        sio.savemat(args.indices_path + '.mat', indices_ssl)
        torch.save(indices_ssl, args.indices_path + '.pth')

        indices_ssl_sup = torch.from_numpy(indices_ssl_sup)
    else:
        print('LOAD PreComputed data indices_ssl_n%d_rng%d' % (args.labels_per_class, args.dataSeed))
        indices_ssl = torch.load(args.indices_path + '.pth')
        indices_ssl_sup = indices_ssl['indices_ssl_sup']
        indices_ssl_sup = torch.from_numpy(indices_ssl_sup)

    return indices_ssl_sup


def save_model_loss(saveFlag, count_epoch, loss_train, eval_valid, vae, args):
    if saveFlag == 'minValLs' or saveFlag == 'maxClsAcc':

        filename = os.path.join(args.save, 'lossTrain_'+saveFlag)
        torch.save(loss_train, filename+'.pth')
        sio.savemat(filename+'.mat', loss_train)

        filename = os.path.join(args.save, 'evalValid_'+saveFlag)
        torch.save(eval_valid, filename+'.pth')
        sio.savemat(filename+'.mat', eval_valid)

        filename = os.path.join(args.save, 'model_'+saveFlag)
        torch.save({'state_dict': vae.state_dict(), 'args': args}, filename+'.pth')

        filename = os.path.join(args.save, 'epoch_'+saveFlag)
        torch.save({'count_epoch':count_epoch}, filename+'.pth')
        sio.savemat(filename + '.mat', {'count_epoch': count_epoch})

    elif saveFlag == 'countEph':
        filename = os.path.join(args.save, 'lossTrain_eph%d' % count_epoch)
        torch.save(loss_train, filename+'.pth')
        sio.savemat(filename+'.mat', loss_train)

        filename = os.path.join(args.save, 'evalValid_eph%d' % count_epoch)
        torch.save(eval_valid, filename+'.pth')
        sio.savemat(filename+'.mat', eval_valid)

        filename = os.path.join(args.save, 'model_eph%d' % count_epoch)
        torch.save({'state_dict': vae.state_dict(), 'args': args}, filename+'.pth')

        filename_prev_loss = os.path.join(args.save, 'lossTrain_eph%d' % (count_epoch - 1))
        filename_prev_eval = os.path.join(args.save, 'evalValid_eph%d' % (count_epoch - 1))
        filename_prev_model = os.path.join(args.save, 'model_eph%d' % (count_epoch - 1))

        if os.path.exists(filename_prev_loss+'.pth'):
            os.remove(filename_prev_loss+'.pth')
            os.remove(filename_prev_loss+'.mat')
            os.remove(filename_prev_eval+'.pth')
            os.remove(filename_prev_eval+'.mat')
            os.remove(filename_prev_model+'.pth')


def update_eval(eval_rec, eval_indi, time_train, time_eval, epoch, iter):

    for k in eval_indi.keys():
        eval_rec[k].append(eval_indi[k])

    eval_rec['time_train'].append(time_train)
    eval_rec['time_eval'].append(time_eval)
    eval_rec['epoch'] = epoch
    eval_rec['iteration'] = iter

    return eval_rec


def update_lossTrain(loss_train, loss_indi_unsup=None, loss_indi_sup=None):
    if loss_indi_sup is not None:
        for k in loss_indi_sup.keys():
            loss_train['s_'+k].append(loss_indi_sup[k])

    if loss_indi_unsup is not None:
        for k in loss_indi_unsup.keys():
            loss_train['u_'+k].append(loss_indi_unsup[k])
    return loss_train

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.0)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)



class VAE_idpVec_catY(nn.Module):
    def __init__(self, z_dim, t_dim, use_cuda=False,
                 tcvae=False, indepLs=False, useSepaUnit=1,
                 modelNum=1, ngf=32, h_dim=256, imCh=1):

        super(VAE_idpVec_catY, self).__init__()

        self.use_cuda = use_cuda

        self.z_dim = z_dim
        self.n_class = t_dim

        self.ngf = ngf
        self.imCh = imCh

        self.tcvae = tcvae
        self.indepLs = indepLs
        self.useSepaUnit = useSepaUnit

        ###################################################################
        # Set latent and data distributions
        self.prior_dist_z = dist.Normal()
        self.q_dist_z = dist.Normal()

        self.prior_dist_y = dist.GumbelSoftmax_catY(nClass=self.n_class)
        self.q_dist_y = dist.GumbelSoftmax_catY(nClass=self.n_class)

        self.x_dist = dist.Bernoulli()

        ###################################################################
        # To be set by set_lossWeight(args, vae) in main code

        self.beta_z, self.beta_y = 1.0, 1.0
        self.alpha_z, self.alpha_y = 1.0, 1.0
        self.gamma_z, self.gamma_y = 1.0, 1.0

        self.beta_h, self.alpha_h = 1.0, 1.0
        self.gamma_h_z, self.gamma_h_y = 1.0, 1.0

        self.lamb_indep = 1.0
        self.lamb_recon = 1.0
        self.temperature = .67

        self.lamb_cls = 1.0
        self.crit_cls = nn.NLLLoss(reduction='none')

        ###################################################################
        self.register_buffer('prior_params_z', torch.zeros(self.z_dim, 2))
        self.register_buffer('prior_params_y', torch.zeros(self.n_class).fill_(1.0 / self.n_class).log())

        # create the encoder and decoder networks
        if modelNum == 1: # im32 mnist, fashion-mnist
            self.encoder = enc_conv_im32(z_dim=z_dim * self.q_dist_z.nparams, y_dim=t_dim,
                                         imCh=imCh, ngf=ngf, h_dim=h_dim, useBias=True)

            self.decoder = dec_conv_im32(z_dim=z_dim, y_dim=t_dim,
                                         imCh=imCh, ngf=ngf, h_dim=h_dim, useBias=True)

        elif modelNum == 2: # im64 shapes
            self.encoder = enc_conv_im64(z_dim=z_dim * self.q_dist_z.nparams, y_dim=t_dim,
                                         imCh=imCh, ngf=ngf, h_dim=h_dim, useBias=True)

            self.decoder = dec_conv_im64(z_dim=z_dim, y_dim=t_dim,
                                         imCh=imCh, ngf=ngf, h_dim=h_dim, useBias=True)

        if use_cuda:
            self.cuda()



    def _get_prior_params(self, flag_latent, batch_size=1):
        if flag_latent == 'z':
            expanded_size = (batch_size,) + self.prior_params_z.size()
            prior_params = Variable(self.prior_params_z.expand(expanded_size))
        elif flag_latent == 'y':
            expanded_size = (batch_size,) + self.prior_params_y.size()
            prior_params = Variable(self.prior_params_y.expand(expanded_size))

        return prior_params

    def encode(self, x):
        z_params, y_params = self.encoder.forward(x)
        z_params = z_params.view(x.size(0), self.z_dim, self.q_dist_z.nparams)
        zs = self.q_dist_z.sample(params=z_params)
        ys = self.q_dist_y.sample(logits=y_params, temperature=self.temperature)

        return zs, z_params, ys, y_params

    def decode(self, zs, ys):
        x_params = self.decoder.forward(zs, ys)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params


    def reconstruct_img(self, x):
        zs, z_params, ys, y_params = self.encode(x)
        xs, x_params = self.decode(zs, ys)
        return xs, x_params, zs, z_params, ys, y_params


    def elbo_modi(self, x, t, dataset_size):
        batch_size = x.size(0)

        prior_params_z = self._get_prior_params('z', batch_size)
        prior_params_y = self._get_prior_params('y', batch_size)

        if t is None:  # for unlabeled samples
            x_recon, x_params, zs, z_params, ys, y_params = self.reconstruct_img(x)

            loss_cls = torch.zeros(batch_size).type_as(x)
            acc_cls = 0.0

        else: # for labeled samples
            zs, z_params, ys, y_params = self.encode(x)

            temp_t = torch.zeros(ys.size())
            temp_t = temp_t.scatter_(1, t.long().cpu().view(-1, 1), 1).type_as(ys)

            x_recon, x_params = self.decode(zs, temp_t) # using TRUE labels for labeled samples

            loss_cls = self.crit_cls(y_params, t)
            pred = y_params.argmax(dim=1, keepdim=False)

            acc_cls = pred.eq(t).sum().item() * 100.0 / batch_size


        ############################################################

        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)

        logpz = self.prior_dist_z.log_density(zs, params=prior_params_z).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist_z.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        logpy = self.prior_dist_y.log_density(sample=ys, logits=prior_params_y, temperature=self.temperature)
        logqy_condx = self.q_dist_y.log_density(sample=ys, logits=y_params, temperature=self.temperature)

        elbo = logpx + logpz - logqz_condx + logpy - logqy_condx


        if (not self.tcvae) and (not self.indepLs): # vanilla/beta VAE Loss + CLS Loss

            if self.useSepaUnit == 1: # 1 for using separate TC
                if self.beta_z == 0 and self.beta_y == 0:  # to avoid ValueError by KL explosion
                    elbo_modi = self.lamb_recon * logpx - self.lamb_cls * loss_cls
                else:
                    elbo_modi = self.lamb_recon * logpx - self.beta_z * (logqz_condx - logpz) - self.beta_y * (logqy_condx - logpy) - \
                                self.lamb_cls * loss_cls

            elif self.useSepaUnit == 2:   # 2 for using collective TC
                if self.beta_h == 0:
                    elbo_modi = self.lamb_recon * logpx - self.lamb_cls * loss_cls
                else:
                    elbo_modi = self.lamb_recon * logpx - self.beta_h * (
                                logqz_condx + logqy_condx - logpz - logpy) - \
                                    self.lamb_cls * loss_cls

            loss_indi = {
                'logpx': logpx.mean().item(),
                'logqz_condx': logqz_condx.mean().item(),
                'logqz': 0.0,
                'logqz_prodmarginals': 0.0,
                'logpz': logpz.mean().item(),
                'logqy_condx': logqy_condx.mean().item(),
                'logqy': 0.0,
                'logqy_prodmarginals': 0.0,
                'logpy': logpy.mean().item(),
                'logqzy': 0.0,
                'loss_cls': loss_cls.mean().item(),
                'acc_cls': acc_cls,
                'elbo_modi': elbo_modi.mean().item(),
                'elbo': elbo.mean().item()
            }


            return elbo_modi, elbo.detach(), loss_indi

        # IndepVector Loss + TC Loss + CLS Loss
        _logqz = self.q_dist_z.log_density(zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist_z.nparams))

        _logqy = self.q_dist_y.log_density(sample=ys.view(batch_size, 1, self.n_class),
                                           logits=y_params.view(1, batch_size, self.n_class),
                                           temperature=self.temperature)

        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        logqy = (logsumexp(_logqy, dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        logqzy = (logsumexp(_logqz.sum(2) + _logqy, dim=1, keepdim=False) - math.log(batch_size * dataset_size))

        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        logqy_prodmarginals = torch.zeros(batch_size).type_as(x)  # dummy

        if self.useSepaUnit == 1:  # 1 for using separate TC
            elbo_modi = self.lamb_recon * logpx - \
                        self.alpha_z * (logqz_condx - logqz) - \
                        self.beta_z * (logqz - logqz_prodmarginals) - \
                        self.gamma_z * (logqz_prodmarginals - logpz) - \
                        self.alpha_y * (logqy_condx - logqy) - \
                        self.beta_y * (logqy - logpy) - \
                        self.lamb_indep * (logqzy - logqz - logqy) - \
                        self.lamb_cls * loss_cls

        elif self.useSepaUnit == 2:  # 2 for using collective TC
            elbo_modi = self.lamb_recon * logpx - \
                        self.alpha_h * (logqz_condx + logqy_condx - logqzy) - \
                        self.beta_h * (logqzy - logqz_prodmarginals - logqy) - \
                        self.gamma_h_z * (logqz_prodmarginals - logpz) - \
                        self.gamma_h_y * (logqy - logpy) - \
                        self.lamb_indep * (logqzy - logqz - logqy) - \
                        self.lamb_cls * loss_cls

        loss_indi = {
            'logpx': logpx.mean().item(),
            'logqz_condx': logqz_condx.mean().item(),
            'logqz': logqz.mean().item(),
            'logqz_prodmarginals': logqz_prodmarginals.mean().item(),
            'logpz': logpz.mean().item(),
            'logqy_condx': logqy_condx.mean().item(),
            'logqy': logqy.mean().item(),
            'logqy_prodmarginals': logqy_prodmarginals.mean().item(),
            'logpy': logpy.mean().item(),
            'logqzy': logqzy.mean().item(),
            'loss_cls': loss_cls.mean().item(),
            'acc_cls': acc_cls,
            'elbo_modi': elbo_modi.mean().item(),
            'elbo': elbo.mean().item()
        }

        return elbo_modi, elbo.detach(), loss_indi


def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def set_lossWeight(args, vae):

    vae.lamb_indep = args.lamb_indep
    vae.lamb_cls = args.lamb_cls
    vae.lamb_recon = args.lamb_recon

    vae.alpha_z = args.alpha_z
    vae.beta_z = args.beta_z
    vae.gamma_z = args.gamma_z
    vae.alpha_y = args.alpha_y
    vae.beta_y = args.beta_y
    vae.gamma_y = args.gamma_y

    vae.alpha_h = args.alpha_h
    vae.beta_h = args.beta_h
    vae.gamma_h_z = args.gamma_h_z
    vae.gamma_h_y = args.gamma_h_y

    vae.temperature = args.temperature



def plot_exImg(args, vae, zs_gen, ys_gen, count_epoch, trainSet=None, testSet=None):

    zs_gen_copy = zs_gen.clone().cuda()
    ys_gen_copy = ys_gen.clone().cuda()
    y_gen = vae.decoder.forward(zs_gen_copy, ys_gen_copy)
    y_gen = y_gen[:, :args.imCh, :, :]

    if args.dataset == 'mnist':
        sidx_arr_train = torch.LongTensor([3,2,1,1020,1010,1003,2002,83,5003,3001,4000,5000])
        sidx_arr_test = torch.LongTensor([3,2,1,1020,1010,1003,2002,83,5003,3001,4000,5000])
    elif args.dataset == 'fashMni':
        sidx_arr_train = torch.LongTensor([19,2,1,13,10,82,7,12,81,0,87,89])
        sidx_arr_test = torch.LongTensor([19,2,1,13,10,82,7,12,81,0,87,89])
    elif args.dataset == 'shapes':
        sidx_arr_train = range(1, args.nPlotData * 24000, 24000)
        sidx_arr_test = [17176, 10729, 60961, 51142, 40595, 29329, 1, 10001, 20001, 30001, 40001, 50001]

    x_plot_train = torch.zeros(len(sidx_arr_train), y_gen.shape[1], y_gen.shape[2], y_gen.shape[3])
    x_plot_test = torch.zeros(len(sidx_arr_test), y_gen.shape[1], y_gen.shape[2], y_gen.shape[3])

    for kk in range(len(sidx_arr_train)):
        x_plot_train[kk], dummy = trainSet[sidx_arr_train[kk]]
        x_plot_test[kk], dummy = testSet[sidx_arr_test[kk]]

    #################################################################################

    n_train = len(sidx_arr_train)
    n_test = len(sidx_arr_test)
    K_z = vae.z_dim
    nparams_z = vae.q_dist_z.nparams

    z_params_train, y_params_train = vae.encoder.forward(x_plot_train.cuda())
    z_params_train = z_params_train.clone().view(n_train, K_z, nparams_z)
    z_params_train = z_params_train.select(-1, 0)

    y_params_train_hard = vae.q_dist_y.sample_test(y_params_train)
    y_recon_train_hard = vae.decoder.forward(z_params_train,
                                             y_params_train_hard).clone()
    y_recon_train_hard = y_recon_train_hard[:, :args.imCh, :, :]
    y_params_train = y_params_train.exp()

    y_recon_train = vae.decoder.forward(z_params_train, y_params_train).clone()
    y_recon_train = y_recon_train[:, :args.imCh, :, :]

    z_params_test, y_params_test = vae.encoder.forward(x_plot_test.cuda())
    z_params_test = z_params_test.clone().view(n_test, K_z, nparams_z)
    z_params_test = z_params_test.select(-1, 0)

    y_params_test_hard = vae.q_dist_y.sample_test(y_params_test)
    y_recon_test_hard = vae.decoder.forward(z_params_test,
                                            y_params_test_hard).clone()
    y_recon_test_hard = y_recon_test_hard[:, :args.imCh, :, :]
    y_params_test = y_params_test.exp()

    y_recon_test = vae.decoder.forward(z_params_test, y_params_test).clone()
    y_recon_test = y_recon_test[:, :args.imCh, :, :]

    #################################################################################
    y_gen = y_gen.sigmoid()
    y_recon_train = y_recon_train.sigmoid()
    y_recon_test = y_recon_test.sigmoid()
    y_recon_train_hard = y_recon_train_hard.sigmoid()
    y_recon_test_hard = y_recon_test_hard.sigmoid()


    ### save reconstruction results
    save_image(torch.cat((y_gen, y_recon_test, y_recon_test_hard, y_recon_train, y_recon_train_hard), 0),
               os.path.join(args.save_fig, 'eph%d.jpg' % count_epoch), nrow=args.nPlotData,
               padding=2, pad_value=1)

    ### save latent traversal results
    for idx_y in range(y_params_test.size(1)):
        y_modi = torch.zeros(y_params_test.size())

        temp_onehot_idx = torch.ones(y_modi.size(0)).mul(idx_y).long().view(-1, 1)
        y_modi = y_modi.scatter_(1, temp_onehot_idx, 1).type_as(z_params_test)
        y_recon_test = vae.decoder.forward(z_params_test, y_modi).clone()
        if idx_y == 0:
            o_collec_modi = y_recon_test[:, :args.imCh, :, :]
        else:
            o_collec_modi = torch.cat((o_collec_modi, y_recon_test[:, :args.imCh, :, :]), 0)

    o_collec_modi = o_collec_modi.sigmoid()
    save_image(o_collec_modi,
               os.path.join(args.save_fig, 'mnp_eph%d.jpg' % count_epoch), nrow=args.nPlotData,
               padding=2, pad_value=1)


    if count_epoch == 0:
        save_image(torch.cat((x_plot_test, x_plot_test, x_plot_train), 0),
                   os.path.join(args.save_fig, 'gt.jpg'), nrow=args.nPlotData,
                   padding=2, pad_value=1)