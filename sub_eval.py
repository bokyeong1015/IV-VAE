import os
import time
import math
from numbers import Number

import torch
import lib.dist as dist

def estimate_entropies_ivvae_catY(qz_samples, qz_params, q_dist_z, qy_samples, qy_params, q_dist_y,
                            temperature=None):
    if temperature is None:
        raise ValueError('catY temperature should be specified')

    if qz_samples.size(1) > 10000:
        temp_idxSet = torch.randperm(qz_samples.size(1))[:10000].cuda()
        qz_samples = qz_samples.index_select(1, temp_idxSet)
        qy_samples = qy_samples.index_select(1, temp_idxSet)

    K_z, S = qz_samples.size()
    N, _, nparams_z = qz_params.size()
    assert(nparams_z == q_dist_z.nparams)
    assert(K_z == qz_params.size(1))

    K_y, S_y = qy_samples.size()
    N_y, _ = qy_params.size()

    assert (K_y == qy_params.size(1))
    assert (S_y == S and N_y == N)

    marginal_entropies_z = torch.zeros(K_z).cuda()
    joint_entropy_z = torch.zeros(1).cuda()

    marginal_entropies_y = torch.zeros(1).cuda() # does NOT exist for a single dimensional catY
    joint_entropy_y = torch.zeros(1).cuda()
    joint_entropy_zy = torch.zeros(1).cuda()


    k = 0

    while k < S:

        batch_size = min(10, S - k)
        logqz_i = q_dist_z.log_density(
            qz_samples.view(1, K_z, S).expand(N, K_z, S)[:, :, k:k + batch_size],
            qz_params.view(N, K_z, 1, nparams_z).expand(N, K_z, S, nparams_z)[:, :, k:k + batch_size])

        logqy_i = q_dist_y.log_density(
            sample=qy_samples.transpose(0, 1).view(1, S, K_y).expand(N, S, K_y)[:, k:k + batch_size, :],
            logits=qy_params.view(N, 1, K_y).expand(N, S, K_y)[:, k:k + batch_size, :],
            temperature=temperature)
        k += batch_size


        marginal_entropies_z += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).detach()).sum(1)

        logqz = logqz_i.sum(1)  #
        joint_entropy_z += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).detach()).sum(0)
        joint_entropy_y += (math.log(N) - logsumexp(logqy_i, dim=0, keepdim=False).detach()).sum(0)
        joint_entropy_zy += (math.log(N) - logsumexp(logqz + logqy_i, dim=0, keepdim=False).detach()).sum(0)

    marginal_entropies_z /= S
    joint_entropy_z /= S
    marginal_entropies_y /= S
    joint_entropy_y /= S
    joint_entropy_zy /= S

    return marginal_entropies_z, joint_entropy_z, marginal_entropies_y, joint_entropy_y, joint_entropy_zy



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




def elbo_decomposition_ivvae_catY(vae, dataset_loader):

    N = len(dataset_loader.dataset)
    S = 1

    K_z = vae.z_dim
    nparams_z = vae.q_dist_z.nparams
    qz_params = torch.Tensor(N, K_z, nparams_z)

    K_y = vae.n_class
    qy_params = torch.Tensor(N, K_y)

    print('  [Eval] Computing q(z|x) & q(y|x) distributions.')

    n, logpx, loss_cls, acc_cls = 0, 0.0, 0.0, 0.0

    result_temp = {}
    for idx_class in range(vae.n_class):
        result_temp['nCorr_c' + str(idx_class)] = 0

    for (xs, t) in dataset_loader:
        batch_size = xs.size(0)

        xs = xs.cuda()
        t = t.cuda()

        z_params, y_params = vae.encoder.forward(xs)
        z_params = z_params.view(batch_size, K_z, nparams_z)
        qz_params[n:n + batch_size] = z_params.detach()
        qy_params[n:n + batch_size] = y_params.detach()

        n += batch_size

        for _ in range(S):

            z = vae.q_dist_z.sample(params=z_params)
            y = vae.q_dist_y.sample(logits=y_params, temperature=vae.temperature)

            x_params = vae.decoder.forward(z, y)
            logpx += vae.x_dist.log_density(xs, params=x_params).view(batch_size, -1).detach().sum()

            loss_cls += vae.crit_cls(y_params, t).sum().item()
            pred = y_params.argmax(dim=1, keepdim=False)
            acc_cls += pred.eq(t).sum().item()

            for idx_class in range(vae.n_class):
                temp = torch.index_select(pred, 0, torch.nonzero(t == idx_class).squeeze())
                result_temp['nCorr_c' + str(idx_class)] += temp.eq(idx_class).sum().item()


    # classification loss
    loss_cls = loss_cls / (N * S)
    acc_cls = acc_cls * 100.0 / (N * S)

    result_temp['nCorr_str'] = ''
    result_temp['nCorr_all'] = 0
    for idx_class in range(vae.n_class):
        result_temp['nCorr_all'] += result_temp['nCorr_c' + str(idx_class)]
        result_temp['nCorr_str'] += 'c%d  %d | ' % (idx_class, result_temp['nCorr_c' + str(idx_class)])

    # reconstruction term
    logpx = logpx / (N * S)

    qz_params = qz_params.cuda()
    qy_params = qy_params.cuda()

    qz_params_expanded = qz_params.view(N, K_z, 1, nparams_z).expand(N, K_z, S, nparams_z)
    qz_samples = vae.q_dist_z.sample(params=qz_params_expanded)
    qz_samples = qz_samples.transpose(0, 1).contiguous().view(K_z, N * S)

    if S != 1:
        qy_params_expanded = qy_params.view(N, K_y, 1).expand(N, K_y, S)
        qy_params_expanded = qy_params_expanded.transpose(0, 1).contiguous().view(K_y, N * S).transpose(0, 1)
        qy_samples = vae.q_dist_y.sample(logits=qy_params_expanded, temperature=vae.temperature).transpose(0, 1)
    else:
        qy_samples = vae.q_dist_y.sample(logits=qy_params, temperature=vae.temperature).transpose(0,1)

    print('  [Eval] Estimating entropies. - takes time...')
    temp_time = time.time()
    marginal_entropies_z, joint_entropy_z, marginal_entropies_y, joint_entropy_y, joint_entropy_zy = \
        estimate_entropies_ivvae_catY(qz_samples, qz_params, vae.q_dist_z, qy_samples, qy_params, vae.q_dist_y,
                                vae.temperature)
    print('    *** takes %.2f sec' % (time.time() - temp_time))


    if hasattr(vae.q_dist_z, 'NLL'):
        nlogqz_condx = vae.q_dist_z.NLL(qz_params).mean(0)
    else:
        nlogqz_condx = - vae.q_dist_z.log_density(qz_samples,
            qz_params_expanded.transpose(0, 1).contiguous().view(K_z, N * S)).mean(1)

    if hasattr(vae.prior_dist_z, 'NLL'):
        pz_params = vae._get_prior_params('z', N * K_z).contiguous().view(N, K_z, -1)
        nlogpz = vae.prior_dist_z.NLL(pz_params, qz_params).mean(0)
    else:
        nlogpz = - vae.prior_dist_z.log_density(qz_samples.transpose(0, 1)).mean(0)

    if S != 1:
        nlogqy_condx = - vae.q_dist_y.log_density(sample=qy_samples.transpose(0, 1),
                                                  logits=qy_params_expanded, temperature=vae.temperature).mean(0)
    else:
        nlogqy_condx = - vae.q_dist_y.log_density(sample=qy_samples.transpose(0, 1),
                                                  logits=qy_params, temperature=vae.temperature).mean(0)

    nlogpy = - vae.prior_dist_y.log_density(sample=qy_samples.transpose(0, 1),
                                            logits=vae._get_prior_params('y', N * S),
                                            temperature=vae.temperature).mean(0)
    nlogqz_condx = nlogqz_condx.detach()
    nlogpz = nlogpz.detach()
    nlogqy_condx = nlogqy_condx.detach()
    nlogpy = nlogpy.detach()


    dependence_z = (- joint_entropy_z + marginal_entropies_z.sum())[0]
    information_z = (- nlogqz_condx.sum() + joint_entropy_z)[0]
    dimwise_kl_z = (- marginal_entropies_z + nlogpz).sum()
    analytical_cond_kl_z = (- nlogqz_condx + nlogpz).sum()
    dependence_zy = (- joint_entropy_zy + joint_entropy_z + joint_entropy_y)[0]

    information_y = (- nlogqy_condx + joint_entropy_y)[0]
    dependence_y = (- joint_entropy_y + nlogpy)
    dimwise_kl_y = torch.zeros(1)
    analytical_cond_kl_y = (- nlogqy_condx + nlogpy)

    if vae.useSepaUnit == 2:
        information_h = (- nlogqz_condx.sum() - nlogqy_condx + joint_entropy_zy)[0]
        dependence_h = (- joint_entropy_zy + marginal_entropies_z.sum() + joint_entropy_y)[0]

    elbo = logpx - analytical_cond_kl_z - analytical_cond_kl_y


    if vae.useSepaUnit == 1:
        if (not vae.tcvae) and (not vae.indepLs):
            elbo_modi = vae.lamb_recon * logpx - vae.beta_z * analytical_cond_kl_z - vae.beta_y * analytical_cond_kl_y - \
                                vae.lamb_cls * loss_cls

        else:
            elbo_modi = vae.lamb_recon * logpx - (
                        vae.alpha_z * information_z + vae.beta_z * dependence_z + vae.gamma_z * dimwise_kl_z) - \
                        (vae.alpha_y * information_y + vae.beta_y * dependence_y) - \
                        vae.lamb_indep * dependence_zy - vae.lamb_cls * loss_cls


    elif vae.useSepaUnit == 2:
        if (not vae.tcvae) and (not vae.indepLs):
            elbo_modi = vae.lamb_recon * logpx - vae.beta_h * (analytical_cond_kl_z + analytical_cond_kl_y) - \
                        vae.lamb_cls * loss_cls

        else:
            elbo_modi = vae.lamb_recon * logpx - \
                        vae.alpha_h * (information_h) - \
                        vae.beta_h * (dependence_h) - \
                        vae.gamma_h_z * dimwise_kl_z - \
                        vae.gamma_h_y * dependence_y - \
                        vae.lamb_indep * dependence_zy - vae.lamb_cls * loss_cls


    print('  [Eval] Result ====================')
    print('    logpx: %.2f | anal KL: z %.2f  y %.2f' % (logpx, analytical_cond_kl_z, analytical_cond_kl_y))
    print('    ELBO: %.2f | Modi ELBO: %.2f' % (elbo, elbo_modi))
    print('    cls loss: %.2f | acc %.1f' % (loss_cls, acc_cls))

    result = {
        'logpx': logpx.item(),
        'dependence_zy': dependence_zy.item(),
        'dependence_z': dependence_z.item(),
        'information_z': information_z.item(),
        'dimwise_kl_z': dimwise_kl_z.item(),
        'analytical_cond_kl_z': analytical_cond_kl_z.item(),
        'dependence_y': dependence_y.item(),
        'information_y': information_y.item(),
        'dimwise_kl_y': dimwise_kl_y.item(),
        'analytical_cond_kl_y': analytical_cond_kl_y.item(),
        'logqzy': - joint_entropy_zy.item(),
        'logqz_prodmarginals': - marginal_entropies_z.sum().item(),
        'logqz': - joint_entropy_z.item(),
        'logpz': - nlogpz.sum().item(),
        'logqz_condx': - nlogqz_condx.sum().item(),
        'logqy_prodmarginals': - marginal_entropies_y.sum().item(),
        'logqy': - joint_entropy_y.item(),
        'logpy': - nlogpy.sum().item(),
        'logqy_condx': - nlogqy_condx.sum().item(),
        'elbo': elbo.item(),
        'elbo_modi': elbo_modi.item(),
        'loss_cls':loss_cls,
        'acc_cls':acc_cls
    }

    for idx_class in range(vae.n_class):
        result['nCorr_c' + str(idx_class)] = result_temp['nCorr_c' + str(idx_class)]


    return result