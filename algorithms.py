# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist

import copy
import numpy as np

import networks
from lib.misc import random_pairs_of_minibatches
from loss import *

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'KL'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class PERM(Algorithm):
    """
    Empirical Risk Minimization (ERM) with probabilistic representation network
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams, probabilistic=True)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.num_samples = hparams['num_samples']

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        all_z_params = self.featurizer(all_x)
        z_dim = int(all_z_params.shape[-1]/2)
        z_mu = all_z_params[:,:z_dim]
        z_sigma = F.softplus(all_z_params[:,z_dim:])

        all_z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        all_z = all_z_dist.rsample()

        loss = F.cross_entropy(self.classifier(all_z), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1]/2)
        z_mu = z_params[:,:z_dim]
        z_sigma = F.softplus(z_params[:,z_dim:])

        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        
        probs = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            probs += F.softmax(self.classifier(z),1)
        probs = probs/self.num_samples
        return probs


class KL(Algorithm):
    """
    KL
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(KL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams, probabilistic=True)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        cls_lr = 100*self.hparams["lr"] if hparams['nonlinear_classifier'] else self.hparams["lr"]


        self.optimizer = torch.optim.Adam(
            #list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            [{'params': self.featurizer.parameters(), 'lr': self.hparams["lr"]},
                {'params': self.classifier.parameters(), 'lr': cls_lr}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.num_samples = hparams['num_samples']
        self.kl_reg = hparams['kl_reg']
        self.kl_reg_aux = hparams['kl_reg_aux']
        self.augment_softmax = hparams['augment_softmax']

    def update(self, minibatches, unlabeled=None):

        x = torch.cat([x for x,y in minibatches])
        y = torch.cat([y for x,y in minibatches])

        x_target = torch.cat(unlabeled)

        total_x = torch.cat([x,x_target])
        total_z_params = self.featurizer(total_x)
        z_dim = int(total_z_params.shape[-1]/2)
        total_z_mu = total_z_params[:,:z_dim]
        total_z_sigma = F.softplus(total_z_params[:,z_dim:]) 

        z_mu, z_sigma = total_z_mu[:x.shape[0]], total_z_sigma[:x.shape[0]]
        z_mu_target, z_sigma_target = total_z_mu[x.shape[0]:], total_z_sigma[x.shape[0]:]

        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample()

        z_dist_target = dist.Independent(dist.normal.Normal(z_mu_target,z_sigma_target),1)
        z_target = z_dist_target.rsample()

        preds = torch.softmax(self.classifier(z),1)
        if self.augment_softmax != 0.0:
            K = 1 - self.augment_softmax * preds.shape[1]
            preds = preds*K + self.augment_softmax
        loss = F.nll_loss(torch.log(preds),y) 

        mix_coeff = dist.categorical.Categorical(x.new_ones(x.shape[0]))
        mixture = dist.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
        mix_coeff_target = dist.categorical.Categorical(x_target.new_ones(x_target.shape[0]))
        mixture_target = dist.mixture_same_family.MixtureSameFamily(mix_coeff_target,z_dist_target)
        
        obj = loss
        kl = loss.new_zeros([])
        kl_aux = loss.new_zeros([])
        if self.kl_reg != 0.0:
            kl = (mixture_target.log_prob(z_target)-mixture.log_prob(z_target)).mean()
            obj = obj + self.kl_reg*kl
        if self.kl_reg_aux != 0.0:
            kl_aux = (mixture.log_prob(z)-mixture_target.log_prob(z)).mean()
            obj = obj + self.kl_reg_aux*kl_aux

        self.optimizer.zero_grad()
        obj.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'kl': kl.item(), 'kl_aux': kl_aux.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1]/2)
        z_mu = z_params[:,:z_dim]
        z_sigma = F.softplus(z_params[:,z_dim:]) 

        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        
        preds = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            preds += F.softmax(self.classifier(z),1)
        preds = preds/self.num_samples

        K = 1 - 0.05 * preds.shape[1]
        preds = preds*K + 0.05
        return preds

class KLUP(Algorithm):
    """
    KLUP
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(KLUP, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams, probabilistic=True)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        cls_lr = 100*self.hparams["lr"] if hparams['nonlinear_classifier'] else self.hparams["lr"]


        self.optimizer = torch.optim.Adam(
            #list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            [{'params': self.featurizer.parameters(), 'lr': self.hparams["lr"]},
                {'params': self.classifier.parameters(), 'lr': cls_lr}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.num_samples = hparams['num_samples']
        self.kl_reg = hparams['kl_reg']
        self.kl_reg_aux = hparams['kl_reg_aux']
        self.augment_softmax = hparams['augment_softmax']

    def update(self, minibatches, unlabeled=None):
        x = torch.cat([x for x,y in minibatches])
        y = torch.cat([y for x,y in minibatches])

        x_target = torch.cat(unlabeled)

        total_x = torch.cat([x,x_target])
        total_z_params = self.featurizer(total_x)
        z_dim = int(total_z_params.shape[-1]/2)
        total_z_mu = total_z_params[:,:z_dim]
        total_z_sigma = F.softplus(total_z_params[:,z_dim:]) + 0.1 

        z_mu, z_sigma = total_z_mu[:x.shape[0]], total_z_sigma[:x.shape[0]]
        z_mu_target, z_sigma_target = total_z_mu[x.shape[0]:], total_z_sigma[x.shape[0]:]

        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample()

        preds = torch.softmax(self.classifier(z),1)
        if self.augment_softmax != 0.0:
            K = 1 - self.augment_softmax * preds.shape[1]
            preds = preds*K + self.augment_softmax
        loss = F.nll_loss(torch.log(preds),y) 

        mix_coeff = x.new_ones(x.shape[0]) / x.shape[0]
        mix_coeff_target = x_target.new_ones(x_target.shape[0]) / x_target.shape[0]

        #reg = kl_upper(z_mu,z_sigma,mix_coeff,z_mu_target,z_sigma_target,mix_coeff_target) \
        reg = kl_upper(z_mu_target,z_sigma_target,mix_coeff_target,z_mu,z_sigma,mix_coeff) \
                #+ kl_lower(z_mu_target,z_sigma_target,mix_coeff_target,z_mu,z_sigma,mix_coeff)

        # moment matching
        m1 = torch.sum(z_mu * mix_coeff[:,None],0)
        m2 = torch.sum((z_mu**2+z_sigma**2)*mix_coeff[:,None],0)
        var =  m2 - m1**2 + 1e-10
        m1_target = torch.sum(z_mu_target * mix_coeff_target[:,None],0)
        m2_target = torch.sum((z_mu_target**2+z_sigma_target**2)*mix_coeff_target[:,None],0)
        var_target =  m2_target - m1_target**2 + 1e-10
        reg2 = 0.5*(torch.log(var) - torch.log(var_target) + (var_target+(m1-m2)**2)/var - 1).sum()

        obj = loss + self.kl_reg*(reg+reg2)/2

        self.optimizer.zero_grad()
        obj.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'reg': reg.item(), 'reg2': reg2.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1]/2)
        z_mu = z_params[:,:z_dim]
        z_sigma = F.softplus(z_params[:,z_dim:]) +0.1

        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        
        preds = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            preds += F.softmax(self.classifier(z),1)
        preds = preds/self.num_samples

        return preds


class WD(Algorithm):
    """Wasserstein Distance guided Representation Learning"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, class_balance=False):

        super(WD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.fw = networks.MLP(self.featurizer.n_outputs,1,self.hparams)

        # Optimizers
        self.wd_opt = torch.optim.Adam(
            self.fw.parameters(),
            lr=self.hparams["lr_wd"],
            weight_decay=self.hparams['weight_decay_wd'])

        self.main_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def wd_loss(self,h_s,h_t,for_fw=True):
        batch_size = h_s.shape[0]
        alpha = torch.rand([batch_size,1]).to(h_s.device)
        h_inter = h_s*alpha + h_t*(1-alpha)
        h_whole = torch.cat([h_s,h_t,h_inter],0)
        critic = self.fw(h_whole)

        critic_s = critic[:h_s.shape[0]]
        critic_t = critic[h_s.shape[0]:h_s.shape[0]+h_t.shape[0]]
        wd_loss = critic_s.mean() - critic_t.mean()

        if for_fw==False:
            return wd_loss
        else:
            epsilon = 1e-10 # for stable torch.sqrt
            grad = autograd.grad(critic.sum(),
                [h_whole], create_graph=True)[0]
            grad_penalty = ((torch.sqrt((grad**2).sum(dim=1)+epsilon)-1)**2).mean(dim=0)
            return -wd_loss + self.hparams['grad_penalty'] * grad_penalty
            

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        features_target = [self.featurizer(xit) for xit in unlabeled]
        total_features = features+features_target
        total_d = len(total_features)

        for _ in range(self.hparams['wd_steps_per_step']):
            # train fw
            fw_loss = 0.0
            for i in range(total_d):
                for j in range(i + 1, total_d):
                    fw_loss += self.wd_loss(total_features[i], total_features[j], True)
            fw_loss /= (total_d * (total_d - 1) / 2)
            self.wd_opt.zero_grad()
            fw_loss.backward(retain_graph=True)
            self.wd_opt.step()


        # Train main network
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
        for i in range(total_d):
            for j in range(i + 1, total_d):
                penalty += self.wd_loss(total_features[i], total_features[j],False)

        objective /= nmb
        if nmb > 1:
            penalty /= (total_d * (total_d - 1) / 2)

        self.main_opt.zero_grad()
        (objective + (self.hparams['lambda_wd']*penalty)).backward()
        self.main_opt.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(Algorithm):
    """Domain-Adversarial Neural Networks"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, class_balance=False):

        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        x_each_domain = [x for x,y in minibatches] + unlabeled
        x = torch.cat([x for x,y in minibatches])
        y = torch.cat([y for x,y in minibatches])
        x_target = torch.cat(unlabeled)
        total_x = torch.cat([x,x_target])
        total_z = self.featurizer(total_x)

        z = total_z[:x.shape[0]]

        disc_input = total_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, x in enumerate(x_each_domain)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            preds = self.classifier(z)
            classifier_loss = F.cross_entropy(preds, y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))



class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        features_target = [self.featurizer(xit) for xit in unlabeled]
        total_features = features+features_target
        total_d = len(total_features)

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
        for i in range(total_d):
            for j in range(i + 1, total_d):
                penalty += self.mmd(total_features[i], total_features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (total_d * (total_d - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class AbstractPMMD(PERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractPMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features, classifs, targets = [], [], []
        for (x,y) in minibatches:
            z_params = self.featurizer(x)
            z_dim = int(z_params.shape[-1]/2)
            z_mu = z_params[:,:z_dim]
            z_sigma = F.softplus(z_params[:,z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
            z = z_dist.rsample()
            features.append(z)
            classifs.append(self.classifier(z))
            targets.append(y)
        
        features_target = []
        for x in unlabeled:
            z_params = self.featurizer(x)
            z_dim = int(z_params.shape[-1]/2)
            z_mu = z_params[:,:z_dim]
            z_sigma = F.softplus(z_params[:,z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
            z = z_dist.rsample()
            features_target.append(z)

        total_features = features+features_target
        total_d = len(total_features)

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
        for i in range(total_d):
            for j in range(i + 1, total_d):
                penalty += self.mmd(total_features[i], total_features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (total_d * (total_d - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class PMMD(AbstractPMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PMMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


