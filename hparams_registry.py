# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from lib import misc

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['RotatedMNIST', 'MNISTUSPS']
    MEDIUM_IMAGES = ['SVHNMNIST']
    RESNET18 = False if dataset == 'VisDA17' else True

    hparams = {}
    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        #assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', RESNET18, lambda r: RESNET18)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, True])))

    _hparam('specify_zdim', True, lambda r: bool(r.choice([False, True])))


    # Network-specific defifitions:
    _hparam('z_dim', 256, lambda r: int(r.choice([16, 128, 256, 512])))



    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        #_hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('lambda', 10.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    if algorithm == 'WD':
        _hparam('weight_decay_wd', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('grad_penalty', 10., lambda r: 10**r.uniform(-2, 1))
        _hparam('lambda_wd', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('wd_steps_per_step', 5, lambda r: int(2**r.uniform(1, 3)))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))


    if algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    if algorithm in ['KL', 'PERM']:
        _hparam('num_samples', 20, lambda r: 20)



    # Dataset-and-algorithm-specific hparam definitions


    if dataset in SMALL_IMAGES or dataset in MEDIUM_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))



    if algorithm != 'KL':
        if dataset in SMALL_IMAGES or dataset in MEDIUM_IMAGES:
            _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)) )
        else:
            _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 5.5)) )


    if algorithm in ['DANN', 'CDANN'] and (dataset in SMALL_IMAGES or dataset in MEDIUM_IMAGES):
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2) )



    if algorithm == 'WD' and (dataset in SMALL_IMAGES or dataset in MEDIUM_IMAGES):
        _hparam('lr_wd', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
    elif algorithm == 'WD':
        _hparam('lr_wd', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )



    if algorithm == 'KL':
        _hparam('augment_softmax', 0.0, lambda r: r.choice([0.0,0.01,0.05]))
        if dataset == 'RotatedMNIST':
            _hparam('kl_reg', 0.2, lambda r: 0.2)
            _hparam('kl_reg_aux', 0.2, lambda r: 0.2)
            _hparam('batch_size', 256, lambda r: 256)
        if dataset == 'MNISTUSPS':
            _hparam('kl_reg', 0.1, lambda r: 0.1)
            _hparam('kl_reg_aux', 0.1, lambda r: 0.1)
            _hparam('batch_size', 256, lambda r: 256)
        elif dataset == 'SVHNMNIST':
            _hparam('kl_reg', 0.055, lambda r: 0.3)
            _hparam('kl_reg_aux', 0.055, lambda r: r.choice([0.0]))
            _hparam('batch_size', 256, lambda r: 256)
            _hparam('z_dim', 16, lambda r: int(r.choice([16])))
            _hparam('augment_softmax', 0.01, lambda r: r.choice([0.0,0.01,0.05]))
        elif dataset == 'VisDA17':
            _hparam('kl_reg', 0.002, lambda r: r.choice([0.05, 0.1, 0.2]))
            _hparam('kl_reg_aux', 0.001, lambda r: r.choice([0.0]))
            _hparam('batch_size', 256, lambda r: int(r.choice([64, 128, 256])))
            _hparam('z_dim', 16, lambda r: int(r.choice([16])))
            _hparam('lr', 1e-5, lambda r: 1e-4)
            _hparam('weight_decay', 0.0, lambda r: 0.)

            _hparam('resnet_dropout', 0., lambda r: r.choice([0.]))
            _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False])))

            _hparam('augment_softmax', 0.05, lambda r: r.choice([0.0,0.01,0.05]))
        else:
            _hparam('kl_reg', 0.1, lambda r: r.choice([0.05, 0.1, 0.2]))
            _hparam('kl_reg_aux', 0.0, lambda r: r.choice([0.0]))
            _hparam('batch_size', 256, lambda r: int(r.choice([64, 128, 256])))
            _hparam('z_dim', 16, lambda r: int(r.choice([16])))
            _hparam('lr', 1e-4, lambda r: 1e-4)
            _hparam('weight_decay', 0.0, lambda r: 0.)

            _hparam('resnet_dropout', 0., lambda r: r.choice([0.]))
            _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False])))

            _hparam('augment_softmax', 0.05, lambda r: r.choice([0.0,0.01,0.05]))

    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, seed).items()}
