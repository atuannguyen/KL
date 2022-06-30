# KL Guided Domain Adaptation

This repository is the official implementation for the ICLR 2022 paper [KL Guided Domain Adaptation](https://openreview.net/forum?id=0JzqUlIVVDd).

Please consider citing our paper as

```
@inproceedings{
nguyen2022kl,
title={{KL} Guided Domain Adaptation},
author={A. Tuan Nguyen and Toan Tran and Yarin Gal and Philip Torr and Atilim Gunes Baydin},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=0JzqUlIVVDd}
}
```

## Credits:

A large part of this repo is modified from the DomainBed codebase https://github.com/facebookresearch/DomainBed

## Requirements:
python3, pytorch 1.7.0 or higher, torchvision 0.8.0 or higher

## How to run:

### To run the experiments with the default hyper-parameters (might get slightly sub-optimal performance)

- To run the experiment for Rotated MNIST: For example, target domain 1 (15 degree rotated) and seed 0

### Alternatively, you can also do a full sweep to find the best hyper-parameters as we did

```
python -m scripts.sweep launch \
    --datasets RotatedMNIST 
    --algorithms KL \
    --output_dir ./results_sweep \
    --data_dir /path/to/your/data/ \
    --gpus 0 1 2 3 4 5 6 7  \
    --command_launcher multi_gpu 
```

You can set the --datasets flag to other datasets such as SVHNMNIST, MNISTUSPS, VisDA17, etc.

You can set the --algorithms flag to other algorithms in the paper such as ERM, PERM, DANN, MMD, CORAL, WD.

The --gpus flag include all possible gpus of your system.
