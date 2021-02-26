#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms

from mainloop import train, test
from network import NET

from dataset import mnist_dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epoch', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--test', action='store_true', help="Make images, project")
parser.add_argument('--read-model', default="")
parser.add_argument('--write-model', default="")
parser.add_argument('--ec_dim', type=int, default=1000,
                    help='Number of CNN hidden layer dimension')

def main():
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    modelname = args.write_model
    
    trainset = mnist_dataset(train=True)
    testset = mnist_dataset(train=False)

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 2,
              'pin_memory': True}

    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(args.seed) # network weightを固定
    mp.set_start_method('spawn')

    model = NET(1000).to(device)

    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device, trainset, kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if args.test:
        test(args, model, device, testset)

    if modelname != "":
        torch.save(model.state_dict(), modelname)
        print("save: " + modelname)

if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
