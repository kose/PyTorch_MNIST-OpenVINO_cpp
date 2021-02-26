#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
import configparser
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import torch
from torchvision import datasets, transforms

from network import NET
from dataset import mnist_dataset


##
## test main function
##
def test(args, model, device):

    testset = mnist_dataset(train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False)

    data, labels_ground_truth = testloader.__iter__().next()
    
    labels_ground_truth = labels_ground_truth.numpy().copy()
    # _pred = model.forward(data).numpy().copy()
    _pred = model(data).numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    result = confusion_matrix(labels_ground_truth, labels_pred)

    print(result)
    


##
## main function
##
def main():
# Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--read-model', default="")
    parser.add_argument('--onnx', default="")

    args = parser.parse_args()

    if args.read_model == "":
        print("Usage: test.py --read-model modelname")
        exit()


    modelname = args.read_model

    device = torch.device("cpu")

    model = NET(1000).to(device)

    model.load_state_dict(torch.load(modelname))

    #
    with torch.no_grad():
        model.eval()
        test(args, model, device)

    # output ONNIX
    onnxfile = args.onnx

    if onnxfile != "":
        input = torch.randn((1, 28 * 28))
        torch.onnx.export(model, input, onnxfile, verbose=False,
                          input_names=['MNIST'], output_names=['digit'])


if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
