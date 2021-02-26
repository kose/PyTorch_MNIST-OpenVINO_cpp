## mode: python ##
## #coding: utf-8 ##

##
## dataset: MNIST
##

import os
import torch
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

# explanatory variable / independent variable 説明変数/独立変数
# object variable /        dependent variable 目的変数/従属変数


##
## MNISTデータセット
##
class mnist_dataset(torch.utils.data.Dataset):

    def __init__(self, train=True):

        if train:
            csvfile = "../data/mnist_train.csv"
        else:
            csvfile = "../data/mnist_test.csv"

        self.transform = transforms.ToTensor()

        df = pd.read_csv(csvfile, header=None)

        self.data = (df.values[:, 1:] / 255.0).astype(np.float32).reshape(-1, 28, 28)
        self.labels = df.values[:, 0]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        exp_val = self.transform(self.data[idx])

        return exp_val.reshape(28 * 28), self.labels[idx]


##
## メイン
##
if __name__ == '__main__':

    ##
    ## データセット確認
    ##

    cols = 24
    rows = 20
    
    dataset = mnist_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cols*rows, shuffle=True)
    data, labels = dataloader.__iter__().next()

    print(data.shape)
    print(labels)
    
    from matplotlib import pyplot as plt

    figure = np.zeros((28 * rows, 28 * cols), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            image = data[y * cols + x].reshape(28, 28) * 255
            figure[y*28:(y+1)*28, x*28:(x+1)*28] = image.detach().numpy()

    plt.figure(figsize=(9, 6))
    plt.imshow(figure, cmap='Greys_r')
    plt.title("MNIST dataset")
    plt.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # plt.pause(10)
    plt.show()

    # import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
