import os
import math
import random
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


def prune(A):
    zero = torch.zeros_like(A).to(device)
    A = torch.where(A < 0.3, zero, A)
    return A


def gumble_dag_loss(A):
    expm_A = torch.exp(F.gumbel_softmax(A))
    l = torch.trace(expm_A) - A.size()[0]
    return l


def filldiag_zero(A):
    mask = torch.eye(A.size()[0], A.size()[0]).byte().to(device)
    A.masked_fill_(mask, 0)
    return A


def matrix_poly(matrix, d):
    x = torch.eye(d).to(device) + torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)


def mask_threshold(x):
    x = (x + 0.5).int().float()
    return x


def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--every_degree', '-N', type=int, default=10,
                        help='every N degree as a partition of dataset')
    args = parser.parse_args()
    return args


def weights_init(m):
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif (type(m) == nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)



class DataLoadWithLabel(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data_item = self.data[idx]

        print("Before getting label tensor, self.labels dimensions:", self.labels.shape)

        label_tensor = self.labels[idx].view(-1, 3)

        print("After getting label tensor, label_tensor dimensions:", label_tensor.shape)

        return data_item, label_tensor

    def __len__(self):
        return len(self.data)



def get_batch_unin_dataset_withlabel(dataset_dir, batch_size, dataset="train"):
    data_csv = pd.read_csv(r'D:\trustworthyAI-master\research\CausalVAE\data\Feature.csv')

    data = torch.from_numpy(data_csv.iloc[:, :-3].values).float()

    labels = torch.from_numpy(data_csv.iloc[:, -3:].values).float()

    print("Data dimensions:", data.shape)
    print("Labels dimensions:", labels.shape)

    label_tensor = labels.view(-1, 3)

    dataset = DataLoadWithLabel(data, label_tensor)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=(dataset == "train"))

    return dataset


if __name__ == "__main__":
    args = get_parse_args()
    dataset_dir = "your_dataset.csv"
    batch_size = 32

    dataset = get_batch_unin_dataset_withlabel(dataset_dir, batch_size)

    for data, label1, label2, label3 in dataset:
        print("Data shape:", data.shape)
        print("Label 1 shape:", label1.shape)
        print("Label 2 shape:", label2.shape)
        print("Label 3 shape:", label3.shape)
