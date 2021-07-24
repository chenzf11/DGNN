import os
import pickle
import argparse
import numpy as np
import mydataset
import torch
from torch.utils.data import DataLoader

Trainset = mydataset.mydataset(root='../data/人体姿态序列分类/data/train')
Testset = mydataset.mydataset(root='../data/人体姿态序列分类/data/val')

import sys
sys.path.extend(['../'])

max_body_true = 2
max_body_kinect = 4
num_joint = 17
max_frame = 100


def gendata(data_path, out_path, benchmark='xview', part='train'):
    ignored_samples = []
    classes = os.listdir(data_path+part+'/')

    sample_name = []
    sample_label = []
    for cls in classes:
        for filename in os.listdir(data_path + part + '/' + cls + '/'):
            if filename in ignored_samples:
                continue
            action_class = int(cls)
            sample_name.append(filename)
            sample_label.append(action_class)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    if part == 'train':
        tmp = DataLoader(Trainset, batch_size=Trainset.__len__())
        for a, sample in tmp:
            sample = torch.squeeze(sample, dim=1)
    else:
        tmp = DataLoader(Testset, batch_size=Testset.__len__())
        for a, sample in tmp:
            sample = torch.squeeze(sample, dim=1)

    np.save('{}/{}_joint'.format(out_path, part), sample.numpy())




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MyNPY Converter.')
    parser.add_argument('--data_path', default='../data/人体姿态序列分类/data/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/人体姿态序列分类/missing_samples.txt')
    parser.add_argument('--out_folder', default='../data/HumanBody/')

    benchmarks = ['xsub']
    parts = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmarks:
        for p in parts:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p)
