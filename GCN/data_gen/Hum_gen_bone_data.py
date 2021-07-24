import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

paris = {
    'HumanBody/xsub': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
        (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

sets = {'train', 'val'}
datasets = {'HumanBody/xsub'}

def gen_bone_data():
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load('../data/{}/{}_joint.npy'.format(dataset, set))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '../data/{}/{}_bone.npy'.format(dataset, set),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))

            # Copy the joints data to bone placeholder tensor
            fp_sp[:, :C, :, :, :] = data


if __name__ == '__main__':
    gen_bone_data()