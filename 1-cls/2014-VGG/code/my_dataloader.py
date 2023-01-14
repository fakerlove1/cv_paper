import torch.nn as nn
import torch
import numpy as np

import pickle

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_train_data(batch_size):
    """
    因为一共10000条数据,所以手写一个按批次读取
    Args:
        batch_size:
    Returns:

    """
    dict_name = unpickle(r"E:\note\cv\data\cifar-10-batches-py\data_batch_1")
    data = dict_name[b'data']
    labels = dict_name[b'labels']
    length = len(labels)
    for i in range(int(length / batch_size)):
        # 如果最后一组数据 小于batch_size数目,则返回最后所有剩余数据
        start = batch_size * i
        end = batch_size * (i + 1)
        if batch_size * (i + 1) > length:
            end = length
        images = torch.FloatTensor(np.array([np.array(x).reshape((3, 32, 32)) for x in data[start:end]]))
        targets = torch.LongTensor(labels[start:end])
        yield images, targets


if __name__ == '__main__':
    for images, targets in get_train_data(10):
        print(images.shape)
        print(targets)
