import os
import pickle
import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

prefix = "processed"

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif dataset == 'SWaT':
        return 51
    elif dataset == 'WADI':
        return 123
    else:
        raise ValueError('unknown dataset '+str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)


    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """returns normalized and standardized data.
    """
    df = np.asarray(df, dtype=np.float32)
    # np.set_printoptions(threshold=99999999999999999)
    # print(df)
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        inx = np.isnan(df)
        df[inx] = 0
    if np.isinf(df).any():
        inx = np.isinf(df)
        print('Data contains inf values. Will be replaced with 100')
        df[inx] = 100
    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def BatchSlidingWindow(values, window_length):
    # data = []
    # for i in range(len(values) - window_length + 1):
    #     a = []
    #     for j in range(window_length):
    #         a = np.concatenate((a, values[i+j]), axis=0)
    #     # a = np.asarray(a)
    #     data.append(a)
    # data = np.array(data)
    # # assert len(data.shape) == 3
    data = []
    for i in range(len(values) - window_length):
        data.append(values[i:i + window_length])
    data = np.array(data)
    # assert len(data.shape) == 3
    return data

def joint(values):
    data = []
    for i in range(values.shape[0]):
        a = []
        for j in range(values.shape[1]):
            a = np.concatenate((a, values[i][j]), axis=0)
        data = np.concatenate((data, a), axis=0)
    return data

def get_loader(values, batch_size, window_length, input_size, shuffle=False):
    if values.shape[0] % batch_size !=0:
        for i in range(batch_size-values.shape[0] % batch_size):
            a = torch.tensor(np.zeros((1, window_length, input_size), dtype='float32'))
            values = np.concatenate((values, a), axis=0)
    values = torch.tensor(values)
    return DataLoader(dataset=values, batch_size=batch_size, shuffle=shuffle)
    # return DataLoader(dataset=values.view(([values.shape[0], values.shape[1] * values.shape[2]])), batch_size=batch_size, shuffle=shuffle)

def load_data(f_name, f_name2):
    # print('We are loading testing data from:', f_name)
    true_edge = []
    false_edge = []
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            true_edge.append((x, y))
    with open(f_name2, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            false_edge.append((x, y))

    return true_edge, false_edge

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        # return np.dot(vector1, vector2)
        return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass

def GCN_Loss(emb):

    emb_true_first = []
    emb_true_second = []
    emb_false_first = []
    emb_false_second = []

    emb = emb.permute(1, 0)

    true_edges, false_edges = load_data('tmp.txt', 'tmp2.txt')
    for edge in true_edges:
        emb_true_first.append(emb[int(edge[0])].detach().numpy())
        emb_true_second.append(emb[int(edge[1])].detach().numpy())

    for edge in false_edges:
        emb_false_first.append(emb[int(edge[0])].detach().numpy())
        emb_false_second.append(emb[int(edge[1])].detach().numpy())

    T1 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)
    T2 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)

    pos_out = torch.tensor(np.diagonal(T1))
    neg_out = torch.tensor(np.diagonal(T2))

    loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

    return loss