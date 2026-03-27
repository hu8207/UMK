# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
from pynndescent import NNDescent

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

prefix = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'input', 'processed'))


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    # for i in range(0, z.shape[1], 20):
    #     with open(filename + '_' + str(i) + '.txt', 'w') as file:
    #         for j in range(0, z.shape[0]):
    #             for k in range(0, z.shape[2]):
    #                 file.write('%f ' % (z[j][i][k]))
    #             file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif dataset == 'NTW':
        return 9
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def load_smd_smap_msl(dataset, batch_size=512, window_size=60, stride_size=10, train_split=0.6, label=False,
                      do_preprocess=True, train_start=0,
                      test_start=0, k=10, alpha=0.5, seed=15):
    """支持SMAP的.npy格式的通用加载函数"""
    x_dim = get_data_dim(dataset)
    global prefix

    # SMAP专用处理逻辑
    if dataset == 'SMAP':
        # 从npy文件加载数据
        train_data = np.load(f"{prefix}/SMAP_train.npy")
        test_data = np.load(f"{prefix}/SMAP_test.npy")
        test_label = np.load(f"{prefix}/SMAP_test_label.npy")
        print('test_data：'+ str(test_data.shape))
        print('train_data:'+ str(train_data.shape))
        print('test_label：' + str(test_label.shape))

        # 标准化处理
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
    elif dataset == 'NTW':
        # 从npy文件加载数据
        train_data = np.load(f"{prefix}/NIPS_TS_Water_train.npy")
        test_data = np.load(f"{prefix}/NIPS_TS_Water_test.npy")
        test_label = np.load(f"{prefix}/NIPS_TS_Water_test_label.npy")
        print(np.array(dataset).shape)


        # 标准化处理
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

    else:
        # 原有pkl文件加载逻辑
        try:
            with open(os.path.join(prefix, dataset + '_test.pkl'), "rb") as f:
                test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        except (KeyError, FileNotFoundError):
            test_data = None

        try:
            with open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb") as f:
                test_label = pickle.load(f).reshape((-1))[test_start:]
        except (KeyError, FileNotFoundError):
            test_label = None

        # 标准化处理
        if do_preprocess:
            test_data = preprocess(test_data)

    # 通用处理逻辑
    n_sensor = x_dim  # 直接从预设维度获取传感器数量
    print(f'n_sensor: {n_sensor}')

    # 划分训练测试集
    if dataset == 'SMAP':
        # SMAP使用独立训练集
        train_df = train_data
        test_df = test_data
    elif dataset == 'NTW':
        train_df = train_data
        test_df = test_data

    else:
        whole_data = test_data  # 其他数据集使用测试数据划分
        train_size = int(len(whole_data) * train_split)
        train_df = whole_data[:train_size]
        test_df = whole_data[train_size:]

        if test_label is not None:

            test_label = np.asarray(test_label).reshape(-1)
            if len(test_label) != len(whole_data):

                min_len = min(len(test_label), len(whole_data))
                whole_data = whole_data[:min_len]
                test_label = test_label[:min_len]

                train_size = int(len(whole_data) * train_split)
                train_df = whole_data[:train_size]
                test_df = whole_data[train_size:]

            test_label = test_label[train_size:]
            print('test_label after split unique:', np.unique(test_label), 'anomaly ration', float(np.sum(test_label)) / float(len(test_label)))

    # 创建数据集
    train_dataset = Smd_smap_msl_dataset(
        train_df,
        np.zeros(len(train_df)),  # 训练标签全0
        window_size,
        stride_size,
        k=k,
        alpha=alpha,
        train_split=train_split,
        train='train',
        dataset=dataset,
        seed=seed
    )

    test_dataset = Smd_smap_msl_dataset(
        test_df,
        test_label,
        window_size,
        stride_size,
        k=k,
        alpha=alpha,
        train_split=train_split,
        train='test',
        dataset=dataset,
        seed=seed
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader, n_sensor


def load_smd_smap_msl_occ(dataset, batch_size=512, window_size=60, stride_size=10, train_split=0.6, label=False,
                          do_preprocess=True, train_start=0,
                          test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """

    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:, :]

    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    n_sensor = train_data.shape[1]
    print('n_sensor', n_sensor)

    train_df = train_data[:]
    train_label = [0] * len(train_df)

    val_df = train_data[int(train_split * len(train_data)):]
    val_label = [0] * len(val_df)

    test_df = test_data[int(train_split * len(test_data)):]
    test_label = test_label[int(train_split * len(test_data)):]
    print('testset size', test_label.shape, 'anomaly ration', sum(test_label) / len(test_label))

    if label:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size),
                                  batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size),
                                  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Smd_smap_msl_dataset(val_df, val_label, window_size, stride_size), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(Smd_smap_msl_dataset(test_df, test_label, window_size, stride_size), batch_size=batch_size,
                             shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


def preprocess(df, mode='Normal'):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    if mode == 'Normal':
        df = StandardScaler().fit_transform(df)
    else:
        df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df



class Smd_smap_msl_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10, k=10, alpha=0.5, train_split=0.8, train='train',
                 dataset='MSL', seed=15) -> None:
        super(Smd_smap_msl_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.train_split = train_split

        self.data, self.idx, self.label = self.preprocess(df, label)
        # self.columns = np.append(df.columns, ["Label"])
        # self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label) / len(self.label))
        print('idx', self.idx.shape)
        print('data', self.data.shape)
        # print(len(self.data), len(self.idx), len(self.label))
        self.k = k
        self.alpha = alpha
        self.train = train
        self.dataset = dataset
        self.seed = seed
        filename = "save_near_index/{}_near{}_train{}_windowsize{}_trainsplit{}_seed{}.pkl".format(self.dataset, k,
                                                                                                   self.train,
                                                                                                   self.window_size,
                                                                                                   self.train_split,
                                                                                                   self.seed)
        if os.path.exists(filename):
            print('load near index from', filename)
            neighbors_index = pickle.load(open(filename, 'rb'))
        else:
            X_rshaped = self.data
            np.random.seed(self.seed)
            index = NNDescent(X_rshaped, n_jobs=-1)
            neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
            neighbors_index = neighbors_index[:, 1:]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pickle.dump(neighbors_index, open(filename, 'wb'))
            print('save near index to', filename)
        self.neighbors_index = neighbors_index
        # 修改数据增强部分为确定性增强
        self.rng = np.random.RandomState(seed)

    def preprocess(self, df, label):
        start_idx = np.arange(0, len(df) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        label = [0 if sum(label[index:index + self.window_size]) == 0 else 1 for index in start_idx]
        return df, start_idx, np.array(label)

    def __len__(self):
        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D

        start = self.idx[index]
        end = start + self.window_size
        data_origin = self.data[start:end].reshape([self.window_size, -1, 1])

        augment_index = self.neighbors_index[start:end]

        # import pdb; pdb.set_trace()
        random_index = augment_index[:, np.random.choice(range(self.k), 1)]

        # data_origin = torch.tensor(self.data[index])
        data_near = self.data[random_index].reshape([self.window_size, -1, 1])
        alpha = np.random.uniform(0, self.alpha)
        augment_data = (alpha) * data_origin + (1 - alpha) * data_near

        return torch.FloatTensor(data_origin).transpose(0, 1), torch.FloatTensor(augment_data).transpose(0, 1), \
               self.label[index], index
