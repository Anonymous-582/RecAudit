from typing import Dict, Union
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color, parser_yaml
from utils import *
import torch
from recstudio.data.dataset import SeqDataset
from recstudio.ann import sampler
import numpy as np
import recstudio.model.loss_func as lfc
import os
import shutil
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from typing import Sized, Dict, Optional, Iterator, Union
import torch.optim as optim
import time
import copy
import logging
from sklearn.metrics import roc_auc_score, auc, roc_curve

class DataSampler(Sampler):
    r"""Data sampler to return index for batch data.

    The datasampler generate batches of index in the `data_source`, which can be used in dataloader to sample data.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=True, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        if self.shuffle:
            output = torch.randperm(
                n, generator=generator).split(self.batch_size)
        else:
            output = torch.arange(n).split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
            
class SortedDataSampler(Sampler):
    r"""Data sampler to return index for batch data, aiming to collect data with similar lengths into one batch.

    In order to save memory in training producure, the data sampler collect data point with similar length into one batch. 

    For example, in sequential recommendation, the interacted item sequence of different users may vary differently, which may cause
    a lot of padding. By considering the length of each sequence, gathering those sequence with similar lengths in the same batch can
    tackle the problem. 

    If `shuffle` is `True`, length of sequence and the random index are combined together to reduce padding without randomness.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            output = torch.div(torch.randperm(
                n), (self.batch_size * 10), rounding_mode='floor')
            output = self.data_source.sample_length + output * \
                (self.data_source.sample_length.max() + 1)
        else:
            output = self.data_source.sample_length
        output = torch.sort(output).indices
        output = output.split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
            
class MIADataset(Dataset):
    def __init__(self, dscore_member, dscore_nonmember, mia_data_mode='mean', gen_fea=True, inter_used_rate=None):
        super(MIADataset, self).__init__()
        self.dscore_member = dscore_member
        self.dscore_nonmember = dscore_nonmember
        self.mode = mia_data_mode
        self.inter_used_rate = inter_used_rate
        if gen_fea:
            self.gen_fea()
            self.tran_fea()
    
    def gen_fea(self):
        self.member_features = []
        self.nonmember_features = []
        self.num_members = len(self.dscore_member)
        self.num_nonmembers = len(self.dscore_nonmember)
        self.user_ids = np.zeros(self.num_members + self.num_nonmembers)
        self.labels = np.zeros(self.num_members + self.num_nonmembers)
        self.labels[:self.num_members] = 1
        self.labels = torch.LongTensor(self.labels)
        self.sample_len = torch.LongTensor(np.zeros(self.num_members + self.num_nonmembers))

        member_list = list(self.dscore_member.keys())
        nonmember_list = list(self.dscore_nonmember.keys())
        self.features = {}
        for i in range(self.num_members+self.num_nonmembers):
            if i < self.num_members:
                key = member_list[i]
                features = torch.Tensor(self.dscore_member[key]['features'])
                sort_index = np.argsort(np.array(self.dscore_member[key]['end']))
            else:
                key = nonmember_list[i-self.num_members]
                features = torch.Tensor(self.dscore_nonmember[key]['features'])
                sort_index = np.argsort(np.array(self.dscore_nonmember[key]['end']))

            self.user_ids[i] = key
            num_fea = len(features[0])
            seq_len = len(features)

            self.sample_len[i] = seq_len
            new_features = torch.zeros((seq_len, num_fea))

            for j in range(seq_len):
                new_features[j] = features[sort_index[j]]
            if self.inter_used_rate is not None:
                use_seq_len = int(seq_len * self.inter_used_rate)
                new_features = new_features[-use_seq_len:]
            if self.mode == 'mean':
                self.features[i] = torch.mean(new_features, dim=0)
            elif self.mode == 'max':
                self.features[i] = torch.max(new_features, dim=0).values
            elif self.mode == 'min':
                self.features[i] = torch.min(new_features, dim=0).values
            elif self.mode == 'sum':
                self.features[i] = torch.sum(new_features, dim=0)
            elif self.mode == 'ccs':
                self.features[i] = new_features[-1,:]
            else:
                self.features[i] = new_features
            
        self.user_ids = torch.LongTensor(self.user_ids)
    
    def tran_fea(self):
        if self.mode == 'all':
            pass
        else:
            new_fea = torch.zeros((len(self.features), len(self.features[0])))
            for i in range(len(self.features)):
                new_fea[i] = self.features[i]
            bn = torch.nn.BatchNorm1d(num_features=len(self.features[0]), eps=0, affine=False, track_running_stats=False)
            self.features = bn(new_fea)

    def __len__(self):
        r"""Return the length of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            index = np.array([index])
        if isinstance(index, torch.Tensor):
            index = index.numpy()
        
        data = {}
        if self.mode=='all':
            data['input'] = pad_sequence(tuple([self.features[id] for id in index]), batch_first=True)
        else:
            data['input'] = torch.vstack(tuple([self.features[id] for id in index]))
        data['label'] = self.labels[index]
        data['seqlen'] = self.sample_len[index]
        data['user_id'] = self.user_ids[index]
        return data

    @property
    def sample_length(self):
        return self.sample_len

    def loader(self, batch_size, shuffle=True, num_workers=0, drop_last=False):
        if self.mode == 'all':
            sampler = SortedDataSampler(self, batch_size, shuffle, drop_last)
        else:
            sampler = DataSampler(self, batch_size, shuffle, drop_last)
        return DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers)

    def build(self, split_ratio=[0.8,0.2]):
        train_dataset = MIADataset(0,0,gen_fea=False)
        val_dataset = MIADataset(0,0,gen_fea=False)
        num_train_samples = int(split_ratio[0] * len(self))
        train_index = np.random.choice(np.arange(len(self)), num_train_samples, replace=False)
        val_index = np.setdiff1d(np.arange(len(self)), train_index)
        train_dataset.features = self.features[train_index]
        train_dataset.labels = self.labels[train_index]
        train_dataset.sample_len = self.sample_len[train_index]
        train_dataset.user_ids = self.user_ids[train_index]

        val_dataset.features = self.features[val_index]
        val_dataset.labels = self.labels[val_index]
        val_dataset.sample_len = self.sample_len[val_index]
        val_dataset.user_ids = self.user_ids[val_index]
        return train_dataset, val_dataset


def labels_acc(pre, labels):
    return (torch.sum(pre == labels) / len(labels)).item()

class MIAModel(torch.nn.Module):
    def __init__(self, num_fea, avg_mode = 'mean', hiddens = [64,32,8], RNN_layers = 2, activate_func = 'Relu'):
        super(MIAModel, self).__init__()
        assert avg_mode in ['mean', 'max', 'min', 'sum', 'LSTM', 'GRU'], 'Avg_mode Error!'
        self.avg_mode = avg_mode
        self.num_fea = num_fea
        self.hiddens = hiddens
        self.RNN_layers = RNN_layers
        self.layers = torch.nn.ModuleList()
        if activate_func == 'Relu':
            self.activate_func = torch.nn.ReLU()
            
        if avg_mode == 'LSTM':
            self.layers.append(torch.nn.LSTM(self.num_fea, self.hiddens[0], self.RNN_layers))
        elif avg_mode == 'GRU':
            self.layers.append(torch.nn.GRU(self.num_fea, self.hiddens[0], self.RNN_layers))
        else:
            self.layers.append(torch.nn.Linear(self.num_fea, hiddens[0]))

        self.layers.append(torch.nn.ReLU())
        for i in range(1,len(hiddens)):
            self.layers.append(torch.nn.Linear(hiddens[i-1], hiddens[i]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hiddens[-1], 2))

    def forward(self, batch):
        if self.avg_mode == 'LSTM':
            h = self.layers[0](batch)[0][:,-1,:]
        elif self.avg_mode == 'GRU':
            h = self.layers[0](batch)[0][:,-1,:]
        else:
            h = batch
            
        if self.avg_mode not in ['LSTM', 'GRU']:
            h = self.layers[0](h)
        for i in range(1,len(self.layers)):
            h = self.layers[i](h)
        return h
    
    def fit(self, train_loader, val_loader=None, early_stop_patience=50, epochs=300, optim='Adam', lr=0.001, val_check=True, start=0, end=-1):
        self.lr = lr
        self.start = start
        self.end = end
        best_parameters = copy.deepcopy(self.state_dict())
        best_val_auc = 0
        early_stop_track = 0
        current_val_auc = 0
        loss_func=torch.nn.CrossEntropyLoss()
        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        for e in range(epochs):
            tik = time.time()
            self.train()

            train_loss = []

            for i, input in enumerate(train_loader):
                if self.avg_mode in ['LSTM', 'GRU']:
                    batch, labels = input['input'][:,:,start:end], input['label']
                else:
                    batch, labels = input['input'][:,start:end], input['label']

                output = self(batch)

                num_sam_each_batch = batch.shape[0]
                loss = loss_func(output, labels)
                if i == 0:
                    pre = output.detach().clone().numpy()
                    target = labels.detach().clone().numpy()
                else:
                    pre = np.vstack((pre, output.detach().clone().numpy()))
                    target = np.hstack((target, labels.detach().clone().numpy()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tok_train = time.time()
                for _ in range(num_sam_each_batch):
                    train_loss.append(loss.item())
            target = np.eye(2,dtype=int)[target]
            current_train_auc = roc_auc_score(target, pre)
            current_train_loss = np.mean(np.array(train_loss))

            if val_loader is not None:
                self.eval()
                for i, input in enumerate(val_loader):
                    if self.avg_mode in ['LSTM', 'GRU']:
                        batch, labels = input['input'][:,:,start:end], input['label']
                    else:
                        batch, labels = input['input'][:,start:end], input['label']
        
                    output = self(batch)
                    num_sam_each_batch = batch.shape[0]

                    if i == 0:
                        pre = output.detach().clone().numpy()
                        target = labels.detach().clone().numpy()
                    else:
                        pre = np.vstack((pre, output.detach().clone().numpy()))
                        target = np.hstack((target, labels.detach().clone().numpy()))
                target = np.eye(2,dtype=int)[target]
                
                current_val_auc = roc_auc_score(target, pre)

                if val_check:
                    if current_val_auc <= best_val_auc:
                        early_stop_track += 1
                    else:
                        early_stop_track = 0
                        best_val_auc = current_val_auc
                        best_parameters = copy.deepcopy(self.state_dict())
                else:
                    best_parameters = copy.deepcopy(self.state_dict())
            else:
                best_parameters = copy.deepcopy(self.state_dict())
        
            tok_valid = time.time()
            if (early_stop_track > early_stop_patience) and val_check:
                print("Early stopped. AUC didn't improve for {} epochs.".format(early_stop_patience))
                self.load_state_dict(best_parameters)
                break
            print("Train time: {:.5f}s. Valid time: {:.5f}s".format((tok_train-tik), (tok_valid-tik)))
            print("Train Auc: {:.5f}. Train loss:{:.5f}. Valid auc: {:.5f}".format(current_train_auc, current_train_loss, current_val_auc))
        self.load_state_dict(best_parameters)

    def evaluate(self, dataloader):
        self.eval()
        for i, input in enumerate(dataloader):
            if self.avg_mode in ['LSTM', 'GRU']:
                batch, labels = input['input'][:,:,self.start:self.end], input['label']
            else:
                batch, labels = input['input'][:,self.start:self.end], input['label']
            output = self(batch)

            if i == 0:
                pre = output.detach().clone().numpy()
                target = labels.detach().clone().numpy()
                seq_len = input['seqlen'].detach().clone().numpy()
            else:
                print(i)
                print(pre.shape)
                print(output.shape)
                print(seq_len.shape)
                print(input['seqlen'].shape)
                print(target.shape)
                print(labels.shape)
                pre = np.vstack((pre, output.detach().clone().numpy()))
                target = np.hstack((target, labels.detach().clone().numpy()))
                seq_len = np.hstack((seq_len, input['seqlen'].detach().clone().numpy()))
            
        target = np.eye(2,dtype=int)[target]

        test_auc = roc_auc_score(target, pre)
        print('Test_auc', test_auc)
        re = print_seq_len(target, pre, seq_len)

        return test_auc, re
    
    def evaluate2(self, dataloader):
        self.eval()
        for i, input in enumerate(dataloader):
            if self.avg_mode in ['LSTM', 'GRU']:
                batch, labels = input['input'][:,:,self.start:self.end], input['label']
            else:
                batch, labels = input['input'][:,self.start:self.end], input['label']
            output = self(batch)
            if i == 0:
                pre = output.detach().clone().numpy()
                target = labels.detach().clone().numpy()
                seq_len = input['seqlen'].detach().clone().numpy()
                user_ids = input['user_id'].detach().clone().numpy()
            else:
                pre = np.vstack((pre, output.detach().clone().numpy()))
                target = np.hstack((target, labels.detach().clone().numpy()))
                seq_len = np.hstack((seq_len, input['seqlen'].detach().clone().numpy()))
                user_ids = np.hstack((user_ids, input['user_id'].detach().clone().numpy()))
            
        target = np.eye(2,dtype=int)[target]
        
        fpr, tpr, _ = roc_curve(np.argmax(target, 1), pre[:,1])
        test_auc = auc(fpr, tpr)
        acc = np.max(1-(fpr+(1-tpr))/2)
        low = 'None'
        return test_auc, acc, low, fpr, tpr, target, pre
    
def id2c(target, pre, user_ids):
    user_ids = list(user_ids)
    target = np.argmax(target, 1)
    pre = np.argmax(pre, 1)
    id_to_classify = {}
    for i in range(len(target)):
        if pre[i] == target[i]:
            id_to_classify[user_ids[i]] = 1
        else:
            id_to_classify[user_ids[i]] = 0
    return id_to_classify

def print_seq_len(target, pre, seq_len):
    target = np.argmax(target, 1)
    pre = np.argmax(pre, 1)
    seq_lens = [[],[],[],[]]
    re = []
    for i in range(len(pre)):
        if pre[i] == 0 and target[i] == 0:
            seq_lens[0].append(seq_len[i])
        elif pre[i] == 0 and target[i] == 1:
            seq_lens[1].append(seq_len[i])
        elif pre[i] == 1 and target[i] == 0:
            seq_lens[2].append(seq_len[i])
        else:
            seq_lens[3].append(seq_len[i])
    for j in range(len(seq_lens)):
        seq_lens[j] = np.array(seq_lens[j])
        re.append((np.mean(seq_lens[j]), len(seq_lens[j])))

    return re


