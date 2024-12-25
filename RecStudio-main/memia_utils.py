import os
import shutil
import time
import copy
import pickle
import math
import torch
import numpy as np
import recstudio.model.loss_func as lfc

from recstudio.ann import sampler
from recstudio.data.dataset import SeqDataset
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color, parser_yaml
from typing import Sized, Dict, Optional, Iterator, Union

from utils import rerank_topk_item

# For ME-MIA
def _score(model, data_loader, fiid, fuid):
    softmax = torch.nn.Softmax(dim=1)
    dscore = {}
    for i, data in enumerate(data_loader):
        data = model._to_device(data, model.device)

        pos_score, all_item_score = model(data, True)
        user_embeddings = model.query_encoder(data)
        top_score, topk_items = model.topk(user_embeddings, k=model.config['cutoff'], user_h=data['user_hist'])

        if hasattr(model, "rerank_mode"):
            rerank_topk_items = rerank_topk_item(
                model,
                topk_items, 
                model.rerank_mode, 
                model.rerank_rate,
            )
            all_item_score_array = all_item_score.cpu().detach().clone().numpy()
            top_score = np.zeros_like(rerank_topk_items).astype(np.float16)
            for i in range(len(top_score)):
                top_score[i] = all_item_score_array[i][rerank_topk_items[i]]
            top_score = torch.tensor(top_score)
        
        softmax_score = softmax(all_item_score)
        mean_score = torch.mean(all_item_score,axis=1)
        std_score = torch.std(all_item_score,axis=1)
        top1_score = torch.max(top_score,dim=1).values
        topk_score = torch.min(top_score,dim=1).values
        mean_topk_score = torch.mean(top_score,axis=1)
        user_ids = data[fuid]
        end = data['end'].cpu().detach().clone().numpy()
        user_embeddings = user_embeddings.cpu().detach().clone().numpy()
        top_score = top_score.cpu().detach().clone().numpy()

        for i in range(len(user_ids)):
            user_id = user_ids[i].item()
            if user_id not in dscore.keys():
                dscore[user_id] = {'pos_score':[],'mean_score':[],'top1_score':[],'topk_score':[],\
                    'mean_topk_score':[], 'softmax_score':[], 'features':[], 'end':[]}
            dscore[user_id]['pos_score'].append(pos_score[i].item())
            dscore[user_id]['mean_score'].append(mean_score[i].item())
            dscore[user_id]['top1_score'].append(top1_score[i].item())
            dscore[user_id]['topk_score'].append(topk_score[i].item())
            dscore[user_id]['mean_topk_score'].append(mean_topk_score[i].item())
            dscore[user_id]['softmax_score'].append(softmax_score[i][data[fiid][i].item()-1].item())
            dscore[user_id]['end'].append(end[i])
            dscore[user_id]['features'].append(np.hstack((user_embeddings[i], pos_score[i].item(), mean_score[i].item(), \
                top_score[i], std_score[i].item())))
    return dscore

# # For ME-MIA
# def score(model, datasets):
#     fiid = datasets[0].fiid
#     fuid = datasets[0].fuid
#     oral_fiid2token2idx = datasets[0].field2token2idx[datasets[0].fiid]
#     fiid2token2idx = model.fiid2token2idx

#     datasets[0].gen_idx2idx(fiid2token2idx)
#     datasets[1].gen_idx2idx(fiid2token2idx)
#     datasets[2].gen_idx2idx(fiid2token2idx)

#     model.eval()

#     sampler = model.sampler
#     model.sampler = None
#     loss_fn = model.loss_fn
#     model.loss_fn = lfc.BinaryCrossEntropyLoss()

#     train_data_loader = datasets[0].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
#     val_data_loader = datasets[1].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
#     test_data_loader = datasets[2].eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])
    
#     train_d_score = _score(model, train_data_loader, fiid, fuid)
#     val_d_score = _score(model, val_data_loader, fiid, fuid)
#     test_d_score = _score(model, test_data_loader, fiid, fuid)

#     datasets[0].gen_idx2idx(oral_fiid2token2idx)
#     datasets[1].gen_idx2idx(oral_fiid2token2idx)
#     datasets[2].gen_idx2idx(oral_fiid2token2idx)

#     model.sampler = sampler
#     model.loss_fn = loss_fn
#     return train_d_score, val_d_score, test_d_score

def vscore(model, datasets):
    fiid = datasets.fiid
    fuid = datasets.fuid
    oral_fiid2token2idx = datasets.field2token2idx[datasets.fiid]
    fiid2token2idx = model.fiid2token2idx

    datasets.gen_idx2idx(fiid2token2idx)

    model.eval()
    sampler = model.sampler
    model.sampler = None
    loss_fn = model.loss_fn
    model.loss_fn = lfc.BinaryCrossEntropyLoss()

    train_data_loader = datasets.eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])

    train_d_score = _score(model, train_data_loader, fiid, fuid)

    datasets.gen_idx2idx(oral_fiid2token2idx)
    model.sampler = sampler
    model.loss_fn = loss_fn
    return train_d_score

# For baseline CCS
def ccs_embed(in_item_id, topk_id, embeddings, avg_mode='sort_sum', topk_shuffle=False):
    assert avg_mode in ['sort_sum', 'mean'], 'Avg mode error!'
    w = np.ones_like(topk_id)
    if avg_mode == 'sort_sum':
        for i in range(len(w)):
            w[i] = len(w) -i 
    else:
        pass
    w = w / np.sum(w)
    if topk_shuffle:
        np.random.shuffle(w)
    return np.dot(w,embeddings[topk_id]) - np.mean(embeddings[in_item_id],axis=0)

def _ccs_score(model, data_loader, fiid, fuid, embeddings):
    dscore = {}
    for i, data in enumerate(data_loader):
        data = model._to_device(data, model.device)
        user_embeddings = model.query_encoder(data)

        _, topk_id = model.topk(user_embeddings, k=model.config['cutoff'], user_h=data['user_hist'])

        if hasattr(model, "rerank_mode"):
            rerank_topk_items = rerank_topk_item(
                model,
                topk_id, 
                model.rerank_mode, 
                model.rerank_rate,
            )
            topk_id = torch.tensor(rerank_topk_items)

        user_ids = data[fuid]
        end = data['end'].cpu().detach().clone().numpy()
        topk_id = topk_id.cpu().detach().clone().numpy()
        in_item_id = data['in_'+ fiid].cpu().detach().clone().numpy()
        
        for i in range(len(user_ids)):
            user_id = user_ids[i].item()
            if user_id not in dscore.keys():
                dscore[user_id] = {'pos_score':[],'mean_score':[],'top1_score':[],'topk_score':[],\
                    'mean_topk_score':[], 'softmax_score':[], 'features':[], 'end':[]}
            dscore[user_id]['end'].append(end[i])
            dscore[user_id]['features'].append(ccs_embed(in_item_id[i], topk_id[i], embeddings))
    return dscore

def ccs_score(model, datasets, embeddings):
    fiid = datasets.fiid
    fuid = datasets.fuid

    oral_fiid2token2idx = datasets.field2token2idx[datasets.fiid]
    fiid2token2idx = model.fiid2token2idx
    datasets.gen_idx2idx(fiid2token2idx)

    model.eval()
    sampler = model.sampler
    model.sampler = None
    loss_fn = model.loss_fn
    model.loss_fn = lfc.BinaryCrossEntropyLoss()

    train_data_loader = datasets.eval_loader(batch_size=model.config['eval_batch_size'], num_workers=model.config['num_workers'])

    train_d_score = _ccs_score(model, train_data_loader, fiid, fuid, embeddings)

    datasets.gen_idx2idx(oral_fiid2token2idx)
    model.sampler = sampler
    model.loss_fn = loss_fn

    return train_d_score
