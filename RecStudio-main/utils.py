import pickle
import torch
import numpy as np
import recstudio.model.loss_func as lfc

from tqdm import tqdm
from recstudio.data.dataset import SeqDataset
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color

def gen_id2token(token2idx):
    idx2token = {}
    for key in token2idx.keys():
        value = token2idx[key]
        idx2token[value] = key
    return idx2token

def save_pkl_data(save_dir, save_value):
    f_save = open(save_dir, 'wb')
    pickle.dump(save_value, f_save)
    f_save.close()

def read_pkl_data(save_dir):
    f_read = open(save_dir, 'rb')
    save_value = pickle.load(f_read)
    f_read.close()
    return save_value

def read_split_information(dataset_name, split_config_file_name):
    save_dirs = f"./metadata/split/{dataset_name}/{split_config_file_name}"
    split_info = read_pkl_data(f"{save_dirs}/split_info.pkl")
    return split_info

def init_datasets(dataset_name, max_seq_len=None):
    if max_seq_len is not None:
        datasets = SeqDataset(name=dataset_name, config={'max_seq_len': max_seq_len}).build()
    else:
        datasets = SeqDataset(name=dataset_name).build()
    print_logger.info(f"{datasets[0]}")
    return datasets

def init_model(model_name, train_configs, _best_ckpt_path=None):
    model_class, model_conf = get_model(model_name)
    if "embed_dim" in train_configs.keys():
        model_conf["embed_dim"] = train_configs["embed_dim"]
    model = model_class(model_conf)
    if "learning_rate" in train_configs.keys():
        model.config["learning_rate"] = train_configs["learning_rate"]
    if "learner" in train_configs.keys():
        model.config["learner"] = train_configs["learner"]
    if "weight_decay" in train_configs.keys():
        model.config["weight_decay"] = train_configs["weight_decay"]
    if "early_stop_patience" in train_configs.keys():
        model.config['early_stop_patience'] = train_configs["early_stop_patience"]
    if "gpu" in train_configs.keys():
        model.config['gpu'] = [train_configs["gpu"]]
    if "train_seed" in train_configs.keys():
        model.config['seed'] = train_configs["train_seed"]
    if "cutoff" in train_configs.keys():
        model.config['cutoff'] = train_configs["cutoff"]
    if _best_ckpt_path is not None:
        model._best_ckpt_path = _best_ckpt_path
    print_logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    return model

def init_model2(model, train_configs, _best_ckpt_path=None):
    if "learning_rate" in train_configs.keys():
        model.config["learning_rate"] = train_configs["learning_rate"]
    if "early_stop_patience" in train_configs.keys():
        model.config['early_stop_patience'] = train_configs["early_stop_patience"]
    if "gpu" in train_configs.keys():
        model.config['gpu'] = [train_configs["gpu"]]
    if "train_seed" in train_configs.keys():
        model.config['seed'] = train_configs["train_seed"]
    if "cutoff" in train_configs.keys():
        model.config['cutoff'] = train_configs["cutoff"]
    if _best_ckpt_path is not None:
        model._best_ckpt_path = _best_ckpt_path
    return model

def train_rec(model, train_set, val_set, train_configs):
    epochs, val_check = train_configs["epochs"], train_configs["val_check"]
    if val_check:
        model.fit(train_set, val_set, run_mode="light", epochs=epochs)
        if "notloadbest" in train_configs.keys() and train_configs["notloadbest"] > 0:
            pass
        else:
            best_ckpt_path = f"{model.config['save_path']}/{model._best_ckpt_path}"
            model.load_checkpoint(best_ckpt_path)
    else:
        model.fit(train_set, run_mode='light', epochs=epochs)

def evaluate_rec(model, dataset):
    dataset.gen_idx2idx(model.fiid2token2idx)
    output = model.evaluate(dataset)
    del dataset.id2id
    return output

def active_by_sample(
    dataset,
    active_indexes,
):
    if hasattr(dataset, 'active_index'):
        del dataset.active_index
    active_indexes = np.sort(np.unique(active_indexes))
    dataset.active_index = active_indexes
    return active_indexes

def active_by_user(
    dataset,
    active_users,
    active_uid2sample=None,
):
    if hasattr(dataset, 'active_index'):
        del dataset.active_index
    active_indexes = []

    if active_uid2sample is None:
        n_samples = len(dataset.data_index)
        sample2uid = np.zeros(n_samples)
        data_loader = dataset.loader(batch_size=1024, num_workers=0, shuffle=False, drop_last=False)
        for _, data in enumerate(data_loader):
            index = data["index"].detach().clone().numpy()
            sample2uid[index] = data["user_id"].detach().clone().numpy()
        sample2uid.astype(int)
        for idx in range(n_samples):
            if sample2uid[idx] in active_users:
                active_indexes.append(idx)
    else:
        for uid in active_users:
            active_indexes += active_uid2sample[uid]

    active_indexes = np.sort(np.unique(np.array(active_indexes)))
    dataset.active_index = active_indexes
    return active_indexes

def get_unforgotten_score(
    model,
    dataset,
    audit_user,
    audit_uid2sample,
    rerank=False,
):
    if len(audit_user) == 0:
        return {}, {}

    # Set fiid2token2idx of dataset
    fiid = dataset.fiid
    oral_fiid2token2idx = dataset.field2token2idx[dataset.fiid]
    dataset.gen_idx2idx(model.fiid2token2idx)

    # Set model's sampler and loss function
    model.eval()
    sampler = model.sampler
    model.sampler = None
    loss_fn = model.loss_fn
    model.loss_fn = lfc.BinaryCrossEntropyLoss()

    # Set active indexes
    active_by_user(dataset, active_users=audit_user, active_uid2sample=audit_uid2sample)
    data_loader = dataset.eval_loader(batch_size=512, num_workers=model.config['num_workers'])

    # Get sample's unforgotten score
    unforgotten_score = np.zeros((len(dataset.data_index), 7)).astype(np.float16)

    # Extract user's unforgotten score
    baseline_unforgotten_score = dict()
    if not rerank:
        recaudit_unforgotten_score = dict()
        unforgotten_score = _get_unforgotten_score(model, data_loader, fiid, unforgotten_score, rerank)
        for uid in audit_user:
            sample_indexes = np.array(audit_uid2sample[uid])
            recaudit_unforgotten_score[uid] = np.mean(unforgotten_score[:,5][sample_indexes],0)
            baseline_unforgotten_score[uid] = np.mean(unforgotten_score[:,6][sample_indexes],0)
    else:
        recaudit_unforgotten_score = init_rerank_dict(num_sequence=None)
        rerank_unforgotten_scores  = _get_unforgotten_score(model, data_loader, fiid, unforgotten_score, rerank)
        for uid in audit_user:
            sample_indexes = np.array(audit_uid2sample[uid])
            baseline_unforgotten_score[uid] = np.mean(rerank_unforgotten_scores["baseline"][sample_indexes],0)
            
            for refine_mode in ["none", "top1", "threshold"]:
                for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
                    if rerank_mode not in ["shuffle", "original"]:
                        for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                            recaudit_unforgotten_score[refine_mode][rerank_mode][rerank_rate][uid] = np.mean(rerank_unforgotten_scores[refine_mode][rerank_mode][rerank_rate][sample_indexes],0)
                    else:
                        recaudit_unforgotten_score[refine_mode][rerank_mode][uid] = np.mean(rerank_unforgotten_scores[refine_mode][rerank_mode][sample_indexes],0)

    # Recover dataset's fiid2token2idx and model's sampler and loss function
    dataset.gen_idx2idx(oral_fiid2token2idx)
    model.sampler = sampler
    model.loss_fn = loss_fn
    return recaudit_unforgotten_score, baseline_unforgotten_score

def _get_unforgotten_score(
    model, 
    data_loader, 
    fiid,
    unforgotten_score,
    rerank=False,
):
    if rerank:
        rerank_unforgotten_scores = init_rerank_dict(num_sequence=len(unforgotten_score))

    softmax = torch.nn.Softmax(dim=1)
    # for _, data in enumerate(data_loader):
    for data in tqdm(data_loader):
        # Obtain all item score
        data  = model._to_device(data, model.device)
        user_embeddings = model.query_encoder(data)
        top_score, topk_items     = model.topk(user_embeddings, k=model.config['cutoff'], user_h=data['user_hist'])
        pos_score, all_item_score = model(data, True)

        # Obtain various unforgotten scores
        pos_score  = pos_score.cpu().detach().clone().numpy().reshape(-1, 1).astype(np.float16)
        mean_score = torch.mean(all_item_score,axis=1).cpu().detach().clone().numpy().reshape(-1, 1).astype(np.float16)
        top1_score = torch.max(top_score,dim=1).values.cpu().detach().clone().numpy().reshape(-1, 1).astype(np.float16)
        topk_score = torch.min(top_score,dim=1).values.cpu().detach().clone().numpy().reshape(-1, 1).astype(np.float16)
        mean_topk_score = torch.mean(top_score,axis=1).cpu().detach().clone().numpy().reshape(-1, 1).astype(np.float16)
        softmax_score = softmax(all_item_score)
        assert softmax_score.dim() == 2, 'error'
        softmax_unforgotten_score = softmax_score.gather(1, data[fiid].reshape(-1, 1)-1).cpu().detach().clone().numpy().astype(np.float16)

        index = data["index"].cpu().detach().clone().numpy()
        unforgotten_score[index] += np.hstack((pos_score, mean_score, top1_score, topk_score, \
        mean_topk_score, pos_score-mean_topk_score, softmax_unforgotten_score))

        if rerank:
            rerank_unforgotten_scores["baseline"][index] += softmax_unforgotten_score.squeeze()
            rerank_unforgotten_scores = rerank_unforgotten_score(
                model,
                rerank_unforgotten_scores,
                index,
                topk_items,
                all_item_score,
                pos_score,
            )

    if rerank:
        return rerank_unforgotten_scores
    else:         
        return unforgotten_score

'''
Post-processing
'''
import copy
def get_novelty_item(item_array, item2class):
    class2item = item2class.T
    num_classes, num_items = class2item.shape
    item_class_mask = np.sum(item2class[item_array],0).astype(bool)
    novelty_class = np.arange(num_classes)[~item_class_mask]
    novelty_items_mask = np.sum(class2item[novelty_class],0).astype(bool)
    novelty_items = np.arange(num_items)[novelty_items_mask]
    return novelty_items

def rerank_topk_item(
    model,
    topk_items, 
    rerank_mode, 
    rerank_rate=0.1,
):
    popular_item_list = model.popular_item_list
    new_item_list = model.new_item_list
    item2class = model.item2class
    rerank_topk_items = copy.deepcopy(topk_items.cpu().detach().clone().numpy()) - 1
    num_samples, num_recommendation = topk_items.shape
    rerank_num = int(num_recommendation * rerank_rate)
    if rerank_mode == "shuffle":
        for i in range(num_samples):
            np.random.shuffle(rerank_topk_items[i,:])
    elif rerank_mode == "popular":
        for i in range(num_samples):
            rerank_topk_items[i,-rerank_num:] = np.random.choice(popular_item_list, size=rerank_num, replace=False)
    elif rerank_mode == "timeliness":
        for i in range(num_samples):
            rerank_topk_items[i,-rerank_num:] = np.random.choice(new_item_list, size=rerank_num, replace=False)
    elif rerank_mode == "novelty":
        for i in range(num_samples):
            novelty_items = get_novelty_item(rerank_topk_items[i,:-rerank_num], item2class)
            rerank_topk_items[i,-rerank_num:] = np.random.choice(novelty_items, size=rerank_num, replace=False)
    elif rerank_mode == "original":
        pass
    else:
        raise ValueError(f"rerank mode  {rerank_mode} is not supported")
    
    # if rerank_mode in ["popular", "timeliness", "novelty"]:
    #     topk_items_copy = copy.deepcopy(topk_items.cpu().detach().clone().numpy()) - 1
    #     rerank_topk_items = np.hstack((topk_items_copy, rerank_topk_items[:,-rerank_num:]))

    return rerank_topk_items

def _rerank_unforgotten_score(pos_score, rerank_topk_items, all_item_score):
    top_score = np.zeros_like(rerank_topk_items).astype(np.float16)
    for i in range(len(top_score)):
        top_score[i] = all_item_score[i][rerank_topk_items[i]]

    mean_score1 = np.mean(top_score,axis=1).astype(np.float16)# no threshold
    mean_score2 = np.zeros_like(pos_score).astype(np.float16) # threshold = top1 * rate (e.g. 80%)
    mean_score3 = np.zeros_like(pos_score).astype(np.float16) # threshold = fix value

    for i in range(len(mean_score2)):
        score = top_score[i]
        #mean_score2[i] = np.mean(score[score>=0.95*np.max(score)]) #ml-100k: 0.5, ml-1m: 0.95
        #mean_score3[i] = np.mean(score[score>=6.82]) #ml-100k:5.35; ml-1m:6.82
        mean_score2[i] = np.max(score)
        mean_score3[i] = np.mean(score[score>=0.9*np.max(score)])
    return pos_score-mean_score1, pos_score-mean_score2, pos_score-mean_score3

def rerank_unforgotten_score(
    model,
    rerank_unforgotten_scores,
    index,
    topk_items,
    all_item_score,
    pos_score,
):
    all_item_score = all_item_score.cpu().detach().clone().numpy()
    pos_score = pos_score.squeeze()
    for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
        if rerank_mode not in ["shuffle", "original"]:
            for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                rerank_topk_items = rerank_topk_item(
                    model,
                    topk_items, 
                    rerank_mode, 
                    rerank_rate,
                )
                score1, score2, score3 = _rerank_unforgotten_score(
                    pos_score, 
                    rerank_topk_items, 
                    all_item_score,
                )
                rerank_unforgotten_scores["none"][rerank_mode][rerank_rate][index] += score1
                rerank_unforgotten_scores["top1"][rerank_mode][rerank_rate][index] += score2
                rerank_unforgotten_scores["threshold"][rerank_mode][rerank_rate][index] += score3
        else:
            rerank_topk_items = rerank_topk_item(
                model,
                topk_items, 
                rerank_mode,
            )
            score1, score2, score3 = _rerank_unforgotten_score(
                pos_score, 
                rerank_topk_items, 
                all_item_score,
            )
            rerank_unforgotten_scores["none"][rerank_mode][index] += score1
            rerank_unforgotten_scores["top1"][rerank_mode][index] += score2
            rerank_unforgotten_scores["threshold"][rerank_mode][index] += score3
    return rerank_unforgotten_scores

def init_rerank_dict(num_sequence=None):
    if num_sequence is not None:
        rerank_unforgotten_scores = dict()
        rerank_unforgotten_scores["baseline"] = np.zeros(num_sequence)
        for refine_mode in ["none", "top1", "threshold"]:
            rerank_unforgotten_scores[refine_mode] = dict()
            for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
                if rerank_mode not in ["shuffle", "original"]:
                    rerank_unforgotten_scores[refine_mode][rerank_mode] = dict()
                    for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        rerank_unforgotten_scores[refine_mode][rerank_mode][rerank_rate] = np.zeros(num_sequence)
                else:
                    rerank_unforgotten_scores[refine_mode][rerank_mode] = np.zeros(num_sequence)
        return rerank_unforgotten_scores
    else:
        recaudit_unforgotten_score = dict()
        for refine_mode in ["none", "top1", "threshold"]:
            recaudit_unforgotten_score[refine_mode] = dict()
            for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
                if rerank_mode not in ["shuffle", "original"]:
                    recaudit_unforgotten_score[refine_mode][rerank_mode] = dict()
                    for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        recaudit_unforgotten_score[refine_mode][rerank_mode][rerank_rate] = dict()
                else:
                    recaudit_unforgotten_score[refine_mode][rerank_mode] = dict()
        return recaudit_unforgotten_score