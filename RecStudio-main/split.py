# !/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from utils import *

'''
    Code for split dataset, including users and sequences/samples.
'''


def get_user2samples(
    dataset,
):
    '''
    Get a dict which map any user id to a complete sample indexes list.
    '''
    n_samples = len(dataset.data_index)
    sample2uid = np.zeros(n_samples)
    data_loader = dataset.loader(batch_size=1024, num_workers=0, shuffle=False, drop_last=False)
    for _, data in enumerate(data_loader):
        index = data['index'].detach().clone().numpy()
        sample2uid[index] = data['user_id'].detach().clone().numpy()
    user_id_list = np.sort(np.unique(sample2uid))
    
    uid2sample = dict()
    for sample_index in tqdm(range(n_samples)):
        uid = sample2uid[sample_index]
        if uid in uid2sample.keys():
            uid2sample[uid].append(sample_index)
        else:
            uid2sample[uid] = [sample_index]
    return user_id_list, uid2sample, sample2uid

def get_user_feature(
    dataset,
    dataset_name,
    user_id_list,
    aux_user_split_mode,
):
    if aux_user_split_mode in ["long", "short"]:
        feature_name = "num_inters"
    elif aux_user_split_mode in ["age", "old", "young"]:
        feature_name = "age"
    else:
        feature_name = None
    if feature_name is not None:
        utoken2id = dataset.field2token2idx[dataset.fuid]
        uid2token = gen_id2token(utoken2id)

        user_features = read_pkl_data(f"../datasets/{dataset_name}/user_features.pkl")
        uid2feature = []
        for user_id in user_id_list:
            user_token = uid2token[user_id]
            value = user_features[feature_name][user_token]
            if feature_name == "age":
                value = int(value)
            uid2feature.append(value)
        return uid2feature
    else:
        return None

def get_user_split(
    user_id_list,
    aux_user_split_mode,
    num_aux_user,
    uid2feature,
):
    '''
    Get training_users, audit_users and auxiliary_users.
    '''
    # Orginal dataset is divided evenly into training, audit and auxiliary users
    n_split_users  = int(len(user_id_list) / 3)
    if uid2feature is not None:
        uid2feature = np.array(uid2feature)
    if aux_user_split_mode == "random":
        np.random.shuffle(user_id_list)
        training_users = user_id_list[:1*n_split_users]
        audit_users    = user_id_list[1*n_split_users:2*n_split_users]
        auxiliary_users= user_id_list[2*n_split_users:]
    elif aux_user_split_mode == "age":
        # age_user_id = user_id_list[uid2feature>=18]
        age_user_id = user_id_list[uid2feature>=30]
        max_num_aux_user = min(n_split_users, len(age_user_id))
        auxiliary_users = np.random.choice(age_user_id, size=max_num_aux_user, replace=False)

        remain_users = np.setdiff1d(user_id_list, auxiliary_users)
        audit_users = np.random.choice(remain_users, size=n_split_users, replace=False)
        training_users = np.setdiff1d(remain_users, audit_users)
    elif aux_user_split_mode == "long" or aux_user_split_mode == "old":
        auxiliary_users = user_id_list[np.argsort(-uid2feature)[:n_split_users]]
        remain_users = np.setdiff1d(user_id_list, auxiliary_users)
        audit_users = np.random.choice(remain_users, size=n_split_users, replace=False)
        training_users = np.setdiff1d(remain_users, audit_users)
    elif aux_user_split_mode == "short" or aux_user_split_mode == "young":
        auxiliary_users = user_id_list[np.argsort(uid2feature)[:n_split_users]]
        remain_users = np.setdiff1d(user_id_list, auxiliary_users)
        audit_users = np.random.choice(remain_users, size=n_split_users, replace=False)
        training_users = np.setdiff1d(remain_users, audit_users)
    else:
        raise ValueError

    if num_aux_user >= 0:
        auxiliary_users = auxiliary_users[:num_aux_user]
    if num_audit_user >= 0:
        audit_users = audit_users[:num_audit_user]
    print_logger.info(f"Number of training users: {len(training_users)}; Number of audit users: {len(audit_users)}; Number of auxiliary users: {len(auxiliary_users)}")
    return training_users, audit_users, auxiliary_users

def get_retain_samples(
    sample_indexes,
    retain_rate,
    retain_mode,
):
    '''
    For the given retain rate, return the recent or outdated samples/sequence of a sample list.
    '''
    assert retain_mode in ["outdated", "recent"], "Retain mode should be outdated or recent."
    np.sort(sample_indexes)
    n_retain_sample = int(retain_rate * len(sample_indexes))
    if retain_mode == "outdated":
        return list(sample_indexes[:n_retain_sample])
    else:
        return list(sample_indexes[-n_retain_sample:])

def get_target_training_indexes(
    uid2sample,
    training_users,
    pos_audit_users,
    neg_audit_users,
    forget_samples_rate,
):
    '''
    Get target model's training indexes. Each index relates to a sample/sequence.
    '''
    target_training_indexes = []
    for uid in training_users+pos_audit_users:
        target_training_indexes += uid2sample[uid]

    if forget_samples_rate >= 1:
        pass
    else:
        for uid in neg_audit_users:
            target_training_indexes += get_retain_samples(
                sample_indexes=uid2sample[uid], 
                retain_rate=1-forget_samples_rate, 
                retain_mode="recent",
            )
    target_training_indexes = np.sort(np.unique(np.array(target_training_indexes)))
    return target_training_indexes

def get_shadow_uid2samples(
    uid2sample,
    pos_audit_users,
    neg_audit_users,
    auxiliary_users,
    forget_samples_rate,
    audit_retain_rate,
    aux_retain_rate,
    retain_mode,
):
    '''
    Shadow user = Audit user + Auxiliary user. Shadow user is used to train shadow/audit models. 
    Get a dict which map shadow user id to a sample indexes list.
    '''
    shadow_uid2samples = dict()
    for uid in auxiliary_users:
        shadow_uid2samples[uid] = get_retain_samples(
            sample_indexes=uid2sample[uid], 
            retain_rate=aux_retain_rate, 
            retain_mode=retain_mode,
        )

    if forget_samples_rate < 1:
        print_logger.info("Note: Argument audit_retain_rate is meaningless when forget_samples_rate < 1.")
        for uid in pos_audit_users:
            shadow_uid2samples[uid] = get_retain_samples(
                sample_indexes=uid2sample[uid], 
                retain_rate=1, 
                retain_mode=retain_mode,
            )
        for uid in neg_audit_users:
            target_retain_samples = get_retain_samples(
                sample_indexes=uid2sample[uid], 
                retain_rate=1-forget_samples_rate, 
                retain_mode="recent",
            )
            target_forget_samples = np.setdiff1d(uid2sample[uid], target_retain_samples)
            np.sort(target_forget_samples)
            shadow_uid2samples[uid] = list(target_forget_samples)
    else:
        for uid in pos_audit_users + neg_audit_users:
            shadow_uid2samples[uid] = get_retain_samples(
                sample_indexes=uid2sample[uid], 
                retain_rate=audit_retain_rate, 
                retain_mode=retain_mode,
            )
    shadow_sample_indexes = []
    for uid in shadow_uid2samples.keys():            
        shadow_sample_indexes += list(shadow_uid2samples[uid])
    shadow_sample_indexes = np.sort(np.unique(np.array(shadow_sample_indexes)))
    return shadow_uid2samples, shadow_sample_indexes

'''
Remove all sessions/sequences from the auxiliary data that are "similar" to the focal user's data.
'''
def get_session_inter_string(dataset, index):
    data = dataset[[index]]
    user_id = data['user_id'].numpy()[0]
    item_id = data['item_id'].numpy()[0]
    in_item_id = data['in_item_id'].numpy()[0]

    session_inter_string = ""
    for iid in in_item_id:
        session_inter_string = session_inter_string + str(iid) + "_"
    session_inter_string = session_inter_string + str(item_id) + "_"
    return user_id, session_inter_string

def get_audit_session_inters(dataset, audit_users):
    audit_session_inters = set()
    for i in tqdm(range(len(dataset))):
        user_id, session_inter_string = get_session_inter_string(dataset, index=i)
        if user_id in audit_users:
            audit_session_inters.add(session_inter_string)
    return audit_session_inters

def filt_shadow_uid2samples(dataset, shadow_uid2samples, auxiliary_users, audit_users):
    audit_session_inters = get_audit_session_inters(dataset, audit_users)
    num_filt = 0
    filt_indexes = set()
    for user in tqdm(auxiliary_users):
        indexes = shadow_uid2samples[user]
        unfilt_indexes = []
        for index in indexes:
            _, session_inter_string = get_session_inter_string(dataset, index)
            if not session_inter_string in audit_session_inters:
                unfilt_indexes.append(index)
            else:
                filt_indexes.add(session_inter_string)
        if len(indexes) != len(unfilt_indexes):
            shadow_uid2samples[user] = unfilt_indexes
            num_filt += len(indexes)-len(unfilt_indexes)
            print_logger.info(f"user: {user}, number of filt indexes: {len(indexes)-len(unfilt_indexes)}")
    print_logger.info("End of removing similar sequences")
    print_logger.info(f"number of filted sequences/sessions: {num_filt}")
    print_logger.info(f"number of filted unique sequences/sessions: {len(filt_indexes)}")
    return shadow_uid2samples


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("RecAudit")
    argparser.add_argument("--dataset", type=str, default="ml-100k", help="Dataset name")
    argparser.add_argument("--split_cf", type=str, default="split", help="Config file for spliting dataset") 
    torch.set_num_threads(5)

    global num_audit_user
    # Init args
    args = argparser.parse_args()
    with open(f"configs/split/{args.split_cf}.yaml", "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    retain_mode = configs["retain_mode"] if "retain_mode" in configs.keys() else "recent"
    dataset_name = args.dataset
    max_seq_len = configs["max_seq_len"]
    split_seed = configs["split_seed"]

    aux_user_split_mode = configs["aux_user_split_mode"]
    num_aux_user = configs["num_aux_user"] if "num_aux_user" in configs.keys() else -1
    num_audit_user = configs["num_audit_user"] if "num_audit_user" in configs.keys() else -1
    aux_retain_rate = configs["aux_retain_rate"]

    neg_audit_rate = configs["neg_audit_rate"]
    forget_samples_rate = configs["forget_samples_rate"]
    audit_retain_rate = configs["audit_retain_rate"]

    # Init log file
    log_path = f"./audit_log/dataset_split/{dataset_name}/{args.split_cf}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=f"{log_path}/split.log", filemode='w')
    logging.info(vars(args))
    
    # Init dataset
    datasets  = init_datasets(dataset_name=dataset_name, max_seq_len=max_seq_len)
    user_id_list, uid2sample, sample2uid = get_user2samples(datasets[0])

    # User split
    np.random.seed(split_seed)
    uid2feature = get_user_feature(
        dataset=datasets[0],
        dataset_name=dataset_name,
        user_id_list=user_id_list,
        aux_user_split_mode=aux_user_split_mode,
    )

    training_users, audit_users, auxiliary_users = get_user_split(
        user_id_list,
        aux_user_split_mode,
        num_aux_user,
        uid2feature,
    )

    # Audit_user is divided evenly into positive and negative users
    shadow_users   = np.append(audit_users, auxiliary_users)
    np.random.shuffle(audit_users)
    n_pos_audit_user = int(len(audit_users) * (1-neg_audit_rate))
    pos_audit_users = audit_users[:n_pos_audit_user]
    neg_audit_users = audit_users[n_pos_audit_user:]

    training_users = list(np.sort(training_users))
    audit_users = list(np.sort(audit_users))
    auxiliary_users = list(np.sort(auxiliary_users))
    shadow_users = list(np.sort(shadow_users))
    pos_audit_users = list(np.sort(pos_audit_users))
    neg_audit_users = list(np.sort(neg_audit_users))


    # Get sample indexes split information
    target_training_indexes = get_target_training_indexes(
        uid2sample,
        training_users,
        pos_audit_users,
        neg_audit_users,
        forget_samples_rate,
    )

    shadow_uid2samples, shadow_sample_indexes = get_shadow_uid2samples(
        uid2sample,
        pos_audit_users,
        neg_audit_users,
        auxiliary_users,
        forget_samples_rate,
        audit_retain_rate,
        aux_retain_rate,
        retain_mode,
    )

    # Remove all sessions/sequences from the auxiliary data that are "similar" to the focal user's data.
    if "mixed_audit_aux" in configs.keys() and configs["mixed_audit_aux"] > 0:
        shadow_uid2samples = filt_shadow_uid2samples(datasets[0], shadow_uid2samples, auxiliary_users, audit_users)

    # Save split information
    split_info = {}
    split_info["training_user"] = training_users
    split_info["audit_users"]   = audit_users
    split_info["shadow_users"]  = shadow_users
    split_info["auxiliary_users"] = auxiliary_users
    split_info["pos_audit_users"] = pos_audit_users
    split_info["neg_audit_users"] = neg_audit_users
    split_info["target_training_indexes"] = target_training_indexes
    split_info["shadow_sample_indexes"] = shadow_sample_indexes
    split_info["shadow_uid2samples"] = shadow_uid2samples

    save_dirs = f"./metadata/split/{dataset_name}/{args.split_cf}"
    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
        save_pkl_data(save_dir=f"{save_dirs}/split_info.pkl", save_value=split_info)
    else:
        raise UserWarning(f"Split information already exists! Path: ./metadata/split/{dataset_name}/{args.split_cf}")

    print_logger.info("End of the code.")
