#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import torch
import argparse

from utils import *
from memia_utils import *
from MIAModel import *

def extract_features_memia(
    datasets,
    target_model,
    shadow_model,
):
    if len(pos_audit_users) == 0:
        dscore_target_member = {}
    else:
        active_by_user(dataset=datasets[0], active_users=pos_audit_users, active_uid2sample=active_uid2sample)
        dscore_target_member = vscore(target_model, datasets[0])

    if len(neg_audit_users) == 0:
        dscore_target_nonmember = {}
    else:
        active_by_user(dataset=datasets[0], active_users=neg_audit_users, active_uid2sample=active_uid2sample)
        dscore_target_nonmember = vscore(target_model, datasets[0])

    active_by_user(dataset=datasets[0], active_users=pos_shadow_users, active_uid2sample=active_uid2sample)
    dscore_shadow_member = vscore(shadow_model, datasets[0])
    active_by_user(dataset=datasets[0], active_users=neg_shadow_users, active_uid2sample=active_uid2sample)
    dscore_shadow_nonmember = vscore(shadow_model, datasets[0])
    print(f"End of MIA feature generation.")
    return dscore_target_member, dscore_target_nonmember, dscore_shadow_member, dscore_shadow_nonmember

def extract_features_recmia(
    datasets,
    target_model,
    shadow_model,
):
    shadow_embeddings = shadow_model.item_encoder.weight.cpu().detach().clone().numpy()
    if len(pos_audit_users) == 0:
        dscore_target_member = {}
    else:
        active_by_user(dataset=datasets[0], active_users=pos_audit_users, active_uid2sample=active_uid2sample)
        dscore_target_member = ccs_score(target_model, datasets[0], shadow_embeddings)

    if len(neg_audit_users) == 0:
        dscore_target_nonmember = {}
    else:
        active_by_user(dataset=datasets[0], active_users=neg_audit_users, active_uid2sample=active_uid2sample)
        dscore_target_nonmember = ccs_score(target_model, datasets[0], shadow_embeddings)

    active_by_user(dataset=datasets[0], active_users=pos_shadow_users, active_uid2sample=active_uid2sample)
    dscore_shadow_member = ccs_score(shadow_model, datasets[0], shadow_embeddings)
    active_by_user(dataset=datasets[0], active_users=neg_shadow_users, active_uid2sample=active_uid2sample)
    dscore_shadow_nonmember = ccs_score(shadow_model, datasets[0], shadow_embeddings)
    print(f"End of MIA feature generation.")
    return dscore_target_member, dscore_target_nonmember, dscore_shadow_member, dscore_shadow_nonmember


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Baseline")
    argparser.add_argument("--dataset", type=str, default='ml-100k', help='Datasets')
    argparser.add_argument("--target_model", type=str, default='BERT4Rec',help='The type of target model')
    argparser.add_argument("--shadow_model", type=str, default='BERT4Rec',help='The type of shadow model')
    argparser.add_argument("--model_seed", type=int, default=1, help='Seed for training RS')

    argparser.add_argument("--split_cf", type=str, default="split", help="Config file for spliting dataset") 
    argparser.add_argument("--train_cf", type=str, default="train", help="Config file for training RS")
    argparser.add_argument("--audit_cf", type=str, default="audit", help="Config file for auditing RS")
    argparser.add_argument("--audit_method", type=str, default="memia", help="Config file for auditing RS")
    argparser.add_argument("--gpu", type=int, default=0, help='The used gpu id')

    torch.set_num_threads(3)

    global pos_audit_users, neg_audit_users, pos_shadow_users, neg_shadow_users, active_uid2sample, train_config_file
    shadow_model_seed = 1

    # Init args
    args = argparser.parse_args()
    dataset_name = args.dataset
    target_model_name = args.target_model
    shadow_model_name = args.shadow_model
    model_seed = args.model_seed

    split_config_file = args.split_cf
    train_config_file = args.train_cf
    audit_config_file = args.audit_cf
    audit_method = args.audit_method

    split_info = read_split_information(dataset_name, split_config_file_name=args.split_cf)
    if "rerank" in train_config_file:
        shadow_split_info = read_pkl_data(f"./metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/rerank/{shadow_model_seed}/split_info.pkl")
        with open(f"configs/train/rerank.yaml", "rb") as f:
            train_configs = yaml.load(f, Loader=yaml.Loader)

    elif "unlearning" in train_config_file:
        shadow_split_info = read_pkl_data(f"./metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/train/{shadow_model_seed}/split_info.pkl")
        with open(f"configs/train/train.yaml", "rb") as f:
            train_configs = yaml.load(f, Loader=yaml.Loader)

    else:
        shadow_split_info = read_pkl_data(f"./metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/{train_config_file}/{shadow_model_seed}/split_info.pkl")
        with open(f"configs/train/{train_config_file}.yaml", "rb") as f:
            train_configs = yaml.load(f, Loader=yaml.Loader)

    with open(f"configs/split/{split_config_file}.yaml", "rb") as f:
        split_configs = yaml.load(f, Loader=yaml.Loader)
    with open(f"configs/audit/{audit_config_file}.yaml", "rb") as f:
        audit_configs0 = yaml.load(f, Loader=yaml.Loader)
    with open(f"configs/audit/{audit_method}.yaml", "rb") as f:
        audit_configs1 = yaml.load(f, Loader=yaml.Loader)
    audit_configs = {**audit_configs0, **audit_configs1}
    train_configs["gpu"] = args.gpu

    classifier_avg_mode = audit_configs["classifier_avg_mode"]
    mia_data_mode = audit_configs["mia_data_mode"]
    mia_batch_size = audit_configs["mia_batch_size"]
    mia_epochs = audit_configs["mia_epochs"]
    mia_fea_start = audit_configs["mia_fea_start"]
    if audit_method == "recmia":
        num_features = audit_configs["num_features"]
    else:
        num_features = train_configs["cutoff"] + audit_configs["extra_num_features"]

    mia_fea_end = num_features + mia_fea_start

    mia_lr = audit_configs["mia_lr"]
    mia_val_check = True if audit_configs["mia_val_check"]>0 else False
    mia_hidden_size = audit_configs["mia_hidden_size"]


    # Init dataset and member/nonmember user
    datasets = init_datasets(dataset_name=dataset_name, max_seq_len=split_configs["max_seq_len"])
    pos_audit_users = split_info["pos_audit_users"]
    neg_audit_users = split_info["neg_audit_users"]
    audit_users = pos_audit_users + neg_audit_users
    active_uid2sample = split_info["shadow_uid2samples"]

    pos_shadow_users = shadow_split_info["shadow_member_users"]
    neg_shadow_users = shadow_split_info["shadow_nonmember_users"]

    # Init target model
    target_model = init_model(target_model_name, train_configs, _best_ckpt_path=None)
    target_model.fit(datasets[1], datasets[2], run_mode="light", epochs=1)

    if "syn" in dataset_name:
        ori_dataset_name = dataset_name.split('_')[0]
        best_ckpt_path = f"./metadata/model_checkpoint/target_model/{ori_dataset_name}/{target_model_name}/{args.split_cf}/{args.train_cf}/model_{model_seed}"
    else:
        best_ckpt_path = f"./metadata/model_checkpoint/target_model/{dataset_name}/{target_model_name}/{args.split_cf}/{args.train_cf}/model_{model_seed}"
    

    if "rerank" in train_config_file:
        best_ckpt_path = f"./metadata/model_checkpoint/target_model/{dataset_name}/{target_model_name}/{args.split_cf}/rerank/model_{model_seed}"
        target_model.load_checkpoint(best_ckpt_path)
        target_model.rerank_mode = train_config_file.split('_')[2]
        target_model.rerank_rate = float(train_config_file.split('_')[3]) if len(train_config_file.split('_')) > 3 else 0.100

        item_freq = datasets[0].item_freq.detach().numpy()
        popular_item_list = np.argsort(-item_freq)[:100]

        item2year = np.load(f"../datasets/{dataset_name}/item2year.npy")
        new_item_list = np.argsort(-item2year)[:100]

        item2class = np.load(f"../datasets/{dataset_name}/item2class.npy")

        target_model.popular_item_list = popular_item_list
        target_model.new_item_list = new_item_list
        target_model.item2class = item2class
    else:
        target_model.load_checkpoint(best_ckpt_path)

    # Init shadow model
    shadow_model = init_model(shadow_model_name, train_configs, _best_ckpt_path=None)
    shadow_model.fit(datasets[1], datasets[2], run_mode="light", epochs=1)
    if "rerank" in train_config_file:
        best_ckpt_path = f"./metadata/model_checkpoint/shadow_model/{dataset_name}/{shadow_model_name}/{args.split_cf}/rerank/member_model_{shadow_model_seed}"
    elif "unlearning" in train_config_file:
        best_ckpt_path = f"./metadata/model_checkpoint/shadow_model/{dataset_name}/{shadow_model_name}/{args.split_cf}/train/member_model_{shadow_model_seed}"
    else:
        best_ckpt_path = f"./metadata/model_checkpoint/shadow_model/{dataset_name}/{shadow_model_name}/{args.split_cf}/{args.train_cf}/member_model_{shadow_model_seed}"
    shadow_model.load_checkpoint(best_ckpt_path)

    # Extract MIA feature
    if audit_method == "memia":
        dscore_target_member, dscore_target_nonmember, dscore_shadow_member, dscore_shadow_nonmember = extract_features_memia(
            datasets=datasets,
            target_model=target_model,
            shadow_model=shadow_model,
        )
    elif audit_method == "recmia":
        dscore_target_member, dscore_target_nonmember, dscore_shadow_member, dscore_shadow_nonmember = extract_features_recmia(
            datasets=datasets,
            target_model=target_model,
            shadow_model=shadow_model,
        )
    else:
        raise ValueError("Audit method {audit_method} is not support for this code!")


    del target_model
    del shadow_model
    torch.cuda.empty_cache()

    train_val_datasets = MIADataset(
        dscore_shadow_member, 
        dscore_shadow_nonmember, 
        mia_data_mode=mia_data_mode
    )
    train_datasets, val_datasets = train_val_datasets.build()
    train_loader = train_datasets.loader(
        batch_size=mia_batch_size, 
        num_workers=0, 
        shuffle=True, 
        drop_last=False
    )

    val_loader = val_datasets.loader(
        batch_size=mia_batch_size, 
        num_workers=0, 
        shuffle=False, 
        drop_last=False
    )
    mia_model = MIAModel(
        num_fea=mia_fea_end - mia_fea_start, 
        hiddens=mia_hidden_size, 
        avg_mode=classifier_avg_mode
    )
    mia_model.fit(
        train_loader, 
        val_loader, 
        epochs=mia_epochs, 
        lr=mia_lr, 
        val_check=mia_val_check, 
        start=mia_fea_start, 
        end=mia_fea_end, 
        optim='Adam'
    )

    # White + Classifier based
    test_datasets = MIADataset(
        dscore_target_member,
        dscore_target_nonmember,
        mia_data_mode=mia_data_mode
    )
    test_loader = test_datasets.loader(
        batch_size=mia_batch_size, 
        num_workers=0, 
        shuffle=False, 
        drop_last=False
    )

    auroc, acc, _, fpr_array, tpr_array, label, prediction  = mia_model.evaluate2(test_loader)
    fnr_array = 1-tpr_array
    fpr_at_fix_fnr = fpr_array[np.where(fnr_array<.1)[0][0]] if len(np.where(fnr_array<.1)[0]) > 0 else -1

    audit_results = {}

    audit_results["users"]  = audit_users
    audit_results["method"] = audit_method
    audit_results["auroc"]  = auroc
    audit_results["label"]  = label
    audit_results["prediction"] = prediction
    audit_results["fpr_at_fix_fnr"]  = fpr_at_fix_fnr
    audit_results["fpr_array"] = fpr_array
    audit_results["tpr_array"] = tpr_array

    map_method = {"memia": "Zhu et al.", "recmia": "Zhang et al."}
    result_save_dirs = f"./audit_results/audit/{dataset_name}/{target_model_name}/{shadow_model_name}/{split_config_file}/{train_config_file}/{audit_config_file}/{model_seed}"
    if not os.path.exists(result_save_dirs):
        os.makedirs(result_save_dirs)
    save_pkl_data(save_dir=f'''{result_save_dirs}/{map_method[audit_method]}''', save_value=audit_results)
    print(f"Method: {audit_results['method']}, AUROC: {auroc}, FPR@10%FNR: {fpr_at_fix_fnr}.")
        










        

        

 