#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import logging
import argparse

from utils import *

'''
    Code for training shadow/audit model.
'''

def save_unforgotten_score(
    recaudit_mem,
    baseline_mem,
    recaudit_nonmem,
    baseline_nonmem,
    train_cf,
):
    score_dirs = f"./metadata/scores/shadow_model/{dataset_name}/{model_name}/{split_cf_file}/{train_cf}/{split_seed}"
    if not os.path.exists(score_dirs):
        os.makedirs(score_dirs)

    save_pkl_data(save_dir=f"{score_dirs}/recaudit_member_score.pkl", save_value=recaudit_mem)
    save_pkl_data(save_dir=f"{score_dirs}/baseline_member_score.pkl", save_value=baseline_mem)
    save_pkl_data(save_dir=f"{score_dirs}/recaudit_nonmember_score.pkl", save_value=recaudit_nonmem)
    save_pkl_data(save_dir=f"{score_dirs}/baseline_nonmember_score.pkl", save_value=baseline_nonmem)

def prepare_shadow_model(
    shadow_model,
    datasets,
    active_users,
    active_uid2sample,
    train_configs,
):
    if "fedtrain" in train_configs and train_configs["fedtrain"] > 0:
        for i in range(3):
            datasets[i].user_batch_size = train_configs["user_batch_size"]
            datasets[i].active_users = active_users
            datasets[i].active_uid2sample = active_uid2sample
    else:
        active_by_user(dataset=datasets[0], active_users=active_users, active_uid2sample=active_uid2sample)
        active_by_user(dataset=datasets[1], active_users=active_users)
        active_by_user(dataset=datasets[2], active_users=active_users)

    # Train shadow model
    train_rec(
        shadow_model, 
        train_set=datasets[0],
        val_set=datasets[1], 
        train_configs=train_configs,
    )

    # Evaluate shadow model
    print_logger.info('Shadow model recommendation performance:')
    evaluate_rec(shadow_model, datasets[2])
    return shadow_model

def prepare_unforgotten_score(
    shadow_mem_model,
    shadow_non_model,
    dataset,
    shadow_member_users,
    shadow_nonmember_users,
    audit_uid2sample,
):
    recaudit_mem0, recaudit_mem1, recaudit_nonmem0, recaudit_nonmem1, baseline_mem, baseline_nonmem = _prepare_unforgotten_score(
        shadow_mem_model,
        shadow_non_model,
        dataset,
        shadow_member_users,
        shadow_nonmember_users,
        audit_uid2sample,
        rerank=False,
    )
    recaudit_mem = {**recaudit_mem0, **recaudit_mem1}
    recaudit_nonmem = {**recaudit_nonmem0, **recaudit_nonmem1}
    return recaudit_mem, baseline_mem, recaudit_nonmem, baseline_nonmem

def _prepare_unforgotten_score(
    shadow_mem_model,
    shadow_non_model,
    dataset,
    shadow_member_users,
    shadow_nonmember_users,
    audit_uid2sample,
    rerank,
):
    recaudit_mem0, baseline_mem0 = get_unforgotten_score(
        model=shadow_mem_model,
        audit_user=shadow_member_users,
        dataset=dataset,
        audit_uid2sample=audit_uid2sample,
        rerank=rerank,
    )
    recaudit_mem1, baseline_mem1 = get_unforgotten_score(
        model=shadow_non_model,
        audit_user=shadow_nonmember_users,
        dataset=dataset,
        audit_uid2sample=audit_uid2sample,
        rerank=rerank,
    )
    recaudit_nonmem0, baseline_nonmem0 = get_unforgotten_score(
        model=shadow_mem_model,
        audit_user=shadow_nonmember_users,
        dataset=dataset,
        audit_uid2sample=audit_uid2sample,
        rerank=rerank,
    )
    recaudit_nonmem1, baseline_nonmem1 = get_unforgotten_score(
        model=shadow_non_model,
        audit_user=shadow_member_users,
        dataset=dataset,
        audit_uid2sample=audit_uid2sample,
        rerank=rerank,
    )
    baseline_mem = {**baseline_mem0, **baseline_mem1}
    baseline_nonmem = {**baseline_nonmem0, **baseline_nonmem1}
    return recaudit_mem0, recaudit_mem1, recaudit_nonmem0, recaudit_nonmem1, baseline_mem, baseline_nonmem


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("RecAudit")
    argparser.add_argument("--dataset", type=str, default='movielens', help='Datasets')
    argparser.add_argument("--model", type=str, default='BERT4Rec',help='The type of sequence model')
    argparser.add_argument("--split_seed", type=int, default=1, help='Seed used to split shadow users')
    argparser.add_argument("--gpu", type=int, default=0, help='The used gpu id')
    argparser.add_argument("--split_cf", type=str, default="split", help="Config file for spliting dataset") 
    argparser.add_argument("--train_cf", type=str, default="train", help="Config file for training RS") 
    #print(afk)
    torch.set_num_threads(3)
    global dataset_name, model_name, split_cf_file, split_seed

    # Init args
    args = argparser.parse_args()
    dataset_name = args.dataset
    model_name = args.model
    split_seed = args.split_seed
    gpu = args.gpu
    split_cf_file = args.split_cf
    torch.cuda.set_device(gpu)

    # Init log file
    log_path = f"./audit_log/train_shadow_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=f"{log_path}/train_{split_seed}.log", filemode='w')
    logging.info(vars(args))

    with open(f"configs/split/{args.split_cf}.yaml", "rb") as f:
        split_configs = yaml.load(f, Loader=yaml.Loader)
    split_info = read_split_information(dataset_name, split_config_file_name=args.split_cf)

    with open(f"configs/train/{args.train_cf}.yaml", "rb") as f:
        train_configs = yaml.load(f, Loader=yaml.Loader)

    train_configs["gpu"] = gpu
    train_configs["val_check"] = True if train_configs["early_stop_patience"] > 0 else False
    if "rerank" in train_configs.keys() and train_configs["rerank"] > 0:
        rerank = True
    else:
        rerank = False
        
    # Init dataset and shadow member/nonmember user
    datasets = init_datasets(dataset_name=dataset_name, max_seq_len=split_configs["max_seq_len"])
    shadow_users = split_info["shadow_users"]
    np.random.seed(split_seed)
    np.random.shuffle(shadow_users)
    n_split_user = int(len(shadow_users) / 2)
    shadow_member_users = shadow_users[:n_split_user]
    shadow_nonmember_users = shadow_users[n_split_user:]

    # Init shadow model
    shadow_mem_model = init_model(model_name, train_configs, _best_ckpt_path=f"member_model_{split_seed}")
    shadow_non_model = init_model(model_name, train_configs, _best_ckpt_path=f"nonmember_model_{split_seed}")
    shadow_mem_model.config["save_path"] = f"./metadata/model_checkpoint/shadow_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}"
    shadow_non_model.config["save_path"] = f"./metadata/model_checkpoint/shadow_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}"

    # Prepare shadow model
    shadow_mem_model = prepare_shadow_model(
        shadow_model=shadow_mem_model,
        datasets=datasets,
        active_users=shadow_member_users,
        active_uid2sample=split_info["shadow_uid2samples"],
        train_configs=train_configs,
    )

    shadow_non_model = prepare_shadow_model(
        shadow_model=shadow_non_model,
        datasets=datasets,
        active_users=shadow_nonmember_users,
        active_uid2sample=split_info["shadow_uid2samples"],
        train_configs=train_configs,
    )

    if not rerank:
        # Prepare unforgotten score of each shadow user including audit user
        recaudit_mem, baseline_mem, recaudit_nonmem, baseline_nonmem = prepare_unforgotten_score(
            shadow_mem_model=shadow_mem_model,
            shadow_non_model=shadow_non_model,
            dataset=datasets[0],
            shadow_member_users=shadow_member_users,
            shadow_nonmember_users=shadow_nonmember_users,
            audit_uid2sample=split_info["shadow_uid2samples"],
        )

        # Save unforgotten score (member and non-member)
        score_dirs = f"./metadata/scores/shadow_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}/{split_seed}"
        if not os.path.exists(score_dirs):
            os.makedirs(score_dirs)

        save_pkl_data(save_dir=f"{score_dirs}/recaudit_member_score.pkl", save_value=recaudit_mem)
        save_pkl_data(save_dir=f"{score_dirs}/baseline_member_score.pkl", save_value=baseline_mem)
        save_pkl_data(save_dir=f"{score_dirs}/recaudit_nonmember_score.pkl", save_value=recaudit_nonmem)
        save_pkl_data(save_dir=f"{score_dirs}/baseline_nonmember_score.pkl", save_value=baseline_nonmem)

    else:
        item_freq = datasets[0].item_freq.detach().numpy()
        popular_item_list = np.argsort(-item_freq)[:100]

        item2year = np.load(f"../datasets/{dataset_name}/item2year.npy")
        new_item_list = np.argsort(-item2year)[:100]

        item2class = np.load(f"../datasets/{dataset_name}/item2class.npy")

        shadow_mem_model.popular_item_list = popular_item_list
        shadow_mem_model.new_item_list = new_item_list
        shadow_mem_model.item2class = item2class

        shadow_non_model.popular_item_list = popular_item_list
        shadow_non_model.new_item_list = new_item_list
        shadow_non_model.item2class = item2class
        
        recaudit_mem0, recaudit_mem1, recaudit_nonmem0, recaudit_nonmem1, baseline_mem, baseline_nonmem = _prepare_unforgotten_score(
            shadow_mem_model=shadow_mem_model,
            shadow_non_model=shadow_non_model,
            dataset=datasets[0],
            shadow_member_users=shadow_member_users,
            shadow_nonmember_users=shadow_nonmember_users,
            audit_uid2sample=split_info["shadow_uid2samples"],
            rerank=True,
        )

        for refine_mode in ["none", "top1", "threshold"]:
            for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
                if rerank_mode not in ["shuffle", "original"]:
                    for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        recaudit_mem = {
                            **recaudit_mem0[refine_mode][rerank_mode][rerank_rate], 
                            **recaudit_mem1[refine_mode][rerank_mode][rerank_rate]
                        }
                        recaudit_nonmem = {
                            **recaudit_nonmem0[refine_mode][rerank_mode][rerank_rate], 
                            **recaudit_nonmem1[refine_mode][rerank_mode][rerank_rate]
                        }
                        save_unforgotten_score(
                            recaudit_mem=recaudit_mem, 
                            baseline_mem=baseline_mem,
                            recaudit_nonmem=recaudit_nonmem,
                            baseline_nonmem=baseline_nonmem,
                            train_cf=f"rerank_{refine_mode}_{rerank_mode}_{rerank_rate:.3f}",
                        )
                else:
                    recaudit_mem = {
                        **recaudit_mem0[refine_mode][rerank_mode], 
                        **recaudit_mem1[refine_mode][rerank_mode]
                    }
                    recaudit_nonmem = {
                        **recaudit_nonmem0[refine_mode][rerank_mode], 
                        **recaudit_nonmem1[refine_mode][rerank_mode]
                    }
                    save_unforgotten_score(
                        recaudit_mem=recaudit_mem, 
                        baseline_mem=baseline_mem,
                        recaudit_nonmem=recaudit_nonmem,
                        baseline_nonmem=baseline_nonmem,
                        train_cf=f"rerank_{refine_mode}_{rerank_mode}",
                    )

    # Save split information
    split_info = {}
    split_info["shadow_users"] = shadow_users
    split_info["shadow_member_users"] = shadow_member_users
    split_info["shadow_nonmember_users"] = shadow_nonmember_users
    score_dirs = f"./metadata/scores/shadow_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}/{split_seed}"
    if not os.path.exists(score_dirs):
        os.makedirs(score_dirs)
    save_pkl_data(save_dir=f"{score_dirs}/split_info.pkl", save_value=split_info)

    print_logger.info("End of the code.")