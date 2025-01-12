#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import logging
import argparse

from utils import *

'''
    Code for training target model.
'''

def save_unforgotten_score(recaudit_unforgotten_score, baseline_unforgotten_score, train_cf):
    score_dirs = f"./metadata/scores/target_model/{dataset_name}/{model_name}/{split_cf_file}/{train_cf}/{train_seed}"
    if not os.path.exists(score_dirs):
        os.makedirs(score_dirs)
    save_pkl_data(save_dir=f"{score_dirs}/recaudit_score.pkl", save_value=recaudit_unforgotten_score)
    save_pkl_data(save_dir=f"{score_dirs}/baseline_score.pkl", save_value=baseline_unforgotten_score)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("RecAudit")
    argparser.add_argument("--dataset", type=str, default='movielens', help='Datasets')
    argparser.add_argument("--model", type=str, default='BERT4Rec',help='The type of sequence model')
    argparser.add_argument("--train_seed", type=int, default=1, help='Seed for training RS')
    argparser.add_argument("--gpu", type=int, default=0, help='The used gpu id')
    argparser.add_argument("--split_cf", type=str, default="split", help="Config file for spliting dataset") 
    argparser.add_argument("--train_cf", type=str, default="train", help="Config file for training RS") 
    #print(afk)
    torch.set_num_threads(3)
    
    global dataset_name, model_name, split_cf_file, train_seed
    # Init args
    args = argparser.parse_args()
    dataset_name = args.dataset
    model_name = args.model
    train_seed = args.train_seed
    gpu = args.gpu
    split_cf_file = args.split_cf
    torch.cuda.set_device(gpu)

    # Init log file
    log_path = f"./audit_log/train_target_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=f"{log_path}/train_{train_seed}.log", filemode='w')
    logging.info(vars(args))

    with open(f"configs/split/{args.split_cf}.yaml", "rb") as f:
        split_configs = yaml.load(f, Loader=yaml.Loader)
    split_info = read_split_information(dataset_name, split_config_file_name=args.split_cf)

    with open(f"configs/train/{args.train_cf}.yaml", "rb") as f:
        train_configs = yaml.load(f, Loader=yaml.Loader)

    train_configs["train_seed"] = train_seed
    train_configs["gpu"] = gpu
    train_configs["val_check"] = True if train_configs["early_stop_patience"] > 0 else False
    if "rerank" in train_configs.keys() and train_configs["rerank"] > 0:
        rerank = True
    else:
        rerank = False

    # Init dataset and member/nonmember user
    datasets = init_datasets(dataset_name=dataset_name, max_seq_len=split_configs["max_seq_len"])
    member_users = split_info["training_user"] + split_info["pos_audit_users"]
    nonmember_users = split_info["neg_audit_users"]

    # Init target model
    target_model_path = f"model_{train_seed}"
    target_model = init_model(model_name, train_configs, _best_ckpt_path=target_model_path)
    target_model.config["save_path"] = f"./metadata/model_checkpoint/target_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}"

    # Train target model
    active_by_sample(dataset=datasets[0], active_indexes=split_info["target_training_indexes"])
    active_by_user(dataset=datasets[1], active_users=member_users)
    active_by_user(dataset=datasets[2], active_users=member_users)

    train_rec(
        target_model, 
        train_set=datasets[0],
        val_set=datasets[1], 
        train_configs=train_configs,
    )

    # Evaluate target model
    print_logger.info('Target model recommendation performance:')
    evaluate_rec(target_model, datasets[2])

    # Get unforgotten score of each audit user
    if rerank:
        item_freq = datasets[0].item_freq.detach().numpy()
        popular_item_list = np.argsort(-item_freq)[:100]

        item2year = np.load(f"../datasets/{dataset_name}/item2year.npy")
        new_item_list = np.argsort(-item2year)[:100]

        item2class = np.load(f"../datasets/{dataset_name}/item2class.npy")

        target_model.popular_item_list = popular_item_list
        target_model.new_item_list = new_item_list
        target_model.item2class = item2class

    recaudit_unforgotten_score, baseline_unforgotten_score = get_unforgotten_score(
        model=target_model,
        dataset=datasets[0],
        audit_user=split_info["audit_users"],
        audit_uid2sample=split_info["shadow_uid2samples"],
        rerank=rerank,
    )

    # Record unforgotten score
    if not rerank:
        score_dirs = f"./metadata/scores/target_model/{dataset_name}/{model_name}/{args.split_cf}/{args.train_cf}/{train_seed}"
        if not os.path.exists(score_dirs):
            os.makedirs(score_dirs)
        save_pkl_data(save_dir=f"{score_dirs}/recaudit_score.pkl", save_value=recaudit_unforgotten_score)
        save_pkl_data(save_dir=f"{score_dirs}/baseline_score.pkl", save_value=baseline_unforgotten_score)
    else:
        for refine_mode in ["none", "top1", "threshold"]:
            for rerank_mode in ["shuffle", "popular", "timeliness", "novelty", "original"]:
                if rerank_mode not in ["shuffle", "original"]:
                    for rerank_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        save_unforgotten_score(
                            recaudit_unforgotten_score=recaudit_unforgotten_score[refine_mode][rerank_mode][rerank_rate], 
                            baseline_unforgotten_score=baseline_unforgotten_score, 
                            train_cf=f"rerank_{refine_mode}_{rerank_mode}_{rerank_rate:.3f}",
                        )
                else:
                    save_unforgotten_score(
                        recaudit_unforgotten_score=recaudit_unforgotten_score[refine_mode][rerank_mode], 
                        baseline_unforgotten_score=baseline_unforgotten_score, 
                        train_cf=f"rerank_{refine_mode}_{rerank_mode}",
                    )
        
    print_logger.info("End of the code.")










        

 