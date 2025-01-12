#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import logging
import argparse
import numpy as np

from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from utils import save_pkl_data, read_pkl_data, read_split_information, print_logger

'''
    Code for auditing recommendation model
'''

AUDIT_METHOD_CONFIG = {
    "Zhang et al.": {}, # ccs
    "Zhu et al.": {}, # www
    "Yeom et al.": {"score_name":"baseline", "prepro_mode":"-loss",     "audit_mode": "global"}, # loss
    "Zhou et al.": {"score_name":"baseline", "prepro_mode":"corr_conf", "audit_mode": "global"}, # nc
    "Sablayrolles et al.": {"score_name":"baseline", "prepro_mode":"-loss", "audit_mode": "2side-nonparam"}, # non
    "Watson et al.": {"score_name":"baseline", "prepro_mode":"-loss", "audit_mode": "1side-nonparam"}, # 1-side-non
    "Carlini et al.": {"score_name":"baseline", "prepro_mode":"lira", "audit_mode": "2side-gauss"}, # lira
    "RecAudit": {"score_name":"recaudit", "prepro_mode":"none", "audit_mode": "2side-gauss"},  # ours
}

def init_log_file():
    log_path = f"./audit_log/audit/{dataset_name}/{target_model_name}/{shadow_model_name}/{split_config_file}/{train_config_file}/{audit_config_file}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=f"{log_path}/audit.log", filemode='w')
    logging.info(vars(args))

def load_target_unforgotten_score(
    train_seed,
    score_name,
):
    score_dirs = f"./metadata/scores/target_model/{dataset_name}/{target_model_name}/{split_config_file}/{train_config_file}/{train_seed}"
    if score_name == "recaudit":
        target_unforgotten_score = read_pkl_data(f"{score_dirs}/recaudit_score.pkl")
    elif score_name == "baseline":
        target_unforgotten_score = read_pkl_data(f"{score_dirs}/baseline_score.pkl")
    else:
        raise ValueError("Score name should be recaudit or baseline.")
    return target_unforgotten_score

def merge_unforgotten_score(
    score_dict1,
    score_dict2,
):
    for uid in score_dict2.keys():
        if uid in score_dict1.keys():
            score_dict1[uid].append(score_dict2[uid])
        else:
            score_dict1[uid] = [score_dict2[uid]]
    return score_dict1

def load_shadow_unforgotten_score(
    split_seeds,
    score_name,
):  
    assert UNLEARNING + AUDIT_BY_USER_FINE_TUNE + (SHADOW_REFINE != "not support") + AUDIT_BY_USER_FINE_TUNE <= 1
    shadow_mem_unforgotten_score = {}
    shadow_non_unforgotten_score = {}
    for split_seed in split_seeds:
        score_dirs = f"metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/{train_config_file}/{split_seed}"
        if UNLEARNING:
            score_dirs = f"metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/train/{split_seed}"
        elif AUDIT_BY_USER_FINE_TUNE:
            score_dirs = f"./audit_by_user_fine_tune/{score_dirs}"
        elif AUDIT_BY_USER:
            score_dirs = f"./audit_by_user/{score_dirs}"
        elif SHADOW_REFINE == "none":
            score_dirs = f"metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/rerank_none_original/{split_seed}"
        elif SHADOW_REFINE == "threshold":
            score_dirs = f"metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/rerank_threshold_original/{split_seed}"
        elif SHADOW_REFINE == "top1":
            score_dirs = f"metadata/scores/shadow_model/{dataset_name}/{shadow_model_name}/{split_config_file}/rerank_top1_original/{split_seed}"
        else:
            pass
            
        if score_name == "recaudit":
            mem_score = read_pkl_data(f"{score_dirs}/recaudit_member_score.pkl")
            nonmem_score = read_pkl_data(f"{score_dirs}/recaudit_nonmember_score.pkl")
        elif score_name == "baseline":
            mem_score = read_pkl_data(f"{score_dirs}/baseline_member_score.pkl")
            nonmem_score = read_pkl_data(f"{score_dirs}/baseline_nonmember_score.pkl")
        else:
            raise ValueError("Score name should be recaudit or baseline.")
        shadow_mem_unforgotten_score = merge_unforgotten_score(shadow_mem_unforgotten_score, mem_score)
        shadow_non_unforgotten_score = merge_unforgotten_score(shadow_non_unforgotten_score, nonmem_score)
    return shadow_mem_unforgotten_score, shadow_non_unforgotten_score

def get_uid2correctness(
    prepro_mode,
    train_seed,
    split_seeds,
):
    if prepro_mode == "corr_conf":
        target_unforgotten_score = load_target_unforgotten_score(
            train_seed=train_seed,
            score_name="recaudit",
        )
        shadow_mem_unforgotten_score, shadow_nom_unforgotten_score = load_shadow_unforgotten_score(
            split_seeds=split_seeds,
            score_name="recaudit",
        )
        target_uid2corr     = unforgotten_score_prepro(target_unforgotten_score, prepro_mode="bool")
        shadow_mem_uid2corr = unforgotten_score_prepro(shadow_mem_unforgotten_score, prepro_mode="bool")
        shadow_non_uid2corr = unforgotten_score_prepro(shadow_nom_unforgotten_score, prepro_mode="bool")
        return target_uid2corr, shadow_mem_uid2corr, shadow_non_uid2corr
    else:
        return None, None, None
    
def _unforgotten_score_prepro(raw_scores, prepro_mode, correctness=None):
    if raw_scores.ndim == 0:
        raw_scores = np.expand_dims(raw_scores,0)
    if prepro_mode == "-loss":
        scores = raw_scores + 1e-3
        scores[scores>1] = 1
        loss = -np.log(scores)
        return -loss
    elif prepro_mode == "lira":
        scores = raw_scores
        scores.astype(np.float64)
        lira_scores = (scores / (1-scores+1e-3)) + 1e-3
        lira_scores.astype(np.float64)
        lira_scores[lira_scores>1e3] = 1e3
        return np.log(lira_scores)
    elif prepro_mode == "corr_conf":
        return raw_scores + correctness
    elif prepro_mode == "bool":
        correctness = raw_scores
        correctness[correctness>0] = 1
        correctness[correctness<0] = 0
        return correctness
    elif prepro_mode == "none":
        return raw_scores
    else:
        raise ValueError("Score preprocess mode error")
    
def unforgotten_score_prepro(uid2score, prepro_mode, uid2correctness=None):
    for uid in uid2score.keys():
        raw_scores = uid2score[uid]
        if uid2correctness is not None:
            correctness = uid2correctness[uid]
        else:
            correctness = None
        prepro_score = _unforgotten_score_prepro(np.array(raw_scores), prepro_mode, correctness).squeeze()
        if prepro_score.ndim == 0:
            prepro_score = float(prepro_score)
        uid2score[uid] = prepro_score
    return uid2score

def _get_unforgotten_likelihood(
    audit_method,
    mem_shadow_scores, 
    non_shadow_scores, 
    target_score,
    mean_target_score,
    mean_shadow_score=None,
):
    audit_mode = AUDIT_METHOD_CONFIG[audit_method]["audit_mode"]
    if NORM:
        mem_shadow_scores = mem_shadow_scores - mean_shadow_score
        non_shadow_scores = non_shadow_scores - mean_shadow_score
        target_score = target_score - mean_target_score

    if audit_mode == "global":
        prediction = 1 if target_score > mean_target_score else 0
        likelihood = target_score - mean_target_score
        
    elif audit_mode == "2side-gauss":
        mean_mem, std_mem = np.mean(mem_shadow_scores), np.std(mem_shadow_scores)
        mean_non, std_non = np.mean(non_shadow_scores), np.std(non_shadow_scores)
        pr_mem = norm.logpdf(target_score, mean_mem, std_mem + 1e-30)
        pr_non = norm.logpdf(target_score, mean_non, std_non + 1e-30)
        prediction = 1 if pr_mem > pr_non else 0
        likelihood = pr_mem - pr_non

    elif audit_mode == "1side-gauss":
        mean_non, std_non = np.mean(non_shadow_scores), np.std(non_shadow_scores)
        pr_non = norm.logpdf(target_score, mean_non, std_non + 1e-30)
        prediction = 1 if target_score > pr_non else 0
        likelihood = - pr_non

    elif audit_mode == "2side-nonparam":
        mean_mem = np.mean(mem_shadow_scores)
        mean_non = np.mean(non_shadow_scores)
        threshold = (mean_mem + mean_non) / 2
        prediction = 1 if target_score > threshold else 0
        likelihood = target_score - threshold

    elif audit_mode == "1side-nonparam":
        mean_non = np.mean(non_shadow_scores)
        prediction = 1 if target_score > mean_non else 0
        likelihood = target_score - mean_non

    else:
        raise ValueError("Audit mode error")

    return likelihood, prediction

def get_unforgotten_likelihood(audit_method):
    # Init config
    split_seeds = list(range(1, int(audit_configs["num_audit_model"]/2)+1))
    score_name  = AUDIT_METHOD_CONFIG[audit_method]["score_name"]
    prepro_mode = AUDIT_METHOD_CONFIG[audit_method]["prepro_mode"]
    
    # Load and preprocess unforgotten scores
    target_unforgotten_score = load_target_unforgotten_score(
        train_seed=train_seed,
        score_name=score_name,
    )

    shadow_mem_unforgotten_score, shadow_nom_unforgotten_score = load_shadow_unforgotten_score(
        split_seeds=split_seeds,
        score_name=score_name,
    )

    target_uid2corr, shadow_mem_uid2corr, shadow_non_uid2corr = get_uid2correctness(
        prepro_mode=prepro_mode,
        train_seed=train_seed,
        split_seeds=split_seeds,
    )
    target_unforgotten_score     = unforgotten_score_prepro(target_unforgotten_score, prepro_mode, target_uid2corr)
    shadow_mem_unforgotten_score = unforgotten_score_prepro(shadow_mem_unforgotten_score, prepro_mode, shadow_mem_uid2corr)
    shadow_nom_unforgotten_score = unforgotten_score_prepro(shadow_nom_unforgotten_score, prepro_mode, shadow_non_uid2corr)

    # Extract mean target unforgotten score
    mean_target_score = np.mean(np.array(list(target_unforgotten_score.values())))

    shadow_score = []
    for i in range(len(audit_users)):
        user_id = audit_users[i]
        mem_shadow_scores=shadow_mem_unforgotten_score[user_id],
        non_shadow_scores=shadow_nom_unforgotten_score[user_id],
        shadow_score += list(mem_shadow_scores)
        shadow_score += list(non_shadow_scores)
    mean_shadow_score = np.mean(np.array(shadow_score))

    # Get unforgotten prediction and likelihood
    user_unforgotten_likelihood  = np.zeros(len(audit_users))
    user_unforgotten_prediction  = np.zeros(len(audit_users))
    for i in range(len(audit_users)):
        user_id = audit_users[i]
        likelihood, prediction = _get_unforgotten_likelihood(
            audit_method=audit_method,
            mem_shadow_scores=shadow_mem_unforgotten_score[user_id],
            non_shadow_scores=shadow_nom_unforgotten_score[user_id],
            target_score=target_unforgotten_score[user_id],
            mean_target_score=mean_target_score,
            mean_shadow_score=mean_shadow_score,
        )
        user_unforgotten_likelihood[i] = likelihood
        user_unforgotten_prediction[i] = prediction
    user_unforgotten_likelihood.astype(np.float64)
    user_unforgotten_prediction.astype(np.int64)

    return user_unforgotten_likelihood, user_unforgotten_prediction

def _get_audit_result(label, likelihood, prediction):
    acc = np.sum(label==prediction)/len(label)
    print(f"Acc:{acc}")

    fpr_array, tpr_array, _ = roc_curve(label, likelihood)
    # max_acc = np.max(1-(fpr+(1-tpr))/2)
    auroc = auc(fpr_array, tpr_array)

    fnr_array = 1-tpr_array
    fpr_at_fix_fnr = fpr_array[np.where(fnr_array<.1)[0][0]] if len(np.where(fnr_array<.1)[0]) > 0 else -1
    return auroc, fpr_at_fix_fnr

def get_audit_results(user_unforgotten_label, audit_methods):
    for audit_method in audit_methods:
        audit_results = {}
        user_unforgotten_likelihood, user_unforgotten_prediction = get_unforgotten_likelihood(audit_method)

        auroc, fpr_at_fix_fnr = _get_audit_result(
            label=user_unforgotten_label,
            likelihood=user_unforgotten_likelihood,
            prediction=user_unforgotten_prediction,
        )
        
        audit_results["users"]  = audit_users
        audit_results["method"] = audit_methods
        audit_results["auroc"]  = auroc
        audit_results["label"]  = user_unforgotten_label
        audit_results["likelihood"] = user_unforgotten_likelihood
        audit_results["prediction"] = user_unforgotten_prediction
        audit_results["fpr_at_fix_fnr"]  = fpr_at_fix_fnr

        if not os.path.exists(result_save_dirs):
            os.makedirs(result_save_dirs)

        save_pkl_data(save_dir=f"{result_save_dirs}/{audit_method}", save_value=audit_results)
        print_logger.info(f"Method: {audit_method}, AUROC: {auroc}, FPR@10%FNR: {fpr_at_fix_fnr}.")

def audit():
    # Set user unforgotten label
    user_unforgotten_label = np.zeros(len(audit_users))
    for i in range(len(audit_users)):
        user_id = audit_users[i]
        if user_id in pos_audit_users:
            user_unforgotten_label[i] = 1
        else:
            user_unforgotten_label[i] = 0
    user_unforgotten_label.astype(np.int64)

    # Get audit results
    audit_methods = ["Yeom et al.", "Sablayrolles et al.", "Watson et al.", "Carlini et al.", "Zhou et al.", "RecAudit"]
    get_audit_results(user_unforgotten_label, audit_methods)

    print_logger.info("End of the code.")
    print_logger.info(100*"#")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("RecAudit")
    argparser.add_argument("--dataset", type=str, default='movielens', help='Datasets')
    argparser.add_argument("--target_model", type=str, default='BERT4Rec',help='The type of target model')
    argparser.add_argument("--shadow_model", type=str, default='BERT4Rec',help='The type of shadow model')
    argparser.add_argument("--train_seed", type=int, default=1, help='Seed for training RS')

    argparser.add_argument("--split_cf", type=str, default="split", help="Config file for spliting dataset") 
    argparser.add_argument("--train_cf", type=str, default="train", help="Config file for training RS") 
    argparser.add_argument("--audit_cf", type=str, default="audit", help="Config file for auditing RS")
    
    torch.set_num_threads(1)

    global dataset_name, target_model_name, shadow_model_name, train_seed
    global audit_config_file, split_config_file, train_config_file, audit_configs
    global audit_users, pos_audit_users, neg_audit_users, result_save_dirs
    global AUDIT_BY_USER, AUDIT_BY_USER_FINE_TUNE, NORM
    global SHADOW_REFINE, UNLEARNING

    # Init args
    args = argparser.parse_args()
    dataset_name = args.dataset
    target_model_name = args.target_model
    shadow_model_name = args.shadow_model
    train_seed = args.train_seed

    audit_config_file = args.audit_cf
    split_config_file = args.split_cf
    train_config_file = args.train_cf
    
    with open(f"configs/audit/{audit_config_file}.yaml", "rb") as f:
        audit_configs = yaml.load(f, Loader=yaml.Loader)
    AUDIT_BY_USER = True if audit_configs["audit_by_user"] > 0 else False
    AUDIT_BY_USER_FINE_TUNE = True if audit_configs["audit_by_user_fine_tune"] > 0 else False

    UNLEARNING = True if "unlearning" in train_config_file else False
    NORM = True if "syn" in dataset_name else False
    SHADOW_REFINE = audit_configs["shadow_refine"] if "shadow_refine" in audit_configs.keys() else "not support"

    # Init log file
    init_log_file()

    # Obtain audit user
    split_info = read_split_information(dataset_name, split_config_file)
    pos_audit_users = split_info["pos_audit_users"]
    neg_audit_users = split_info["neg_audit_users"]
    audit_users = pos_audit_users + neg_audit_users

    # Perform audit
    result_save_dirs = f"./audit_results/audit/{dataset_name}/{target_model_name}/{shadow_model_name}/{split_config_file}/{train_config_file}/{audit_config_file}/{train_seed}"

    audit()
