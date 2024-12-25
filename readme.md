# Forget Me If You Can: Auditing User Data Revocation in Recommendation Systems

This code is the official implementation of RecAudit.

## Dependencies

```python

# python==3.7.13
pip install -r requirements.txt
```

## Why do you need RecAudit?

Recommendation systems have become integral components of modern e-commerce and streaming platforms, enhancing user experience by personalizing content and product suggestions. While users benefit from personalized recommendations by allowing platforms to utilize their data in training these systems, regulations such as the General Data Protection Regulation (GDPR) uphold the user’s right to be forgotten, entitling them to request the removal of their personal data from recommendation systems at any time. However, while deleting user data from storage devices is straightforward, removing personal information from a recommendation system without retraining the model from scratch is challenging.

To address the new design problem of user data revocation auditing—examining whether the model has forgotten or still retains a user’s data in sequential recommendation systems—we propose **RecAudit**: a privacy-preserving IT artifact designed to effectively and efficiently infer the likelihood of individual data being forgotten.

---

## How to use RecAudit?

**Note**:  Before executing code, ensure that the path is set to ./RecStudio-main. We provide the required data, trained models, and evaluation results on the **ml-100k** dataset as a demo. In our source code, audit and surrogate models are called shadow models refer to other open-sourced codes in membership inference attack literature.

### 1. First, to train target and audit models, you can run:

```python

bash train_base_model.sh
```

The checkpoints of the target and audit models, along with the users' forgotten scores in these models, are stored in the **RecStudio-main/metadata** folder.

### 2. Perform user data revocation auditing

```python

bash audit.sh
```

The auditing results, including AUROC, FPR@10% FNR, and prediction outcomes, are saved in the **RecStudio-main/audit_results** folder.

**Note:** In the demo, we used only 32 audit models instead of the 256 mentioned in the paper. As a result, the performance of most audit methods is expected to be slightly lower than the results presented in the paper.

## More details about RecAudit

### Configs

We use the configuration files to control the data partitioning (located at RecStudio-main/configs/split), model training (RecStudio-main/configs/train), and the audit process (RecStudio-main/configs/audit) for the User Data Revocation Auditing process. For example, you can pass different split configuration files to `split.py` to obtain various user data partitions, or pass a different training configuration file to `target.py` to train the target model with different hyperparameters.

### Algorithm implementation

We use functions from [RecStudio](https://github.com/USTCLLM/RecStudio) to build recommendation models. All implementations of the audit algorithm are based on the original paper.
