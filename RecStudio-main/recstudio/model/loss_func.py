from turtle import forward
import torch
import torch.nn.functional as F

class FullScoreLoss(torch.nn.Module):
    r"""Calculate loss with positive scores and scores on all items.

    The loss need user's perference scores on positive items(ground truth) and all other items. 
    However, due to the item numbers are very huge in real-world datasets, calculating scores on all items
    may be very time-consuming. So the loss is seldom used in large-scale dataset.
    """
    def forward(self, label, pos_score, all_score):
        r"""
        """
        pass

class PairwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pass

class PointwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score):
        raise NotImplementedError(f'{type(self).__name__} is an abstrat class, \
            this method would not be implemented' )

class SquareLoss(PointwiseLoss):
    def forward(self, label, pos_score):
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.square(label - pos_score), dim=-1))
        else:
            return torch.mean(torch.square(label - pos_score))

class SoftmaxLoss(FullScoreLoss):
    def forward(self, label, pos_score, all_score):
        if all_score.dim() > pos_score.dim():
            return torch.mean(torch.logsumexp(all_score, dim=-1) - pos_score)
        else:
            output = torch.logsumexp(all_score, dim=-1, keepdim=True) - pos_score
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
            return torch.mean(output)

class BPRLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return -torch.mean((loss * weight).sum(-1))
        else:
            loss = -torch.mean(
                F.logsigmoid(pos_score - torch.max(neg_score, dim=-1)))
            return loss

class Top1Loss(BPRLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = torch.sigmoid(neg_score - pos_score.view(*pos_score.shape, 1)) 
            loss += torch.sigmoid(neg_score ** 2)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return torch.mean((loss * weight).sum(-1))
        else:
            max_neg_score = torch.max(neg_score, dim=-1)
            loss = torch.sigmoid(max_neg_score-pos_score)
            loss = loss + torch.sigmoid(max_neg_score ** 2)
        return loss

class SampledSoftmaxLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        return torch.mean(output)

class WeightedBPRLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        weight = F.softmax(neg_score - log_neg_prob, -1)
        return -torch.mean((loss * weight).sum(-1))

class BinaryCrossEntropyLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns or pos_score.dim() > 1:
            weight = F.softmax(torch.ones_like(neg_score), -1)
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(F.logsigmoid(pos_score), nan=0.0).sum(-1)\
                 / notpadnum
            return torch.mean(-output + \
                torch.sum(F.softplus(neg_score) * weight, dim=-1))
        else:
            return torch.mean(-F.logsigmoid(pos_score) + \
                F.softplus(torch.max(neg_score, dim=-1)))

class ModelStealingLoss(PairwiseLoss):
    def __init__(self, lamda1, lamda2):
        super().__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.activate_func = torch.nn.ReLU()

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, topk_score):
        loss1 = 0
        loss2 = 0
        _, k = topk_score.shape
        loss1 = self.activate_func(topk_score[:,1:] - topk_score[:,:-1] + self.lamda1)
        loss2 = self.activate_func(neg_score - topk_score + self.lamda2)
        return torch.mean(loss1) + torch.mean(loss2)


class RankingLoss(PairwiseLoss):
    def __init__(self, mode, lamda1):
        super().__init__()
        assert mode in ['hinge', 'bpr']
        self.mode = mode
        self.lamda1 = lamda1
        self.activate_func = torch.nn.ReLU()

    def forward(self, topk_score):
        if self.mode == 'hinge':
            loss1 = self.activate_func(topk_score[:,1:] - topk_score[:,:-1] + self.lamda1)
        else:
            loss1 = -F.logsigmoid(topk_score[:,:-1] - topk_score[:,1:])

        return torch.mean(loss1)

class posloss():
    pass

class TopkLoss(PairwiseLoss):
    def __init__(self, mode, lamda2, temp):
        super().__init__()
        assert mode in ['hinge', 'bpr', 'info_nce']
        self.mode = mode
        self.lamda2 = lamda2
        self.temp = temp
        self.activate_func = torch.nn.ReLU()

    def forward(self, neg_score, topk_score):
        if self.mode == 'hinge':
            loss2 = self.activate_func(neg_score - topk_score + self.lamda2)
        elif self.mode == 'bpr':
            loss2 = -F.logsigmoid(topk_score - neg_score)
        else:
            logsoftmax = torch.nn.LogSoftmax()
            _, k = topk_score.shape
            topk_score = topk_score / self.temp
            neg_score = neg_score / self.temp
            for i in range(k):
                score = torch.hstack((topk_score[:,i].unsqueeze(1), neg_score))
                if i == 0:
                    loss2 = -logsoftmax(score)[:,0]
                else:
                    loss2 += -logsoftmax(score)[:,0]
        return torch.mean(loss2)

class ExtractionLoss(PairwiseLoss):
    def __init__(self, lamda1, lamda2, mode1, mode2, temp):
        super().__init__()
        assert mode1 in ['hinge', 'bpr', 'no_rank']
        assert mode2 in ['hinge', 'bpr', 'info_nce']
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.mode1 = mode1
        self.mode2 = mode2
        self.temp = temp

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, topk_score):
        if self.mode1 == 'no_rank':
            pass
        else:
            rankloss = RankingLoss(self.mode1, self.lamda1)
        topkloss = TopkLoss(self.mode2, self.lamda2, self.temp)
        return rankloss(topk_score) + topkloss(neg_score, topk_score) if self.mode1 != 'no_rank' else topkloss(neg_score, topk_score)

class WightedBinaryCrossEntropyLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        weight = F.softmax(neg_score - log_neg_prob, -1)
        if pos_score.dim() > 1:
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(F.logsigmoid(pos_score), nan=0.0).sum(-1)\
                 / notpadnum
        else:
            output = F.logsigmoid(pos_score)
        return torch.mean(-output + torch.sum(F.softplus(neg_score) * weight, dim=-1))

class HingeLoss(PairwiseLoss):
    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, label, pos_score, log_pos_prob, neg_score, neg_prob):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score +\
            self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)

class InfoNCELoss(SampledSoftmaxLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return super().forward(label, pos_score, torch.zeros_like(pos_score), \
            neg_score, torch.zeros_like(neg_score))


class NCELoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        loss = F.logsigmoid(new_pos) + (new_neg - F.softplus(new_neg)).sum(1)
        return -loss.mean()


class CCLLoss(PairwiseLoss):
    def __init__(self, margin=0.9, neg_weight=0.3) -> None:
        super().__init__()
        self.margin = 0.9
        self.neg_weight = neg_weight

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return (1 - pos_score) + \
            self.neg_weight * torch.sum(torch.relu(neg_score - self.margin), dim=1)
        