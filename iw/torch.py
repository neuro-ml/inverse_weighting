import torch

from dpipe.torch.utils import sequence_to_var, to_np
from dpipe.torch.model import optimizer_step
from dpipe.torch.functional import weighted_cross_entropy_with_logits, focal_loss_with_logits


# ============ Generalized Dice Loss ==================================================================================


def generalized_dice_loss(logit, target):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)
    preds_bg = 1 - preds  # bg = background
    preds = torch.cat([preds, preds_bg], dim=1)

    target_bg = 1 - target
    target = torch.cat([target, target_bg], dim=1)

    sp_dims = list(range(2, logit.dim()))
    weight = 1 / (1 + torch.sum(target, dim=sp_dims) ** 2)

    generalized_dice = 2 * torch.sum(weight * torch.sum(preds * target, dim=sp_dims), dim=-1) \
        / torch.sum(weight * torch.sum(preds ** 2 + target ** 2, dim=sp_dims), dim=-1)

    loss = 1 - generalized_dice

    return loss.mean()


# ========== Asymmetric Similarity Loss ===============================================================================


def asymmetric_similarity_loss_orig(logit, target, beta=1.5):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    f_beta = (1 + beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
             / ((1 + beta ** 2) * torch.sum(preds * target, dim=sum_dims) +
                beta ** 2 * torch.sum((1 - preds) * target, dim=sum_dims) +
                torch.sum(preds * (1 - target), dim=sum_dims))

    loss = 1 - f_beta

    return loss.mean()


def asymmetric_similarity_loss(logit, target, beta=1.5):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    f_beta = (1 + beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
             / torch.sum(beta ** 2 * target ** 2 + preds ** 2, dim=sum_dims)

    loss = 1 - f_beta

    return loss.mean()


# ================ Dice Loss ==========================================================================================


def dice_loss(logit: torch.Tensor, target: torch.Tensor):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
    loss = 1 - dice

    return loss.mean()


# ============== Inverse Weights ======================================================================================


def cc2weights(logit, target, cc):
    weight = torch.ones_like(logit)
    for i, (target_single, cc_single) in enumerate(zip(target, cc)):
        n_cc = int(torch.max(cc_single).data)
        if n_cc > 0:
            n_positive = torch.sum(cc_single > 0).type(torch.FloatTensor)
            for n in range(1, n_cc + 1):
                weight[i][cc_single == n] = n_positive / (n_cc * torch.sum(cc_single == n))
    return weight


def iwbce(logit: torch.Tensor, target: torch.Tensor, cc: torch.Tensor = None, adaptive=False):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    weight = cc2weights(logit, target, cc)

    loss = weighted_cross_entropy_with_logits(logit, target, weight, adaptive=adaptive)
    return loss


def iwdl(logit: torch.Tensor, target: torch.Tensor, cc: torch.Tensor = None):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)
    weight = cc2weights(logit, target, cc)

    sum_dims = list(range(1, logit.dim()))
    dice = 2 * torch.sum(weight * preds * target, dim=sum_dims) \
           / torch.sum(weight * (preds ** 2 + target ** 2), dim=sum_dims)
    loss = 1 - dice
    return loss.mean()


def iwasl(logit, target, cc: torch.Tensor = None, beta=1.5):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    # re-weighting components
    weight = cc2weights(logit, target, cc)

    sum_dims = list(range(1, logit.dim()))
    
    f_beta = (1 + beta ** 2) * torch.sum(weight * preds * target, dim=sum_dims) \
             / torch.sum(weight * (beta ** 2 * target ** 2 + preds ** 2), dim=sum_dims)

    loss = 1 - f_beta

    return loss.mean()


def iwfl(logit: torch.Tensor, target: torch.Tensor, cc: torch.Tensor = None, gamma: float = 2, alpha: float = 0.25):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    weight = cc2weights(logit, target, cc)

    loss = focal_loss_with_logits(logit, target, weight=weight, gamma=gamma, alpha=alpha)
    return loss


# ======= Train Step to pass connected components through =============================================================


def train_step_with_cc(*inputs, architecture, criterion, optimizer, with_cc=False, **optimizer_params):
    architecture.train()
    if with_cc:
        n_inputs = len(inputs) - 2  # target and cc
        inputs = sequence_to_var(*inputs, device=architecture)
        inputs, target, cc = inputs[:n_inputs], inputs[-2], inputs[-1]
        loss = criterion(architecture(*inputs), target, cc)
    else:
        n_inputs = len(inputs) - 1  # target
        inputs = sequence_to_var(*inputs, device=architecture)
        inputs, target = inputs[:n_inputs], inputs[-1]
        loss = criterion(architecture(*inputs), target)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)
