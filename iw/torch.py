import torch

from dpipe.torch.utils import sequence_to_var, to_np
from dpipe.torch.model import optimizer_step


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


def asymmetric_similarity_loss(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                               beta: float = 1.5):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    if weight is None:
        f_beta = (1 + beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
                 / torch.sum(beta ** 2 * target ** 2 + preds ** 2, dim=sum_dims)
    else:
        f_beta = (1 + beta ** 2) * torch.sum(weight * preds * target, dim=sum_dims) \
                 / torch.sum(weight * (beta ** 2 * target ** 2 + preds ** 2), dim=sum_dims)
    loss = 1 - f_beta

    return loss.mean()


# ================ Dice Loss ==========================================================================================


def dice_loss(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    if weight is None:
        dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
    else:
        dice = 2 * torch.sum(weight * preds * target, dim=sum_dims) \
               / torch.sum(weight * (preds ** 2 + target ** 2), dim=sum_dims)
    loss = 1 - dice

    return loss.mean()


# ======= Train Step to pass weights (or old: connected components) through ===========================================


def train_step_with_cc(*inputs, architecture, criterion, optimizer, with_cc=False, **optimizer_params):
    architecture.train()
    if with_cc:
        n_inputs = len(inputs) - 2  # target and weight
        inputs = sequence_to_var(*inputs, device=architecture)
        inputs, target, weight = inputs[:n_inputs], inputs[-2], inputs[-1]
        loss = criterion(architecture(*inputs), target, weight)
    else:
        n_inputs = len(inputs) - 1  # target
        inputs = sequence_to_var(*inputs, device=architecture)
        inputs, target = inputs[:n_inputs], inputs[-1]
        loss = criterion(architecture(*inputs), target)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)
