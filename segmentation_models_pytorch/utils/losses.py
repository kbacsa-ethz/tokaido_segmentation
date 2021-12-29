import torch
import torch.nn as nn
import torch.nn.functional as F_torch

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class FocalLoss(base.Loss):
    """
    Implementation from https://github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, gamma=0., alpha=None, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    @staticmethod
    def flatten(tensor):
        tensor = tensor.view(tensor.size(0), tensor.size(1), -1)  # N,C,H,W => N,C,H*W
        tensor = tensor.transpose(1, 2)  # N,C,H*W => N,H*W,C
        tensor = tensor.contiguous().view(-1, tensor.size(2))  # N,H*W,C => N*H*W,C
        return tensor

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        if y_pr.dim() > 2:
            y_pr = self.flatten(y_pr)

        if y_gt.dim() > 2:
            y_gt = self.flatten(y_gt)
            y_gt = torch.argmax(y_gt, dim=-1, keepdim=True)

        logpt = F_torch.log_softmax(y_pr)
        indexes = torch.LongTensor(list(set(list(range(8))) - set(self.ignore_channels)))
        logpt = torch.index_select(logpt, 1, indexes)
        logpt = logpt.gather(1, y_gt.long())
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != y_pr.data.type():
                self.alpha = self.alpha.type_as(y_pr.data)
            at = self.alpha.gather(0, y_gt.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()


class MeanIOU(base.Loss):
    """
    Implementation from https://github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        intersection = y_gt * y_pr
        not_true = torch.ones_like(y_gt) - y_gt
        union = y_gt + (not_true * y_pr)
        return -torch.log(torch.sum(intersection)/torch.sum(union))


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
