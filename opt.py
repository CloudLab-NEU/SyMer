import numpy as np
import torch
import torch.nn.functional as F
import time

from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
# from torch.cuda.amp import autocast
from tqdm import trange

from common import target2outputs, Common
from utils.eval_result import greedy_decoder, get_eval_result, calculate_f1_scores, print_predict


class NoamOpt:
    def __init__(self, d_model, warmup, optimizer, start_step=0, scale=1):
        self.optimizer = optimizer
        self._step = start_step
        self.warmup = warmup
        self.d_model = d_model
        self._rate = 0
        self.scale = scale

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return (self.d_model ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5))) * self.scale

    def get_step(self):
        return self._step

    def load_optimizer(self, load_optimizer):
        self.optimizer.load_state_dict(load_optimizer)


class LossCompute:
    def __init__(self, target_dict, criterion, opt=None, step=None, amp=False):
        self.target_dict = target_dict
        self.criterion = criterion
        self.opt = opt
        self.step = step
        self.amp = amp

    def __call__(self, x, y, norm):
        if self.amp:
            with autocast():
                y = target2outputs(y, self.target_dict)
                x = F.softmax(x, dim=1)
                loss = self.criterion(x.log(), y.view(-1)) / norm
        else:
            y = target2outputs(y, self.target_dict)
            x = F.softmax(x, dim=1)
            loss = self.criterion(x.log(), y.view(-1)) / norm
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data.eq(self.padding_idx), as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == '__main__':
    opts = [NoamOpt(512, 4000, None),
            NoamOpt(512, 4000, None, scale=2),
            NoamOpt(512, 8000, None),
            NoamOpt(1024, 4000, None)]
    plt.plot(np.arange(1, 100000), [[opt.rate(i) for opt in opts] for i in range(1, 100000)])
    plt.legend(["512:4000", "512:4000 scaled", "512:8000", "1024:4000"])
    plt.show()
