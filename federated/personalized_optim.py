# -*- coding: utf-8 -*-
"""
Created on Thu Jau 25th 2024
@author: yamadaaiki
"""

# reference
# https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/optimizers/fedoptimizer.py

import torch
from torch.optim import Optimizer


# for ditto (https://proceedings.mlr.press/v139/li21h.html)
class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr:float=0.01, mu:float=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)
        
    @torch.no_grad()
    def step(self, global_params, device, loss_path=None, is_first=False):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                
                if loss_path is not None:
                    with open(loss_path, 'a') as fp:
                        fp.write(f'{(torch.linalg.norm(p.grad.data)).item()},{(torch.linalg.norm(p.data - g.data)).item()}\n')
                
                if is_first:
                    p.data.add_(p.grad.data, alpha=-group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])


class ProposedPerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr:float=0.01, ratio:float=0.0):
        default = dict(lr=lr, ratio=ratio)
        super().__init__(params, default)
        
    @torch.no_grad()
    def step(self, global_params, device, loss_path=None, is_first=False):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                
                if torch.linalg.norm(p.data - g.data).item() != 0.0:
                    mu = (torch.linalg.norm(p.grad.data)).item() * group['ratio'] / (torch.linalg.norm(p.data - g.data).item())
                else:
                    mu = -float('inf')
                
                if torch.linalg.norm(p.data - g.data).item() == 0.0:
                    d_p = p.grad.data + 1.0 * (p.data - g.data)
                else:
                    d_p = p.grad.data + mu * (p.data - g.data)
                    
                    assert mu != -float('inf')
                
                if loss_path is not None:
                    with open(loss_path, 'a') as fp:
                        fp.write(f'{(torch.linalg.norm(p.grad.data)).item()},{(torch.linalg.norm(p.data - g.data)).item()},{mu},{group["ratio"]}\n')
                        
                if is_first:
                    p.data.add_(p.grad.data, alpha=-group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])
