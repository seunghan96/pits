import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):     
        if isinstance(x, tuple):
            x1 = x[1][0]
            x2 = x[1][1]
            if mode == 'norm':
                self._get_statistics(x1)
                x1 = self._normalize(x1)
            elif mode == 'denorm':
                x1 = self._denormalize(x1)
            else: raise NotImplementedError
            
            if mode == 'norm':
                self._get_statistics(x2)
                x2 = self._normalize(x2)
            elif mode == 'denorm':
                x2 = self._denormalize(x2)
            else: raise NotImplementedError
            return x[0], (x1,x2), x[2]
            
        else:            
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
                
            elif mode == 'denorm':
                x = self._denormalize(x)
            else: raise NotImplementedError
            return x    

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
