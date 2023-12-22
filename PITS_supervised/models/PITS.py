__all__ = ['PatchTST_ours']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PITS_backbone import PITS_backbone
from layers.PITS_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, 
                  verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        c_in = configs.c_in
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        shared_embedding = configs.shared_embedding
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            
        else:
            self.model = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x