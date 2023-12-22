
__all__ = ['PatchTST','PITS']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor

from ..models.layers.basics import *
from ..models.losses import *


class PITS(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, num_patch:int, 
                 d_model=128, shared_embedding=True, 
                 head_type = "prediction", aggregate='max', individual = False, 
                 instance_CL = False, temporal_CL = True,
                 ft=False,  pretrain_task='PI', head_dropout = 0, 
                 mean_norm_pretrain = 0, mean_norm_for_cls = 0, 
                 y_range:Optional[tuple]=None,  **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        
        # Backbone
        self.backbone = MLPencoder(c_in = c_in, patch_len = patch_len, 
                                   d_model = d_model, shared_embedding = shared_embedding, 
                                   **kwargs)
        
        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.ft = ft
        self.pretrain_task = pretrain_task
        self.mean_norm_pretrain = mean_norm_pretrain
        self.mean_norm_for_cls = mean_norm_for_cls
        
        self.instance_CL = instance_CL
        self.temporal_CL = temporal_CL

        if instance_CL & temporal_CL :
            self.contrastive_loss = hard_inst_hard_temp
        elif instance_CL :
            self.contrastive_loss = hard_inst
        elif temporal_CL :
            self.contrastive_loss = hard_temp

        if head_type == "pretrain":
            # y : [bs x num_patch x nvars x patch_len]
            self.head = PretrainHead(d_model, patch_len, head_dropout) 
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            # y: [bs x output_dim]
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            # y: [bs x n_classes]
            if aggregate == 'max':
                self.head = ClassificationHead_max(self.n_vars, d_model, target_dim, head_dropout)
            elif aggregate == 'avg':
                self.head = ClassificationHead_avg(self.n_vars, d_model, target_dim, head_dropout)
            elif aggregate == 'concat':
                self.head = ClassificationHead_concat(self.n_vars, d_model, num_patch,  target_dim, head_dropout)

    def forward(self, z, mask):        
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        #####################################################################################
        # (1) Pretrain
        #####################################################################################
        if self.head_type == 'pretrain':
            if self.pretrain_task=='PI':
                mask = ~mask
            mask = mask.unsqueeze(-1)
            B, num_patch, C, _ = z.shape
            
            if self.mean_norm_pretrain:
                z_mean = z.mean(axis=1).mean(axis=-1).unsqueeze(1).unsqueeze(-1)
                z = z-z_mean
            
            z1_CL, z1_MTM = self.backbone(z*(~mask))   # (B,C,D,overlap_margin)
            z2_CL, z2_MTM = self.backbone(z*mask)   # (B,C,D,overlap_margin)
            
            D = z1_CL.shape[2]
            z1_CL = z1_CL.view(B, C * D, num_patch).permute(0,2,1)
            z2_CL = z2_CL.view(B, C * D, num_patch).permute(0,2,1)
            loss_contrastive = self.contrastive_loss(z1_CL, z2_CL)
            z1_MTM = self.head(z1_MTM)  
            z2_MTM = self.head(z2_MTM)
            
            if self.mean_norm_pretrain:
                z1_MTM += z_mean      
                z2_MTM += z_mean
            
            return (z1_MTM, z2_MTM), loss_contrastive
        
        #####################################################################################
        # (2) Finetune
        #####################################################################################
        # --- 2-1) Forecasting
        elif self.head_type != 'classification':
            if self.mean_norm_pretrain:
                z_mean = z.mean(axis=1).mean(axis=-1).unsqueeze(1)
                z = z-z_mean.unsqueeze(-1)
            _, z = self.backbone(z)  # (64,125,1,12) -> (64,1,128,125)
            out = self.head(z)
            if self.mean_norm_pretrain:
                out += z_mean
            return out
        
        # --- 2-2) Classification
        elif self.head_type == 'classification':
            if self.mean_norm_for_cls:
                z_mean = z.mean(axis=1).mean(axis=-1).unsqueeze(1)
                z = z-z_mean.unsqueeze(-1)
            _, z = self.backbone(z) 
            out = self.head(z) 
            return out
        


    
class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: 
            y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        x, _ = torch.max(x.squeeze(1),dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y

class ClassificationHead_max(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = torch.max(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_avg(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.mean(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_concat(nn.Module):
    def __init__(self, n_vars, d_model, num_patch_new, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model*num_patch_new, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                     
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class MLPencoder(nn.Module):
    def __init__(self, c_in, patch_len,  d_model=128, shared_embedding=True, **kwargs):
        super().__init__()
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.act = nn.ReLU(inplace=True)
        if not shared_embedding: 
            self.W_P1 = nn.ModuleList()
            self.W_P2 = nn.ModuleList()
            for _ in range(self.n_vars): 
                self.W_P1.append(nn.Linear(patch_len, d_model))
                self.W_P2.append(nn.Linear(d_model, d_model))
        else:
            self.W_P1 = nn.Linear(patch_len, d_model)      
            self.W_P2 = nn.Linear(d_model, d_model)      

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out1 = []
            x_out2 = []
            for i in range(n_vars):
                z = self.W_P1[i](x[:,:,i,:])
                x_out1.append(z)
                z = self.act(z)
                z = self.W_P2[i](z) 
                x_out2.append(z)
            x1 = torch.stack(x_out1, dim=2)
            x2 = torch.stack(x_out2, dim=2)
        else:
            x1 = self.W_P1(x)                                                      # x: [bs x num_patch x nvars x d_model]
            x2 = self.act(x1)
            x2 = self.W_P2(x2)                                                      # x: [bs x num_patch x nvars x d_model]
        x1 = x1.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        
        x2 = x2.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        
        x1 = x1.permute(0,1,3,2)
        x2 = x2.permute(0,1,3,2)
        return x1,x2
    
    