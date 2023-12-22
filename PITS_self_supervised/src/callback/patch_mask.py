
import torch
from torch import nn

from .core import Callback

def right_fill_2d(tensor):
    
    B,L,C = tensor.size()
    tensor = tensor.permute(0,2,1)
    new_B = B*C
    tensor = tensor.reshape(new_B * L, 1)

    # Find indices of non-zero elements
    non_zero_indices = torch.nonzero(tensor)

    # Find the indices of the first zero elements
    zero_indices = torch.nonzero(tensor == 0)

    # Compute the differences between zero indices and non-zero indices
    differences = zero_indices.unsqueeze(1) - non_zero_indices.unsqueeze(0)

    # Find the indices of the closest non-zero elements on the right for each zero element
    closest_indices = torch.argmin(differences.clamp(min=0), dim=1)

    # Replace zeros with the closest non-zero values on the right
    filled_tensor = tensor.clone()
    filled_tensor[zero_indices[:, 0], zero_indices[:, 1]] = tensor[
        non_zero_indices[closest_indices[:, 0], 0], non_zero_indices[closest_indices[:, 0], 1]
    ]
    return filled_tensor.view(new_B, L).view(B, L, C).view(B, L, C)


def get_weight(mask_):
    val = mask_.int()#[idx,:,0]
    a = torch.cumsum(mask_,axis=1)#[idx,:,0]
    temp = torch.hstack([mask_.int(),
                            torch.zeros(mask_.size(0),1,mask_.size(2)).to(mask_.device)])
    temp2 = torch.hstack([torch.zeros(mask_.size(0),1,mask_.size(2)).to(mask_.device),
                            mask_.int()])
    b = torch.diff(temp,axis=1)==-1
    c = a*b
    d = right_fill_2d(c)
    e = d-a+1
    f = e*(torch.diff(temp2,axis=1)==1)
    numerator = (e*val)
    denominator = right_fill_2d(f.flip(0,1)).flip(0,1)
    return numerator, denominator
        
# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): 
        self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio, mask_schedule, overlap,
                        mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio        
        self.mask_schedule = mask_schedule  
        self.overlap = overlap      

    def before_fit(self):
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): 
        self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        if self.mask_schedule:
            mask_ratio = self.mask_schedule + ((self.mask_ratio-self.mask_schedule) * self.epoch)/self.n_epochs
        else:
            mask_ratio = self.mask_ratio
        
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, mask, _ = random_masking(xb_patch, mask_ratio, self.overlap)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        
        self.mask = mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_patch       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
        self.learner.mask = mask.bool()

        
        
    def _loss(self, outs, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        preds1, preds2 = outs[0]
        mask1 = self.mask
        mask2 = ~self.mask
        
        loss1 = (preds1 - target) ** 2
        loss2 = (preds2 - target) ** 2

        loss1 = loss1.mean(dim=-1)
        loss2 = loss2.mean(dim=-1)
        
        loss1 = (loss1 * mask1).sum() / mask1.sum()
        loss2 = (loss2 * mask2).sum() / mask2.sum()
        
        return loss1 + loss2   


def create_patch(xb, patch_len, stride):

    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x

def random_masking(xb, mask_ratio, overlap=None):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
    
    if overlap is not None:
        overlap_margin = int((L/(2-overlap)))
        margin = L - overlap_margin
        noise[:,margin:-margin]=0
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    
    
    return x_masked, x_kept, mask, ids_restore

def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,20,4,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()


