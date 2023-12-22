
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import random
from src.models.PITS import PITS
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cls', type=int, default=0, help='classification or not')
parser.add_argument('--pretrain_task', type=str, default='PI', help='PI vs. PD')
parser.add_argument('--CI', type=int, default=1, help='channel independence or not')

parser.add_argument('--instance_CL', type=int, default=0, help='Instance-wise contrastive learning')
parser.add_argument('--temporal_CL', type=int, default=1, help='Temporal contrastive learning')

parser.add_argument('--is_finetune_cls', type=int, default=0, help='(classification) do finetuning or not')
parser.add_argument('--is_finetune', type=int, default=0, help='(forecasting) do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')

# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='etth1', help='pretrain dataset name')
parser.add_argument('--dset_finetune', type=str, default='etth1', help='finetune dataset name')

parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--num_patches', type=int, default=42, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
parser.add_argument('--mean_norm', type=int, default=0, help='reversible instance normalization')
parser.add_argument('--mean_norm_for_cls', type=int, default=0)

# Model args
parser.add_argument('--d_model', type=int, default=128, help='hidden dimension of MLP')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.5, help='masking ratio for the input')
parser.add_argument('--mask_schedule', type=float, default=0, help='mask_schedule')

# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--n_epochs_load', type=int, default=100, help='number of loading pre-training epochs')

parser.add_argument('--n_epochs_finetune_head', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_finetune_entire', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# Device Id
parser.add_argument('--device_id', type=int, default=7, help='Device ID')
parser.add_argument('--seed', type=int, default=1, help='Random Seed')

parser.add_argument('--aggregate', type=str, default='max')



args = parser.parse_args()

print('args:', args)
assert args.aggregate in ['max','avg','concat']

assert args.pretrain_task in ['PI', 'PD']
########################################################################
# (1) Pretrain weight path
########################################################################
pretain_path1 = f'saved_models/{args.dset_pretrain}/PITS_{args.pretrain_task}/based_model'
pretain_path2 = f'PITS_pretrained_D{args.d_model}_cw{args.context_points}_patch{args.patch_len}_stride{args.stride}_epochs-pretrain{args.n_epochs_pretrain}_mask{args.mask_ratio}_model1'
if args.CI:
    pretain_path2 += '_CI'
else:
    pretain_path2 += '_CD' 

if args.mean_norm:
    pretain_path2 += '_mean_norm' 
    
args.pretrained_model = os.path.join(pretain_path1, pretain_path2, 
                                     f'{pretain_path2}_{args.n_epochs_pretrain}.pth')

########################################################################
# (2) Fineteune path
########################################################################        
args.ft_path = f'saved_models/{args.dset_pretrain}2{args.dset_finetune}/PITS_{args.pretrain_task}/{args.model_type}/{args.aggregate}/' 
args.ft_path = os.path.join(args.ft_path, pretain_path2) + '/'
if not os.path.exists(args.ft_path): 
    os.makedirs(args.ft_path)


suffix_name = '_ep' + str(args.n_epochs_finetune_entire) + '_model' + str(args.finetuned_model_id) 

if args.is_finetune or args.is_finetune_cls: 
    args.save_finetuned_model = 'tw'+str(args.target_points) + '_ft'+ suffix_name + f'_load_ep{args.n_epochs_pretrain}'
    args.ft = True
elif args.is_linear_probe: 
    args.save_finetuned_model = 'tw'+str(args.target_points) +'_lp'+ suffix_name + f'_load_ep{args.n_epochs_pretrain}'
    args.ft = False

if args.mean_norm_for_cls:
    args.save_finetuned_model += 'mean_norm_cls'

########################################################################        
set_device(args.device_id)

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    if args.ft:
        num_patch = args.num_patches
    else:
        num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)

    # get model
    model = PITS(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                d_model=args.d_model,
                shared_embedding=args.CI,
                head_dropout=args.head_dropout,
                head_type=head_type,
                aggregate=args.aggregate,
                ft =args.ft,
                mean_norm_for_cls = args.mean_norm_for_cls,
                )    
    
    if weight_path: 
        model = transfer_weights(weight_path, model, exclude_head=True)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    #--------------------------------------------------------#
    # (1) Dataloader & Model
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    
    #--------------------------------------------------------#
    # (2) Transfer Weight
    model = transfer_weights(args.pretrained_model, model, exclude_head=True)
    
    #--------------------------------------------------------#
    # (3) Define Loss function
    if head_type=='classification':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_func = nn.MSELoss(reduction='mean')
    
    #--------------------------------------------------------#
    # (4) Call backs
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
    #--------------------------------------------------------#
    # (5) Define Learner
    if head_type=='classification':
        learn = Learner(args, dls, model, loss_func, lr=args.lr, 
                        cbs=cbs, metrics=[acc], ft = True)        
    else:
        learn = Learner(args, dls, model, loss_func, lr=args.lr, 
                        cbs=cbs, ft = True)                        
    
    #--------------------------------------------------------#
    # (6) Fit the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.ft_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction')
    model = transfer_weights(args.pretrained_model, model, exclude_head=True)
    loss_func = nn.MSELoss(reduction='mean')   
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.ft_path)
        ]
    learn = Learner(args, dls, model, loss_func, lr=lr, 
                    cbs=cbs, metrics=[mse], ft=True)                            
    learn.fine_tune(n_epochs=args.n_epochs_finetune_entire, base_lr=lr, freeze_epochs=args.n_epochs_finetune_head)
    save_recorders(learn)


def finetune_cls_func(lr=args.lr):
    print('end-to-end finetuning')
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='classification')
    model = transfer_weights(args.pretrained_model, model, exclude_head=True)
    loss_func = nn.CrossEntropyLoss(reduction='mean')   
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.ft_path)
        ]
    learn = Learner(args, dls, model, loss_func, lr=lr, 
                    cbs=cbs, metrics=[acc],ft=True)                            
    learn.fine_tune(n_epochs=args.n_epochs_finetune_entire, base_lr=lr, freeze_epochs=args.n_epochs_finetune_head)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction')
    model = transfer_weights(args.pretrained_model, model, exclude_head=True)
    loss_func = nn.MSELoss(reduction='mean')    
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.ft_path)
        ]
    learn = Learner(args, dls, model, loss_func, 
                    lr=lr, cbs=cbs,metrics=[mse], ft=True)                            
    learn.linear_probe(n_epochs=args.n_epochs_finetune_entire, base_lr=lr)
    save_recorders(learn)


def test_func_cls(args, weight_path,head_type):
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type).to('cuda')
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    loss_func = nn.CrossEntropyLoss(reduction='mean')   
    learn = Learner(args, dls, model, loss_func = loss_func,cbs=cbs,ft = True)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', 
                      scores=[accuracy,weighted_f1_score,micro_f1_score,macro_f1_score,precision,recall])
    _, _, score = out # preds, targets, score
    print('score:', score)
    df = pd.DataFrame(np.array(score).reshape(1,-1), 
                      columns=['acc','weighted_F1','micro_F1',
                               'macro_F1','precision','recall'])
    df.to_csv(args.ft_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out


def test_func(args, weight_path,head_type):
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type).to('cuda')
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(args, dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse,mae]) 
    _, _, score = out # preds, targets, score
    print('score:', score)
    df = pd.DataFrame(np.array(score).reshape(1,-1), columns=['mse','mae'])
    df.to_csv(args.ft_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':
        
    if args.is_finetune:
        print('-'*50)
        print('(Option 1) Fine-tune + Test')
        print('-'*50)
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        finetune_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args, args.ft_path+args.save_finetuned_model,head_type='prediction')         
        print('----------- Complete! -----------')

    elif args.is_finetune_cls:
        print('-'*50)
        print('(Option 1) Fine-tune + Test')
        print('-'*50)
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='classification')        
        finetune_cls_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func_cls(args, args.ft_path+args.save_finetuned_model,head_type='classification')         
        print('----------- Complete! -----------')
        
    elif args.is_linear_probe:
        print('-'*50)
        print('(Option 2) Linear Probing + Test')
        print('-'*50)
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args, args.ft_path+args.save_finetuned_model,head_type='prediction')        
        print('----------- Complete! -----------')

    else:
        print('-'*50)
        print('(Option 3) Test')
        print('-'*50)
        args.dset = args.dset_finetune
        weight_path = args.ft_path+args.dset_finetune+'_pits_finetuned'+suffix_name
        # Test
        out = test_func(args, weight_path,head_type='prediction')        
        print('----------- Complete! -----------')