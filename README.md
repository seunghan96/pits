# Patch Independence for Time Series

### Seunghan Lee, Taeyoung Park, Kibok Lee

<br>

This repository contains the official implementation for the paper [Patch Independence for Time Series](https://arxiv.org/abs/xxxxx) 

This work is accepted in [NeurIPS 2023 Workshop: Self-Supervised Learning - Theory and Practice](https://sslneurips23.github.io/), and has been selected for an oral presentation.

<br>

# 0. Dataset

## (1) TS forecasting

Download datasets according to **PatchTST**

Put the data files under

- `\PITS_self_supervised\data\`
- `\PITS_supervised\data\`

<br>

## (2) TS classification

Download datasets according to xxxxx

Put the data files under

- `\PITS_self_supervised\data\`

<br>

<br>

# 1. Self-supervised PITS

## (1) TS forecasting

### Dataset & Hyperparameters

```python
ds_pretrain = 'etth1'
ds_finetune = 'etth1'

# (1) Model Size
d_model = 128

# (2) Input Size
context_points = 512
patch_len = stride = 12
num_patches = context_points//patch_len

# (3) Finetune Epoch
ep_ft_head = 5
ep_ft_entire = ep_ft_head * 2
```

<br>

### 1) Pretrain

```python
!python PITS_pretrain.py --dset_pretrain {ds_pretrain} \
    --context_points {context_points} --d_model {d_model} --patch_len {patch_len} --stride {stride} \
```

<br>

### 2) Finetune

```python
for pred_len in [96, 192, 336, 720]:
  !python PITS_finetune.py --dset_pretrain {ds_pretrain} --dset_finetune {ds_finetune} \
    --n_epochs_finetune_head {ep_ft_head} --n_epochs_finetune_entire {ep_ft_entire} \
    --target_points {pred_len} --num_patches {num_patches} --context_points {context_points} \
    --d_model {d_model} --patch_len {patch_len} --stride {stride} \
      --is_finetune 1 
```

<br>

## (2) TS classification

### Dataset & Hyperparameters

```python
# ep_pretrain = xx
# ep_ft_head = xx
# ep_ft_entire = ep_ft_head * 2
# d_model = xx
# patch_len = stride = xx
# aggregate = xx

context_points = 176
num_patches = int(cp/stride)
batch_size = 128

# ft_data_length = xx
# num_classes = xx
```

```
ds_pretrain = 'SleepEEG'
ds_finteune = 'Epilepsy' # ['Epilepsy','FD_B','Gesture','EMG']
```

<br>

### 1) Pretrain

```python
!python PITS_pretrain.py --dset_pretrain {ds_pretrain} \
    --n_epochs_pretrain {ep_pretrain}  --context_points {context_points} \
	--d_model {d_model} --patch_len {patch_len} --stride {stride} 
```

<br>

### 2) Finetune

```python
!python PITS_finetune.py --dset_pretrain {ds_pretrain} --dset_finetune {ds_finetune} \
    --n_epochs_finetune_head {ep_ft_head} --n_epochs_finetune_entire {ep_ft_entire} \
    --target_points {num_classes} --num_patches {num_patches} --context_points {context_points} \
    --d_model {d_model} --patch_len {patch_len} --stride {stride} --aggregate {aggregate} \
    --is_finetune_cls 1 --cls 1 
```

<br>

<br>

# 2. Supervised PITS

Refer to `scripts/`

<br>

# Contact

If you have any questions, please contact **seunghan9613@yonsei.ac.kr**

<br>

# Acknowledgement

We appreciate the following github repositories for their valuable code base & datasets:

- https://github.com/yuqinie98/PatchTST