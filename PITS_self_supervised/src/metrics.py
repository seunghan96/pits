
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)

def acc(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    accuracy = (y_pred==y_true).float().mean()
    print('accuracy',accuracy)
    return accuracy
    
def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    print('accuracy',acc)
    return acc
    
def weighted_f1_score(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    print('Weighted F1 score:', f1)
    return f1

def micro_f1_score(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro F1 score:', f1)
    return f1

def macro_f1_score(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Macro F1 score:', f1)
    return f1

def precision(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    prec = precision_score(y_true, y_pred, average='weighted')
    print('Precision:', prec)
    return prec

def recall(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rec = recall_score(y_true, y_pred, average='weighted')
    print('Recall:', rec)
    return rec
    
#[acc,weighted_f1_score,micro_f1_score,macro_f1_score,precision,recall]