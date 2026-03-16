import numpy as np
import torch.nn as nn
import torch
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)




def RMSE(pred, true):

    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))


def MSPE(pred, true):
    # return np.mean(np.square((pred - true) / (true + 1e-8)))
    return np.mean(np.abs((pred - true) / true)) * 100

def SMAPE(pred, true):
    # return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))
    return 2.0 * np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true))) * 100

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))



def accuracy(pred, true):
    """
    计算分类准确率
    Args:
        pred: 预测标签 (0或1), shape=(n_samples,)
        true: 真实标签 (0或1), shape=(n_samples,)
    Returns:
        acc: 准确率 [0, 1]
    """
    return np.mean(pred == true)

def precision(pred, true):
    """
    计算精准率（Precision）
    Args:
        pred: 预测标签 (0或1), shape=(n_samples,)
        true: 真实标签 (0或1), shape=(n_samples,)
    Returns:
        prec: 精准率 [0, 1]
    """
    tp = np.sum((pred == 1) & (true == 1))  # 真阳性
    fp = np.sum((pred == 1) & (true == 0))  # 假阳性
    return tp / (tp + fp + 1e-8)  # 防止分母为0

def recall(pred, true):
    """
    计算召回率（Recall）
    Args:
        pred: 预测标签 (0或1), shape=(n_samples,)
        true: 真实标签 (0或1), shape=(n_samples,)
    Returns:
        rec: 召回率 [0, 1]
    """
    tp = np.sum((pred == 1) & (true == 1))  # 真阳性
    fn = np.sum((pred == 0) & (true == 1))  # 假阴性
    return tp / (tp + fn + 1e-8)  # 防止分母为0

def f1_score(pred, true):
    """
    计算F1-score
    Args:
        pred: 预测标签 (0或1), shape=(n_samples,)
        true: 真实标签 (0或1), shape=(n_samples,)
    Returns:
        f1: F1-score [0, 1]
    """
    prec = precision(pred, true)
    rec = recall(pred, true)
    return 2 * (prec * rec) / (prec + rec + 1e-8)  # 防止分母为0
def R2(pred, true):
    """
    计算决定系数 R²（R-squared）
    Args:
        pred: 预测值, shape=(n_samples,)
        true: 真实值, shape=(n_samples,)
    Returns:
        r2: R²值，范围通常在(-∞, 1]，1表示完美预测
    """
    # 总平方和 (Total Sum of Squares)
    tss = np.sum((true - true.mean()) ** 2)
    # 残差平方和 (Residual Sum of Squares)
    rss = np.sum((true - pred) ** 2)
    # R² = 1 - RSS/TSS
    return 1 - (rss / (tss + 1e-8))  # 防止分母为0

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)
    acc=accuracy(pred,true)
    pre=precision(pred,true)
    re=recall(pred,true)
    F1=f1_score(pred,true)

    return mae, mse, rmse, smape,r2
