import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_recall_curve,auc


f_mean = lambda l: sum(l)/len(l)


def performance_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', 'precision', 'recall', 'ap']

    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)

    return performances_pd


def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def performance(y_true, y_pred, y_prob, print_=True):
    # f00,f01,f10,f11
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()

    # 1.准确率
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    # 2.马修斯相关系数
    try:
        mcc = ((tp * tn) - (fn * fp)) / np.sqrt(np.float((tp + fn) * (tn + fp) * (tp + fp) * (tn + fn)))
    except:
        print('MCC Error: ', (tp + fn) * (tn + fp) * (tp + fp) * (tn + fn))
        mcc = np.nan

    # 3.灵敏度(召回率)
    try:
        sensitivity = tp / (tp + fn)
    except:
        sensitivity = np.nan

    # 4.特异度
    specificity = tn / (tn + fp)

    # 5.召回率
    try:
        recall = tp / (tp + fn)
    except:
        recall = np.nan

    # 6.精确率
    try:
        precision = tp / (tp + fp)
    except:
        precision = np.nan

    # 7.f1分数
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = np.nan

    # AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    # precision,recall
    pre, rec, _ = precision_recall_curve(y_true, y_prob)
    # AP:PR曲线下面积
    ap = auc(rec, pre)

    if print_:
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity,specificity, accuracy,mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|ap={:.4f}'.format(precision, recall, f1, ap))
    return roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, ap