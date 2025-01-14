import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

def save_and_close(fig, filename):
    """保存图像到指定文件并关闭图像"""
    save_dir = os.path.dirname(filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    if save_path:
        save_and_close(fig, save_path)
    else:
        plt.show()

def plot_multiclass_roc(model, X, y, classes, title="Multi-class ROC Curve", save_path=None):
    """
    为多类别问题绘制 One-vs-Rest ROC 曲线：
      - 对每个类别计算 ROC 曲线
      - 计算 macro-average ROC 曲线
    参数:
      model: 已训练的模型，必须支持 predict_proba()
      X: 特征数据
      y: 真实标签（非二值化）
      classes: 类别列表，顺序应与 model.classes_ 保持一致（或可排序的）
      title: 图像标题
      save_path: 保存路径
    """
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    y_score = model.predict_proba(X)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC (AUC = {0:0.2f})'.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    colors = plt.cm.get_cmap('Dark2', n_classes)
    for i, cls in enumerate(classes):
        ax.plot(fpr[i], tpr[i], color=colors(i),
                lw=2, label='ROC of class {0} (AUC = {1:0.2f})'.format(cls, roc_auc[i]))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    if save_path:
        save_and_close(fig, save_path)
    else:
        plt.show()

def plot_multiclass_precision_recall(model, X, y, classes, title="Multi-class Precision-Recall Curve", save_path=None):
    """
    为多类别问题绘制 One-vs-Rest Precision-Recall 曲线：
      - 对每个类别计算 Precision-Recall 曲线，并计算 Average Precision（AP）
      - 计算 macro-average Precision-Recall 曲线
    参数:
      model: 已训练的模型，必须支持 predict_proba()
      X: 特征数据
      y: 真实标签（非二值化）
      classes: 类别列表
      title: 图像标题
      save_path: 保存路径
    """
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    y_score = model.predict_proba(X)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])
    
    all_ap = np.mean(list(average_precision.values()))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap('Dark2', n_classes)
    for i, cls in enumerate(classes):
        ax.plot(recall[i], precision[i], color=colors(i),
                lw=2, label='PR of class {0} (AP = {1:0.2f})'.format(cls, average_precision[i]))
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title + f'\nMacro-average AP = {all_ap:0.2f}')
    ax.legend(loc="best")
    ax.grid(True)
    
    if save_path:
        save_and_close(fig, save_path)
    else:
        plt.show()


