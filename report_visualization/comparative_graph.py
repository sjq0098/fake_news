import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_classification_report(file_path):
    """解析分类报告txt文件为结构化数据"""
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 提取数据部分
    data_lines = [line.split() for line in lines[2:-3]]
    columns = lines[1].split()
    
    # 构建DataFrame
    df = pd.DataFrame(data_lines, columns=columns).apply(pd.to_numeric)
    df['class'] = df['class'].astype(int)
    
    # 提取整体指标
    accuracy = float(lines[-3].split()[-1])
    macro_avg = list(map(float, lines[-2].split()[-3:]))
    weighted_avg = list(map(float, lines[-1].split()[-3:]))
    
    return df, accuracy, macro_avg, weighted_avg

def plot_class_metrics(df, model_name, ax=None):
    """绘制每个类别的三指标柱状图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    
    df.plot(x='class', y=['precision', 'recall', 'f1-score'],
            kind='bar', ax=ax, alpha=0.8)
    ax.set_title(f'{model_name} - Per Class Metrics')
    ax.set_xticklabels(df['class'], rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    return ax

def plot_heatmap(df, model_name, ax=None):
    """绘制热力图显示各类别指标"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = df[['precision', 'recall', 'f1-score']].T
    sns.heatmap(metrics, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar=False, ax=ax)
    ax.set_title(f'{model_name} - Metrics Heatmap')
    ax.set_xticklabels(df['class'], rotation=45)
    ax.set_yticklabels(['Precision', 'Recall', 'F1'])
    return ax

def plot_macro_comparison(all_reports, ax=None):
    """对比不同模型的宏观指标"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    data = []
    for name, (_, _, macro, _) in all_reports.items():
        models.append(name)
        data.append(macro)
    
    df = pd.DataFrame(data, index=models, columns=['Precision', 'Recall', 'F1'])
    df.plot(kind='bar', ax=ax, rot=45, alpha=0.8)
    ax.set_title('Macro Average Comparison')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    return ax

def visualize_reports(report_paths):
    """主可视化函数"""
    all_reports = {}
    
    # 解析所有报告
    for path in report_paths:
        model_name = Path(path).stem
        df, accuracy, macro, weighted = parse_classification_report(path)
        all_reports[model_name] = (df, accuracy, macro, weighted)
    
    # 创建画布
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 绘制各模型详细指标
    for idx, (name, (df, _, _, _)) in enumerate(all_reports.items()):
        ax1 = fig.add_subplot(gs[idx, 0])
        plot_class_metrics(df, name, ax1)
        
        ax2 = fig.add_subplot(gs[idx, 1])
        plot_heatmap(df, name, ax2)
    
    # 绘制宏观对比
    ax_comp = fig.add_subplot(gs[2, :])
    plot_macro_comparison(all_reports, ax_comp)
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 假设有多个报告文件，例如：
    # report_files = ['CNN_report.txt', 'RNN_report.txt', 'Transformer_report.txt']
    report_files = ['CNN_classification_report.txt']  # 示例单个文件
    visualize_reports(report_files)