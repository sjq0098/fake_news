import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jieba

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)
from data_factory.data_processing import load_data_txt, jieba_cut_text,veclization

train_file_path = os.path.join(root_dir, "data", "train.txt")
test_file_path = os.path.join(root_dir, "data", "test1.txt")
stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")

train_df = load_data_txt(train_file_path)
test_df = load_data_txt(test_file_path)

train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)
test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

from collections import Counter

# 定义特殊标记
PAD_TOKEN = "<PAD>"  # 填充标记
UNK_TOKEN = "<UNK>"  # 未知词标记

# 统计词频
all_words = [word for text in train_df['cut_text'] for word in text.split()]
word_counts = Counter(all_words)

# 构建词汇表（取前N个高频词）
vocab_size = 100000  # 根据数据量调整
vocab = word_counts.most_common(vocab_size - 2)  # 保留位置给PAD和UNK

# 创建词到索引的映射
word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, _ in vocab:
    word_to_idx[word] = len(word_to_idx)

max_sequence_length = 100  # 根据文本长度分布调整

def text_to_indices(text):
    # 将分词后的文本转为索引列表
    indices = [word_to_idx.get(word, word_to_idx[UNK_TOKEN]) 
               for word in text.split()[:max_sequence_length]]
    # 填充或截断
    if len(indices) < max_sequence_length:
        indices += [word_to_idx[PAD_TOKEN]] * (max_sequence_length - len(indices))
    else:
        indices = indices[:max_sequence_length]
    return indices

# 转换训练集和测试集
X_train_seq = [text_to_indices(text) for text in train_df['cut_text']]
X_test_seq = [text_to_indices(text) for text in test_df['cut_text']]

from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]

# 创建训练集和验证集
train_labels = train_df['label'].astype(int).tolist()
X_train, X_val, y_train, y_val = train_test_split(
    X_train_seq, train_labels, test_size=0.2, random_state=42
)

# 创建Dataset
train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)
test_dataset = TextDataset(X_test_seq)

# 检查词汇表覆盖率
train_words = set(word for text in train_df['cut_text'] for word in text.split())
coverage = len([word for word in train_words if word in word_to_idx]) / len(train_words)
print(f"词汇表覆盖率: {coverage:.2%}")

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 num_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * directions, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, emb)
        output, (hidden, cell) = self.lstm(embedded)
        
        # 拼接双向最后隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# 初始化改进后的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(vocab_size=len(word_to_idx),embed_dim=256,hidden_dim=128,output_dim=len(train_df['label'].unique()) ,bidirectional=True,
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs=15
train_losses, val_accuracies = [], []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # 验证评估
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)
    accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}")

accuracy = accuracy_score(val_labels, val_preds)
print(f"验证集准确率: {accuracy:.4f}")


# Save classification report
result_dir = os.path.join(root_dir, "result")
model_name = "LSTM"
classes = [str(label) for label in sorted(np.unique(val_labels))]
report = classification_report(val_labels, val_preds, target_names=classes)

report_save_path = os.path.join(result_dir, f"{model_name}_classification_report.txt")
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"分类报告已保存到: {report_save_path}")

# Confusion Matrix
cm = confusion_matrix(val_labels, val_preds)
cm_save_path = os.path.join(result_dir, f"{model_name}_confusion_matrix.png")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(cm_save_path)
plt.close()
print(f"混淆矩阵已保存到: {cm_save_path}")

# Plot training curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Training and Validation Performance')
training_curve_path = os.path.join(result_dir, f"{model_name}_training_curve.png")
plt.savefig(training_curve_path)
plt.close()
print(f"训练过程曲线已保存到: {training_curve_path}")

# Test predictions
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch
        inputs = inputs.to(device)  # Move test data to device
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU

test_result_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")
result_df = pd.DataFrame({'id': range(len(test_preds)), 'prediction': test_preds})
result_df.to_csv(test_result_path, index=False, encoding="utf-8")
print(f"测试集预测结果已保存到: {test_result_path}")

# 保存分类报告
result_dir = os.path.join(root_dir, "result")
model_name = "RNN"
classes = [str(label) for label in sorted(np.unique(val_labels))]
report = classification_report(val_labels, val_preds, target_names=classes)

report_save_path = os.path.join(result_dir, f"{model_name}_classification_report.txt")
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"分类报告已保存到: {report_save_path}")

# 混淆矩阵
cm = confusion_matrix(val_labels, val_preds)
cm_save_path = os.path.join(result_dir, f"{model_name}_confusion_matrix.png")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(cm_save_path)
plt.close()
print(f"混淆矩阵已保存到: {cm_save_path}")

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Training and Validation Performance')
training_curve_path = os.path.join(result_dir, f"{model_name}_training_curve.png")
plt.savefig(training_curve_path)
plt.close()
print(f"训练过程曲线已保存到: {training_curve_path}")

# 测试集预测并保存结果
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch
        inputs = inputs.to(device)  # 确保输入数据在正确的设备上
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())  # 将预测结果移回CPU并转换为numpy数组

# 结果保存路径
test_result_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")

# 创建 DataFrame 并保存预测结果
result_df = pd.DataFrame({'id': range(len(test_preds)), 'prediction': test_preds})
result_df.to_csv(test_result_path, index=False, encoding="utf-8")

print(f"测试集预测结果已保存到: {test_result_path}")