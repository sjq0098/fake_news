import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)
from utils.visualization import plot_confusion_matrix
from data_factory.data_processing import load_data_txt, jieba_cut_text, veclization
from data_factory.data_loader import prepare_data_loader

train_file_path = os.path.join(root_dir, "data", "train.txt")
test_file_path = os.path.join(root_dir, "data", "test1.txt")
stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")

train_df = load_data_txt(train_file_path)
print("[INFO] 训练数据预览:")
print(train_df.head())
train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

test_df = load_data_txt(test_file_path)
print("\n[INFO] 测试数据预览:")
print(test_df.head())
test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

X_train, y_train, tfidf_vec = veclization(train_df, text_col="cut_text",
                                          label_col="label", max_features=5000,
                                          ngram_range=(1, 1), stop_words=None)
y_tr = [int(label) for label in y_train if pd.notna(label)]
# 划分训练集和验证集
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_tr, test_size=0.2, random_state=42)
# 确保标签是整数列表

y_val = [int(label) for label in y_val if pd.notna(label)]
# 转换为PyTorch的Tensor格式
X_tr_tensor = torch.tensor(X_tr.toarray(), dtype=torch.float32)
y_tr_tensor = torch.tensor(np.array(y_tr), dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.long)

# 创建DataLoader
train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 不需要 softmax，因为 CrossEntropyLoss 已经包含 softmax

model = SimpleNN(input_dim=X_tr_tensor.shape[1], output_dim=len(np.unique(y_tr)))
criterion = nn.CrossEntropyLoss()  # 分类问题需要使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")

# 在验证集上进行预测
model.eval()
val_preds = []
val_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        val_preds.extend(predicted.numpy())
        val_labels.extend(labels.numpy())

# 输出验证集准确率
print("验证集准确率:", accuracy_score(val_labels, val_preds))

# 保存分类报告
result_dir = os.path.join(root_dir, "result")
model_name = "SimpleNN"

classes = [str(label) for label in sorted(np.unique(y_tr))]
report = classification_report(val_labels, val_preds, target_names=classes)

report_save_path = os.path.join(result_dir, f"{model_name}_classification_report.txt")  # 修正格式化字符串
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"分类报告已保存到: {report_save_path}")

# 混淆矩阵
cm_save_path = os.path.join(result_dir, f"{model_name}_confusion_matrix.png")  # 修正格式化字符串
plot_confusion_matrix(val_labels, val_preds, classes=classes, title="Confusion Matrix", save_path=cm_save_path)

# 对测试集进行预测
X_test = tfidf_vec.transform(test_df["cut_text"].tolist())
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_pred_labels = torch.max(test_outputs, 1)

# 保存测试集预测结果
if "id" in test_df.columns:
    result_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_pred_labels.numpy()
    })
else:
    result_df = pd.DataFrame({'prediction': test_pred_labels.numpy()})

test_result_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")  # 修正格式化字符串
result_df.to_csv(test_result_path, index=False, encoding="utf-8")
print("测试集预测结果已保存到:", test_result_path)

# 绘制学习曲线
learning_curve_save_path = os.path.join(result_dir, f"{model_name}_learning_curve.png")  # 修正格式化字符串

# 绘制训练过程的学习曲线
train_sizes = np.linspace(0.1, 1.0, 5)
train_accuracies = []
val_accuracies = []

for size in train_sizes:
    # 用不同的训练集大小训练模型
    X_train_subset, _, y_train_subset, _ = train_test_split(X_tr, y_tr, train_size=size, random_state=42)
    X_train_tensor = torch.tensor(X_train_subset.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_subset, dtype=torch.long)
    train_dataset_subset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader_subset = DataLoader(train_dataset_subset, batch_size=32, shuffle=True)
    
    # 重新初始化模型和优化器
    model_subset = SimpleNN(input_dim=X_tr_tensor.shape[1], output_dim=len(np.unique(y_train)))
    optimizer_subset = optim.Adam(model_subset.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model_subset.train()
        for inputs, labels in train_loader_subset:
            optimizer_subset.zero_grad()
            outputs = model_subset(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_subset.step()
    
    # 在训练集和验证集上评估准确率
    model_subset.eval()
    with torch.no_grad():
        train_preds_subset = []
        for inputs, labels in train_loader_subset:
            outputs = model_subset(inputs)
            _, predicted = torch.max(outputs, 1)
            train_preds_subset.extend(predicted.numpy())
        train_accuracy = accuracy_score(y_train_subset, train_preds_subset)
        train_accuracies.append(train_accuracy)
        
        val_preds_subset = []
        for inputs, labels in val_loader:
            outputs = model_subset(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds_subset.extend(predicted.numpy())
        val_accuracy = accuracy_score(y_val, val_preds_subset)
        val_accuracies.append(val_accuracy)

pig_path=os.path.join(result_dir, f"{model_name}_learning_curve.png")

# 日志字符串
log_str = '''
Epoch [1], Loss: 0.5999
Epoch [2], Loss: 0.4815
Epoch [3], Loss: 0.7088
Epoch [4], Loss: 0.5332
Epoch [5], Loss: 0.1308
Epoch [6], Loss: 0.4640
Epoch [7], Loss: 0.9849
Epoch [8], Loss: 0.6427
Epoch [9], Loss: 0.0547
Epoch [10], Loss: 0.0539
Epoch [11], Loss: 0.3556
Epoch [12], Loss: 0.6814
Epoch [13], Loss: 0.3439
Epoch [14], Loss: 0.3905
Epoch [15], Loss: 0.1139
'''

# 按行分割日志
lines = log_str.strip().split('\n')

# 提取损失值
losses = []
for line in lines:
    parts = line.split()
    loss = float(parts[-1])
    losses.append(loss)

# 生成epoch列表
epochs = range(1, len(losses) + 1)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(epochs)
plt.show()

plt.savefig(pig_path)