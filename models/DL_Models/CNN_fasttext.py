import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jieba
from collections import Counter
from gensim.models import KeyedVectors

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)
from data_factory.data_processing import load_data_txt, jieba_cut_text

# 配置参数
class Config:
    w2v_model_path = os.path.join(root_dir, "models", "cc.zh.300.vec")
    vocab_size = 100000
    max_sequence_length = 100
    embed_dim = 300  # 需与预训练模型维度一致
    num_filters = 200
    filter_sizes = [2, 3, 4]
    batch_size = 64
    lr = 0.001
    epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特殊标记
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def main():
    config = Config()
    
    # 加载数据
    train_df = load_data_txt(os.path.join(root_dir, "data", "train.txt"))
    test_df = load_data_txt(os.path.join(root_dir, "data", "test1.txt"))

    # 标签预处理
    # 移除空标签并转换类型
    train_df = train_df.dropna(subset=['label'])
    train_df['label'] = train_df['label'].astype(int)
    
    # 创建标签映射
    unique_labels = sorted(train_df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    output_dim = len(unique_labels)
    train_df['label'] = train_df['label'].map(label_mapping)

    # 分词处理
    stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")
    train_df = jieba_cut_text(train_df, "text", "cut_text", stop_word_path)
    test_df = jieba_cut_text(test_df, "text", "cut_text", stop_word_path)

    # 加载预训练词向量
    print("正在加载预训练词向量...")
    w2v_model = KeyedVectors.load_word2vec_format(config.w2v_model_path, binary=False)
    
    # 构建词汇表
    all_words = [word for text in train_df['cut_text'] for word in text.split()]
    word_counts = Counter(all_words)
    vocab = word_counts.most_common(config.vocab_size - 2)
    
    # 创建嵌入矩阵
    embedding_matrix = np.zeros((config.vocab_size, config.embed_dim))
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    
    for word, _ in vocab:
        if word in w2v_model:
            embedding_matrix[len(word_to_idx)] = w2v_model[word]
        else:
            np.random.seed(len(word_to_idx))
            embedding_matrix[len(word_to_idx)] = np.random.normal(scale=0.6, size=(config.embed_dim,))
        word_to_idx[word] = len(word_to_idx)
    
    # 转换为PyTorch张量
    embedding_matrix = torch.FloatTensor(embedding_matrix)

    # 文本转索引函数
    def text_to_indices(text):
        indices = [word_to_idx.get(word, word_to_idx[UNK_TOKEN]) 
                  for word in text.split()[:config.max_sequence_length]]
        if len(indices) < config.max_sequence_length:
            indices += [word_to_idx[PAD_TOKEN]] * (config.max_sequence_length - len(indices))
        return indices[:config.max_sequence_length]

    # 创建数据集
    class TextDataset(Dataset):
        def __init__(self, sequences, labels=None):
            self.sequences = torch.LongTensor(sequences)
            self.labels = torch.LongTensor(labels) if labels is not None else None
            
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            if self.labels is not None:
                return self.sequences[idx], self.labels[idx]
            return self.sequences[idx]

    # 数据转换
    X_train_seq = [text_to_indices(text) for text in train_df['cut_text']]
    X_test_seq = [text_to_indices(text) for text in test_df['cut_text']]
    y_train = train_df['label'].tolist()

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_seq, y_train, test_size=0.2, random_state=42
    )

    # 创建DataLoader
    train_loader = DataLoader(TextDataset(X_train, y_train), 
                             batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), 
                           batch_size=config.batch_size)
    test_loader = DataLoader(TextDataset(X_test_seq), 
                            batch_size=config.batch_size)

    # 定义CNN模型
    class CNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, 
                                                         padding_idx=0,
                                                         freeze=False)
            self.convs = nn.ModuleList([
                nn.Conv1d(config.embed_dim, config.num_filters, fs)
                for fs in config.filter_sizes
            ])
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), output_dim)
            
        def forward(self, x):
            embedded = self.embedding(x).permute(0, 2, 1)
            conved = [torch.relu(conv(embedded)) for conv in self.convs]
            pooled = [torch.max(conv, dim=2)[0] for conv in conved]
            cat = self.dropout(torch.cat(pooled, dim=1))
            return self.fc(cat)

    # 初始化模型
    model = CNNClassifier().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    train_losses, val_accs = [], []
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证评估
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs.cpu(), dim=1)
                val_preds.extend(preds.numpy())
                val_true.extend(labels.numpy())
        
        train_loss = total_loss / len(train_loader)
        val_acc = accuracy_score(val_true, val_preds)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(root_dir, "models", "best_cnn_model.pth"))

    # 生成报告
    result_dir = os.path.join(root_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 分类报告
    report = classification_report(val_true, val_preds, 
                                  target_names=[str(k) for k in label_mapping.keys()])
    with open(os.path.join(result_dir, "CNN_classification_report.txt"), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 混淆矩阵
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(val_true, val_preds), 
                annot=True, fmt='d', 
                xticklabels=label_mapping.keys(),
                yticklabels=label_mapping.keys())
    plt.savefig(os.path.join(result_dir, "CNN_confusion_matrix.png"))
    
    # 测试集预测
    model.load_state_dict(torch.load(os.path.join(root_dir, "models", "best_cnn_model.pth")))
    model.eval()
    test_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.cpu(), dim=1)
            test_preds.extend(preds.numpy())
    
    # 映射回原始标签
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    final_preds = [inverse_mapping[p] for p in test_preds]
    
    result_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else test_df.index,
        'prediction': final_preds
    })
    result_df.to_csv(os.path.join(result_dir, "CNN_test_predictions.csv"), index=False)

if __name__ == "__main__":
    main()