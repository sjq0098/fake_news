import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))  # 修改为项目的根目录
sys.path.append(root_dir)

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from data_factory.data_processing import load_data_txt, jieba_cut_text, veclization

class TextDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {'text': self.X[idx].squeeze()}  # 直接使用Tensor，无需toarray()
        if self.y is not None:  # 如果有标签，返回标签
            data['label'] = self.y[idx]
        return data
    
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {'text': self.sequences[idx], 'label': self.labels[idx]}
        else:
            return {'text': self.sequences[idx]}
    
def prepare_data_loader(train_file_path, test_file_path, stop_word_path, max_features=5000, batch_size=32):
    # 加载和处理数据
    train_df = load_data_txt(train_file_path)
    test_df = load_data_txt(test_file_path)
    train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)
    test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)
    
    # 向量化
    tfidf_vec = TfidfVectorizer(max_features=max_features)
    X_train = tfidf_vec.fit_transform(train_df['cut_text'])
    X_test = tfidf_vec.transform(test_df['cut_text'])
    
    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    
    # 划分训练集和验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 转换为Tensor
    X_tr_tensor = torch.tensor(X_tr.toarray(), dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    
    # 创建Dataset
    train_dataset = TextDataset(X_tr_tensor, y_tr_tensor)
    val_dataset = TextDataset(X_val_tensor, y_val_tensor)
    test_dataset = TextDataset(X_test_tensor)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("数据集已经准备完成")
    return train_loader, val_loader, test_loader, tfidf_vec, le


from sklearn.model_selection import train_test_split
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import LabelEncoder

def prepare_sequence_data_loader(train_file_path, test_file_path, stop_word_path, max_seq_length=200, batch_size=32):
    # 加载数据
    train_df = load_data_txt(train_file_path)
    test_df = load_data_txt(test_file_path)
    
    # 分词
    train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)
    test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)
    
    # 分割训练集和验证集
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 构建词汇表
    word_to_idx = {'<PAD>': 0}
    for tokens in train_df['cut_text']:
        for token in tokens:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)
    
    # 将文本转换为词索引序列
    def text_to_indices(text_list, word_to_idx, max_seq_length):
        sequences = []
        for tokens in text_list:
            indices = [word_to_idx[token] for token in tokens if token in word_to_idx]
            if len(indices) < max_seq_length:
                indices += [0] * (max_seq_length - len(indices))
            else:
                indices = indices[:max_seq_length]
            sequences.append(torch.tensor(indices, dtype=torch.long))
        return sequences
    
    train_sequences = text_to_indices(train_df['cut_text'], word_to_idx, max_seq_length)
    val_sequences = text_to_indices(val_df['cut_text'], word_to_idx, max_seq_length)
    test_sequences = text_to_indices(test_df['cut_text'], word_to_idx, max_seq_length)
    
    # 标签编码
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['label'])
    val_labels = le.transform(val_df['label'])
    test_labels = le.transform(test_df['label'])
    
    # 创建Dataset
    train_dataset = TensorDataset(torch.stack(train_sequences), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.stack(val_sequences), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.stack(test_sequences), torch.tensor(test_labels, dtype=torch.long))
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, word_to_idx, le