import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ctypes

# 禁用torchvision的beta警告（可选）
# from torchvision import disable_beta_transforms_warning
# disable_beta_transforms_warning()

# 系统防休眠设置（Windows专用）
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)  # 阻止系统休眠

# 环境优化配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer多线程
torch.backends.cudnn.benchmark = True           # 启用加速算法
torch.backends.cuda.matmul.allow_tf32 = True    # 启用TF32加速

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)
from data_factory.data_processing import load_data_txt

# 超参数配置
class Config:
    max_sequence_length = 128      # 文本最大长度
    batch_size = 16                # 批量大小
    grad_accum_steps = 2           # 梯度累积步数
    epochs = 5                     
    learning_rate = 2e-5
    warmup_ratio = 0.1             # 学习率预热比例
    max_grad_norm = 1.0            # 梯度裁剪阈值

# 数据加载
def load_datasets():
    train_df = load_data_txt(os.path.join(root_dir, "data", "train.txt"))
    test_df = load_data_txt(os.path.join(root_dir, "data", "test1.txt"))
    
    # 数据集划分
    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # 数据编码
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    encode = lambda texts: tokenizer(
        texts, 
        padding='max_length', 
        truncation=True, 
        max_length=Config.max_sequence_length, 
        return_tensors="pt"
    )
    
    return (
        encode(X_train), torch.tensor(y_train),
        encode(X_val), torch.tensor(y_val),
        encode(test_df['text'].tolist()),
        tokenizer
    )

# 优化数据集类
class OptimizedBERTDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx] if self.labels is not None else -1
        }

# 模型初始化（移除torch.compile）
def initialize_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    
    # 冻结前6层（平衡速度与性能）
    for layer in model.bert.encoder.layer[:6]:
        for param in layer.parameters():
            param.requires_grad = False
            
    return model

def save_results(model_name, val_labels, val_preds, test_preds, result_dir):
    """统一保存所有结果"""
    os.makedirs(result_dir, exist_ok=True)
    
    # 分类报告
    classes = [str(label) for label in sorted(np.unique(val_labels))]
    report = classification_report(val_labels, val_preds, target_names=classes)
    report_path = os.path.join(result_dir, f"{model_name}_classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分类报告已保存到: {report_path}")

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(val_labels, val_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(result_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {cm_path}")

    # 测试结果
    test_df = pd.DataFrame({
        'id': range(len(test_preds)),
        'prediction': test_preds
    })
    test_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    print(f"测试集预测结果已保存到: {test_path}")

def optimized_training():
    # 加载数据
    (train_encodings, y_train, 
     val_encodings, y_val, 
     test_encodings, tokenizer) = load_datasets()
    
    # 创建数据集
    train_dataset = OptimizedBERTDataset(train_encodings, y_train)
    val_dataset = OptimizedBERTDataset(val_encodings, y_val)
    test_dataset = OptimizedBERTDataset(test_encodings)

    # 优化数据加载配置
    loader_args = {
        'batch_size': Config.batch_size,
        'num_workers': min(4, os.cpu_count()),  # 根据CPU核心数调整
        'pin_memory': True,
        'persistent_workers': True
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, **loader_args)
    test_loader = DataLoader(test_dataset, **loader_args)

    # 初始化模型
    model = initialize_model(len(torch.unique(y_train)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 优化器配置
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.learning_rate,
        correct_bias=False
    )
    
    # 学习率预热
    total_steps = len(train_loader) * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # 数据准备
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            
            # 混合精度前向
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs).logits
                loss = criterion(outputs, labels) / Config.grad_accum_steps
                
            # 梯度累积反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if (step + 1) % Config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            total_loss += loss.item() * Config.grad_accum_steps
            
            # 系统心跳保持（每50步）
            if step % 50 == 0:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
            # 在训练循环内添加
            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # 验证流程
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                outputs = model(**inputs).logits
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        # 输出验证结果
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # 测试集预测
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs).logits
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # 保存结果
    result_dir = os.path.join(root_dir, "result")
    save_results("OptimizedBERT", val_labels, val_preds, test_preds, result_dir)

if __name__ == "__main__":
    optimized_training()