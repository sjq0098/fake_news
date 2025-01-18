import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data_loader(X_train, y_train, X_val, y_val, batch_size=32):
    """
    将训练数据和验证数据转换为Tensor，并创建DataLoader。
    Parameters:
    - X_train: 训练特征数据
    - y_train: 训练标签数据
    - X_val: 验证特征数据
    - y_val: 验证标签数据
    - batch_size: 每个batch的大小
    Returns:
    - train_loader: 训练数据的DataLoader
    - val_loader: 验证数据的DataLoader
    """
    # 转换为Tensor格式
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader