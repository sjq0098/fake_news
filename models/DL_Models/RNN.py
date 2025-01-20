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
from data_factory.data_processing import load_data_txt, jieba_cut_text

train_file_path = os.path.join(root_dir, "data", "train.txt")
test_file_path = os.path.join(root_dir, "data", "test1.txt")
stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")

# Load and tokenize data
train_df = load_data_txt(train_file_path)
train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

test_df = load_data_txt(test_file_path)
test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

# Build vocabulary
def build_vocabulary(train_texts, max_features=200):
    word_freq = {}
    for tokens in train_texts:
        for token in tokens:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [word for word, freq in sorted_words[:max_features]]
    vocab_to_idx = {word: idx + 2 for idx, word in enumerate(vocab)}
    vocab_to_idx['<PAD>'] = 0
    vocab_to_idx['<UNK>'] = 1
    return vocab_to_idx

vocab_to_idx = build_vocabulary(train_df['cut_text'], max_features=5000)

# Convert text to sequences
def text_to_sequence(texts, vocab_to_idx, max_seq_length=5000):
    sequences = []
    for tokens in texts:
        seq = []
        for token in tokens:
            if token in vocab_to_idx:
                seq.append(vocab_to_idx[token])
            else:
                seq.append(vocab_to_idx['<UNK>'])
        if len(seq) < max_seq_length:
            seq += [vocab_to_idx['<PAD>']] * (max_seq_length - len(seq))
        else:
            seq = seq[:max_seq_length]
        sequences.append(seq)
    return sequences

# Prepare sequences
train_sequences = text_to_sequence(train_df['cut_text'], vocab_to_idx, max_seq_length=200)
test_sequences = text_to_sequence(test_df['cut_text'], vocab_to_idx, max_seq_length=200)

# Split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(train_sequences, train_df['label'], test_size=0.2, random_state=42)

# Convert to tensors
X_tr_tensor = torch.tensor(X_tr, dtype=torch.long)
y_tr_tensor = torch.tensor(y_tr.tolist(), dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.tolist(), dtype=torch.long)
X_test_tensor = torch.tensor(test_sequences, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(X_test_tensor, batch_size=32, shuffle=False)

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # Convert to word embeddings
        output, _ = self.rnn(embedded)  # RNN output
        output = output[:, -1, :]  # Get last hidden state
        output = self.fc(output)  # Fully connected layer
        return output

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer
model = RNNModel(vocab_size=len(vocab_to_idx), embedding_dim=100, hidden_dim=256, output_dim=len(train_df['label'].unique()))
model.to(device)  # Move model to the selected device

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 15
train_losses, val_accuracies = [], []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
        
        optimizer.zero_grad()
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU for storing
            val_labels.extend(labels.cpu().numpy())  # Move labels back to CPU

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Final validation accuracy
accuracy = accuracy_score(val_labels, val_preds)
print(f"验证集准确率: {accuracy:.4f}")

# Save classification report
result_dir = os.path.join(root_dir, "result")
model_name = "RNN"
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
