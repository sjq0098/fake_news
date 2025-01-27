import matplotlib.pyplot as plt

epochs = list(range(1, 16))
train_loss = [0.6576, 0.3478, 0.2173, 0.1472, 0.1138, 0.0930, 
              0.0800, 0.0739, 0.0693, 0.0641, 0.0615, 0.0580, 
              0.0550, 0.0543, 0.0521]
val_acc = [0.8609, 0.8616, 0.8558, 0.8514, 0.8518, 0.8481, 
           0.8457, 0.8456, 0.8427, 0.8388, 0.8399, 0.8372, 
           0.8357, 0.8386, 0.8366]

plt.figure(figsize=(12, 5))

# 训练损失曲线（左图）
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=8)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.7)

# 验证准确率曲线（右图）
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, 'r-s', linewidth=2, markersize=8)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0.82, 0.87)  # 突出变化细节
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("CNN_training_curves.png", dpi=300)
plt.show()