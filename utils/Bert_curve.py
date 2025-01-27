import matplotlib.pyplot as plt
import os

# 从日志中提取的指标数据
epochs = 5
train_losses = [0.7136, 0.4182, 0.3275, 0.2570, 0.1995]
val_accuracies = [0.8650, 0.8774, 0.8831, 0.8823, 0.8820]

# 设置可视化参数
plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-o', linewidth=2, markersize=8)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, epochs+1))

# 绘制验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), val_accuracies, 'r-s', linewidth=2, markersize=8)
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.8, 0.9)  # 设置Y轴范围以突出变化
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, epochs+1))

# 保存和显示结果
plt.tight_layout()
result_dir = "d:\\fake_news\\result"
training_curve_path = os.path.join(result_dir, "OptimizedBERT_training_curves.png")
plt.savefig(training_curve_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"训练过程可视化图表已保存到: {training_curve_path}")