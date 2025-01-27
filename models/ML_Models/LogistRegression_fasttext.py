import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

from data_factory.data_processing import load_data_txt, jieba_cut_text, fasttext_vectorization
from utils.visualization import plot_confusion_matrix, plot_multiclass_precision_recall, plot_multiclass_roc


train_file_path = os.path.join(root_dir, "data", "train.txt")
test_file_path = os.path.join(root_dir, "data", "test1.txt")
stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")
w2v_model_path = os.path.join(root_dir, "models", "cc.zh.300.vec")  # 确保路径正确

# 加载数据
train_df = load_data_txt(train_file_path)
print("[INFO] 训练数据预览:")
print(train_df.head())
train_df = jieba_cut_text(train_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

test_df = load_data_txt(test_file_path)
print("\n[INFO] 测试数据预览:")
print(test_df.head())
test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

# 加载 FastText 模型（二进制格式）
try:
    w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)
    print("模型加载成功（文本格式）。")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

vector_size = w2v_model.vector_size  # 获取向量维度（通常为 300）

# 向量化
X_train = fasttext_vectorization(train_df["cut_text"], model=w2v_model, vector_size=vector_size)
X_label = train_df["label"].values
X_test = fasttext_vectorization(test_df["cut_text"], model=w2v_model, vector_size=vector_size)

# 划分训练集和验证集
X_tr, X_val, y_tr, y_val = train_test_split(X_train, X_label, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)

# 验证集评估
y_val_pred = model.predict(X_val)
print("验证集准确率:", accuracy_score(y_val, y_val_pred))

# 保存分类报告
result_dir = os.path.join(root_dir, "result")
os.makedirs(result_dir, exist_ok=True)

model_name = "LogisticRegression_fasttext_vectorization"
report = classification_report(y_val, y_val_pred, target_names=[str(label) for label in sorted(np.unique(X_label))])
report_save_path = os.path.join(result_dir, f"{model_name}_classification_report.txt")
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"分类报告已保存到: {report_save_path}")

# 可视化
classes = sorted(np.unique(X_label))
plot_confusion_matrix(y_val, y_val_pred, classes=classes, title="Confusion Matrix", save_path=os.path.join(result_dir, f"{model_name}_confusion_matrix.png"))
plot_multiclass_roc(model, X_val, y_val, classes=classes, title="Multi-class ROC Curve", save_path=os.path.join(result_dir, f"{model_name}_multiclass_roc_curve.png"))
plot_multiclass_precision_recall(model, X_val, y_val, classes=classes, title="Multi-class Precision-Recall Curve", save_path=os.path.join(result_dir, f"{model_name}_multiclass_pr_curve.png"))
print("所有图形已保存到:", result_dir)

# 测试集预测
test_predictions = model.predict(X_test)
result_df = pd.DataFrame({'prediction': test_predictions})
test_result_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")
result_df.to_csv(test_result_path, index=False, encoding="utf-8")
print("测试集预测结果已保存到:", test_result_path)