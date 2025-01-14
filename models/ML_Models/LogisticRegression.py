import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目的根目录路径（假定根目录为 fake_news 的目录）
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将根目录添加到 sys.path
sys.path.append(root_dir)

from data_factory.data_processing import load_data_txt, jieba_cut_text, veclization
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

from utils.visualization import plot_confusion_matrix, plot_multiclass_precision_recall, plot_multiclass_roc

train_file_path = os.path.join(root_dir, "data", "train.txt")
test_file_path = os.path.join(root_dir, "data", "test1.txt")
stop_word_path = os.path.join(root_dir, "data", "cn_stopwords.txt")

train_df = load_data_txt(train_file_path)
print("[INFO] 训练数据预览:")
print(train_df.head())
train_df=jieba_cut_text(train_df,text_col="text",new_col="cut_text",stopwords_path=stop_word_path)

test_df = load_data_txt(test_file_path)
print("\n[INFO] 测试数据预览:")
print(test_df.head())
test_df = jieba_cut_text(test_df, text_col="text", new_col="cut_text", stopwords_path=stop_word_path)

X_train,X_label,tfidf_vec=veclization(train_df,text_col="cut_text",
                                      label_col="label",max_features=5000,
                                      ngram_range=(1, 1), stop_words=None)
# 划分训练集和验证集
X_tr, X_val, y_tr, y_val = train_test_split(X_train, X_label, test_size=0.2, random_state=42)
# 建立逻辑回归模型并训练
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)

# 在验证集上预测
y_val_pred = model.predict(X_val)
print("验证集准确率:", accuracy_score(y_val, y_val_pred))
print("分类报告:")
print(classification_report(y_val, y_val_pred))


result_dir = os.path.join(root_dir, "result")
model_name = "LogisticRegression"  # 可根据实际情况修改

# 取出所有类别标签（确保顺序与模型预测结果一致）
classes = sorted(np.unique(X_label))

# 混淆矩阵
cm_save_path = os.path.join(result_dir, f"{model_name}_confusion_matrix.png")
plot_confusion_matrix(y_val, y_val_pred, classes=classes,
                      title="Confusion Matrix", save_path=cm_save_path)

# 多类别 ROC 曲线（OvR）
roc_save_path = os.path.join(result_dir, f"{model_name}_multiclass_roc_curve.png")
plot_multiclass_roc(model, X_val, y_val, classes=classes,
                    title="Multi-class ROC Curve", save_path=roc_save_path)

# 多类别 Precision-Recall 曲线（OvR）
pr_save_path = os.path.join(result_dir, f"{model_name}_multiclass_pr_curve.png")
plot_multiclass_precision_recall(model, X_val, y_val, classes=classes,
                                 title="Multi-class Precision-Recall Curve", save_path=pr_save_path)

print("所有图形已保存到:", result_dir)


# -------------------------------
# 对没有标签的测试集进行预测并保存结果
# -------------------------------

# 使用训练时 fit 得到的 tfidf_vec 对测试数据进行向量化
# 注意：测试集文本已经存放在 "cut_text" 列中
X_test = tfidf_vec.transform(test_df["cut_text"].tolist())

# 用训练好的模型对测试集进行预测
test_predictions = model.predict(X_test)

# 构造 DataFrame 保存预测结果
# 此处可根据需求将ID或者其他信息一并保存，假设测试集中有 "id" 列
if "id" in test_df.columns:
    result_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_predictions
    })
else:
    result_df = pd.DataFrame({'prediction': test_predictions})

# 保存预测结果至 result 文件夹中，命名文件中包含模型名称
test_result_path = os.path.join(result_dir, f"{model_name}_test_predictions.csv")
result_df.to_csv(test_result_path, index=False, encoding="utf-8")

print("测试集预测结果已保存到:", test_result_path)


