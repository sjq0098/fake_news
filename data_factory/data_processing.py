import  pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def load_data_txt(file_path):
    data_rows=[   ]
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split("_separator_")
            if len(parts)==4:
                ids,text,keywords,labels=parts
                try:
                    idx = int(ids)
                except ValueError:
                    idx = ids  
                
                try:
                    label = int(labels)
                except ValueError:
                    label = labels 
                if keywords.lower() == "nan" or not keywords.strip():
                    keywordss = []
                else:
                    keywordss = keywords.split(",")
                row_dict = {
                    "id": ids,
                    "text": text,
                    "keywords": keywordss,
                    "label": label
                }
                data_rows.append(row_dict)
            else:
                if len(parts) == 3:
                    id_, text, keywords = parts
                    data_rows.append({
                        'id': id_,
                        'text': text,
                        'keywords': keywords
                    })
                else:
                    print("测试数据格式不一致：", line)
    
    # 用 pandas.DataFrame 打包
    df = pd.DataFrame(data_rows, columns=["id", "text", "keywords", "label"])
    return df


def jieba_cut_text(df,
                       text_col="text",
                       new_col="cut_text",
                       stopwords_path=None):
    if stopwords_path:
        stopwords=set()
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word=line.strip()
                stopwords.add(word)
    def cut_and_filter(text):
        # 如果这一行文本为空或类型不对，就返回空字符串
        if not isinstance(text, str):
            return ""

        # jieba 分词
        words = jieba.lcut(text)

        # 如果需要去停用词
        if stopwords_path:
            words = [w for w in words if w not in stopwords and w.strip()]

        # 拼接成空格分隔的字符串
        return " ".join(words)

    # 3. 对 df[text_col] 逐行 apply 分词函数
    df[new_col] = df[text_col].apply(cut_and_filter)

    return df
        

def veclization(df, 
                text_col="cut_text", 
                label_col="label",
                max_features=5000,
                ngram_range=(1,1),
                stop_words=None):
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    tfidf_vec = TfidfVectorizer(max_features=max_features,
                                ngram_range=ngram_range,
                                stop_words=stop_words)
    X = tfidf_vec.fit_transform(texts)
    return X, labels, tfidf_vec

def fasttext_vectorization(texts, model, vector_size=300):
    """
    将分词后的文本列表转换为 FastText 平均向量
    :param texts: 分词后的文本列表（例如 ["我", "喜欢", "自然语言处理"]）
    :param model: 加载的 FastText 模型
    :param vector_size: 向量维度
    :return: 向量矩阵 (n_samples, vector_size)
    """
    features = []
    for text in texts:
        vectors = []
        for word in text:
            if word in model:  # 忽略未登录词
                vectors.append(model[word])
        if len(vectors) > 0:
            avg_vector = np.mean(vectors, axis=0)
        else:
            avg_vector = np.zeros(vector_size)  # 处理空文本
        features.append(avg_vector)
    unknown_words = [word for text in texts for word in text if word not in model]
    print(f"未登录词比例: {len(unknown_words) / sum(len(text) for text in texts):.2%}")
    return np.array(features)





