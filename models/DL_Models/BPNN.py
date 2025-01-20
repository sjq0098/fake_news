import numpy as np

# Sigmoid激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 网络参数
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1

# 初始化权重
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# 训练数据（XOR示例）
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 训练过程
for epoch in range(10000):
    # 前向传播
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)
    output = sigmoid(np.dot(hidden_output, W2))
    
    # 计算误差
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    # 反向传播更新权重
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

print("预测结果：", output)