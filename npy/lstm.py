import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# LSTM类定义
class LSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化权重和偏置
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bf = np.zeros((hidden_dim, 1))

        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bi = np.zeros((hidden_dim, 1))

        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bo = np.zeros((hidden_dim, 1))

        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bc = np.zeros((hidden_dim, 1))

    def forward(self, x, prev_h, prev_c):
        # 拼接输入和隐藏状态
        combined = np.vstack((prev_h, x))

        # 遗忘门
        f = sigmoid(np.dot(self.Wf, combined) + self.bf)

        # 输入门
        i = sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_hat = tanh(np.dot(self.Wc, combined) + self.bc)

        # 更新细胞状态
        c = f * prev_c + i * c_hat

        # 输出门
        o = sigmoid(np.dot(self.Wo, combined) + self.bo)
        h = o * tanh(c)

        return h, c

# 使用示例
input_dim = 3  # 输入维度
hidden_dim = 5  # 隐藏层维度

# 初始化LSTM
lstm = LSTM(input_dim, hidden_dim)

# 示例输入数据
x = np.random.randn(input_dim, 1)  # 单步输入
prev_h = np.zeros((hidden_dim, 1))  # 初始隐藏状态
prev_c = np.zeros((hidden_dim, 1))  # 初始细胞状态

# 前向传播
h, c = lstm.forward(x, prev_h, prev_c)

print("Hidden state (h):", h)
print("Cell state (c):", c)
