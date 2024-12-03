import numpy as np

# 激活函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 卷积操作
def conv2d(x, kernel, stride=1, padding=0):
    # 对输入进行零填充
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # 获取输入和卷积核的尺寸
    x_h, x_w = x.shape
    k_h, k_w = kernel.shape

    # 输出尺寸计算
    out_h = (x_h - k_h) // stride + 1
    out_w = (x_w - k_w) // stride + 1

    # 卷积操作
    out = np.zeros((out_h, out_w))
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            out[i // stride, j // stride] = np.sum(x[i:i+k_h, j:j+k_w] * kernel)

    return out

# 最大池化
def max_pooling(x, size=2, stride=2):
    x_h, x_w = x.shape
    out_h = (x_h - size) // stride + 1
    out_w = (x_w - size) // stride + 1

    out = np.zeros((out_h, out_w))
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            out[i // stride, j // stride] = np.max(x[i:i+size, j:j+size])

    return out

# 全连接层
def fully_connected(x, weights, bias):
    return np.dot(weights, x.flatten()) + bias

# CNN类定义
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, num_fc_units):
        self.input_size = input_size  # 输入大小 (高度, 宽度)
        self.num_filters = num_filters  # 卷积核数量
        self.kernel_size = kernel_size  # 卷积核大小
        self.num_fc_units = num_fc_units  # 全连接层单元数

        # 卷积层权重初始化
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1

        # 全连接层权重初始化
        fc_input_size = (input_size - kernel_size + 1) // 2  # 假设stride=1, padding=0, pool_size=2
        self.fc_weights = np.random.randn(num_fc_units, fc_input_size * fc_input_size * num_filters) * 0.1
        self.fc_bias = np.zeros((num_fc_units, 1))

    def forward(self, x):
        # 卷积层 + 激活 + 池化
        conv_outputs = []
        for i in range(self.num_filters):
            conv_output = conv2d(x, self.filters[i])  # 卷积
            relu_output = relu(conv_output)  # ReLU 激活
            pool_output = max_pooling(relu_output)  # 最大池化
            conv_outputs.append(pool_output)

        # 拼接所有卷积层输出
        combined = np.concatenate([o.flatten() for o in conv_outputs], axis=0)

        # 全连接层
        fc_output = fully_connected(combined, self.fc_weights, self.fc_bias)

        return fc_output

# 使用示例
input_size = 8  # 输入尺寸 8x8
num_filters = 3  # 使用 3 个卷积核
kernel_size = 3  # 卷积核大小 3x3
num_fc_units = 4  # 全连接层 4 个单元

# 初始化 CNN
cnn = CNN(input_size, num_filters, kernel_size, num_fc_units)

# 示例输入（8x8 图像）
x = np.random.randn(input_size, input_size)

# 前向传播
output = cnn.forward(x)

print("Output of CNN:", output)
