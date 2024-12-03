import numpy as np

class SimpleCNN:
    def __init__(self, num_classes):
        # 初始化卷积层参数
        self.num_classes = num_classes

        # 卷积层 1 参数 (3x3 卷积核，输出 16 个通道)
        self.conv1_weights = np.random.randn(16, 3, 3, 3)  # 16 个 3x3x3 卷积核
        self.conv1_bias = np.zeros(16)  # 每个卷积核一个偏置

        # 卷积层 2 参数 (3x3 卷积核，输出 32 个通道)
        self.conv2_weights = np.random.randn(32, 16, 3, 3)  # 32 个 3x3x16 卷积核
        self.conv2_bias = np.zeros(32)  # 每个卷积核一个偏置

        # 全连接层 1 参数
        self.fc1_weights = np.random.randn(32 * 56 * 56, 128)  # 从卷积后的特征图展平到 128 维
        self.fc1_bias = np.zeros(128)

        # 全连接层 2 参数
        self.fc2_weights = np.random.randn(128, num_classes)  # 输出分类结果
        self.fc2_bias = np.zeros(num_classes)

    def conv2d(self, input_image, filters, bias, stride=1, padding=1):
        """
        手动实现卷积操作

        参数:
        - input_image: 输入图像，形状为 (height, width, channels)
        - filters: 卷积核，形状为 (num_filters, filter_height, filter_width, num_channels)
        - bias: 卷积核的偏置，形状为 (num_filters,)
        - stride: 步幅
        - padding: 填充

        返回:
        - output: 卷积后的输出图像
        """
        # 填充输入图像
        input_image_padded = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

        filter_height, filter_width = filters.shape[1], filters.shape[2]
        output_height = (input_image_padded.shape[0] - filter_height) // stride + 1
        output_width = (input_image_padded.shape[1] - filter_width) // stride + 1

        output = np.zeros((output_height, output_width, filters.shape[0]))  # 输出尺寸为 (height, width, num_filters)

        # 执行卷积操作
        for i in range(output_height):
            for j in range(output_width):
                region = input_image_padded[i * stride:i * stride + filter_height, j * stride:j * stride + filter_width, :]
                for k in range(filters.shape[0]):  # 每个卷积核
                    # 修改：逐通道乘法后求和，确保维度匹配
                    output[i, j, k] = np.sum(region * filters[k, :, :, :]) + bias[k]  # 计算卷积结果并加上偏置

        return output


    def max_pool2d(self, input_image, size=2, stride=2):
        """
        手动实现最大池化操作

        参数:
        - input_image: 输入图像，形状为 (height, width, channels)
        - size: 池化窗口的尺寸
        - stride: 池化操作的步幅

        返回:
        - output: 池化后的输出图像
        """
        output_height = (input_image.shape[0] - size) // stride + 1
        output_width = (input_image.shape[1] - size) // stride + 1

        output = np.zeros((output_height, output_width, input_image.shape[2]))  # 输出尺寸为 (height, width, channels)

        for i in range(output_height):
            for j in range(output_width):
                region = input_image[i * stride:i * stride + size, j * stride:j * stride + size, :]
                output[i, j, :] = np.max(region, axis=(0, 1))  # 对每个通道做池化

        return output

    def flatten(self, input_image):
        """
        将输入图像展平为一维向量

        参数:
        - input_image: 输入图像，形状为 (height, width, channels)

        返回:
        - flattened: 展平后的向量
        """
        return input_image.flatten()

    def relu(self, x):
        """
        ReLU 激活函数

        参数:
        - x: 输入

        返回:
        - 输出
        """
        return np.maximum(0, x)

    def forward(self, x):
        """
        执行前向传播

        参数:
        - x: 输入图像，形状为 (height, width, channels)

        返回:
        - output: 最终的分类输出
        """
        # 第一层卷积
        x = self.conv2d(x, self.conv1_weights, self.conv1_bias)
        x = self.relu(x)

        # 最大池化
        x = self.max_pool2d(x)

        # 第二层卷积
        x = self.conv2d(x, self.conv2_weights, self.conv2_bias)
        x = self.relu(x)

        # 最大池化
        x = self.max_pool2d(x)

        # 展平操作
        x = self.flatten(x)

        # 第一层全连接
        x = np.dot(x, self.fc1_weights) + self.fc1_bias
        x = self.relu(x)

        # 第二层全连接
        x = np.dot(x, self.fc2_weights) + self.fc2_bias

        return x

# 使用模型
model = SimpleCNN(num_classes=10)  # 假设有 10 个类别

# 假设输入一个 224x224 RGB 图像（输入的形状是 (224, 224, 3)）
input_image = np.random.randn(224, 224, 3)

# 获取预测输出
output = model.forward(input_image)
print(output)
