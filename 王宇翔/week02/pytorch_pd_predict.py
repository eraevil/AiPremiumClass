import torch
import numpy as np


class IrisClassifier:
    def __init__(self, model_path):
        # 加载模型参数
        checkpoint = torch.load(model_path)
        self.w = checkpoint['weights']
        self.b = checkpoint['bias']
        self.input_shape = checkpoint['input_shape']

    def predict(self, x):
        """
        预测函数
        参数：
            x: 输入数据 (numpy数组或列表)，形状为(n_samples, 4)
        返回：
            预测类别 (0或1) 和概率
        """
        # 输入验证
        if x.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"输入特征维度错误，期望 {self.input_shape[1:]}，实际 {x.shape[1:]}")

        # 转换为张量
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # 前向计算
        with torch.no_grad():
            logits = torch.nn.functional.linear(x_tensor, self.w, self.b)
            probabilities = torch.sigmoid(logits)

        # 转换为numpy
        prob_np = probabilities.numpy().flatten()
        classes = (prob_np >= 0.5).astype(int)

        return classes, prob_np


if __name__ == "__main__":
    # 使用示例
    classifier = IrisClassifier("model_params.pt")

    # 测试样本（实际使用时可替换为真实数据）
    test_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # 类别0
        [6.2, 2.9, 4.3, 1.3]  # 类别1（模拟数据）
    ])

    # 进行预测
    classes, probs = classifier.predict(test_samples)

    # 输出结果
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        print(f"样本 {i + 1}:")
        print(f"特征值: {test_samples[i]}")
        print(f"预测类别: {cls} | 属于类别1的概率: {prob:.4f}")
        print("-" * 40)