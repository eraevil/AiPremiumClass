import numpy as np
# 加载保存的模型参数
def load_model(param_path):
    params = np.load(param_path)
    theta = params['theta']
    bias = params['bias']
    return theta, bias

# 前向传播函数
def forward(x, theta, bias):
    z = np.dot(theta, x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 预测函数
def predict(x, theta, bias):
    # 添加维度处理单个样本
    if x.ndim == 1:
        x = x.reshape(1, -1)
    pred = forward(x, theta, bias)
    return 1 if pred > 0.5 else 0

if __name__ == "__main__":
    # 加载模型
    theta, bias = load_model('model_params.npz')
    # 创建虚拟测试样本（实际使用时替换为真实数据）
    test_sample = np.array([5.1, 3.5, 1.4, 0.2])
    # 进行预测
    prediction = predict(test_sample, theta, bias)
    print(f"模型预测结果: {prediction}")