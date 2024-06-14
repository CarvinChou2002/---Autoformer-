import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from regression import rolling_fit_predict as rp
import pandas as pd

df = pd.read_excel('C:/Users/linglingyu/Desktop/案例/alldata.xlsx')

# 参数设定
step = 1  # 步长，每次窗口前进的单位
start_idx = 0  # 起始位置索引
end_idx = 4000  # 结束位置索引，确保窗口可以遍历整个数据集

window_sizes = [24, 24*7, 24*30, 24*10, 24*60]  # 示例窗口大小列表
X_multi_windows = []

# 确保所有窗口预测值列表长度一致，最小长度为最短窗口的长度
min_length = min([end_idx - ws - start_idx for ws in window_sizes])

for ws in window_sizes:
    X_window,_, _, _ = rp(df, ws, step, start_idx, end_idx)
    # 调整窗口预测值长度以匹配最小长度
    X_window = X_window[:min_length] if len(X_window) > min_length else np.concatenate([X_window, [X_window[-1]]*(min_length-len(X_window))])
    X_multi_windows.append(X_window)

# 将所有窗口的预测值合并为一个二维数组，每行代表一个样本，每列代表一个窗口的预测值
X_concat = np.column_stack(X_multi_windows)
Y = df['DA'][start_idx:start_idx+min_length]  # 确保Y与X的样本数一致

# 添加截距项
X_with_intercept = sm.add_constant(X_concat)

# 定义分位数
quantiles = [0.1, 0.5, 0.9]

# 对每个分位数进行回归并打印系数
models = {}
predictions = {}
for q in quantiles:
    model = sm.QuantReg(Y, X_with_intercept)
    result = model.fit(q=q)
    models[q] = result
    predictions[q] = result.predict(X_with_intercept)
    print(result.params)
    intercept = result.params[0]
    slopes = result.params[1:]
    print(f"Quantile {q} Coefficients: Intercept={intercept}, Slopes={slopes}")

# 绘制实际值与预测值的折线图
plt.figure(figsize=(14, 7))

# 绘制实际Y值的折线图
plt.plot(range(min_length), Y, label='Actual DA', color='blue', linewidth=2)

# 为每个分位数的预测值绘制折线图
for q, pred in predictions.items():
    plt.plot(range(min_length), pred, label=f'Quantile {q}', alpha=0.6)

# 设置图表标题和坐标轴标签
plt.title('Quantile Regression Predictions vs Actual DA')
plt.xlabel('Sample Index')
plt.ylabel('DA Value')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()





# 继续绘图部分，现在X_concat已经是正确的Numpy数组格式
plt.figure(figsize=(10, 6))
plt.scatter(X_concat, Y, color='blue', label='Actual Data')
for q in quantiles:
    # 确保预测也正确对应，此处未改动，因为预测应该已经基于正确的X_with_intercept进行了
    plt.plot(X_concat, predictions[q], linestyle='--', label=f'Quantile {q}')
plt.title('Quantile Regression using statsmodels')
plt.xlabel('Feature X')
plt.ylabel('Response Y')
plt.legend()
plt.show()