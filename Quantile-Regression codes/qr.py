import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tools.eval_measures import bic
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from regression import rolling_fit_predict as rp
from scipy.stats import chi2
from random import shuffle
from scipy.stats import t
from arch.univariate import GARCH
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题

# 数据读取


df = pd.read_excel('C:/Users/linglingyu/Desktop/案例/alldata.xlsx')

# 参数设定
step = 1  # 步长，每次窗口前进的单位
start_idx = 0  # 起始位置索引
end_idx = 8000  # 结束位置索引，确保窗口可以遍历整个数据集


window_sizes = [24, 24*2, 24*3, 24*4, 24*5,24*6, 24*10, 24*30, 24*60, 24*90, 24*120, 24*15 ] # 示例窗口大小列表
#window_sizes = [24 * i for i in range(1, 61)] + [24*90, 24*120, 24*180]
window_sizes = [24]
#window_sizes = [24, 24*7, 24*15, 24*90, 24*120, 24*180]
X_multi_windows = []

# 确保所有窗口预测值列表长度一致，最小长度为最短窗口的长度`
min_length = min([end_idx + ws - ws - start_idx for ws in window_sizes])

cc = 0
for ws in window_sizes:
    cc += 1
    print(cc)
    X_window,_, coefficients, intercept,_ = rp(df, ws, step, start_idx, end_idx)# 保存系数
    # 调整窗口预测值长度以匹配最小长度
    X_window = X_window[:min_length] if len(X_window) > min_length else np.concatenate([X_window, [X_window[-1]]*(min_length-len(X_window))])
    X_multi_windows.append(X_window)


X_concat = np.column_stack(X_multi_windows)
scaler = StandardScaler()

# 将所有窗口的预测值合并为一个二维数组，每行代表一个样本，每列代表一个窗口的预测值


# 数据标准化

X_scaled = scaler.fit_transform(X_concat)

# 应用PCA
pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)  # 使用PCA转换原始数据

Y = df['DA'][start_idx:start_idx+min_length]  # 确保Y与X的样本数一致

# 应用PCA并找到最优主成分数目
max_factors = min(X_scaled.shape[1], 100)  # 设定最大尝试的主成分数
bic_values = []
for num_factors in range(1, max_factors + 1):
    # 应用n_components参数指定主成分数
    pca = PCA(n_components=num_factors)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_with_intercept = sm.add_constant(X_pca)
    model = sm.OLS(Y, X_pca_with_intercept).fit()
    # 直接提取BIC值而不是整个params
    bic_value = model.bic
    bic_values.append((num_factors, bic_value))

optimal_factors = min(bic_values, key=lambda x: x[1])[0]
print(f"Optimal number of factors according to BIC: {optimal_factors}")

# 使用最优主成分数重构PCA
pca_optimal = PCA(n_components=optimal_factors)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)
X_pca_with_intercept_final = sm.add_constant(X_pca_optimal)

# 分位数回归
quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
models = {}
predictions = {}

for q in quantiles:
    model_qr = sm.QuantReg(Y, X_pca_with_intercept_final)
    result = model_qr.fit(q=q)
    models[q] = result
    predictions[q] = result.predict(X_pca_with_intercept_final)
    print(f"Quantile {q} Coefficients: Intercept={result.params.iloc[0]}, Slopes={list(result.params.iloc[1:])}")


low = 0.25
high = 0.75
target_coverage = high - low
alpha = high - low
# 可视化
min_length = 240
# 绘制实际值与预测值的折线图
plt.figure(figsize=(14, 7))
plt.plot(range(min_length), Y[720:720+min_length], label='日前电价真实值', color='blue', linewidth=2)

for q, pred in predictions.items():
    plt.plot(range(min_length), pred[720:720+min_length], label=f'分位数 {q}', alpha=0.6)

plt.title('FQRA模型预测值 vs 真实值')
plt.xlabel('时间索引')
plt.ylabel('日前电价')
plt.legend()
plt.grid(True)
plt.show()

print("PCA dimensionality reduction and Quantile Regression analysis completed.")
q = [0.05 * i for i in range(1, 10)]
coverage = []
COV_A = []
MAD = []
T= []
p_v = []
for qq in q:
#构建MAD
 low = round(qq, 2)
 high = 1 - low
 print(high - low)
 pre_e = np.zeros((2,end_idx))
 for i in range (0,end_idx):
    pre_e[0,i] = min(predictions[low][i],predictions[0.5][i],predictions[high][i])
    pre_e[1,i] = max(predictions[low][i],predictions[0.5][i],predictions[high][i])

 J = np.zeros((1,end_idx))
 for i in range (0,end_idx):
    if Y[i] <= pre_e[1,i] and Y[i] >= pre_e[0,i]:
        J[0,i] = 1
    else:
        J[0,i] = 0
#print(J)
 ones_count = np.sum(J)
 coverage.append(ones_count)
 print(f"J 中占{end_idx}有{ones_count}个1")

 cov_h = np.zeros(24)  # 修改维度为直接的24，不再包裹成(1, 24)
 count_per_hour = np.zeros(24)  # 类似地，直接使用24作为大小

 for i in range(end_idx):
    hour = i % 24  # 当前小时索引
    count_per_hour[hour] += 1  # 每过一个小时，计数加1
    if J[0,i] == 1:  # 如果J[i]为真，即预测正确
        cov_h[hour] += 1  # 对应小时的覆盖率计数加1

# 计算覆盖率
 for i in range(24):
    if count_per_hour[i] != 0:  # 防止除以0错误
        cov_h[i] /= count_per_hour[i]  # 计算覆盖率

 #print(cov_h)
 cov_a = np.mean(cov_h)
 print(f"平均覆盖率{cov_a}")
 COV_A.append(cov_a)
# 假设您想计算cov_h中每个元素与0.8之间的绝对偏差，然后取平均

 deviations = np.abs(cov_h - target_coverage)
 MAD_a = np.mean(deviations)
 MAD.append(MAD_a)
 print(f"Mean Absolute Deviation: {MAD_a}")




# 假设我们已知残差平方序列
 residuals = Y - predictions[0.5]  # 以中位数分位数回归预测为例计算残差

# 计算残差的平方
 residuals_squared = (cov_h - alpha)**2

# 计算残差平方的一阶自相关系数
 autocorr_1 = np.corrcoef(residuals_squared[:-1], residuals_squared[1:])[0, 1]

# Christoffersen检验统计量的一个简化版本（注意：这不是严格的Christoffersen检验）
 test_statistic_simplified = autocorr_1 * np.sqrt(len(residuals_squared)) 
 T.append(test_statistic_simplified)

# 由于实际的p值计算较为复杂，这里仅给出一个简化的处理方法
# 实际应用中应使用更精确的方法，如查阅表或使用专门的统计函数
 p_value_simplified = 2 * (1 - t.cdf(abs(test_statistic_simplified), len(residuals_squared) - 1))
 p_v.append(p_value_simplified)
 print(f"Simplified Christoffersen Test Statistic: {test_statistic_simplified}")
 print(f"Simplified P-value: {p_value_simplified}")







# 初始化一个DataFrame用于存储各分位数回归的系数和其它指标
coefficients_df = pd.DataFrame(columns=['Quantile', 'Intercept'] + [f'Slope_{i}' for i in range(1, optimal_factors + 1)])
#coverage_df = pd.DataFrame(columns=['Quantile', 'Coverage'])
#mad_df = pd.DataFrame(columns=['Quantile', 'MAD'])

# 循环每个分位数，填充DataFrame
for q, model in models.items():
    coefficients = model.params.tolist()  # 获取系数列表，包含截距和斜率
    coefficients_df.loc[len(coefficients_df)] = [q] + coefficients  # 存储系数


# 显示DataFrame
#print(coefficients_df)
print("样本中多少被覆盖",coverage)
print('每小时的覆盖率',COV_A)
print('每小时的MAD',MAD)
print('统计量',T)
print('p值',p_v)


#print(predictions[0.1][0:24])
#print(predictions[0.9][0:24])
#print(predictions[0.5][1270:1320])
#交易策略评价模型
Bd = 1

s = np.zeros(int(90))
c = 0
start = int(10)
out = 0
q = [0.05 * i for i in range(1, 10)]
ab = []
for qq in  q:
 low  = round(qq, 2)
 high = 1 - low
 for i in range (start , start + int(90)):
    #d = i #day
    idx_max = predictions[0.5][(i * 24):(i + 1) * 24].argmax()
    idx_min = predictions[0.5][(i * 24):(i + 1) * 24].argmin()
    max = predictions[0.5][(i * 24) + idx_max]
    min = predictions[0.5][(i * 24) + idx_min]
    ss = max - min#记录一天的波动以确定买卖情况
    ss = 0
    sell = predictions[low][(i * 24) + idx_max] + ss
    buy = predictions[high][(i * 24) + idx_min] - ss

    if (sell - buy) < 0:
        #print(ss)
        #print(sell - buy)
        out = out + 1
        #print("sell is lower than buy:",out)
    if Bd == 1:
        #print((i*24)+idx_min,(i*24)+idx_max)
        #print(sell,buy)
        if sell <= Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 1
            s[i-start] = 0.9*sell - (1/0.9)*buy
        elif sell > Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 2
            s[i-start] = -(1/0.9)*buy
        elif sell <= Y[(i*24)+idx_max] and buy < Y[(i*24)+idx_min]:
            Bd = 0
            s[i-start] = 0.9*sell
        else:
            Bd = 1
            s[i-start] = 0
        #print(Bd)
    elif Bd == 0:
        if idx_max == 0:
            idx = idx_max
        else :
            idx = predictions[0.5][(i * 24):(i * 24) + idx_max].argmin()

        buy_1 = predictions[high][(i*24)+idx]
        #print((i*24)+idx_min,(i*24)+idx_max)
        #print(sell,buy)
        if sell <= Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 1
            s[i-start] = 0.9*sell - (1/0.9)*buy -(1/0.9)*buy_1
        elif sell > Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 2
            s[i-start] = -(1/0.9)*buy-(1/0.9)*buy_1
        elif sell <= Y[(i*24)+idx_max] and buy < Y[(i*24)+idx_min]:
            Bd = 0
            s[i-start] = 0.9*sell-(1/0.9)*buy_1
        else:
            Bd = 1
            s[i-start] = 0-(1/0.9)*buy_1
    elif Bd == 2:
        if idx_min == 0:
            idx = idx_min
        else :
            idx = predictions[0.5][(i * 24):(i * 24) + idx_min].argmax()

        #print(idx)
        sell_1 = predictions[low][(i*24)+idx]
        #print((i*24)+idx_min,(i*24)+idx_max)
        #print(sell,buy)
        if sell <= Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 1
            s[i-start] = 0.9*sell - (1/0.9)*buy +0.9*sell_1
        elif sell > Y[(i*24)+idx_max] and buy >= Y[(i*24)+idx_min]:
            Bd = 2
            s[i-start] = -(1/0.9)*buy+0.9*sell_1
        elif sell <= Y[(i*24)+idx_max] and buy < Y[(i*24)+idx_min]:
            Bd = 0
            s[i-start] = 0.9*sell+0.9*sell_1
        else:
            Bd = 1
            s[i-start] = 0+0.9*sell_1
    if Bd != 1:
        c = c + 1
 ab.append(np.mean(s))
#print(s)
print(c)
print(np.mean(s))
print(ab)

os = np.zeros(int(24*7))
for i in range (start , start + int(90)):
    #d = i #day
    idx_max = predictions[0.5][(i * 24):(i + 1) * 24].argmax()
    idx_min = predictions[0.5][(i * 24):(i + 1) * 24].argmin()
    max = predictions[0.5][(i * 24) + idx_max]
    min = predictions[0.5][(i * 24) + idx_min]
    ss = max - min#记录一天的波动以确定买卖情况
    ss = 0
    sell = predictions[low][(i * 24) + idx_max] + ss
    buy = predictions[high][(i * 24) + idx_min] - ss
    os[i-start] = 0.9*sell-1/0.9*buy
#print(os)
print(np.mean(os))

confidence_levels = [0.1 * (10-i) for i in range(1, 10)]  # 置信度水平
plt.plot(confidence_levels, ab, label='电力交易模型下的盈利水平')
plt.axhline(np.mean(os), color='r', linestyle='--', label='理想情况下的盈利水平')

plt.title('交易策略盈利水平 vs. 理想水平盈利')
plt.xlabel('置信区间')
plt.ylabel('盈利')
plt.legend()
plt.xlim(0.1, 0.9)  # 设置x轴范围从0.1到0.9
plt.show()