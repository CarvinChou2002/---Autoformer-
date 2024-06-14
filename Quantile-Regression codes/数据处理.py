import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
from scipy.interpolate import UnivariateSpline


# 读取Excel文件
df = pd.read_excel('C:/Users/linglingyu/Desktop/DA.xlsx')
print(df.columns)
df.dropna(subset=['DA'], inplace=True)  # 这将删除任何含有'DA'列中NaN值的行


# 对'da'列应用N-PIT变换
def normalise_data(series):
    # 计算经验累积分布函数
    ecdf = series.rank(pct=True)
    # 使用标准正态分布的逆函数转换
    return norm.ppf(ecdf)

df['DA_normalized'] = normalise_data(df['DA'])

# Z-score异常值检测
z_scores = stats.zscore(df['DA'])
abs_z_scores = np.abs(z_scores)
threshold = 5  # 设定异常值阈值，可根据实际情况调整
outliers = abs_z_scores > threshold

# 标记异常值
df['outlier'] = outliers

# 使用样条插值修正异常值
# 首先，找出异常值的位置
outlier_indices = np.where(outliers)[0]

# 排除异常值进行插值
good_data = df.loc[~outliers, 'DA']
x_good = np.arange(len(good_data))  # 假设索引作为x轴，根据实际情况调整
spline = UnivariateSpline(x_good, good_data, k=3)  # k为样条的次数，可根据数据特性调整

# 用插值结果替换异常值
df['DA_corrected'] = df['DA'].copy()# 复制原始列以保留原始数据
df['DA_act'] = df['DA'].copy()
for i in outlier_indices:
    df.at[i, 'DA_corrected'] = spline(i)  # 用插值结果替换异常值
df['DA'] = df['DA_corrected']
print(len(df))

df['DA_1'] = df['DA']
df['DA_2'] = df['DA']
df['DA_7'] = df['DA']
df['DA_1_min'] = df['DA_1']
df['DA_1_max'] = df['DA_1']
df['DA_1_24'] = df['DA_1']
print(df.columns) 



for i in range(24, len(df)):  # 改变循环起始点为24
    df.loc[i, 'DA_1'] = df.loc[i-24, 'DA']  # 使用.loc[]访问避免错误并提高代码可读性
    if i >= 48:
        df.loc[i, 'DA_2'] = df.loc[i-48, 'DA']
    if i >= 168:
        df.loc[i, 'DA_7'] = df.loc[i-168, 'DA']

# DA_1_min 和 DA_1_max 的计算部分可以简化，如下所示：
# 计算每天的最小值和最大值
for i in range(0, 8761):
    window_start = 24 * (i // 24)  # 当前窗口的起始索引
    window_end = window_start + 24  # 下一个窗口的起始索引（当前窗口的结束索引+1）
    
    # 确保window_end不会超出索引范围
    window_end = min(window_end, len(df))
    
    # 使用numpy的min和max函数提高效率（如果不怕引入外部库的话）
    df.loc[i, 'DA_1_min'] = np.min(df.loc[window_start:window_end - 1, 'DA_1'])
    df.loc[i, 'DA_1_max'] = np.max(df.loc[window_start:window_end - 1, 'DA_1'])
for i in range (8761):
    for j in range(365):
        if i >= j*24 and i < (j+1)*24:
             df.loc[i, 'DA_1_24'] = df.loc[(j+1)*24-1, 'DA_1']
                


df.to_excel('alldata.xlsx', index=False)  # 不保存索引
