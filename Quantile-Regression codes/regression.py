import pandas as pd
import numpy as np
from rich import columns
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LinearRegression was fitted with feature names")
    # 在这里调用你的模型拟合和预测代码

# 假设df是包含所有必要列的时间序列数据（'DA', 'DA_d-1', 'DA_d-2', 'DA_d-7', 'DA_d-1_min', 'DA_d-1_max'）
# 请确保在实际使用前，df中的列名与模型公式中的变量相对应，并已完成必要的数据预处理

def rolling_fit_predict(df, window_size, step, start_idx, end_idx, predict_new_data=None):
  """
  使用滚动窗口方法拟合并预测时间序列模型。
  """
  predictions = []
  failed_windows = []
  final_model = None
  end_idx = end_idx + window_size

  for idx in tqdm(range(start_idx, end_idx - window_size, step), desc="Rolling Window Fit"):
      # 提取当前窗口的数据
      window_data = df.iloc[idx:idx + window_size][['W_DA','DA', 'DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max','DA_1_24','oil', 'LNG']]

      # 分割特征和目标变量'Load','S_DA','W_DA',
      X_window = window_data[['W_DA','DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max','DA_1_24','oil', 'LNG']]
      y_window = window_data['DA']
      #print(X_window)
      # 拟合模型
      try:
          model = LinearRegression()
          model.feature_names_in_ = None  # 清除特征名称
          model.fit(X_window[:-1], y_window[:-1])

          # 预测下一个点
          # 预测下一个点'Load','S_DA','W_DA',
          next_exog = pd.DataFrame([window_data.iloc[-1][['W_DA','DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max','DA_1_24','oil', 'LNG']].values],
                                   columns=X_window.columns)
          next_prediction = model.predict(next_exog)[0]

          predictions.append(next_prediction)
          #print(model.coef_)
          # 保存最后一个成功拟合的模型实例
          final_model = model
      except Exception as e:
          print(f"Error fitting window at index {idx}: {e}")
          failed_windows.append(idx)

  # 在循环结束后，从final_model中提取参数
  if final_model is not None:
      coefficients = final_model.coef_  # 系数
      intercept = final_model.intercept_  # 截距
      #print(coefficients)'Load', 'S_DA', 'W_DA',
      if predict_new_data is not None:
          new_predictions = final_model.predict(predict_new_data[
                                                    [ 'W_DA','DA_1', 'DA_2', 'DA_7', 'DA_1_min',
                                                     'DA_1_max', 'DA_1_24', 'oil', 'LNG']])
         # print("\nPredictions on new data:")
          #print(new_predictions)
          return predictions, failed_windows, coefficients, intercept, final_model, new_predictions
      else:
          return predictions, failed_windows, coefficients, intercept, final_model
  else:
      print("No successful model fit. Unable to provide coefficients and intercept.")
      return predictions, failed_windows, None, None, final_model # 如果没有成功拟合的模型，则返回None

df = pd.read_excel('C:/Users/linglingyu/Desktop/案例/alldata.xlsx')




# 绘制每个特征与DA之间的散点图，组织成3x3网格布局
plt.figure(figsize=(14, 14))

variables_to_plot = [ 'W_DA','DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max', 'DA_1_24', 'oil', 'LNG']

for idx, var in enumerate(variables_to_plot):
    plt.subplot(3, 4, idx + 1)  # 修正了索引以正确对应子图位置
    plt.scatter(df[var], df['DA'], alpha=0.5)
    plt.title(f'{var} vs DA', fontsize=9)  # 调整字体大小以适应可能的空间限制
    plt.xlabel(var, fontsize=8)
    plt.ylabel('DA', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True)

# 防止子图之间的重叠
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局参数以留出标题空间

# 添加一个总的标题
plt.suptitle('Scatter Plots of Variables against DA', fontsize=12, y=0.98)  # y参数控制标题距离顶部的距离

plt.show()





# 参数设定
window_size = 24*15# 根据数据特性和需求设定窗口大小
step = 1  # 步长，每次窗口前进的单位
start_idx = 0  # 起始位置索引
end_idx = 8000 # 结束位置索引，确保窗口可以遍历整个数据集

# 执行滚动窗口拟合预测
predictions, failed_windows, coefficients, intercept ,final_model = rolling_fit_predict(df, window_size, step, start_idx, end_idx)

# 如果需要使得预测序列长度与原始序列一致，可以考虑重复最后一个预测值
# 假设`predictions`列表包含了所有窗口的预测值，且其长度与可以执行预测的窗口数量相符


print(np.shape(predictions), coefficients, intercept) # 应该与df的行数相等





# 假设df是原始数据框，并且'predictions'列表包含了对应于df中每个预测点的预测值
# 注意：这里的假设是预测和实际值的索引是对齐的，即predictions[i]对应df['DA'][i+len(predictions)]

# 将预测值添加到DataFrame以便于一起绘图（这里假设df的索引是连续的，且可以用于对齐预测值）
df['Predicted_DA'] = np.nan  # 初始化一个新的列用于存放预测值
for i, pred in enumerate(predictions):
    if i + window_size - 1 < len(df):
      df.loc[df.index[i + window_size - 1], 'Predicted_DA'] = pred  # 假设最后一个预测值对应于窗口的最后一个实际时间点


#'Load', 'S_DA', 'W_DA',
# 确保模型成功拟合并且获得了系数和截距
if coefficients is not None and intercept is not None:
    # 对于起始部分无法通过滚动窗口预测的点，直接使用模型的系数和截距预测
    start_predictions = []
    for i in range(window_size - 1):  # 这里假设前window_size-1个点无法通过滚动预测
        # 你需要确保df中包含这些起始索引对应的所有特征列数据
        # 例如，如果第一个点的索引为0，那么你需要提取df.loc[0]['Load', 'S_DA', ...]等特征
        # 下面的代码是一个示意，你需要根据实际情况调整
        X_start = df.iloc[i][
            ['W_DA','DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max', 'DA_1_24', 'oil', 'LNG']]
        start_prediction = np.dot(coefficients, X_start) + intercept  # 直接预测
        df.loc[df.index[i], 'Predicted_DA'] = start_prediction
else:
    print("Model did not fit successfully. Cannot make predictions using coefficients.")

# 现在，predictions列表应该包含了整个序列的预测值，包括起始部分直接使用系数预测的值
# 接下来，将整个predictions列表填充到df中相应的位置，这部分逻辑保持不变




# 绘制图表
# 确保df中的'Predicted_DA'列已经根据预测结果更新完毕

# 绘制图表，仅显示前3000个点
plt.figure(figsize=(14, 7))

# 截取前3000个数据点
df_slice = df.iloc[24*30:24*60]

plt.plot(df_slice.index, df_slice['DA'], label='真实值', color='blue')
plt.plot(df_slice.index, df_slice['Predicted_DA'], label='预测值', color='red', linestyle='--')

plt.title('日前电价 vs 预测值 ')
plt.xlabel('时间索引')
plt.ylabel('日前电价价格')
plt.legend()
plt.grid(True)
plt.xlim(left=24*30)  # 确保x轴从0开始
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# 计算预测误差
mse = mean_squared_error(df_slice['DA'], df_slice['Predicted_DA'])
rmse = np.sqrt(mse)
r2 = r2_score(df_slice['DA'], df_slice['Predicted_DA'])

print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

residuals = df_slice['DA'] - df_slice['Predicted_DA']

# 绘制残差图
plt.figure(figsize=(10, 5))
plt.scatter(df_slice['Predicted_DA'], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted DA')
plt.ylabel('Residuals')
plt.show()

# 可选：正态性检验
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print(f"Shapiro-Wilk Test for Residual Normality, Statistic={stat:.3f}, p-value={p:.3f}")
if p > 0.05:
    print("Residuals are likely normally distributed.")
else:
    print("Residuals may not be normally distributed.")

# 确保我们有最终拟合的模型
if final_model is not None:
    # 获取df中指定时间段的数据，对每行分别进行预测，并打印预测值与实际值
    for idx, row_data in df.iloc[100:200].iterrows():  # iterrows()会迭代每一行
        prediction = final_model.predict(pd.DataFrame([row_data[['W_DA', 'DA_1', 'DA_2', 'DA_7', 'DA_1_min', 'DA_1_max', 'DA_1_24', 'oil', 'LNG']]]))[0]
        actual_da = row_data['Predicted_DA']
        #actual_da = row_data['DA']
        #print(f"Predicted DA for day at index {idx}: {prediction:.2f}, Actual DA: {actual_da:.2f}")
else:
    print("No valid model to make a new prediction.")

#'Load', 'S_DA', 'W_DA',


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 假设df是包含数据的数据框

# 计算相关系数
corr_load_da, _ = pearsonr(df['DA_1'], df['DA'])
print(f"Correlation between Load and DA: {corr_load_da}")

# 绘制散点图并添加回归线
sns.lmplot(x='DA_1', y='DA', data=df, ci=None)  # 使用seaborn绘制带有回归线的散点图
plt.title('Scatter Plot with Regression Line for Load vs DA')
plt.xlabel('DA_1')
plt.ylabel('DA')

# 绘制残差图
#sns.residplot('Load', 'DA', data=df, lowess=True, color="g")
#plt.title('Residual Plot for Load vs DA')
#plt.xlabel('Fitted Values')
#plt.ylabel('Residuals')

# 输出决定系数（如果使用statsmodels等包进行线性回归可以直接获得）
# 以下为简单示例，实际应用中可直接从线性回归模型结果中提取
from sklearn.linear_model import LinearRegression

X = df[['DA_1']]  # 特征变量
y = df['DA']      # 目标变量
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
print(f"Coefficient of Determination (R²): {r_squared}")

plt.show()