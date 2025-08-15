import sys
import time  # 导入 time 模块
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 注意: 此路径为用户本地路径，可根据您的环境进行修改或删除
# sys.path.append('D:\\anaconda\\lib\\site-packages')

# --- 数据加载与预处理 (您的原始代码) ---
print("正在加载和预处理数据...")
data_list = np.load('data_list_1500变1500.npy', allow_pickle=True)
data_array = np.vstack(data_list).astype(np.float32)
data_length = len(data_array)
print(f"第一个数据集形状: {data_array.shape}")

first_half = data_array[:data_length // 2]
second_half = data_array[data_length // 2:]
np.random.seed(42)
first_half_selected = first_half[np.random.choice(len(first_half), len(first_half) // 4, replace=False)]
second_half_selected = second_half[np.random.choice(len(second_half), len(second_half) // 4, replace=False)]
data_array = np.vstack((first_half_selected, second_half_selected))

data_list2 = np.load('data_list_1500变1500_2.npy', allow_pickle=True)
data_array2 = np.vstack(data_list2).astype(np.float32)
data_length2 = len(data_array2)
print(f"第二个数据集形状: {data_array2.shape}")

first_half2 = data_array2[:data_length2 // 2]
second_half2 = data_array2[data_length2 // 2:]
np.random.seed(42)
first_half_selected2 = first_half2[np.random.choice(len(first_half2), len(first_half2) // 4, replace=False)]
second_half_selected2 = second_half2[np.random.choice(len(second_half2), len(second_half2) // 4, replace=False)]
data_array2 = np.vstack((first_half_selected2, second_half_selected2))

X1 = data_array[:, :1248]
y1 = np.array([[np.max(sample[1250:]), np.min(sample[1250:])] for sample in data_array])
X2 = data_array2[:, :1248]
y2 = np.array([[sample[1250], sample[1251]] for sample in data_array2])

X = np.vstack((X1, X2))
y = np.vstack((y1, y2))

scaler = MinMaxScaler()
X_normalized = np.array([scaler.fit_transform(sample.reshape(-1, 1)).flatten() for sample in X])
X = X_normalized
print(f"合并后特征形状: {X.shape}")
print(f"合并后标签形状: {y.shape}")

X = np.expand_dims(X, axis=-1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'训练集: {X_train.shape}, {y_train.shape}')
print(f'验证集: {X_val.shape}, {y_val.shape}')
print(f'测试集: {X_test.shape}, {y_test.shape}')

print("\n正在加载模型 'best_model.h5'...")
model = load_model('best_model_big_1500变1500.h5')
print("模型加载完毕。")

# ===================================================================
# ============= 新增: 单样本预测时间测试 (Single Sample Inference Time Test) =============
# ===================================================================
print("\n--- 开始单样本预测时间测试 ---")
# 1. 模型“预热”，以排除首次运行的额外开销
print("正在进行模型预热...")
_ = model.predict(X_test[0:1], verbose=0)
print("预热完成。")

# 2. 循环测试多个样本以获得稳定的平均值
times = []
num_samples_to_test = 100  # 定义要测试的样本数量
# 确保测试样本数不超过测试集总数
if num_samples_to_test > len(X_test):
    num_samples_to_test = len(X_test)

print(f"将对 {num_samples_to_test} 个独立样本进行计时...")
for i in range(num_samples_to_test):
    single_sample = X_test[i:i + 1]  # 使用切片来保持维度正确

    start_time = time.perf_counter()
    model.predict(single_sample, verbose=0)  # verbose=0 关闭预测过程的进度条
    end_time = time.perf_counter()

    duration = end_time - start_time
    times.append(duration)

# 3. 计算并打印统计结果
average_time_s = np.mean(times)
average_time_ms = average_time_s * 1000
std_dev_ms = np.std(times) * 1000

print(f"\n在 {num_samples_to_test} 次测试中:")
print(f"平均单条数据预测时间: {average_time_ms:.4f} ms ({average_time_s:.6f} s)")
print(f"预测时间标准差: {std_dev_ms:.4f} ms")
print("--- 单样本预测时间测试结束 ---\n")
# ===================================================================
# ========================= 新增代码结束 =========================
# ===================================================================


# --- 模型评估 (您的原始代码) ---
print("正在进行批量预测以评估模型整体性能...")
y_pred = model.predict(X_test)

print("保存预测结果到 y_pred.npy...")
np.save("y_pred.npy", y_pred)

predicted_max = y_pred[:, 0]
predicted_min = y_pred[:, 1]
true_max = y_test[:, 0]
true_min = y_test[:, 1]

metrics = {}


def calculate_mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))


mae_max = calculate_mae(true_max, predicted_max)
mae_min = calculate_mae(true_min, predicted_min)
metrics['MAE'] = [mae_max, mae_min]


def calculate_std(true_values, predicted_values):
    return np.std(true_values - predicted_values)


std_max = calculate_std(true_max, predicted_max)
std_min = calculate_std(true_min, predicted_min)
metrics['STD'] = [std_max, std_min]


def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))


rmse_max = calculate_rmse(true_max, predicted_max)
rmse_min = calculate_rmse(true_min, predicted_min)
metrics['RMSE'] = [rmse_max, rmse_min]

metrics_df = pd.DataFrame(metrics, index=['Max Value', 'Min Value'])
print("\n模型在测试集上的整体评估指标:")
print(metrics_df)

rmse_samples_max = np.abs(true_max - predicted_max)
rmse_samples_min = np.abs(true_min - predicted_min)

thresholds = [5, 10, 15]
proportions_max = {f'RMSE ≤ {th}': np.mean(rmse_samples_max <= th) for th in thresholds}
proportions_min = {f'RMSE ≤ {th}': np.mean(rmse_samples_min <= th) for th in thresholds}

proportions_max_df = pd.DataFrame(list(proportions_max.items()), columns=['Threshold', 'Proportion_Max'])
proportions_min_df = pd.DataFrame(list(proportions_min.items()), columns=['Threshold', 'Proportion_Min'])

print("\n最大值预测误差在阈值内的样本占比:")
print(proportions_max_df)
print("\n最小值预测误差在阈值内的样本占比:")
print(proportions_min_df)

print("\n脚本执行完毕。")