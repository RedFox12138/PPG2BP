import sys
sys.path.append('D:\\anaconda\\lib\\site-packages')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# 读取数据
X = np.load("老数据/X.npy")
y = np.load("老数据/y.npy")

# 取每个样本的最大值和最小值作为SBP和DBP
y = np.stack((np.max(y, axis=1), np.min(y, axis=1)), axis=1)

scaler = MinMaxScaler()
X_normalized = np.array([scaler.fit_transform(sample.reshape(-1, 1)).flatten() for sample in X])
X = X_normalized
print(X.shape)
print(y.shape)

# 调整输入数据的形状以适应模型
X = np.expand_dims(X, axis=-1)

seed = 42
tf.random.set_seed(seed)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'训练集: {X_train.shape}, {y_train.shape}')
print(f'验证集: {X_val.shape}, {y_val.shape}')
print(f'测试集: {X_test.shape}, {y_test.shape}')

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.best_loss = np.Inf
        self.best_mae = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_mae = logs.get('val_mae')

        if val_loss < self.best_loss and val_mae < self.best_mae:
            self.best_loss = val_loss
            self.best_mae = val_mae
            self.model.save(self.filepath)
            print(f'\nEpoch {epoch + 1}: val_loss improved to {val_loss}, val_mae improved to {val_mae}, saving model to {self.filepath}')

class MetricsCallback(Callback):
    def __init__(self, validation_data, interval=10):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            val_X, val_y = self.validation_data
            y_pred = self.model.predict(val_X)
            predicted_max = y_pred[:, 0]
            predicted_min = y_pred[:, 1]
            true_max = val_y[:, 0]
            true_min = val_y[:, 1]

            mae_max = calculate_mae(true_max, predicted_max)
            mae_min = calculate_mae(true_min, predicted_min)
            std_max = calculate_std(true_max, predicted_max)
            std_min = calculate_std(true_min, predicted_min)
            rmse_max = calculate_rmse(true_max, predicted_max)
            rmse_min = calculate_rmse(true_min, predicted_min)

            rmse_samples_max = abs(true_max - predicted_max)
            rmse_samples_min = abs(true_min - predicted_min)
            thresholds = [5, 10, 15]
            proportions_max = {f'RMSE ≤ {threshold}': np.mean(rmse_samples_max <= threshold) for threshold in thresholds}
            proportions_min = {f'RMSE ≤ {threshold}': np.mean(rmse_samples_min <= threshold) for threshold in thresholds}

            print(f'\nEpoch {epoch + 1} validation metrics:')
            print(f'MAE Max: {mae_max}, MAE Min: {mae_min}')
            print(f'STD Max: {std_max}, STD Min: {std_min}')
            print(f'RMSE Max: {rmse_max}, RMSE Min: {rmse_min}')
            print(f'Proportions Max: {proportions_max}')
            print(f'Proportions Min: {proportions_min}')

def calculate_mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

def calculate_std(true_values, predicted_values):
    y_diff_mean = np.mean(true_values - predicted_values)
    squared_diff = (true_values - predicted_values - y_diff_mean) ** 2
    std = np.sqrt(np.sum(squared_diff) / (len(true_values) - 1))
    return std

def calculate_rmse(true_values, predicted_values):
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    return rmse


length = 1248
n_channel = 1

# 加载预训练模型
best_model = load_model('best_model.h5')

# 设置学习率衰减
initial_learning_rate = 0.0001
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-6, verbose=1)

# 编译模型
optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999)
best_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 定义自定义回调函数
checkpoint = CustomModelCheckpoint('best_model.h5')
metrics_callback = MetricsCallback((X_val, y_val), interval=10)

# 继续训练模型
history = best_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint, metrics_callback])

# 评估模型
y_pred = best_model.predict(X_test)
# 保存模型
best_model.save('model_new.h5')
# 读取npy文件
np.save("y_pred.npy", y_pred)

predicted_max = y_pred[:, 0]
predicted_min = y_pred[:, 1]
true_max = y_test[:, 0]
true_min = y_test[:, 1]

metrics = {}

# MAE calculation
mae_max = calculate_mae(true_max, predicted_max)
mae_min = calculate_mae(true_min, predicted_min)
metrics['MAE Max'] = mae_max
metrics['MAE Min'] = mae_min

# Calculate the STD for max and min values
std_max = calculate_std(true_max, predicted_max)
std_min = calculate_std(true_min, predicted_min)
metrics['STD Max'] = std_max
metrics['STD Min'] = std_min

# RMSE calculation
rmse_max = calculate_rmse(true_max, predicted_max)
rmse_min = calculate_rmse(true_min, predicted_min)
metrics['RMSE Max'] = rmse_max
metrics['RMSE Min'] = rmse_min

# 转换为DataFrame
metrics_df = pd.DataFrame(metrics, index=['Value'])
print(metrics_df)

# 计算每个样本的RMSE
rmse_samples_max = abs(true_max - predicted_max)
rmse_samples_min = abs(true_min - predicted_min)

# 定义阈值
thresholds = [5, 10, 15]

# 计算每个阈值内的样本占比
proportions_max = {}
proportions_min = {}

for threshold in thresholds:
    proportions_max[f'RMSE ≤ {threshold}'] = np.mean(rmse_samples_max <= threshold)
    proportions_min[f'RMSE ≤ {threshold}'] = np.mean(rmse_samples_min <= threshold)

# 转换为DataFrame以便显示
proportions_max_df = pd.DataFrame(list(proportions_max.items()), columns=['Threshold', 'Proportion'])
proportions_min_df = pd.DataFrame(list(proportions_min.items()), columns=['Threshold', 'Proportion'])

print(proportions_max_df)
print(proportions_min_df)

# 绘制训练结果的可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失随训练轮次的变化')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.legend()

plt.tight_layout()
plt.show()

# 可视化预测结果与真实值的对比
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(true_max, predicted_max, alpha=0.5)
plt.title('最大值: 真实值 vs 预测值')
plt.xlabel('真实最大值')
plt.ylabel('预测最大值')

plt.subplot(1, 2, 2)
plt.scatter(true_min, predicted_min, alpha=0.5)
plt.title('最小值: 真实值 vs 预测值')
plt.xlabel('真实最小值')
plt.ylabel('预测最小值')

plt.tight_layout()
plt.show()
