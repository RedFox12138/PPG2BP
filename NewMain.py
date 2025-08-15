import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, concatenate, Flatten, Dense, UpSampling2D, Reshape, Activation, add
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
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


def LightweightUNetDS16(length, n_channel=1):
    x = 16  # Number of filters

    inputs = Input((length, n_channel))
    conv1 = Conv1D(x, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x * 2, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x * 2, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x * 4, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x * 4, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x * 8, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x * 8, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x * 16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x * 16, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Manual upsampling and reshaping
    shape = conv5.shape
    up6 = Reshape((shape[1], 1, shape[2]))(conv5)
    up6 = UpSampling2D(size=(2, 1))(up6)
    up6 = Reshape((shape[1] * 2, shape[2]))(up6)
    up6 = Conv1D(x * 8, 3, activation='relu', padding='same')(up6)
    up6 = concatenate([up6, conv4], axis=2)

    conv6 = Conv1D(x * 8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x * 8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    shape = conv6.shape
    up7 = Reshape((shape[1], 1, shape[2]))(conv6)
    up7 = UpSampling2D(size=(2, 1))(up7)
    up7 = Reshape((shape[1] * 2, shape[2]))(up7)
    up7 = Conv1D(x * 4, 3, activation='relu', padding='same')(up7)
    up7 = concatenate([up7, conv3], axis=2)

    conv7 = Conv1D(x * 4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x * 4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    shape = conv7.shape
    up8 = Reshape((shape[1], 1, shape[2]))(conv7)
    up8 = UpSampling2D(size=(2, 1))(up8)
    up8 = Reshape((shape[1] * 2, shape[2]))(up8)
    up8 = Conv1D(x * 2, 3, activation='relu', padding='same')(up8)
    up8 = concatenate([up8, conv2], axis=2)

    conv8 = Conv1D(x * 2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x * 2, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    shape = conv8.shape
    up9 = Reshape((shape[1], 1, shape[2]))(conv8)
    up9 = UpSampling2D(size=(2, 1))(up9)
    up9 = Reshape((shape[1] * 2, shape[2]))(up9)
    up9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    up9 = concatenate([up9, conv1], axis=2)

    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    out = Conv1D(1, 1, activation='linear', name="out")(conv9)

    model = Model(inputs=[inputs], outputs=[out])
    return model

def MultiResUNet1D(length, n_channel=1):
    def conv2d_bn(x, filters, kernel_size, padding='same', activation='relu', name=None):
        x = Conv1D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation, name=name)(x)
        return x

    def MultiResBlock(U, inp, alpha=1.67):  # Use alpha for flexible filter sizes
        W = alpha * U
        shortcut = conv2d_bn(inp, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, activation=None)
        conv3x3 = conv2d_bn(inp, int(W * 0.167), 3)
        conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3)
        conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3)
        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = add([shortcut, out])
        out = Activation('relu')(out)
        return out

    def ResPath(filters, length, inp):
        shortcut = conv2d_bn(inp, filters, 1, activation=None)
        out = conv2d_bn(inp, filters, 3, activation='relu')
        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)
        for _ in range(length - 1):
            shortcut = conv2d_bn(out, filters, 1, activation=None)
            out = conv2d_bn(out, filters, 3, activation='relu')
            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)
        return out

    inputs = Input((length, n_channel))

    mresblock1 = MultiResBlock(16, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(16, 4, mresblock1)

    mresblock2 = MultiResBlock(16*2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(16*2, 3, mresblock2)

    mresblock3 = MultiResBlock(16*4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(16*4, 2, mresblock3)

    mresblock4 = MultiResBlock(16*4, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(16*4, 1, mresblock4)

    mresblock5 = MultiResBlock(16*8, pool4)

    shape = mresblock5.shape
    up6 = Reshape((shape[1], 1, shape[2]))(mresblock5)
    up6 = UpSampling2D(size=(2, 1))(up6)
    up6 = Reshape((shape[1] * 2, shape[2]))(up6)
    up6 = Conv1D(16 * 4, 3, activation='relu', padding='same')(up6)
    up6 = concatenate([up6, mresblock4], axis=-1)
    mresblock6 = MultiResBlock(16*4, up6)

    shape = mresblock6.shape
    up7 = Reshape((shape[1], 1, shape[2]))(mresblock6)
    up7 = UpSampling2D(size=(2, 1))(up7)
    up7 = Reshape((shape[1] * 2, shape[2]))(up7)
    up7 = Conv1D(16 * 4, 3, activation='relu', padding='same')(up7)
    up7 = concatenate([up7, mresblock3], axis=-1)
    mresblock7 = MultiResBlock(16*4, up7)

    shape = mresblock7.shape
    up8 = Reshape((shape[1], 1, shape[2]))(mresblock7)
    up8 = UpSampling2D(size=(2, 1))(up8)
    up8 = Reshape((shape[1] * 2, shape[2]))(up8)
    up8 = Conv1D(16 * 2, 3, activation='relu', padding='same')(up8)
    up8 = concatenate([up8, mresblock2], axis=-1)
    mresblock8 = MultiResBlock(16*2, up8)

    shape = mresblock8.shape
    up9 = Reshape((shape[1], 1, shape[2]))(mresblock8)
    up9 = UpSampling2D(size=(2, 1))(up9)
    up9 = Reshape((shape[1] * 2, shape[2]))(up9)
    up9 = Conv1D(16, 3, activation='relu', padding='same')(up9)
    up9 = concatenate([up9, mresblock1], axis=-1)
    mresblock9 = MultiResBlock(16, up9)

    flatten = Flatten()(mresblock9)
    dense1 = Dense(16, activation='relu')(flatten)
    out = Dense(2, activation='linear')(dense1)

    model = Model(inputs=[inputs], outputs=[out])

    return model

def CombinedModel(length, n_channel=1):
    inputs = Input((length, n_channel))

    # First UNet model
    unet_model = LightweightUNetDS16(length, n_channel)
    unet_output = unet_model(inputs)

    # Second MultiResUNet model
    multiresunet_model = MultiResUNet1D(length, 1)
    final_output = multiresunet_model(unet_output)

    model = Model(inputs=[inputs], outputs=[final_output])
    return model

length = 1248
n_channel = 1
model = CombinedModel(length, n_channel)

# 设置学习率衰减
initial_learning_rate = 0.001
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=20, min_lr=1e-6, verbose=1)

# 编译模型
optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 定义自定义回调函数
checkpoint = CustomModelCheckpoint('best_model.h5')
metrics_callback = MetricsCallback((X_val, y_val), interval=10)

# 训练模型
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint, metrics_callback])

# 评估模型
y_pred = model.predict(X_test)
# 保存模型
model.save('model_new.h5')
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
