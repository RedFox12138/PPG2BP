import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Reshape, UpSampling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, concatenate, Flatten, Dense, UpSampling1D, Activation, add
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

# 检查GPU是否可用
print(tf.config.list_physical_devices('GPU'))

# --- 数据加载与预处理 ---
# 读取数据
X = np.load("老数据/X.npy")
y = np.load("老数据/y.npy")

# 取每个样本的最大值和最小值作为SBP和DBP
y = np.stack((np.max(y, axis=1), np.min(y, axis=1)), axis=1)

# 数据归一化
scaler = MinMaxScaler()
X_normalized = np.array([scaler.fit_transform(sample.reshape(-1, 1)).flatten() for sample in X])
X = X_normalized
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 调整输入数据的形状以适应模型
X = np.expand_dims(X, axis=-1)

# 设置随机种子以保证结果可复现
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'训练集: {X_train.shape}, {y_train.shape}')
print(f'验证集: {X_val.shape}, {y_val.shape}')
print(f'测试集: {X_test.shape}, {y_test.shape}')


# --- 回调函数定义 ---
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
            print(f'\nEpoch {epoch + 1}: val_loss improved to {val_loss:.4f}, val_mae improved to {val_mae:.4f}, saving model to {self.filepath}')

class MetricsCallback(Callback):
    def __init__(self, validation_data, interval=10):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            val_X, val_y = self.validation_data
            y_pred = self.model.predict(val_X, verbose=0)
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

            error_max = np.abs(true_max - predicted_max)
            error_min = np.abs(true_min - predicted_min)
            thresholds = [5, 10, 15]
            proportions_max = {f'Error ≤ {t}': np.mean(error_max <= t) for t in thresholds}
            proportions_min = {f'Error ≤ {t}': np.mean(error_min <= t) for t in thresholds}

            print(f'\n--- Epoch {epoch + 1} Validation Metrics ---')
            print(f'SBP -> MAE: {mae_max:.4f}, STD: {std_max:.4f}, RMSE: {rmse_max:.4f}')
            print(f'DBP -> MAE: {mae_min:.4f}, STD: {std_min:.4f}, RMSE: {rmse_min:.4f}')
            print(f'SBP Proportions: {proportions_max}')
            print(f'DBP Proportions: {proportions_min}')
            print('-------------------------------------\n')

# --- 评估指标计算函数 ---
def calculate_mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

def calculate_std(true_values, predicted_values):
    error = true_values - predicted_values
    return np.std(error)

def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

# =================================================================================
#  1. 定义升级后的第一阶段模型 (参考 UNetDS64)
# =================================================================================
# 导入所有需要的 Keras 层 (这些层在 TF.js 中都有对应)
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, UpSampling2D, Reshape, concatenate, add, Flatten, Dense
)
from tensorflow.keras.models import Model


# =================================================================================
#  辅助函数 (无需改动)
# =================================================================================
def do_upsampling_1d_for_js(x):
    """
    使用 UpSampling2D 模拟 UpSampling1D(size=2) 的功能，以确保 TensorFlow.js 兼容性。
    """
    input_shape = x.shape
    length = input_shape[1]
    channels = input_shape[2]

    x_reshaped = Reshape((length, 1, channels))(x)
    x_upsampled_2d = UpSampling2D(size=[2, 1], interpolation='nearest')(x_reshaped)
    x_restored = Reshape((length * 2, channels))(x_upsampled_2d)
    return x_restored


# =================================================================================
#  1. 定义第一阶段模型 (Upgraded_UNetDS64)，应用简化逻辑
# =================================================================================
def Upgraded_UNetDS64(length, n_channel=1):
    """
    参考 UNetDS64 升级后的 U-Net 模型，内部逻辑已简化以减少参数量并兼容 TF.js
    - MODIFIED: 将每个阶段的两个 Conv1D + BN 块减少到一个。
    - MODIFIED: 将基础滤波器数量从 x=16 减少到 x=12。
    """
    # 优化点 1: 减少基础滤波器数量
    x = 14

    inputs = Input((length, n_channel))

    # --- 编码器路径 ---
    # 优化点 2: 每个 block 从两个 Conv1D 减少到一个
    conv1 = Conv1D(x, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(x * 2, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(x * 4, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x * 8, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(x * 16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)

    # --- 解码器路径 ---
    up6_upsampled = do_upsampling_1d_for_js(conv5)
    up6 = concatenate([up6_upsampled, conv4], axis=2)
    conv6 = Conv1D(x * 8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)

    up7_upsampled = do_upsampling_1d_for_js(conv6)
    up7 = concatenate([up7_upsampled, conv3], axis=2)
    conv7 = Conv1D(x * 4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)

    up8_upsampled = do_upsampling_1d_for_js(conv7)
    up8 = concatenate([up8_upsampled, conv2], axis=2)
    conv8 = Conv1D(x * 2, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)

    up9_upsampled = do_upsampling_1d_for_js(conv8)
    up9 = concatenate([up9_upsampled, conv1], axis=2)
    conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)

    out = Conv1D(1, 1, activation='linear', name="out")(conv9)
    model = Model(inputs=[inputs], outputs=[out])
    return model


# =================================================================================
#  2. 定义第二阶段模型 (Upgraded_MultiResUNet1D)，应用简化逻辑
# =================================================================================
def Upgraded_MultiResUNet1D(length, n_channel=1):
    """
    参考新的 MultiResUNet 升级后的模型，内部逻辑已简化以减少参数量并兼容 TF.js
    - MODIFIED: 简化 MultiResBlock，移除了最长的卷积路径。
    - MODIFIED: 缩小最终全连接层的规模。
    """

    def conv_bn(x, filters, kernel_size, activation='relu', padding='same'):
        x = Conv1D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x

    # 优化点 1: MultiResBlock 内部结构被简化
    def MultiResBlock(U, inp, alpha=2.5):
        W = alpha * U
        # 调整 shortcut 维度以匹配简化后的输出
        shortcut_filters = int(W * 0.167) + int(W * 0.333)
        shortcut = conv_bn(inp, shortcut_filters, 1, activation=None)

        conv3 = conv_bn(inp, int(W * 0.167), 3)
        conv5 = conv_bn(conv3, int(W * 0.333), 3)
        # 移除了最长的 conv7 路径

        # 只拼接 conv3 和 conv5
        out = concatenate([conv3, conv5], axis=-1)
        out = BatchNormalization()(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)
        return out

    def ResPath(filters, path_length, inp):
        shortcut = conv_bn(inp, filters, 1, activation=None)
        out = conv_bn(inp, filters, 3)
        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)
        for _ in range(path_length - 1):
            shortcut = out
            shortcut = conv_bn(shortcut, filters, 1, activation=None)
            out = conv_bn(out, filters, 3)
            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)
        return out

    inputs = Input((length, n_channel))
    U = 14

    mres1 = MultiResBlock(U, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mres1)
    res_path1 = ResPath(mres1.shape[-1], 4, mres1)

    mres2 = MultiResBlock(U * 2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mres2)
    res_path2 = ResPath(mres2.shape[-1], 3, mres2)

    mres3 = MultiResBlock(U * 4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mres3)
    res_path3 = ResPath(mres3.shape[-1], 2, mres3)

    mres4 = MultiResBlock(U * 8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mres4)
    res_path4 = ResPath(mres4.shape[-1], 1, mres4)

    mres5 = MultiResBlock(U * 16, pool4)

    up6_upsampled = do_upsampling_1d_for_js(mres5)
    up6 = concatenate([up6_upsampled, res_path4], axis=-1)
    mres6 = MultiResBlock(U * 8, up6)

    up7_upsampled = do_upsampling_1d_for_js(mres6)
    up7 = concatenate([up7_upsampled, res_path3], axis=-1)
    mres7 = MultiResBlock(U * 4, up7)

    up8_upsampled = do_upsampling_1d_for_js(mres7)
    up8 = concatenate([up8_upsampled, res_path2], axis=-1)
    mres8 = MultiResBlock(U * 2, up8)

    up9_upsampled = do_upsampling_1d_for_js(mres8)
    up9 = concatenate([up9_upsampled, res_path1], axis=-1)
    mres9 = MultiResBlock(U, up9)

    flatten = Flatten()(mres9)
    # 优化点 2: 减小全连接层的规模
    dense1 = Dense(64, activation='relu')(flatten)  # 128 -> 64
    dense2 = Dense(32, activation='relu')(dense1)  # 64 -> 32
    out = Dense(2, activation='linear')(dense2)

    model = Model(inputs=[inputs], outputs=[out])
    return model


# =================================================================================
#  3. 定义最终的串联模型 (无需改动)
# =================================================================================
def CombinedModel(length, n_channel=1):
    """
    此函数无需改动，它会自动调用上面已被优化的模型函数。
    """
    inputs = Input((length, n_channel))
    stage1_model = Upgraded_UNetDS64(length, n_channel)
    stage1_output = stage1_model(inputs)
    stage2_model = Upgraded_MultiResUNet1D(length, 1)
    final_output = stage2_model(stage1_output)
    model = Model(inputs=[inputs], outputs=[final_output])
    return model



# --- 模型构建与训练 ---
length = X_train.shape[1]
n_channel = 1

# 使用最终的、由两个升级后组件构成的 CombinedModel
model = CombinedModel(length, n_channel)
print("--- Combined Model Summary ---")
model.summary()

# 设置学习率衰减
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-6, verbose=1)

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 定义回调函数
checkpoint = CustomModelCheckpoint('best_combined_upgraded_model.h5')
metrics_callback = MetricsCallback((X_val, y_val), interval=10)

# 训练模型
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=128,  # 由于模型变大，可以适当减小batch_size防止显存不足
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, metrics_callback, lr_schedule])

# --- 模型评估与可视化 ---
print("\n--- Final Test Set Evaluation ---")
# 加载表现最好的模型进行最终评估
model.load_weights('best_combined_upgraded_model.h5')
y_pred = model.predict(X_test)

# 保存预测结果
np.save("y_pred_upgraded.npy", y_pred)

predicted_max = y_pred[:, 0]
predicted_min = y_pred[:, 1]
true_max = y_test[:, 0]
true_min = y_test[:, 1]

metrics = {}
metrics['MAE Max'] = calculate_mae(true_max, predicted_max)
metrics['MAE Min'] = calculate_mae(true_min, predicted_min)
metrics['STD Max'] = calculate_std(true_max, predicted_max)
metrics['STD Min'] = calculate_std(true_min, predicted_min)
metrics['RMSE Max'] = calculate_rmse(true_max, predicted_max)
metrics['RMSE Min'] = calculate_rmse(true_min, predicted_min)

metrics_df = pd.DataFrame(metrics, index=['Value'])
print("\nTest Set Metrics:")
print(metrics_df)

# 计算误差在特定阈值内的样本比例
error_max = np.abs(true_max - predicted_max)
error_min = np.abs(true_min - predicted_min)
thresholds = [5, 10, 15]

proportions_max = {f'Error ≤ {t}': np.mean(error_max <= t) for t in thresholds}
proportions_min = {f'Error ≤ {t}': np.mean(error_min <= t) for t in thresholds}

print("\nSBP Prediction Accuracy:")
print(pd.DataFrame(list(proportions_max.items()), columns=['Threshold', 'Proportion']))
print("\nDBP Prediction Accuracy:")
print(pd.DataFrame(list(proportions_min.items()), columns=['Threshold', 'Proportion']))

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(history.history['loss'], label='训练损失 (Training Loss)')
plt.plot(history.history['val_loss'], label='验证损失 (Validation Loss)')
plt.title('模型损失随训练轮次的变化', fontsize=16)
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('均方误差损失 (MSE Loss)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 可视化预测结果与真实值的对比
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(true_max, predicted_max, alpha=0.6, edgecolors='w', s=50)
plt.plot([min(true_max), max(true_max)], [min(true_max), max(true_max)], '--', color='red', lw=2, label='理想情况 (y=x)')
plt.title('收缩压(SBP): 真实值 vs 预测值', fontsize=16)
plt.xlabel('真实 SBP (mmHg)', fontsize=12)
plt.ylabel('预测 SBP (mmHg)', fontsize=12)
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(true_min, predicted_min, alpha=0.6, edgecolors='w', s=50, c='green')
plt.plot([min(true_min), max(true_min)], [min(true_min), max(true_min)], '--', color='red', lw=2, label='理想情况 (y=x)')
plt.title('舒张压(DBP): 真实值 vs 预测值', fontsize=16)
plt.xlabel('真实 DBP (mmHg)', fontsize=12)
plt.ylabel('预测 DBP (mmHg)', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()