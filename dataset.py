import pandas as pd
import numpy as np
import os

# 定义文件夹路径 (请确保这里是您的实际路径)
folder_path = 'D:\\Matlab\\PPG2BP_ABP_第二批'

# 初始化用于存储 x 和 y 数据的列表
x_list = []
y_list = []

# 初始化计数器
ii = 0

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否为 .xlsx 文件
    if file_name.endswith('.xlsx'):
        print(f"正在处理文件: {ii}")
        ii += 1

        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 读取 Excel 文件，假设没有表头
        data = pd.read_excel(file_path, header=None)

        # 从 DataFrame 中提取 NumPy 数组
        data_values = data.values

        # 切片数据以分离 x 和 y
        # x 是前 1250 个点
        x_data = data_values[:, :1250]
        # y 是接下来的 1250 个点 (从索引 1250 到 2499)
        y_data = data_values[:, 1250:2500]

        # 将切片后的数据添加到相应的列表中
        x_list.append(x_data)
        y_list.append(y_data)

# 将列表转换为 NumPy 数组
# 使用 np.vstack 将多个 (1, 1250) 的数组堆叠成一个 (n, 1250) 的数组，n是文件数量
x_array = np.vstack(x_list)
y_array = np.vstack(y_list)

# 将 x 和 y 数组分别保存到 .npy 文件
np.save('老数据/X.npy', x_array)
np.save('老数据/y.npy', y_array)

print("处理完成！x.npy 和 y.npy 文件已成功生成。")