clear all;
% 读取MAT文件
data1 = load('Part_1.mat');
data2 = load('Part_2.mat');
data3 = load('Part_3.mat');
data4 = load('Part_4.mat');

% 定义窗长和间隔
window_length = 1250;
step_size = 625;
%%
% 初始化保存所有PPG片段的数组
all_ppg_segments = {};
folder_name = 'PPG2BP_ABP';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end
% 读取每组PPG数 据并进行分段
ppg_data1 = data1.Part_1;  % 获取PPG数据
ppg_data2 = data2.Part_2;  % 获取PPG数据
ppg_data3 = data3.Part_3;  % 获取PPG数据
ppg_data4 = data4.Part_4;  % 获取PPG数据
ppg_data=[ppg_data1 ppg_data2 ppg_data3 ppg_data4];
%%
%%上一次停留在1372
ii=0;
for i = 1:length(ppg_data)
    ppg_signal = ppg_data{i};  % 获取单个cell中的PPG信号
    i
    % 进行分段
    start_idx = 1;
    while start_idx + window_length - 1 <= length(ppg_signal)
        segment = ppg_signal(1,start_idx:start_idx + window_length - 1);
        segment = HPF(segment,125,0.15);
        segment = LPF(segment,125,15);
        ABP=ppg_signal(2,start_idx:start_idx + window_length - 1);
        %SBP=max(ppg_signal(2,start_idx:start_idx + window_length - 1));
        %DBP=min(ppg_signal(2,start_idx:start_idx + window_length - 1));
        segment=[segment ABP];
        all_ppg_segments{end + 1} = [segment];  % 添加到数组中
        % 将片段保存为Excel文件
        filename = sprintf('%s/segment_%d.xlsx', folder_name,ii);
        ii=ii+1;
        writematrix(segment, filename);
        start_idx = start_idx + step_size;
    end
end

% % 保存所有PPG片段到MAT文件
% save('ppg_segments.mat', 'all_ppg_segments');
%%
% folder_path = 'D:\Matlab\bin\PPG2BP_ABP_12s_6s';
% output_file = 'merged_data.xlsx';
% 
% % 获取文件夹中的所有Excel文件
% file_list = dir(fullfile(folder_path, '*.xlsx'));
% 
% % 初始化空表格
% all_data = [];
% 
% % 遍历所有Excel文件并读取数据
% for i = 1:length(file_list)
%     i
%     file_path = fullfile(folder_path, file_list(i).name);
%     data = readtable(file_path, 'ReadVariableNames', false);
%     all_data = [all_data; data];
% end
% 
% % 将合并后的数据写入一个新的Excel文件
% writetable(all_data, output_file, 'WriteVariableNames', false);
folder_path = 'D:\Matlab\PPG2BP_BP_第二批';
output_file = 'merged_data.xlsx';

% 获取文件夹中的所有Excel文件
file_list = dir(fullfile(folder_path, '*.xlsx'));

% 初始化空表格
all_data = [];

% 使用并行处理
parfor i = 1:length(file_list)
    i
    file_path = fullfile(folder_path, file_list(i).name);
    data = readtable(file_path, 'ReadVariableNames', false);
    
    % 使用 cell 数组来存储每个文件的数据
    data_cells{i} = data;
end

% 将所有数据合并到一起
all_data = vertcat(data_cells{:});

% 将合并后的数据写入一个新的Excel文件
writetable(all_data, output_file, 'WriteVariableNames', false);