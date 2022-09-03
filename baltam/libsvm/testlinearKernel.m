% 注意先初始化插件 svminit();
%% 测试预计算的核函数 (precomputed kernel)
%   改自: testheartScaleDataset.m: 使用预计算的线性核函数
load('heart_scale.mat')
% .mat 导出两个变量: heart_scale_label, heart_scale_inst

%% 线性核函数预计算
n = size(heart_scale_inst, 1);
K = heart_scale_inst * heart_scale_inst';
K1 = [(1:n)', K];  % include sample serial number as first column

%% 训练模型
disp('训练模型');
% note: 
%	-t kernel_type=4;   Use precomputed kernel
model = svmtrain(heart_scale_label, K1, '-t 4');
% ..*.*
% optimization finished, #iter = 1010
% nu = 0.350371
% obj = -92.473356, rho = -1.050690
% nSV = 101, nBSV = 88
% Total nSV = 101

%% 用模型预测
disp('用模型预测');
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, K1, model);
% Accuracy = 84.8148% (229/270) (classification)
