% 注意先初始化插件 svminit();
%% 测试加载 heart_scale 数据集并训练
load('heart_scale.mat')
% .mat 导出两个变量: heart_scale_label, heart_scale_inst

%% 训练模型
disp('训练模型');
% note: 
%	-c cost=1; parameter C of C-SVC
%	-g gamma=0.07; gamma in kernel function
model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');
% *
% optimization finished, #iter = 134
% nu = 0.433785
% obj = -101.855060, rho = 0.426412
% nSV = 130, nBSV = 107
% Total nSV = 130

%% 用模型预测
disp('用模型预测');
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model);
% Accuracy = 86.6667% (234/270) (classification)
