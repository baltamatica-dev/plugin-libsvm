% 注意先初始化插件 svminit();
%% 测试 probability estimates
%   改自: testheartScaleDataset.m: 训练、预测增加了 '-b 1' 选项
load('heart_scale.mat')
% .mat 导出两个变量: heart_scale_label, heart_scale_inst

%% 训练模型
disp('训练模型');
% note: 
%   -c cost=1;      parameter C of C-SVC
%   -g gamma=0.07;  gamma in kernel function
%   -b 1;           train a SVC model for probability estimates
model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 2 -b 1');
% .*
% optimization finished, #iter = 325
% nu = 0.764961
% obj = -87.613595, rho = 0.026995
% nSV = 208, nBSV = 71
% Total nSV = 208
%
% ...
%
% .*
% optimization finished, #iter = 410
% nu = 0.729836
% obj = -106.601572, rho = 0.055112
% nSV = 252, nBSV = 71
% Total nSV = 252

%% 用模型预测
disp('用模型预测');
% note: 
%   - option `-b 1` see note for svmtrain
%   - 这里返回的是 prob_estimates 而不是 dec_values
[predict_label, accuracy, prob_estimates] = svmpredict(heart_scale_label, heart_scale_inst, model, '-b 1');
% Accuracy = 99.2593% (268/270) (classification)
