% 注意先初始化插件 svminit();
%% 测试写入并读取 svm 格式
load('heart_scale.mat')
% .mat 导出两个变量: heart_scale_label, heart_scale_inst

%% 写入问题到 .svm 文件
svmwrite('heart_scale.svm', heart_scale_label, heart_scale_inst);
% 目录下出现 heart_scale.svm 文件
ls();

%% 从 .svm 文件读取问题，并与输入比较做验证
[label_vector0, instance_matrix0] = svmread('heart_scale.svm');
% note: instance_matrix0 为稀疏矩阵. 不支持直接比较稀疏与稠密矩阵. 需要转成稠密矩阵再比较
all(label_vector0 == heart_scale_label) && all(full(instance_matrix0) == heart_scale_inst, 'all')
% 1
