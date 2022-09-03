function model = svmtrain(label_vector, instance_matrix, libsvm_options)
%svmread - 训练 SVM 模型
%
% Usage:
%   model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
%
%    libsvm_options:
%    -s svm_type    : set type of SVM (default 0)
%        0 -- C-SVC         (multi-class classification)
%        1 -- nu-SVC        (multi-class classification)
%        2 -- one-class SVM
%        3 -- epsilon-SVR   (regression)
%        4 -- nu-SVR        (regression)
%    -t kernel_type : set type of kernel function (default 2)
%        0 -- linear: u'*v
%        1 -- polynomial: (gamma*u'*v + coef0)^degree
%        2 -- radial basis function: exp(-gamma*|u-v|^2)
%        3 -- sigmoid: tanh(gamma*u'*v + coef0)
%        4 -- precomputed kernel (kernel values in training_instance_matrix)
%    -d degree      : set degree in kernel function (default 3)
%    -g gamma       : set gamma in kernel function (default 1/num_features)
%    -r coef0       : set coef0 in kernel function (default 0)
%    -c cost        : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%    -n nu          : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%    -p epsilon     : set the epsilon in loss function of epsilon-SVR (default 0.1)
%    -m cachesize   : set cache memory size in MB (default 100)
%    -e epsilon     : set tolerance of termination criterion (default 0.001)
%    -h shrinking   : whether to use the shrinking heuristics, 0 or 1 (default 1)
%    -b probability_estimates : 
%           whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%    -wi weight     : set the parameter C of class i to weight*C, for C-SVC (default 1)
%    -v n           : n-fold cross validation mode
%    -q             : quiet mode (no outputs)

%% 输入参数检查
% -- 参数个数
if 3 ~= nargin
    disp("输入参数应为 3 个.");
    % 引发错误输出帮助
    __svmwrite_impl();
    return;
end

% -- 参数维度
% TODO: 
%   - size() 可以正确计算稀疏数组
%   - transpose() 支持稀疏数组时
_todo_instance_matrix_full = full(instance_matrix);
[label_M, label_N] = size(full(label_vector));
[instance_M, instance_N] = size(_todo_instance_matrix_full);
if label_M ~= instance_M
    disp(sprintf( ...
        "输入标签和模型维度不匹配。请检查输入参数。\n\t" + ...
            "label_length:%d ~= instance_length:%d", ...
        label_M, instance_M ...
    ));
    return;
end % if label_M ~= instance_M

%% 输入预处理
% 转置 instance_matrix 便于后续使用 (稠密形式无需转置)
% instance_matrix_T = sparse(transpose(_todo_instance_matrix_full));
% 文件名兼容字符串
libsvm_options_chars = char(libsvm_options);

%% 调用 bex 函数
% note: instance_matrix 输入稠密形式。避开预先计算核函数的 full 调用
% TODO: 优化 full 调用：仅针对需要的情况展开参数为稠密形式
_model = __svmtrain_impl(label_vector, _todo_instance_matrix_full, libsvm_options_chars);

%% 返回值后处理
% 转置 .SVs
% TODO: transpose() 支持稀疏数组时: 消除 full/sparse 调用
validation_mode = _parse_opt(libsvm_options);
if ~validation_mode
    _model.SVs = sparse(transpose(full(_model.SVs)));
end
model = _model;

end % SPDX-License-Identifier: BSD-3-Clause

%% 解析 libsvm_options 判断是否处于交叉验证模式下
function has_v = _parse_opt(libsvm_options)
has_v = false;
char_libsvm_options = char(libsvm_options);
is_flag = false;

for c = char_libsvm_options
    if ' ' == c
        continue;
    end
    if '-' == c
        is_flag = true;
        continue;
    end
    if is_flag
        if 'v' == c
            has_v = true;
            break;
        end
        is_flag = false;
        continue;
    end
end

end % function has_v = _parse_opt(libsvm_options)