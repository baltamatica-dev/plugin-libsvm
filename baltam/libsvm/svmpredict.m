function [predicted_label, accuracy, prob_estimates] = svmpredict(label_vector, instance_matrix, model, libsvm_options)
%svmpredict - 依据模型进行预测
%
% Usage: 
%   [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
%   [predicted_label, accuracy, prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, '-b 1')
%   [predicted_label] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
%
%    Parameters:
%       model: SVM model structure from svmtrain.
%       libsvm_options:
%           -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
%           -q : quiet mode (no outputs)
%
%    Returns:
%       predicted_label:    SVM prediction output vector.
%       accuracy:           a vector with accuracy, mean squared error, squared correlation coefficient.
%       prob_estimates:     If selected, probability estimate vector.

%% 输入参数检查+预处理
% 转置 instance_matrix 便于后续使用
% instance_matrix_T = sparse(transpose(_todo_instance_matrix_full));
if 3 == nargin
    char_libsvm_options = '-b 0';
elseif 4 == nargin
    char_libsvm_options = char(libsvm_options);
else
    % 参数过少，自动引发错误
    __svmpredict_impl();
    return;
end

% TODO: 
%   - size() 可以正确计算稀疏数组
%   - transpose() 支持稀疏数组时
_todo_instance_matrix_full = full(instance_matrix);

[label_M, label_N] = size(full(label_vector));
[instance_M, instance_N] = size(_todo_instance_matrix_full);
if label_M ~= instance_M
    error(sprintf( ...
        "输入标签和模型维度不匹配。请检查输入参数。\n\t" + ...
            "label_length:%d ~= instance_length:%d", ...
        label_M, instance_M ...
    ));
end % if label_M ~= instance_M

% 预转置 SVs
model.SVs = sparse(transpose(full(model.SVs)));

%% 输入参数检查 + 调用 bex 函数
if 3 == nargout
    [predicted_label, accuracy, prob_estimates] =  __svmpredict_impl(label_vector, _todo_instance_matrix_full, model, char_libsvm_options);
elseif 1 == nargout
    [predicted_label] =  __svmpredict_impl(label_vector, _todo_instance_matrix_full, model, char_libsvm_options);
elseif 0 == nargout
    __svmpredict_impl(label_vector, _todo_instance_matrix_full, model, char_libsvm_options);
else
    % 参数过少，自动引发错误
    __svmpredict_impl();
end

%% 返回值后处理

end % SPDX-License-Identifier: BSD-3-Clause