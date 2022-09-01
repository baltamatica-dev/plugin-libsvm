function svmwrite(filename, label_vector, instance_matrix)
%svmwrite - 将问题写入 .svm 文件
%
% Usage:
%   svmwrite('filename.svm', label_vector, instance_matrix);
%

% TODO: 
%   - size() 可以正确计算稀疏数组
%   - transpose() 支持稀疏数组时
_todo_instance_matrix_full = full(instance_matrix);

%% 输入参数检查
[label_M, label_N] = size(full(label_vector));
[instance_M, instance_N] = size(_todo_instance_matrix_full);
if label_M ~= instance_M
    error(sprintf( ...
        "输入标签和模型维度不匹配。请检查输入参数。\n\t" + ...
            "label_length:%d ~= instance_length:%d", ...
        label_M, instance_M ...
    ));
end % if label_M ~= instance_M

%% 输入预处理
% 转置 instance_matrix 便于后续使用
instance_matrix_T = sparse(transpose(_todo_instance_matrix_full));
__svmwrite_impl(char(filename), label_vector, instance_matrix_T);

end