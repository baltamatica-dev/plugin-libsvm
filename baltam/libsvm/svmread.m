function [label_vector, instance_matrix] = svmread(filename)
%svmread - 从 .svm 文件中读取问题
%
% Usage:
%   [label_vector, instance_matrix] = svmread('filename.svm');
%

%% 输入参数检查
% -- 参数个数
if 1 ~= nargin
    disp("输入参数应为 1 个.");
    % 引发错误输出帮助
    __svmread_impl();
    return;
end

%% 输入预处理
% 文件名兼容字符串
filename_chars = char(filename);

%% 调用 impl 函数
[label_vector, _instance_matrix] = __svmread_impl(filename_chars);

%% 输出后处理
% TODO:
%   - transpose() 支持稀疏数组时
_todo_instance_matrix_full = full(_instance_matrix);
instance_matrix = sparse(transpose(_todo_instance_matrix_full));

end % SPDX-License-Identifier: BSD-3-Clause