function [label_vector, instance_matrix] = svmread(filename)
%svmread - 从 .svm 文件中读取问题
%
% Usage:
%   [label_vector, instance_matrix] = svmread('filename.svm');
%

[label_vector, _instance_matrix] = __svmread_impl(char(filename));
_todo_instance_matrix_full = full(_instance_matrix);
instance_matrix = sparse(transpose(_todo_instance_matrix_full));

end