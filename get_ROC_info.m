function [ TP, FP ] = get_ROC_info( result, all_labels )
%GET_ROC_INFO Summary of this function goes here
%   Detailed explanation goes here
TP = zeros(1, size(result, 2));
FP = zeros(1, size(result, 2));

for i=1:size(result, 2),
    if mod(i, 100) == 0,
        display(strcat(['Getting: ', num2str(i)]));
    end
    TP(i) = sum((result > result(i)) & all_labels);
    FP(i) = sum((result > result(i)) & all_labels == 0);
end

[TP, order] = sort(TP);
FP = FP(order);
end

