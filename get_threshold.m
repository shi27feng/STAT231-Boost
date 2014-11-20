function [ error_min, threshold, operation ] = get_threshold( diff_sorted, order, label, weight )
%GET_THRESHOLD Summary of this function goes here
%   Detailed explanation goes here
[unique_diff, unique_diff_index] = unique(diff_sorted);
n = size(unique_diff, 2);
label_ordered = label(order);
weight_ordered = weight(order);
errors = zeros(1,n);
operations = zeros(1,n);

total_positive = sum(weight_ordered(label_ordered==1));
total_negative = 1 - total_positive;
positive_so_far = 0;
negative_so_far = 0;

prev = 0;

for i=1:n,
    for j=prev+1:unique_diff_index(i),
        if label_ordered(j) == 0,
            negative_so_far = negative_so_far + weight_ordered(j);
        else
            positive_so_far = positive_so_far + weight_ordered(j);
        end
    end
    prev = unique_diff_index(i);
    [errors(i), operations(i)] = min([positive_so_far + total_negative - negative_so_far, negative_so_far + total_positive - positive_so_far]);
end
    [error_min, error_min_index] = min(errors);
    threshold = unique_diff(error_min_index);
    operation = operations(error_min_index);
end

