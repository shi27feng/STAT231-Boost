function [ result ] = get_prediction( fragments, all_features, weak_learners )
%GET_PREDICTION Summary of this function goes here
%   Detailed explanation goes here
result = zeros(1, size(fragments, 3));
for i=1:size(weak_learners, 2),
    display(strcat(['Done with weak learner ', num2str(i)]));
    weak_learner = weak_learners{i};
    error = weak_learner.error;
    threshold = weak_learner.threshold;
    operation = weak_learner.operation;
    feature = all_features{weak_learner.feature};
    diff = get_difference(fragments, feature, false);
    alpha = log((1-error)/error);
    if operation == 1,
        result = result + alpha * sign(diff - threshold);
    else
        result = result + alpha * sign(threshold - diff);
    end
end

end

