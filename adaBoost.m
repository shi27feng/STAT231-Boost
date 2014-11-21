function [ weak_learners ] = adaBoost( all_features, label, weight, loops, path )
%ADABOOST Summary of this function goes here
%   Detailed explanation goes here
n = size(all_features, 2);
weak_learners = cell(1, loops);
for i=1:loops,
    weight = weight / sum(weight);
    errors = zeros(1,n);
    thresholds = zeros(1,n);
    operations = zeros(1,n);
    for j=1:size(all_features, 2),
        if mod(j, 100) == 0,
            display(strcat(['Get error for iteration: ', num2str(i), ', feature ', num2str(j)]));
        end
        feature = all_features{j};
        feature_path = strcat([path, 'feature_', num2str(feature.type), '_', num2str(feature.index), '.mat']);
        load(feature_path);
        [errors(j), thresholds(j), operations(j)] = get_threshold(diff_sorted, order, label, weight);
    end
    
    [error_min, error_index_min] = min(errors);
    
    weak_learners{i} = struct('feature', error_index_min, 'error', error_min, 'threshold', thresholds(error_index_min), 'operation', operations(error_index_min));
    save(strcat(['../iterations/iteration_', num2str(i), '.mat']), 'weight', 'errors', 'thresholds', 'operations');       

    beta = error_min/(1-error_min);
    operation = operations(error_index_min);
    threshold = thresholds(error_index_min);
    
    feature = all_features{error_index_min};
    feature_path = strcat(['../features/feature_', num2str(feature.type), '_', num2str(feature.index), '.mat']);
    load(feature_path);
    [tem, original_order] = sort(order);
    diff = diff_sorted(original_order);
    
    if operation==1,
        predicted = ((diff - threshold)>0);
    else
        predicted = ((diff - threshold)<=0);
    end
    
    ei = abs(label - predicted);
    weight = weight .*(beta.^(1 - ei));
    
end

end

