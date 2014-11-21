% Preprocess images
image_dimension = [16, 16];
face_image_path = '../data/newface16/';
face_images = get_images(face_image_path, 'bmp', image_dimension);
face_images_integral = get_integral(face_images);
nonface_image_path = '../data/nonface16/';
nonface_images = get_images(nonface_image_path, 'bmp', image_dimension);
nonface_images_integral = get_integral(nonface_images);
all_images_integral = cat(3, face_images_integral, nonface_images_integral);
save('input.mat', 'face_images_integral', 'nonface_images_integral');

% Get all features
feature_set_1 = get_feature_set_1(image_dimension);
feature_set_2 = get_feature_set_2(image_dimension);
feature_set_3 = get_feature_set_3(image_dimension);
feature_set_4 = get_feature_set_4(image_dimension);
all_features = [feature_set_1, feature_set_2, feature_set_3, feature_set_4];
save('features.mat', 'all_features');

% Get features from image
for i=1:size(all_features,2),
    if mod(i, 50) == 0,
        display(strcat(['Get difference for ', num2str(i), ' features']));
    end
    get_difference(all_images_integral, all_features{i}, true);
end

% Adaboost
m = size(face_images_integral,3);
n = size(nonface_images_integral,3);
all_labels = [ones(1,m),zeros(1,n)];
all_weights = [ones(1,m)/m, ones(1,n)/n];
weak_learners = adaBoost(all_features, all_labels, all_weights, 100, '../iterations/');
save('weak_learners.mat', 'weak_learners')

% Plot histograms for sample features
plot_histogram_sample_feature( all_features, weak_learners{1}, m, '../plots/1-1.png', 'Sample histogram for feature type 1', [-300, 300], [0, 7000], -300:10:300);
plot_histogram_sample_feature( all_features, weak_learners{3}, m, '../plots/1-2.png', 'Sample histogram for feature type 2', [-1500, 1500], [0, 6000], -1500:50:1500);
plot_histogram_sample_feature( all_features, weak_learners{13}, m, '../plots/1-3.png', 'Sample histogram for feature type 3', [-2500, 500], [0, 3000], -2500:50:500);
plot_histogram_sample_feature( all_features, weak_learners{2}, m, '../plots/1-4.png', 'Sample histogram for feature type 4', [-1500, 1500], [0, 6000], -1500:40:1500);

% Plot the top 10 features
for i=1:10,
    plot_feature(all_features, weak_learners{i}, i);
end
plot_feature(all_features, weak_learners{13}, 13);


% Plot the loweest 1000 errors for weak learners from T = 0, 50, 100
iter0 = load('../iterations/iteration_1.mat');
iter10 = load('../iterations/iteration_10.mat');
iter50 = load('../iterations/iteration_50.mat');
iter100 = load('../iterations/iteration_100.mat');
error0 = sort(iter0.errors);
error10 = sort(iter10.errors);
error50 = sort(iter50.errors);
error100 = sort(iter100.errors);

figure2 = figure('visible','off');
plot(1:1000, error0(1:1000), '-', 1:1000, error10(1:1000), '--', 1:1000, error50(1:1000), ':', 1:1000, error100(1:1000), '-.');
legend('Iteration 0','Iteration 10', 'Iteration 50', 'Iteration 100', 'Location','southeast');
legend boxoff;
title('Error rate for top 1000 weak learners Ada Boost');
print(figure2, '-dpng', '../plots/2-2.png', '-r50');


% Test on the training set
result_train_100 = get_prediction(all_images_integral, all_features, weak_learners);
plot_histogram(result_train_100, m, '../plots/2-3.png', 'Histogram for 100 weak learners Ada Boost', [-15,15], [0,7000], -15:0.5:15);
[TP_100, FP_100] = get_ROC_info(result_train_100, all_labels);

result_train_50 = get_prediction(all_images_integral, all_features, weak_learners(1:50));
plot_histogram(result_train_50, m, '../plots/2-4.png', 'Histogram for 50 weak learners Ada Boost', [-15,15], [0,7000], -15:0.5:15);
[TP_50, FP_50] = get_ROC_info(result_train_50, all_labels);

result_train_10 = get_prediction(all_images_integral, all_features, weak_learners(1:10));
plot_histogram(result_train_10, m, '../plots/2-5.png', 'Histogram for 10 weak learners Ada Boost', [-15,15], [0,7000], -15:0.5:15);
[TP_10, FP_10] = get_ROC_info(result_train_10, all_labels);

figure3 = figure('visible','off');
plot(TP_100, FP_100, '-', TP_50, FP_50, '--', TP_10, FP_10, '-.');
legend('Iteration 100', 'Iteration 50', 'Iteration 10', 'Location', 'northwest');
legend boxoff;
title('Ada Boost ROC curves');
print(figure3, '-dpng', '../plots/2-6.png', '-r50');

% Test on classroom image
classroom_image = rgb2gray(imread('../data/class_photo_2014.JPG'));

figure;
imshow(classroom_image)
hold on;

scales = 0.3:0.05:1.3;
for i=1:size(scales,2),
    [classroom_fragments, loc] = get_fragments(classroom_image, image_dimension, scales(i));
    classroom_fragments_integral = get_integral(classroom_fragments);
    result = get_prediction(classroom_fragments_integral, all_features, weak_learners);
        
    face_index = find(result > 5.5);
    for j=1:size(face_index, 2),
        current_loc = loc{face_index(j)};
        rectangle('position', [current_loc.c, current_loc.r, image_dimension(1), image_dimension(2)]/current_loc.scale, 'EdgeColor', 'r')    
    end
end

hold off;

% Test on background image
background_image = rgb2gray(imread('../data/class_photo_2014.JPG'));

figure;
imshow(classroom_image)
hold on;

[background_fragments, loc] = get_fragments(background_image, image_dimension, 1);
classroom_fragments_integral = get_integral(classroom_fragments);
result = get_prediction(classroom_fragments_integral, all_features, weak_learners);
face_index = find(result > 0);
background_images = background_fragments(:,:,face_index);

more_image_integral = cat(3, all_image_integral, background_images);

for i=1:size(all_features,2),
    if mod(i, 50) == 0,
        display(strcat(['Get difference for ', num2str(i), ' features']));
    end
    get_difference(more_images_integral, all_features{i}, true);
end

m = size(face_images_integral,3);
n = size(nonface_images_integral,3);
t = size(background_images, 3);
all_labels = [ones(1,m),zeros(1,n), zeros(1,t)];
all_weights = [ones(1,m)/m, ones(1,n)/n, ones(1,t)/t];
weak_learners = adaBoost(all_features, all_labels, all_weights, 100, '../iterations/');
save('weak_learners_more.mat', 'weak_learners')


% Real Boost on the image
all_weights_real = ones(1,m+n)/(m+n);
bins_100 = realBoost(all_features, weak_learners, all_labels, all_weights_real, 100, '../features/');

result_train_10_real = get_prediction_real(all_images_integral, all_features, bins_100(1:10));
plot_histogram(result_train_10_real, m, '../plots/3-1.png', 'Histogram for 10 weak learners Real Boost', [-20,10], [0,7000], -20:0.5:10);
[TP_10_real, FP_10_real] = get_ROC_info(result_train_10_real, all_labels);

result_train_50_real = get_prediction_real(all_images_integral, all_features, bins_100(1:50));
plot_histogram(result_train_50_real, m, '../plots/3-2.png', 'Histogram for 50 weak learners Real Boost', [-20,10], [0,7000], -20:0.5:10);
[TP_50_real, FP_50_real] = get_ROC_info(result_train_50_real, all_labels);

result_train_100_real = get_prediction_real(all_images_integral, all_features, bins_100);
plot_histogram(result_train_100_real, m, '../plots/3-3.png', 'Histogram for 100 weak learners Real Boost', [-20,10], [0,7000], -20:0.5:10);
[TP_100_real, FP_100_real] = get_ROC_info(result_train_100_real, all_labels);

figure4 = figure('visible','off');
plot(TP_100_real, FP_100_real, '-', TP_50_real, FP_50_real, '--', TP_10_real, FP_10_real, '-.');
legend('Iteration 100', 'Iteration 50', 'Iteration 10', 'Location', 'northwest');
legend boxoff;
title('Real Boost ROC curves');
print(figure4, '-dpng', '../plots/3-4.png', '-r50');

% Test on classroom image
figure;
imshow(classroom_image)
hold on;

scales = 0.3:0.2:1.3;
for i=1:size(scales,2),
    [classroom_fragments, loc] = get_fragments(classroom_image, image_dimension, scales(i));
    classroom_fragments_integral = get_integral(classroom_fragments);
    result = get_prediction_real(classroom_fragments_integral, all_features, bins_100);
        
    face_index = find(result > 5);
    for j=1:size(face_index, 2),
        current_loc = loc{face_index(j)};
        rectangle('position', [current_loc.c, current_loc.r, image_dimension(1), image_dimension(2)]/current_loc.scale, 'EdgeColor', 'r')    
    end
end



