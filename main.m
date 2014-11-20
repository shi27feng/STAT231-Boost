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
weak_learners = adaBoost(all_features, all_labels, all_weights);
save('weak_learners.mat', 'weak_learners')

% Plot the top 10 features
for i=1:10,
    plot_feature(all_features, weak_learners{i}, i);
end

% Plot the loweest 1000 errors for weak learners from T = 0, 50, 100
iter0 = load('../iterations/iteration_1.mat');
iter10 = load('../iterations/iteration_50.mat');
iter100 = load('../iterations/iteration_100.mat');
error0 = sort(iter0.errors);
error10 = sort(iter10.errors);
error100 = sort(iter100.errors);
figure;
plot(1:1000, error0(1:1000), 1:1000, error10(1:1000), 1:1000, error100(1:1000));
legend;


% Test on the training set
result_train_100 = get_prediction(all_images_integral, all_features, weak_learners);
[TP_100, FP_100] = get_ROC_info(result_train_100, all_labels);

result_train_50 = get_prediction(all_images_integral, all_features, weak_learners(1:50));
[TP_50, FP_50] = get_ROC_info(result_train_50, all_labels);

result_train_10 = get_prediction(all_images_integral, all_features, weak_learners(1:10));
[TP_10, FP_10] = get_ROC_info(result_train_10, all_labels);


figure;
plot(TP_100, FP_100, TP_50, FP_50, TP_10, FP_10);

% Test on classroom image
classroom_image = rgb2gray(imread('../data/class_photo_2014.JPG'));


figure;
imshow(classroom_image)
hold on;

for i=1:10,
    [classroom_fragments, loc] = get_fragments(classroom_image, image_dimension, 1.3);
    classroom_fragments_integral = get_integral(classroom_fragments);
    result = get_prediction(classroom_fragments_integral, all_features, weak_learners);
        
    face_index = find(result > 5.5);
    for j=1:size(face_index, 2),
        current_loc = loc{face_index(j)};
        rectangle('position', [current_loc.c, current_loc.r, image_dimension(1), image_dimension(2)]/current_loc.scale, 'EdgeColor', 'r')    
    end
end

hold off;








