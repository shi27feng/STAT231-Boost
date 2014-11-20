function [ images ] = get_images( dir_path, type, dim )
%GET_IMAGES Summary of this function goes here
%   Detailed explanation goes here
image_files = dir(strcat([dir_path, '*.', type]));
images = zeros(dim(1), dim(2), size(image_files, 1));
for i=1:size(image_files, 1),
   image_path = strcat([dir_path, image_files(i).name]);
   images(:, :, i) = double(rgb2gray(imread(image_path)));
end
end

