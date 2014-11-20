function [ images_integral ] = get_integral( images )
%GET_INTEGRAL Summary of this function goes here
%   Detailed explanation goes here
images_integral = zeros(size(images));
for i=1:size(images, 3),
    images_integral(:,:,i) = cumsum(cumsum(images(:,:,i)),2);
end
end

