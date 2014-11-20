function [ fragment, loc ] = get_fragments( image, dim, scale )
%GET_FRAGMENTS Summary of this function goes here
%   Detailed explanation goes here
count = 0;
image_scaled = double(imresize(image, scale));
n = (size(image_scaled, 1) - dim(1) + 1) * (size(image_scaled, 2) -dim(2) + 1);

fragment = zeros(dim(1), dim(2), n);
loc = cell(1, n);

dim1 = size(image_scaled, 1);
dim2 = size(image_scaled, 2);
for j=1:(dim1 - dim(1)+1),
    for k=1:(dim2-dim(2)+1),
        count = count + 1;
        if mod(count, 2000) == 0,
            display(strcat(['Created ', num2str(count), ' fragments']));
        end
        fragment(:,:,count) = image_scaled(j:(j+dim(1)-1),k:(k+dim(2)-1));
        loc{count} = struct('r', j, 'c', k, 'scale', scale); 
    end
end

end

