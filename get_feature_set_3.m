function [ feature_set_3 ] = get_feature_set_3( dim )
%GET_FEATRUE_SET_3 Summary of this function goes here
%   Detailed explanation goes here
n = 0;
count = 0;
for c1=1:(dim(2)-2),
    for c2=c1+2:3:dim(2),
        for r1=1:(dim(1)-1),
            for r2=r1+1:dim(1),
                n = n + 1;
            end
        end
    end
end

feature_set_3 = cell(1,n);

for c1=1:(dim(2)-1),
    for c2=c1+2:3:dim(2),
        for r1=1:(dim(1)-1),
            for r2=r1+1:dim(1),
                count = count + 1;
                feature_set_3{1,count} = struct('type', 3, 'index', count, 'data', struct('c1', c1, 'r1', r1, 'c2', c2, 'r2', r2));
            end
        end
    end
end

end

