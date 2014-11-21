function [] = plot_feature( all_features, weak_learner, index )
%PLOT_FEATURE Summary of this function goes here
%   Detailed explanation goes here
feature = all_features{weak_learner.feature};
c1 = (feature.data.c1-1)*16;
c2 = feature.data.c2*16;
r1 = (feature.data.r1-1)*16;
r2 = feature.data.r2*16;

a = (ones(256, 256)*255);
a(r1+1,(c1+1):c2) = 0;
a(r2,(c1+1):c2) = 0;
a((r1+1):r2, c1+1) = 0;
a((r1+1):r2, c2) = 0;

a(1,:) = 0;
a(256,:) = 0;
a(:,1) = 0;
a(:, 256) = 0;

if feature.type == 3,
    a((r1+1):r2, (2*c1+c2)/3) = 0;
    a((r1+1):r2, (c1+2*c2)/3) = 0;
end    

if feature.type == 1 || feature.type == 4 ,
    a((r1+1):r2, (c1+c2)/2) = 0;
end

if feature.type == 2 || feature.type == 4,
    a((r1+r2)/2, (c1+1):c2)=0;
end

imwrite(uint8(a), strcat(['../plots/feature_', num2str(index), '.png']));

end

