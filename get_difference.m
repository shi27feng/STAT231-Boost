function [ diff ] = get_difference( images, feature, save_data)
%GET_DIFFERENCE Summary of this function goes here
%   Detailed explanation goes here
n = size(images, 3);
diff = zeros(1, n);

if feature.type == 1,
    for i=1:n,
        image = images(:,:,i);
        c1 = feature.data.c1;
        c2 = feature.data.c2;
        r1 = feature.data.r1;
        r2 = feature.data.r2;
        area = zeros(1,6);
        if c1 ~= 1 && r1 ~= 1,
            area(1) = image(r1-1, c1-1);
        end
        if c1 ~= 1,
            area(4) = image(r2, c1-1);
        end
        if r1 ~= 1, 
            area(2) = image(r1-1, (c1+c2-1)/2);
            area(3) = image(r1-1, c2);
        end
        area(5) = image(r2, (c1+c2-1)/2);
        area(6) = image(r2, c2);
        diff(i) = area(6) - 2*area(5) + area(4) - area(3) + 2*area(2) - area(1);
    end
    
elseif feature.type == 2,
    for i=1:n,
        image = images(:,:,i);
        c1 = feature.data.c1;
        c2 = feature.data.c2;
        r1 = feature.data.r1;
        r2 = feature.data.r2;
        area = zeros(1,6);
        if c1 ~= 1 && r1 ~= 1,
            area(1) = image(r1-1, c1-1);
        end
        if c1 ~= 1,
            area(2) = image((r1+r2-1)/2, c1-1);
            area(3) = image(r2, c1-1);
        end
        if r1 ~= 1, 
            area(4) = image(r1-1, c2);
        end
        area(5) = image((r1+r2-1)/2, c2);
        area(6) = image(r2, c2);
        diff(i) = -area(6) + 2*area(5) - area(4) + area(3) - 2*area(2) + area(1);  
    end
    
elseif feature.type == 3,
    for i=1:n,
        image = images(:,:,i);
        c1 = feature.data.c1;
        c2 = feature.data.c2;
        r1 = feature.data.r1;
        r2 = feature.data.r2;
        area = zeros(1,8);
        if c1 ~= 1 && r1 ~= 1,
            area(1) = image(r1-1, c1-1);
        end
        if c1 ~= 1,
            area(5) = image(r2, c1-1);
        end
        if r1 ~= 1, 
            area(2) = image(r1-1, (c1-1)*2/3+c2/3);
            area(3) = image(r1-1, (c1-1)/3+c2*2/3);
            area(4) = image(r1-1, c2);
        end
        area(6) = image(r2, (c1-1)*2/3+c2/3);
        area(7) = image(r2, (c1-1)/3+c2*2/3);
        area(8) = image(r2, c2);
        diff(i) = -area(8) + 2*area(7) - 2*area(6) + area(5) + area(4) - 2*area(3) + 2*area(2) - area(1);
    end
    
elseif feature.type == 4,
    for i=1:n,
        image = images(:,:,i);
        c1 = feature.data.c1;
        c2 = feature.data.c2;
        r1 = feature.data.r1;
        r2 = feature.data.r2;
        area = zeros(1,9);
        if c1 ~= 1 && r1 ~= 1,
            area(1) = image(r1-1, c1-1);
        end
        if c1 ~= 1,
            area(4) = image((r1+r2-1)/2, c1-1);
            area(7) = image(r2, c1-1);
        end
        if r1 ~= 1, 
            area(2) = image(r1-1, (c1+c2-1)/2);
            area(3) = image(r1-1, c2);
        end
        area(5) = image((r1+r2-1)/2, (c1+c2-1)/2);
        area(6) = image((r1+r2-1)/2, c2);
        area(8) = image(r2, (c1+c2-1)/2);
        area(9) = image(r2, c2);
        diff(i) = -area(9) + 2*area(8) - area(7) + 2* area(6) - 4*area(5) + 2*area(4) - area(3) + 2*area(2) - area(1);
    end
end

if save_data,
    [diff_sorted, order] = sort(diff);
    save(strcat(['../features/feature_', num2str(feature.type), '_', num2str(feature.index), '.mat']), 'diff_sorted', 'order');       
end
end

