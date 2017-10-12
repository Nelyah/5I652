function [matConf, txCat] = multiClassPrediction( predictclassifiers, imCatTest)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    ntest = size(predictclassifiers, 1); 
    ncat = size(predictclassifiers, 2); 
    matConf = zeros(ncat, ncat);
    start = 1;
    txCat = zeros(ncat, 1);
    
    for cat = 1:ncat
        bound = start + imCatTest(cat,1) - 1;
        for j = start: bound
           [val, idx] = min(predictclassifiers(j, :));
           matConf(cat, idx) = matConf(cat, idx) + 1;
        end
        start = bound + 1;
    end
    for cls = 1:ncat
        tot = sum(matConf(:,cls));
        matConf(:,cls) = matConf(:,cls) ./ tot;
        txCat(cls) = matConf(cls, cls);
    end
    
end

