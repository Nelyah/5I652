function [ values ] = trainTest(train, test, y)
    % @y : train labels
    % @train : train split (BoW of category)
    % @test : test split (BoW of category)
    
    model = svmtrain(y, train, '-c 1000 -t linear');
    [w,b] = getPrimalSVMParameters(model);
    values = test * w + b;
end

