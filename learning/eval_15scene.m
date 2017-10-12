T = 15; % Number of split
nTrain = 100; % Number of image to train per category
K = 1001; % Number of words
pathBow='./bow_files/';
[ imCat,imCatTest ] = NbImCatAllTest( pathBow , nTrain);
cate = categories();
tx_split = []

for iter = 1:T
    [ train , test ] = loadData( nTrain , imCat , pathBow, K);
    ntest = size(test, 1); 
    %arr_acc = []; 
    
    predictclassifiers = [];
    for idx_cate = 1:15
        idx_cate
        cat = cate{idx_cate};
        
        [y, ytest] = labelsTrainTest(nTrain, ntest, imCat , idx_cate); 
        [values] = trainTest(train, test, y);
        classif = values.*ytest;
        
        ACC = sum(classif(:)>0) / ntest;
        predictclassifiers = [predictclassifiers classif];
        %arr_acc = [arr_acc ACC];
    end
    [matConf, txCat] = multiClassPrediction(predictclassifiers, imCatTest);
    tx_split = [ tx_split; mean(txCat) ]
    %avg_acc = avg_acc / 15; %  #categories
    %avg_T_split = [ avg_T_split avg_acc ];
end
mean(tx_split)
std(tx_split)
