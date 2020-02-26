clc
clear
close all
data = load('diabetes.mat');

xtrainOriginal = data.x_train;
ytrainOriginal = data.y_train;
xtestOriginal = data.x_test;
ytestOriginal = data.y_test;

data = [[xtrainOriginal ytrainOriginal]; [xtestOriginal ytestOriginal]];

Ndata = length(data);

data = data(randperm(Ndata),:);
fold =5;
indices = crossvalind('Kfold', data(:,end), fold);
lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
bestlambdatest = zeros(1, length(lambda));
bestlambdatrain = zeros(1, length(lambda));
figure(1)
hold on
for i =1 : fold
    test = (indices == i);
    train = ~test;
    test = data(test,:);
    train = data(train,:);
    xtest = test(:, 1:end-1);
    ytest = test(:,end);
    xtrain = train(:, 1:end-1);
    ytrain = train(:,end);
    testerror= zeros(1, length(lambda));
    trainerror = zeros(1, length(lambda));
    for j = 1 : length(lambda)
        w = RidgeRegression(xtrain, ytrain, lambda(j));
        testerror(j) = MeanSquareError(xtest, ytest, w); 
        trainerror(j) = MeanSquareError(xtrain, ytrain, w);
    end
    plot(log(lambda), testerror)
    [~, bestidxtest] = min(testerror);
    bestlambdatest(i)= lambda(bestidxtest);
    [~, bestidxtrain] = min(trainerror);
    bestlambdatrain(i)= lambda(bestidxtrain);
    
end

% KfoldLambdaTest = mode(bestlambdatest);
% KfoldLambdaTrain = mode(bestlambdatrain);



