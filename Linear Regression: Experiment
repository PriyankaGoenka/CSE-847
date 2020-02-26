clc
clear
close all
data = load('diabetes.mat');

xtrainOriginal = data.x_train;
ytrainOriginal = data.y_train;
xtestOriginal = data.x_test;
ytestOriginal = data.y_test;

% lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
% lambda = logspace(-5, 15, 100);
lambda=logspace(-5, 5, 20);
testerror= zeros(1, length(lambda));
trainerror = zeros(1, length(lambda));
for i = 1 : length(lambda)
    w = RidgeRegression(xtrainOriginal, ytrainOriginal, lambda(i));
    testerror(i) = MeanSquareError(xtestOriginal, ytestOriginal, w);
    trainerror(i) = MeanSquareError(xtrainOriginal, ytrainOriginal, w);
end

figure(1)

plot(log10(lambda), testerror, 'Linewidth', 1, 'Linestyle', '--')
hold on
plot(log10(lambda), trainerror, 'Linewidth', 1, 'Linestyle', '-.')
xlabel('Log \lambda')
ylabel('MSE')
legend('test', 'train')

%%
clc
% clear
% data = load('diabetes.mat');

xtrainOriginal = data.x_train;
ytrainOriginal = data.y_train;
xtestOriginal = data.x_test;
ytestOriginal = data.y_test;

data = [[xtrainOriginal ytrainOriginal]; [xtestOriginal ytestOriginal]];

Ndata = length(data);
% rng('default');
% rng(1);
data = data(randperm(Ndata),:);
fold =5;
indices = crossvalind('Kfold', data(:,end), fold);


bestlambdatest = zeros(1, length(lambda));
bestlambdatrain = zeros(1, length(lambda));
figure(2)
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
    plot(log10(lambda), testerror)
    [~, bestidxtest] = min(testerror);
    bestlambdatest(i)= lambda(bestidxtest);
    plot(log10(bestlambdatest(i)), min(testerror), '*')
    [~, bestidxtrain] = min(trainerror);
    bestlambdatrain(i)= lambda(bestidxtrain);
    
end






