function w = RidgeRegression(Xtrain, Ytrain, Lambda)
[~, ncols] = size(Xtrain);
w = (Xtrain'*Xtrain + Lambda* eye(ncols))\ (Xtrain'*Ytrain);