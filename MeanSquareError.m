function MSE = MeanSquareError(Xtest, Ytest, w)
Ypredict = Xtest*w;
MSE = mean(sqrt((Ypredict-Ytest).^2));


