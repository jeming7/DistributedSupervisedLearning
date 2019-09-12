function [X, y] = GetData(n)

load('ex4data1.mat');

[X, y] = Shufl(X,y);


X1 = X(1:n,:);
y1 = y(1:n,:);

clear X y

X = X1;
y = y1;
