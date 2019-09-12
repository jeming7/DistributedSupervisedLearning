function [Err, Err_indx] = Testing(Attributes, Classifications, W1, W2)

e = size(Attributes);Err = 0;Err_indx=[];

% activation function
f = @(x) (1./(1+exp(-x)));

for i=1:e(1)
    D = Classifications(i,:).';
    I = Attributes(i,:).';
    Y0 = f(W2*[1;f(W1*I)]);
    Y = zeros(size(Y0));
    Y(find(Y0==max(Y0))) = 1;
    Err = Err + ceil(norm(D-Y,2)/10);
    if (ceil(norm(D-Y,2)/10))==1
        Err_indx = [Err_indx;i];
    end
end

Err= Err/e(1)*100; % 100 because we are computing %