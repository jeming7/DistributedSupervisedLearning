%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% File Name:Main01.m
%% Description: Impliment distributed SGD for supervised learning 
%% Inputs: None
%% Output: 1)   PLOTS & 
%% SubFunctions:    1) GetLaplacian
%%                  2) GetData
%%                  3) CentralizedLearning
%%                  4) DistributedLearning
%% Last modified: 09/12/19 by Jemin George
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
rng(0,'twister');
clear all;close all

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Netowrk Design ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = 10;                         % number of nodes
type = 1;                       % 1 = ring, 2 = path, 3 = some random
Lap = GetLaplacian(n,type);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

alpha0 = 0.1;                   % initial value for alpha
beta0 = 1.01/max(svd(Lap));     % initial value for beta
(abs(eig(eye(n)-beta0*Lap)))    % making sure (I-\beta*L) is stable
% keyboard

%***************************** Data Uploading *****************************
DataSize = 5e3;
[X, y] = GetData(DataSize);

% One hot encoding Y
Y = zeros(length(X),10);
for i=1:length(X)    Y(i,y(i))=1; end

% Saving Data
Attributes0 = X; Classifications0 = Y;
clear X Y
%**************************************************************************


e0 = round(size(Attributes0)*0.5); % data used for trining (using only half)

Attributes = [ones(e0(1),1) Attributes0(1:e0(1),:)];
Classifications = Classifications0(1:e0(1),:);

disp('*******************************************************************')
disp(['Total data: ' num2str(length(Attributes0(:,1)))])
disp(['Total trining data: ' num2str(e0(1))])
disp('                               ')


%~~~~~~~~~~~~~~~~~~~~~~~~~~~ Neural Nets Design ~~~~~~~~~~~~~~~~~~~~~~~~~~~       
nbrOfNodes = 50;     % nodes per layer
nbrOfEpochs = 400;   % opt. iterations 

% initialize the Ws (network NN weights)
W10_avg = zeros( nbrOfNodes, length(Attributes(1,:)) );
W20_avg = zeros( length(Classifications(1,:)), nbrOfNodes + 1 );
for i=1:n
    % Initialize matrices with random weights (weights for each layer)
    W1{i} = randn( nbrOfNodes, length(Attributes(1,:)) );
    W2{i} = randn( length(Classifications(1,:)), nbrOfNodes + 1 );
    W10_avg = W10_avg + 1/n*W1{i};
    W20_avg = W20_avg + 1/n*W2{i};
end
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Centralized Learning (stochastic)
[Err_c, Err_indx_c, cnt_c, MSE_c, W1_c, W2_c] = CentralizedLearning(Attributes, Classifications, ...
        W10_avg, W20_avg, Attributes0, Classifications0, nbrOfEpochs*1, alpha0);

disp('*******************************************************************')
disp(['Centralized Stochastic Gradient Error: ' num2str(Err_c)])
disp('                               ')


figure(1)
loglog(cnt_c, 1/e0(1)*MSE_c)
%yticks([10^-3 10^-2 10^-1 1 10 10^2])
grid on;box on
xlabel('k')
ylabel('J(k)')
title('Centralized SGD')
ax = gca; % current axes
ax.FontSize = 12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Divide the data amoung agents

%~~~~~~~~~~~~ For dividing data equally amoung n agents

count = zeros(1,n);
for i=1:n
    InputData{i} = Attributes( (i-1)*e0(1)/n + 1 : i*e0(1)/n, :);
    InputLabel{i} = Classifications( (i-1)*e0(1)/n + 1 : i*e0(1)/n, :);
    count(i) = count(i) + length(InputLabel{i});
end

disp('*******************************************************************')
disp(['Total trining data across all Agents: ' num2str(e0(1))])
disp('Training data for agent 1-10: ')
disp(num2str(count))
disp('                               ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Distributed learning
[Err_SGD, Err_indx_SGD, cnt_SGD, MSE_SGD, W1_d, W2_d] = DistributedLearning(InputData, InputLabel, W1, W2,...
                  Attributes0, Classifications0, nbrOfEpochs, alpha0, beta0, e0, n, Lap);


disp('*******************************************************************')
disp('Stochastic Gradient Error for Agents: ')
disp(num2str(Err_SGD))
disp(['Error % : Mean ' num2str(mean(Err_SGD)) ' Max: ' num2str(max(Err_SGD)) ' Min: ' num2str(min(Err_SGD))]);
disp('                               ')

figure(2)
loglog(cnt_SGD, n/e0(1)*MSE_SGD, 'LineWidth',2)
grid on;box on
xlabel('k')
ylabel('J(k)')
title('Distributed SGD')
ax = gca; % current axes
ax.FontSize = 12;









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Divide the data amoung agents

%~~~~~~~~~~~~ For dividing data one class per agent 

count = ones(1,n);
for k=1:e0(1)
    if y(k)==0
        ind = 10;
    else
        ind = y(k);
    end
    InputData{ind}(count(ind),:) = Attributes(k, :);
    InputLabel{ind}(count(ind),:) = Classifications(k, :);
    count = count + Classifications(k, :);
end
count = count - ones(1,n);


disp('*******************************************************************')
disp(['Total trining data across all Agents: ' num2str(e0(1))])
disp('Training data for agent 1-10: ')
disp(num2str(count))
disp('                               ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Distributed learning
[Err_SGD1, Err_indx_SGD1, cnt_SGD1, MSE_SGD1, W1_d1, W2_d1] = DistributedLearning(InputData, InputLabel, W1, W2,...
                  Attributes0, Classifications0, nbrOfEpochs, alpha0, beta0, e0, n, Lap);


disp('*******************************************************************')
disp('Stochastic Gradient Error for Agents: ')
disp(num2str(Err_SGD1))
disp(['Error % : Mean ' num2str(mean(Err_SGD1)) ' Max: ' num2str(max(Err_SGD1)) ' Min: ' num2str(min(Err_SGD1))]);
disp('                               ')

for i=1:10
    MSE_SGD1(:,i) = 1/count(i)*MSE_SGD1(:,i);
end

figure(3)
loglog(cnt_SGD1, MSE_SGD1, 'LineWidth',2)
grid on;box on
xlabel('k')
ylabel('J(k)')
title('Distributed SGD')
ax = gca; % current axes
ax.FontSize = 12;