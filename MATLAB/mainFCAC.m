% 
% Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php
% 
clear all

rng(10);


% set data name
data_name = 'OptDigits';

% Load dataset
tmpData = load(strcat(data_name,'.mat'));
original_data = tmpData.data ;
original_target = tmpData.target;


% Avoid label value == 0
if min(original_target)==0
    original_target = original_target+1;
end
tempData = [original_data, original_target];


% Number of clients
numClients = 100;

% Privacy Budget (0,inf] for Differential Privacy
epsilon = 15;






% Parameters of FCAC ======================================================
% for Client CAplus
clientNet = cell(1,numClients);
for k = 1:numClients
    clientNet{k}.numNodes = 0;           % the number of nodes
    clientNet{k}.weight = [];            % node position
    clientNet{k}.CountNode = [];         % winner counter for each node
    clientNet{k}.adaptiveSig = [];       % kernel bandwidth for CIM in each node
    clientNet{k}.V_thres_ = [];          % similarlity thresholds
    clientNet{k}.activeNodeIdx = [];     % nodes for SigmaEstimation
    clientNet{k}.numSample = 0;          % counter for input sample
    clientNet{k}.flag_set_lambda = false;% flag for a calculation of lambda
    clientNet{k}.numActiveNodes = inf;     % the number of active nodes
    clientNet{k}.div_lambda = inf;           % numActiveNodes * 2
    clientNet{k}.divMat = [];   % matrix for a pairwise similarity
    clientNet{k}.sigma = [];           % sigma defined by initial nodes
end

% for Server CAE
serverNet.numNodes    = 0;   % the number of nodes
serverNet.weight      = [];  % node position
serverNet.CountNode = [];    % winner counter for each node
serverNet.edge = [];         % Initial connections (edges) matrix
serverNet.adaptiveSig = [];  % kernel bandwidth for CIM in each node
serverNet.LabelCluster = []; % Cluster label for connected nodes
serverNet.V_thres_ = [];     % similarlity thresholds
serverNet.activeNodeIdx = [];% nodes for SigmaEstimation
serverNet.CountLabel = [];   % counter for labels of each node
serverNet.numSample = 0;     % number of samples 
serverNet.flag_set_lambda = false; % a flag for setting lambda
serverNet.numActiveNode = inf; % number of active nodes
serverNet.divMat(1,1) = 1;     % a matrix for diversity via determinants
serverNet.div_lambda = inf;    % \lambda determined by diversity via determinants
serverNet.lifetime_d_edge = 0; % average lifetime of deleted edges
serverNet.n_deleted_edge = 0;  % number of deleted edges
serverNet.sigma = 0;           % an estimated sigma for CIM
% =========================================================================
    


% Data preprocessing (i.i.d.)
% Randomization
ran = randperm(size(tempData,1));
tempData = tempData(ran,:);

% Divide Dataset
% https://jp.mathworks.com/matlabcentral/fileexchange/35085-mat2tiles-divide-array-into-equal-sized-sub-arrays?s_tid=srchtitle
divided_tempData = mat2tiles(tempData,[round(size(tempData,1)/numClients),size(tempData,2)]);

for k = 1:numClients
    divided_data{k} = divided_tempData{k}(:, 1:end-1);
    divided_target{k} = divided_tempData{k}(:, end);
end


% Local epsilon-Differential Privacy (Laplace Distribution)
% epsilon-differential privacy lets you balance the privacy and accuracy level with a positive value named epsilon.
% If epsilon is small, then more privacy is preserved but data accuracy gets worse.
% If epsilon is large, privacy will be worse but data accuracy will be preserved.
for k = 1:numClients
    noised_divided_data{k} = addLaplacianNoise(divided_data{k}, epsilon);
end
    

% record processing time
time_cap_client_train = 0;
time_cae_server_train = 0;



% Training clients by CAplus
tic
parfor k = 1:numClients
% for k = 1:numClients
    clientNet{k} = Client_CAplus_train(noised_divided_data{k}, clientNet{k});
end
time_cap_client_train = time_cap_client_train + toc;


% Data Preparation for a server
% Sort training data for a server based on CountNode
data_geq_Q3 = [];
data_less_Q3 = [];

for k = 1:size(clientNet,2)

    tmp_nodes = clientNet{k}.weight;
    tmp_counts = clientNet{k}.CountNode;
    
    Q3 = quantile(tmp_counts, 0.75);
    indices_geq_Q3 = tmp_counts >= Q3;
    indices_less_Q3 = tmp_counts < Q3;
    
    tmp_data_geq_Q3 = tmp_nodes(indices_geq_Q3, :);
    tmp_data_less_Q3 = tmp_nodes(indices_less_Q3, :);

    % Randamization
    ran = randperm(size(tmp_data_geq_Q3,1));
    tmp_data_geq_Q3 = tmp_data_geq_Q3(ran,:);
    ran = randperm(size(tmp_data_less_Q3,1));
    tmp_data_less_Q3 = tmp_data_less_Q3(ran,:);

    data_geq_Q3 = [data_geq_Q3; tmp_data_geq_Q3];
    data_less_Q3 = [data_less_Q3; tmp_data_less_Q3];
end

% Randamization
ran = randperm(size(data_geq_Q3,1));
data_geq_Q3 = data_geq_Q3(ran,:);
ran = randperm(size(data_less_Q3,1));
data_less_Q3 = data_less_Q3(ran,:);

% data for a server
server_train_data = [data_geq_Q3; data_less_Q3];

% Training a server by CAE
tic
serverNet = Server_CAE_train(server_train_data, serverNet);
time_cae_server_train = time_cae_server_train + toc;

% Test a server
predicted_labels = Server_CAE_test(original_data, serverNet);

% Evaluation
[NMI, AMI, ARI] = Evaluate_Clustering_Performance(original_target, predicted_labels);




% Result
disp(['   Dataset: ', data_name]);
disp(['   # nodes: ', num2str(serverNet.numNodes)])
disp(['# clusters: ', num2str(max(serverNet.LabelCluster))])
disp(['       ARI: ', num2str(ARI)])
disp(['       AMI: ', num2str(AMI)])
disp(['       NMI: ', num2str(NMI)])
disp(['Time Total: ', num2str(time_cap_client_train + time_cae_server_train)])
disp(' ')



