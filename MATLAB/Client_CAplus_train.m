% 
% Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php
% 
function net = Client_CAplus_train(DATA, net)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node

divMat = net.divMat; % matrix for a pairwise similarity
V_thres_ = net.V_thres_; % similarlity thresholds
div_lambda = net.div_lambda;
numActiveNodes = net.numActiveNodes; % the number of active nodes
activeNodeIdx = net.activeNodeIdx; % nodes for SigmaEstimation

adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
numSample = net.numSample; % counter for input sample
flag_set_lambda = net.flag_set_lambda; % flag for a calculation of lambda
sigma = net.sigma; % an estimated sigma for CIM

div_threshold = 1.0e-6; % a threshold for diversity via determinants
n_init_data = 10; % number of signals for initialization of sigma


if numSample == 0
    sigma = SigmaEstimationByNode(DATA(1:n_init_data,:),1:n_init_data);
end

for sampleNum = 1:size(DATA,1)
    
    % Current data sample.
    input = DATA(sampleNum,:);
    numSample = numSample+1;
    
    if flag_set_lambda == false || numNodes < numActiveNodes   
        % Generate 1st to bufferNode-th node from inputs.
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
        CountNode(numNodes) = 1;
        
        if size(weight ,1) >= n_init_data && flag_set_lambda == false
            Corr = correntropy(weight(numNodes,:),weight,median(sigma));
            Corr = flip(Corr);
            divMat = toeplitz(Corr);
            Div = det(exp(divMat));
            
            if  Div < div_threshold && size(weight ,1) >= n_init_data
                numActiveNodes = numNodes;
                div_lambda = numActiveNodes*2;
            end
        end
        
        
        % Calculate the initial similarlity threshold to the initial nodes.
        if numNodes == numActiveNodes
            flag_set_lambda = true;
            initSig = SigmaEstimationByNode(weight, activeNodeIdx(1:min(div_lambda,numActiveNodes)));
            adaptiveSig = repmat(initSig,1,numNodes); % Assign the same initSig to the all nodes.
            
            tmpTh = zeros(1,numActiveNodes);
            for k = 1:numActiveNodes
                tmpCIMs1 = CIM(weight(k,:), weight, mean(adaptiveSig));
                [~, s1] = min(tmpCIMs1);
                tmpCIMs1(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:).
                tmpTh(k) = min(tmpCIMs1);
            end
            V_thres_ = mean(tmpTh);
        end
    else
               
        % Calculate CIM based on global mean adaptiveSig.
        globalCIM = CIM(input, weight, mean(adaptiveSig));
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(globalCIM);
        globalCIM(s1) = inf;
        [Vs2, s2] = min(globalCIM);
        
        %numbufferNod
        if V_thres_ < Vs1 || numNodes < numActiveNodes% Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
            CountNode(numNodes) = 1;
            
            adaptiveSig(numNodes) = SigmaEstimationByNode(weight, activeNodeIdx(1:min(div_lambda,numActiveNodes)));
            
        else % Case 2 i.e., V >= CIM_k1
                      
            % Update s1 weight
            CountNode(s1) = CountNode(s1) + 1;
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            activeNodeIdx = updateActiveNode(activeNodeIdx, s1);
            
            if V_thres_ >= Vs2 % Case 3 i.e., V >= CIM_k2
                % Update weight of s2 node.
                weight(s2,:) = weight(s2,:) + (1/(100*CountNode(s2))) * (input - weight(s2,:));
            end     
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
end % for sampleNum = 1:size(DATA,1)





net.numNodes = numNodes;      % Number of nodes
net.weight = weight;          % Mean of nodes
net.CountNode = CountNode;    % Counter for each node
net.numSample = numSample;

net.Correntropy_Mat = divMat;
net.CIMthreshold = V_thres_;
net.numActiveNodes = numActiveNodes;
net.Lambda = div_lambda;

net.activeNodeIdx = activeNodeIdx;
net.adaptiveSig = adaptiveSig;

net.setActiveNodes = flag_set_lambda;
net.sig_div = sigma;

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimationByNode(weight, activeNodeIdx)

exNodes = weight(activeNodeIdx,:);

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end


% Correntropy induced Metric
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))';
end

% Correntropy
function corr = correntropy(X,Y,sig)
% X : 1 x n
% Y : m x n
corr = mean(exp(-(X-Y).^2/(2*sig^2)), 2);
end


function activeNodeIdx = updateActiveNode(activeNodeIdx, winnerIdx)
%activeNodeIdx
activeNodeIdx(activeNodeIdx == winnerIdx)= [];
activeNodeIdx = [winnerIdx,activeNodeIdx];
end

