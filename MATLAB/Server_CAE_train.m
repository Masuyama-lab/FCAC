% 
% Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php
% 
function net = Server_CAE_train(DATA, net)


numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
edge = net.edge;                 % edge matrix

adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
LabelCluster = net.LabelCluster; % Cluster label for connected nodes
V_thres_ = net.V_thres_;         % similarlity thresholds
activeNodeIdx = net.activeNodeIdx; % indexes of active nodes
CountLabel = net.CountLabel;     % a label counter
numSample = net.numSample;  % number of samples 

flag_set_lambda = net.flag_set_lambda; % a flag for setting lambda
numActiveNode = net.numActiveNode; % number of active nodes
divMat = net.divMat;             % a matrix for diversity via determinants
div_lambda = net.div_lambda;       % \lambda determined by diversity via determinants
lifetime_d_edge = net.lifetime_d_edge; % average lifetime of deleted edges
n_deleted_edge = net.n_deleted_edge; % number of deleted edges
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
    
    if flag_set_lambda == false || numNodes < numActiveNode   
        % Generate 1st to bufferNode-th node from inputs.
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
        CountNode(numNodes) = 1;
        edge(numNodes, numNodes) = 0;
        
      
        if size(weight ,1) >= 2 && flag_set_lambda == false
            Corr = correntropy(weight(numNodes,:),weight,median(sigma));
            Corr = flip(Corr);
            divMat = toeplitz(Corr);
            Div = det(exp(divMat));
            if  Div < div_threshold && size(weight ,1) >= n_init_data
                numActiveNode = numNodes;
                div_lambda = numActiveNode*2;
            end
        end
        
        % Calculate the initial similarlity threshold to the initial nodes.
        if numNodes == numActiveNode
            flag_set_lambda = true;
            initSig = SigmaEstimationByNode(weight, activeNodeIdx(1:min(div_lambda,numActiveNode)));
            adaptiveSig = repmat(initSig,1,numNodes); % Assign the same initSig to the all nodes.
            
            tmpTh = zeros(1,numActiveNode);
            for k = 1:numActiveNode
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
        if V_thres_ < Vs1 || numNodes < numActiveNode% Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimationByNode(weight, activeNodeIdx(1:min(div_lambda,numActiveNode)));
            edge(numNodes, numNodes) = 0;
            
        else % Case 2 i.e., V >= CIM_k1
                      
            % Update s1 weight
            CountNode(s1) = CountNode(s1) + 1;
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            activeNodeIdx = updateActiveNode(activeNodeIdx, s1);
            
            s1Neighbors = find(edge(s1,:));  
            edge(s1Neighbors,s1) = edge(s1Neighbors,s1) + 1;
            edge(s1,s1Neighbors) = edge(s1,s1Neighbors) + 1;
            
            % Update s1 neighbor
            if V_thres_ >= Vs2 % Case 3 i.e., V >= CIM_k2
                % Create an edge between s1 and s2 nodes
                edge(s1,s2) = 1.0;
                edge(s2,s1) = 1.0;  

                % Update weight of s2 node.
                for k = s1Neighbors
                    weight(k,:) = weight(k,:) + ( 1/(10*CountNode(k) )) * (input - weight(k,:));
                end
                           
            end
            
           % delete adge by function of soinnplus
           [Delete_Edge_Indexes,n_deleted_edge,lifetime_d_edge] = DeleteAdge(edge,s1,n_deleted_edge,lifetime_d_edge);
           edge(s1,Delete_Edge_Indexes) = 0;
           edge(Delete_Edge_Indexes,s1) = 0;          

            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    
    % Topology Adjustment
    % If activate the following function, CAEAF shows a noise reduction ability.
    if mod(sampleNum, div_lambda) == 0 && size(weight,1) > 1
        % -----------------------------------------------------------------
        % Delete Node based on number of neighbors
        nNeighbor = sum(edge);
        deleteNode = (nNeighbor == 0);
        
        % Delete process
        numNodes = numNodes - sum(deleteNode);
        weight(deleteNode, :) = [];
        CountNode(deleteNode) = [];
        edge(deleteNode, :) = [];
        edge(:, deleteNode) = [];
        adaptiveSig(deleteNode) = [];

        
        % activeNodeIdx denotes a node index, so the node index must be changed
        % after deleating some nodes.   
        diffArray = setdiff(activeNodeIdx, find(deleteNode),'stable');
        [~, sortIdx] = sort(diffArray);
        [~, activeNodeIdx] = sort(sortIdx);
      
        
    end % if mod(sampleNum, Lambda) == 0
    
end % for sampleNum = 1:size(DATA,1)



connection = graph(edge~= 0);
LabelCluster = conncomp(connection);
if isempty(LabelCluster)
    LabelCluster = 0;
end


net.numNodes = numNodes;
net.weight = weight;
net.CountNode = CountNode;
net.edge = edge;
net.adaptiveSig = adaptiveSig;
net.LabelCluster = LabelCluster;
net.V_thres_ = V_thres_;
net.activeNodeIdx = activeNodeIdx;
net.CountLabel = CountLabel;
net.numSample = numSample;
net.flag_set_lambda = flag_set_lambda;
net.numActiveNode = numActiveNode;
net.divMat = divMat;
net.div_lambda = div_lambda;
net.lifetime_d_edge =lifetime_d_edge;
net.n_deleted_edge = n_deleted_edge;
net.sigma = sigma;

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

function [Delete_Edge_Indexes,edgeDeleted,edgeAvgLtDel]= DeleteAdge(edge,s1,edgeDeleted,edgeAvgLtDel)

connectedIdx = edge(s1,:)>0;
edgeAges = edge(s1, connectedIdx);


% rank the data
sortedEA = sort(edgeAges);
% compute 50th percentile (second quartile)
medianSortedEA = median(sortedEA);
% compute 25th percentile (first quartile)
q1 = median(sortedEA(sortedEA < medianSortedEA));
% compute 75th percentile (third quartile)
q3 = median(sortedEA(sortedEA > medianSortedEA));
% compute Interquartile Range (IQR)
dif = q3 - q1;
c = q3;


% paramEdge = length(edgeAges)/length(edge);  %changing parameter
paramEdge = 1;  %We would like to change here as no-param.


th = paramEdge * dif;
age = edge(:,s1);

curTh = c + th;
ratio = edgeDeleted / ( edgeDeleted + length(edgeAges));

% Check if there are any edges to be deleted
delThreshold = (edgeAvgLtDel*ratio + curTh*(1-ratio) );
% % delThreshold = edgeAvgLtDel;
delFlag = age > delThreshold;
Delete_Edge_Indexes = find(delFlag);
% Update average lifetime of deleted edges
if ~isempty(Delete_Edge_Indexes)
    edgeAvgLtDel = (edgeDeleted*edgeAvgLtDel + sum(age(delFlag))) / (edgeDeleted + length(Delete_Edge_Indexes));
    edgeDeleted = edgeDeleted + length(Delete_Edge_Indexes);
end
end