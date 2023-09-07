% 
% Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php
% 
function estimated_labels = Server_CAE_test(Xt, net)

weight = net.weight;
LabelCluster = net.LabelCluster;
adaptiveSig = net.adaptiveSig;

% Evaluate clustering performance
LabelClusterCC = LabelCluster';

% Classify test data by disjoint clusters
estimated_labels = zeros(size(Xt,1),1);
for sampleNum = 1:size(Xt,1)
    
    % Current data sample
    pattern = Xt(sampleNum,:); % Current Input
    
    % Find 1st winner node
    clusterCIM = CIM(pattern, weight, mean(adaptiveSig));
    [~, orderCIM] = sort(clusterCIM, 'ascend');
    s1 = orderCIM(1);
    
    estimated_labels(sampleNum, 1) = LabelClusterCC(s1, 1);
end

end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))';
end
