function label = LSC(data,k,opts)
% label = LSC(data,k,opts): Landmark-based Spectral Clustering
% Input:
%       - data: the data matrix of size nSmp x nFea, where each row is a sample
%               point
%       - k: the number of clusters
%       opts: options for this algorithm
%           - p: the number of landmarks picked (default 1000)
%           - r: the number of nearest landmarks for representation (default 5)
%           - numRep: the number of replicates for the final kmeans (default 10)
%           - maxIter: the maximum number of iterations for final kmeans (default 100)
%           - mode: landmark selection method, currently support
%               - 'kmeans': use centers of clusters generated by kmeans (default)
%               - 'random': use randomly sampled points from the original
%                           data set 
%           The following parameters are effective ONLY in mode 'kmeans'
%           - kmNumRep: the number of replicates for initial kmeans (default 1)
%           - kmMaxIter: the maximum number of iterations for initial kmeans (default 5)
% Output:
%       - label: the cluster assignment for each point
% Requre:
%       litekmeans.m
% Usage:
%       data = rand([100,50]);
%       label = LSC(data,10);
%Reference:
%
%	Xinlei Chen, Deng Cai, "Large Scale Spectral Clustering with
%	Landmark-Based Representation," AAAI 2011. 
%
%   version 2.0 --Dec./2011 
%   version 1.0 --Oct./2010 
%
%   Written by Xinlei Chen (endernewton AT gmail.com)
%              Deng Cai (dengcai AT gmail.com)



% Set and parse parameters
if (~exist('opts','var'))
   opts = [];
end


p = 1000;
if isfield(opts,'p')
    p = opts.p;
end

r = 5;
if isfield(opts,'r')
    r = opts.r;
end

maxIter = 100;
if isfield(opts,'maxIter')
    maxIter = opts.maxIter;
end

numRep = 10;
if isfield(opts,'numRep')
    numRep = opts.numRep;
end

mode = 'kmeans';
if isfield(opts,'mode')
    mode = opts.mode;
end

nSmp=size(data,1);

% Landmark selection
if strcmp(mode,'kmeans')
    kmMaxIter = 5;
    if isfield(opts,'kmMaxIter')
        kmMaxIter = opts.kmMaxIter;
    end
    kmNumRep = 1;
    if isfield(opts,'kmNumRep')
        kmNumRep = opts.kmNumRep;
    end
    [dump,marks]=litekmeans(data,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
    clear kmMaxIter kmNumRep
elseif strcmp(mode,'random')
    indSmp = randperm(nSmp);
    marks = data(indSmp(1:p),:);
    clear indSmp
else
    error('mode does not support!');
end

% Z construction
D = EuDist2(data,marks,0);

if isfield(opts,'sigma')
    sigma = opts.sigma;
else
    sigma = mean(mean(D));
end

dump = zeros(nSmp,r);
idx = dump;
for i = 1:r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100; 
end

dump = exp(-dump/(2*sigma^2));
sumD = sum(dump,2);
Gsdx = bsxfun(@rdivide,dump,sumD);
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);

% Graph decomposition
feaSum = full(sqrt(sum(Z,1)));
feaSum = max(feaSum, 1e-12);
Z = Z./feaSum(ones(size(Z,1),1),:);
U = mySVD(Z,k+1);
U(:,1) = [];

U=U./repmat(sqrt(sum(U.^2,2)),1,k);

% Final kmeans
label=litekmeans(U,k,'MaxIter',maxIter,'Replicates',numRep);




