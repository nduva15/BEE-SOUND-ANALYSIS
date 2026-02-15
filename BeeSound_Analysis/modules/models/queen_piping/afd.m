function [v, d, lmd] = afd(data, indices)
% [V, D] = afd(data, indices)
% compute discriminant factor analysis from data
%
%
% INPUT:
% data : matrix where each line is a unit and each colum is a factor (parameter or descriptor)
% indices: matrix which contains the bounds of each class (lines correspond to each class)
%
% OUTPUT:
% v: eigen vectors
% d: diagonal matrix which contain eigen values
% lmd: lambda de Wilk (deprecated)(see. http://www.jybaudot.fr/Inferentielle/lambdawilks.html)
%
% Author: Dominique Fourer
% Date: 21-03-2014

data = real(data);

nb_desc = size(data, 2);
nb_unit = size(data, 1);
nb_clust = size(indices, 1);

mu = mean(data);        %% gravity center
% S = sqrt(var(data));   %% 
%data = data - repmat(mu, nb_unit, 1);  %%centrage



%% compute total covariance-
V = cov(data);
% dV = det(V);

W = zeros(nb_desc, nb_desc); % compute intra-class variance-covariance matrix

B = zeros(nb_desc, nb_desc);
nk = zeros(1, nb_clust);
for i = 1:nb_clust
 I = indices(i,1):indices(i,2);
 nk(i) = length(I);
 W = W +  nk(i) * cov(data(I, :));
 B = B +  nk(i) * (mean(data(I, :)) - mu)' * (mean(data(I, :)) - mu);       %% matrice des moyenne de chaque classe
end

W = W / nb_unit;
B = B / nb_unit;

%lambda de wilk
lmd = det(W) / (det(V)+eps);

% compute inter-class variance-covariance matrix
%* cov(W_tmp); % - repmat(mean(W_tmp), nb_clust, 1) );
%B = 1/nb_unit * cov(W_tmp - repmat(mean(W_tmp), nb_clust, 1) );


%[v, d] = eig(pinv(B+W) * B);
[v, d] = eig(pinv(V) * B);



