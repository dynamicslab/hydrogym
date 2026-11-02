function new_skel = expand_support(W_skel, R, R0, gamma )
%EXPAND_SUPPORT Expand W_skel towards making ||R||_2^2 <= gamma

M = intp.find_support(R,gamma);
new_skel = W_skel | M;

bad_row = logical(sum(M & W_skel, 2)); nbad = sum(bad_row);
if nbad==0; return; end

[nf,nc] = size(W_skel);
ibr = find(bad_row);
X = R0 - (R0 .* W_skel);
X = abs(X(bad_row,:));

if false

% add the worst new element from each bad row
[~,j]=max(X,[],2);
M = sparse(ibr,j,ones(nbad,1),nf,nc);
new_skel = new_skel | M;

else

% add the largest new elements from each bad row,
%   whose collective sum is at least half the row sum (of all new elements)

% rank elements of X by descending magnitude, in each row
vec = @(x) reshape(x,[],1);
[i j v] = find(X); sx = [vec(i) vec(j) vec(v)];
[~,idx] = sort(-sx(:,3)); sx = sx(idx,:);
[~,idx] = sort( sx(:,1)); sx = sx(idx,:);

[i j v] = find(sort(X,2,'descend')); sy = [vec(i) vec(j) vec(v)];
[~,idx] = sort(sy(:,1)); sy = sy(idx,:);

w = max(sy(:,2));  % max # of nonzeros in any row
X = sparse(sy(:,2),sx(:,1),sx(:,3),w,nbad);
J = sparse(sy(:,2),sx(:,1),sx(:,2),w,nbad);
% X_ij  is the ith largest value in row j,
% J_ij  is the column X_ij was in
%   let t_j = sum_{k=1}^w X_{kj}     (total of row j)
S = cumsum(X);
V = ones(w,1) * sum(X)/2;  % V_ij = t_j/2
% (S-V)_ij < 0 when sum_{k=1}^i X_{kj} < 1/2 t_j
N = ones(w,1) * (1+sum((S-V)<0));
% N_ij = n_j, where n_j is min s.t. sum_{k=1}^(n_j) X_{kj} >= t_j
L = sparse( ((1:w)' * ones(1,nbad)) <= N);
% L_ij = { 1  i <= n_j
%        { 0  i >  n_j
[~,i,j] = find(L .* J);
N = sparse(ibr(i),j,1,nf,nc);
new_skel = new_skel | N;

end

end

