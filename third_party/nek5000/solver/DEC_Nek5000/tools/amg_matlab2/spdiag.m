function D = spdiag( d )
%SPDIAG Sparse diagonal matrix D, D_ii = d_i

[n,m] = size(d);
if m==1
    [i, ~, v] = find(d);
    D = sparse(i,i,v,n,n);
elseif n==1
    [~, j, v] = find(d);
    D = sparse(j,j,v,m,m);
else
    D = sparse([]);
end

end

