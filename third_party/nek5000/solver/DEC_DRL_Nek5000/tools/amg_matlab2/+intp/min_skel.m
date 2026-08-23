function W_skel = min_skel( R )
%MIN_SKEL Minimal interpolation skeleton

[nf,nc] = size(R);
[y,j]=max(R,[],2);
W_skel = logical(sparse(find(y>0),j(y>0),1,nf,nc));

end

