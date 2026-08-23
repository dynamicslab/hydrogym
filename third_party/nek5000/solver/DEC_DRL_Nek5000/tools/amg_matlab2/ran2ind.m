function r = ran2ind(sz, ir, jr, kr )
%RAN2IND logical vector of linear indices from ranges
%   Returns sparse logical vector of length prod(sz).
%   In 3D, if [i,j,k] = ind2sub(sz,l) then r(l) is true exactly when
%      ir contains i, jr contains j, and kr contains k.

if nargin==4
    [i,j,k] = ndgrid(ir,jr,kr);
    i = sub2ind(sz,i,j,k);
else
    [i,j] = ndgrid(ir,jr);
    i = sub2ind(sz,i,j);
end

r = logical(sparse(reshape(i,[],1),1,1,prod(sz),1));
    
end

