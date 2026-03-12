function skel = find_support( R, goal )
%FIND_SUPPORT Finds skel such that  || R-R.*skel ||_2^2 <= goal
%   assumes R is entry-wise nonnegative
%   skel is logical; the routine attempts to minimize nnz(skel)

[nf,nc] = size(R);
one = ones(nc,1);

nskel=0;
skel = zeros(nnz(R),2);

theta = 0.5;

while true

rs = R*one;
w = (rs'*R)';
w2 = ((R*w)'*R)';
v = w2./w; v(w==0) = 0;

%plot([w v]); pause(0.5);
if max( v ) < goal || max( w ) < goal; break; end

mw = max( w );
while mw <= (1+theta)*goal; theta=theta/2; end

%r = (v > goal) & sum(logical(R))' ;   % bad columns
r = (w > (1+theta)*goal) & sum(logical(R))' ;   % bad columns
nbad = sum(r);
X = spdiag(rs) * R(:,r);  % X_ij = R_ij ( e_i' R 1 )
if nf>1; [~,i] = max(X); else i=ones(1,nbad); end
mask = [ i' find(r) ];
M = logical(sparse(mask(:,1),mask(:,2),1,nf,nc));
R = R - (R.*M);

skel(nskel+(1:nbad),:) = mask;
nskel=nskel+nbad;

end

%fprintf(1,'find_support: %d nnzs\n', nskel);
skel = skel(1:nskel,:);
skel = logical(sparse(skel(:,1),skel(:,2),1,nf,nc));

end

