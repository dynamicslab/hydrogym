function lam = solve_constraint( W_skel, Af, W0, alpha, u, v, tol, lam )
%SOLVE_CONSTRAINT Finds Lagrange multiplier to enforce W*u = v

[nf,nc] = size(W_skel);

if nargin<8; lam=zeros(nf,1); end
if nargin<7; tol=1e-16; end

S = cinterp_lmop(Af,alpha.*(u.*u),W_skel,logical( (+W_skel)*(+W_skel)' ));

resid = v - W0*u;
i = logical(diag(S));
if ~all(i); S = S(i,i); lam(~i) = 0; end
%lam(i) = S(i,i) \ resid(i);
%lam(i) = gmres_diag(S(i,i),1./full(diag(S(i,i))),resid(i),tol);
%[x,k] = pcg(spdiag(1./diag(S)),S,resid(i),lam(i),tol);
[x,k] = pcg(S,resid(i)-S*lam(i),1./diag(S),tol,resid(i));
lam(i) = lam(i)+x;
%[x,k] = pcg(S,resid(i),1./diag(S),tol);
%lam(i) = x;
if tol==1e-16; fprintf(1,'Lagrange multiplier found in %d PCG iterations.\n',k); end

end

