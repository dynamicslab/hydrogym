function W = intp(A, C, gamma2, u, tol1 )
%INTP Summary of this function goes here
%   Detailed explanation goes here

nc = sum(C); one = ones(nc,1);
if nargin<4; u = ones(length(C),1); end
if nargin<5; tol1 = 1e-4; end
%gamma2=gamma*gamma;

show_plot = usejava('desktop');

F = ~C; nf = sum(F);
Af = A(F,F);
Ar = A(F,C);
Ac = A(C,C);

if nc==0; W=sparse(nf,nc); return; end

d = 1./full(diag(Af));
uc = u(C);
v = pcg(Af,full(-Ar*uc),d,1e-16);
%v = gmres(Af ,full(-Ar*uc),[],d      ,abs(d));

d = diag(Af);
Df = spdiag(d);
Dfsqrti = spdiag(sqrt(1./d));

dc = abs(full(diag(Ac)));
Dc = spdiag(dc);

W_skel = intp.min_skel( (Ar/Dc) .* (Df\Ar) );

alpha = dc;

lam=zeros(nf,1);

fprintf(1,'Interpolation weights:\n');

while true

[W,W0,lam] = intp.solve_weights(W_skel,Af,Ar,alpha,uc,v,tol1,lam);

Arhat0 = Af*W0+Ar;
Arhat  = Af*W +Ar;

%Achat  = W'*Arhat + Ar'*W + Ac;
dchat  = full(sum(W .* Arhat + Ar .* W, 1).' + diag(Ac));
Dcsqrti = spdiag(1./sqrt(dchat));

R  = abs(Dfsqrti*Arhat )*Dcsqrti;
R0 = abs(Dfsqrti*Arhat0)*Dcsqrti;

w1 = full(((R*one)'*R)');
w2 = full(((R*w1 )'*R)');
r = w2./w1; r(w1==0) = 0;

if show_plot; plot([w1 r]); pause(0.01); end

n = sum(r>gamma2);

fprintf(1,'  %d nzs, %d cols > %g, worst = %g\n', nnz(W_skel), n, sqrt(gamma2), sqrt(max(r)));

if (n==0 || max(w1)<=gamma2)
    % Polish weights
    W = intp.solve_weights(W_skel,Af,Ar,alpha,uc,v,1e-16,lam);
    i = logical(W*uc);
    if sum(i)>0; W(i,:) = spdiag(v(i) ./ (W(i,:)*uc)) * W(i,:); end
    break;
end

alpha = dc./max(w2,1e-6);

W_skel = intp.expand_support(W_skel,R,R0,gamma2);

end

end

