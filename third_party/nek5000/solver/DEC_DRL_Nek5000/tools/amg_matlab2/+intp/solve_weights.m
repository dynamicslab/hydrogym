function [W,W0,lam] = solve_weights(W_skel,Af,Ar,alpha,u,v,tol,lam)
%SOLVE_WEIGHTS Summary of this function goes here
%   Detailed explanation goes here

[nf,~] = size(W_skel);
%[Xi,b,Xf] = setup(W_skel);
%W0 = intp.solve_columns(W_skel,Af,Ar,Xi,b,alpha,u,zeros(nf,1));
W0 = cinterp(Af,-Ar,alpha.*u,zeros(nf,1),W_skel);
if nargin<8; lam=zeros(nf,1); end;
lam = intp.solve_constraint(W_skel,Af,W0,alpha,u,v,tol,lam);
%W = intp.solve_columns(W_skel,Af,Ar,Xi,b,alpha,u,lam);
W = cinterp(Af,-Ar,alpha.*u,lam,W_skel);

end

