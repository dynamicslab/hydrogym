function [x hist] = iterate_hist(A, b, Ml, Mr, X, opts)
%ITERATE_HIST Simple linear iteration on A x = b with B = Mr*Ml
%   Records || Ml (b - A x) ||_X  in hist  ( X should be SPD )

n=length(b);

if nargin<3; Ml=[]; end
if nargin<4; Mr=[]; end
if nargin<5; X =[]; end
if nargin<6; opts=[]; end

if isnumeric(opts) && isscalar(opts)
    tol = opts;
elseif isfield(opts,'tol');
    tol = opts.tol;
else
    tol = 0;
end

if isfield(opts,'stall'); stall=opts.stall; else stall=[]; end
if isfield(opts,'maxsteps'); maxsteps=opts.maxsteps; else maxsteps=n; end

A  = make_fun(A);
Ml = make_fun(Ml);
Mr = make_fun(Mr);
X  = make_fun(X);

x = zeros(n,1);
r = Ml(b);
hist = sqrt( r' * X(r) );
if hist==0; return; end

tol = max(tol,1e-16);
tol = tol * hist;

k=0;
maxsteps = min(n,maxsteps);
if maxsteps==0; return; end

while 1
    k=k+1;
    x = x + Mr(r);
    r = Ml(b - A(x));
    hist = [hist; sqrt( r' * X(r) )];
    if k==maxsteps || hist(k+1) <= tol; break; end
    if isscalar(stall) && hist(k+1)/hist(k) >= stall; break; end
end

end

