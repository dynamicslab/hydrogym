function [x hist] = gmres(A, b, Ml, Mr, X, opts)
%GMRES GMRES on A x = b with preconditioner Mr*Ml
%   Minimizes || Ml (b - A x) ||_X    ( X should be SPD )

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

if isfield(opts,'stall'); stall=opts.stall; else stall=.99; end
if isfield(opts,'maxsteps'); maxsteps=opts.maxsteps; else maxsteps=n; end
if maxsteps==0; x=zeros(n,1); return; end

A  = make_fun(A);
Ml = make_fun(Ml);
Mr = make_fun(Mr);
X  = make_fun(X);

Q = Ml(b);
g = sqrt( Q' * X(Q) );
if g==0; x=zeros(n,1); return; end
Q = Q/g;
hist = g;

%tol = max(tol,1e-16);
tol = tol * g;

Z = zeros(n,0);
%H = zeros(0,0);
R = zeros(0,0);
c = zeros(0);
s = zeros(0);
k=0;
maxsteps = min(n,maxsteps);

while 1
    k=k+1;
    z = Mr(Q(:,k)); Z = [Z z];
    q = Ml(A(z));
    h = zeros(k,1);
    for i=1:k
        h(i) = q' * X( Q(:,i) );
        q = q - h(i) * Q(:,i);
    end
    alpha = sqrt( q' * X( q ) );
    %H = [H, h; zeros(1,k-1), alpha];
    Q = [Q, q/alpha];
    %Q'*X*Q % == I
    %Q'*X*Ml*A*Z % == H
    %H
    for i=1:k-1
        t = [c(i) s(i); -s(i) c(i)] * [h(i); h(i+1)];
        h(i)=t(1); h(i+1)=t(2);
    end
    l = norm([h(k); alpha]);
    c = [c; h(k)/l];
    s = [s; alpha/l];
    h(k) = l;
    R = [[R;zeros(1,k-1)], h];
    if condest(R) > 1e+14; k=k-1; break; end
    g = [g; -g(k)*s(k)]; g(k) = g(k)*c(k);
    hist = [hist; abs(g(k+1))];
    %abs(g(k+1))
    if k==maxsteps || abs(g(k+1)) <= tol || abs(s(k)) <= 1e-15; break; end
    if hist(k+1)/hist(k) >= stall; break; end
    %if (k>15 && c(k) < 1e-2); break; end
    %pause
end

%g
y = R(1:k,1:k) \ g(1:k,1);
x = Z(:,1:k)*y;

end

