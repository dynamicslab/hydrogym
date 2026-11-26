function x = gmres_diag(A, d, b, tol )
%GMRES_DIAG GMRES on A x = b with diagonal preconditioner D=diag(d)
%   The residual is minimized in the ||.||_D norm

g = sqrt( b' * (d .* b) );
if g(1)==0; x = 0*b; return; end
Q = b/g(1);

if nargin<4; tol=0; end
tol = max(tol,1e-16);
tol = tol * g;

%H = zeros(0,0);
R = zeros(0,0);
c = zeros(0);
s = zeros(0);
k=0;
[~,n]=size(A);
while 1
    k=k+1;
    w = A * ( d .* Q(:,k) );
    h = zeros(k,1);
    for i=1:k
        h(i) = w' * ( d .* Q(:,i) );
        w = w - h(i) * Q(:,i);
    end
    alpha = w' * ( d .* w );
    %if alpha==0; break; end
    alpha = sqrt(alpha);
    %H = [H, h; zeros(1,k-1), alpha];
    Q = [Q, w/alpha];
    %X = diag(sparse(d));
    %Q'*X*Q % == I
    %Q'*X*A*X*Q(:,1:end-1) % == H
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
    %abs(g(k+1))
    if abs(g(k+1)) <= tol || abs(s(k)) <= 1e-15; break; end
    if k==n || (k>15 && c(k) < 1e-2); break; end
    %pause
end

%g
y = R(1:k,1:k) \ g(1:k,1);
x = d .* (Q(:,1:k)*y);

end

