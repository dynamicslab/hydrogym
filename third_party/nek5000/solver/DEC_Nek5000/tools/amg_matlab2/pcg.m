function [x, k] = pcg(A,r,M,tol,b)
    A = make_fun(A);
    M = make_fun(M);
    x = zeros(size(r)); p=x;
    z = M(r);
    rho = r'*z;
    if nargin<5
        rho_0 = rho;
    else
        rho_0 = b'*M(b);
    end
    rho_stop=tol*tol*rho_0;
    k = 0;
    n = min(length(r),100);
    if n==0; return; end
    rho_old = 1;
    while rho > rho_stop && k<n
        k = k+1;
        beta = rho / rho_old;
        p = z + beta * p;  % when k=1, p=0, so this reduces to p=z
        w = A(p);
        alpha = rho / (p'*w);
        x = x + alpha*p;
        r = r - alpha*w;
        z = M(r);
        rho_old = rho; rho = r'*z;
    end
	%fprintf(1, ' %d iterations.\n', k);
end
