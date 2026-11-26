function lambda = skew_lanczos(n,A,Bi,tol)
    if nargin<4; tol=1e-4; end
    A = make_fun(A);
    Bi = make_fun(Bi);

    r = rand(n,1);
    q = Bi(r);
    beta = sqrt(abs(q'*r));
    wj = zeros(n,1); vj = wj;
    a = zeros(0,1); b=a;
    l = 0; y = 1;
    for j=1:300
        wjm1 = wj; vjm1 = vj;
        wj = r/beta; vj = q/beta;
        r = A(vj);
        r = -(r - wjm1*beta);
        %reorthogonalize locally
        r = r - wjm1*(vjm1'*r);
        r = r - wj*(vj'*r);
        %
        q = Bi(r);
        beta = sqrt(abs(q'*r)); b=[b;beta];
        %T = spdiags(b,-1,j,j); T = T' - T;
        %eig(full(T))
        c = beta * y(end);
        [l y] = tdeig([0; real(l); 0],real([0; beta*y]),j);
        fprintf(1,'Skew symmetric Lanczos iteration %d : %g at %g (%g)\n',j,l(end),y(end),c);
        if c<tol; break; end
    end
    lambda = l(end);
end

