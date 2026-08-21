function lambda = glanczos(n,A,Bi,tol)
    if nargin<4; tol=1e-4; end
    A = make_fun(A);
    Bi = make_fun(Bi);

    r = rand(n,1);
    q = Bi(r);
    beta = sqrt(abs(q'*r));
    wj = zeros(n,1); vj = wj;
    a = zeros(0,1); b=a;
    for j=1:300
        wjm1 = wj; vjm1 = vj;
        wj = r/beta; vj = q/beta;
        r = A(vj);
        r = r - wjm1*beta;
        alpha = vj'*r; a=[a;alpha];
        r = r - wj*alpha;
        %reorthogonalize locally
        r = r - wjm1*(vjm1'*r);
        r = r - wj*(vj'*r);
        %
        if j==1
            l = alpha; y = 1;
        else
            [l y] = tdeig([0; real(l); 0],real([alpha; beta*y]),j-1);
        end
        %
        q = Bi(r);
        beta = sqrt(abs(q'*r)); b=[b;beta];
        %T = spdiags(b,-1,j,j); T = T' + T; T = T + spdiag(a); full(T)
        %eig(full(T))
        c = beta * y(end);
        fprintf(1,'Lanczos iteration %d : %g at %g (%g)\n',j,l(end),y(end),c);
        if (abs(y(end)<0.01) && c<tol) || beta==0; break; end
    end
    lambda = l(end);
end

