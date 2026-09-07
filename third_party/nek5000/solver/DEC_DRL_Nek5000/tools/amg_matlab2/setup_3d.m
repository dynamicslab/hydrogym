
xn=nodes(n);
zn=linspace(-1,1,n);
%xn=[-1:2/(n-1):1];
Af = sdop_3d(eps,xn,xn,zn);
%mesh2d(xn',xn');

% make z periodic
Qp = speye(n*n*n);
Qp(ran2ind([n n n],1:n,1:n,n), ran2ind([n n n],1:n,1:n,1)) = speye(n*n);
Qp = Qp(:,ran2ind([n n n],1:n,1:n,1:n-1));
Af = Qp'*Af*Qp;

% Direchlet boundaries at x,y = +/- 1
Qd = speye(n*n*(n-1));
Qd = Qd(:,ran2ind([n n (n-1)],2:n-1,2:n-1,1:n-1));
A = Qd'*Af*Qd;

% right hand side from inhomogeneous Direchlet boundary
ub = zeros(n*n*(n-1),1);
ub(ran2ind([n n (n-1)],n,1:n,1:floor(n/2))) = 1;
f = - Qd'*(Af*ub);

Q = Qp*Qd;
ir = ran2ind([n n n],2:n-1,2:n-1,1:n-1);

u = A\f;

% velocity
[XN,YN,ZN] = meshgrid(xn,xn,zn);
cx =  2*(YN.*(1-XN.*XN));
cy = -2*(XN.*(1-YN.*YN));
cz = 0*XN + .05;

%xs = [-.9:.1:0];
%streamline(XN,YN,ZN,cx,cy,cz,xs,0*xs,0*xs-1);