global A iint

xn=nodes(n);
%xn=[-1:2/(n-1):1];
%Af = Afull(1,eps,eps,xn);
%Af = Afull(0,eps,eps,xn);
%Af = Afull(0,100*eps,eps,xn);
Af = sdop(eps,xn,xn);
mesh2d(xn',xn');

iint = sort(reshape(ones(n-2,1)*[2:n-1] + (n*[1:n-2]')*ones(1,n-2),1,[]));
A = Af(iint,iint);

ub = zeros(n*n,1);
ub(n + n*[0:n-1])=1;

f = -Af(iint,:)*ub;

u = A\f;

%reshape(u,n-2,n-2)

uf = ub; uf(iint)=u;

surf(xn,xn,reshape(uf,n,n)')

% more information

% velocity
cx =  2*((0*xn+1)'*xn).*((1-xn.*xn)'*(0*xn+1));
cy = -2*(xn'*(0*xn+1)).*((0*xn+1)'*(1-xn.*xn));
%quiver(xn,xn,cx',cy');

xnm = .5*(xn(1:end-1)+xn(2:end)); % midpoints
hx = diff(xn); % mesh spacing
cxm =  2*((0*xnm+1)'*xnm).*((1-xnm.*xnm)'*(0*xnm+1));
cym = -2*(xnm'*(0*xnm+1)).*((0*xnm+1)'*(1-xnm.*xnm));
cm = sqrt(cxm.*cxm + cym.*cym);
%surf(xnm,xnm,cm');
% h = mesh spacing in direction of c
h = min( (hx'*(0*hx+1)).*cm ./ abs(cxm) , ((0*hx+1)'*hx).*cm ./ abs(cym) );
peclet = cm.*h/eps;
%surf(xnm,xnm,peclet');