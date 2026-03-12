function plotint(u,uf)
n = sqrt(length(u))+2;
if nargin==1; uf=zeros(n*n,1); end
uf = reshape(uf,n,n);
uf(2:end-1,2:end-1) = reshape(u,n-2,n-2);
vplot(reshape(uf,[],1));
%xn=nodes(n);
%xn=xn(2:end-1);
%x=reshape(xn'*ones(1,n-2),[],1);
%y=reshape(ones(n-2,1)*xn,[],1);
%tri=delaunay(x,y)
%patch(tri,x,y,reshape(u,n-2,n-2));
end
