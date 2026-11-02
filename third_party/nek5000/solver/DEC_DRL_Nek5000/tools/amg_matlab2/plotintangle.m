function plotintangle(u,uf)
n = sqrt(length(u))+2;
if nargin==1; uf=zeros(n*n,1); end
uf = reshape(uf,n,n);
uf(2:end-1,2:end-1) = reshape(u,n-2,n-2);
uf=reshape(uf,[],1);
uf = hsv2rgb([(pi+angle(uf))/(2*pi) (0*uf+1) (0*uf+1)]);

global conn vert
clf
patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]), ...
      'FaceVertexCData',uf,'FaceColor','interp','EdgeColor','none');

end
