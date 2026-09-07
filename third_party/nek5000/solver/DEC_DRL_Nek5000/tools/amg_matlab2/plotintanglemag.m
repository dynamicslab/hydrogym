function plotintanglemag(u,uf)
n = sqrt(length(u))+2;
if nargin==1; uf=zeros(n*n,1); end
uf = reshape(uf,n,n);
uf(2:end-1,2:end-1) = reshape(u,n-2,n-2);
uf=reshape(uf,[],1);
a = max(abs(uf));
m = abs(uf)/a;
a = (angle(uf))/(2*pi) * 360;
a = ((a<0) .* (360+a)) + ((a>=0).*a);
%uf = hsv2rgb([(pi+angle(uf))/(2*pi) (0*uf+1) abs(uf)/a]);
%uf = colorspace('<-HSI',[a (0*a+1) m]);

%uf = colorspace('XYZ<-HSV',[a (0*a+1) (0*a+1)]);
%uf = ([m m m].^2) .* uf;
%uf = colorspace('<-XYZ',uf);

uf = colorspace('XYZ<-LCH',[(50+0*a) (30+0*a) a]);
uf = ([m m m].^1.5) .* uf;
uf = colorspace('<-XYZ',uf);

%uf = colorspace('<-LCH',[(50*m) (30*m) a]);

global conn vert
clf
patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]), ...
      'FaceVertexCData',uf,'FaceColor','interp','EdgeColor','none');

end
