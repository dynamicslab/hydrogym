function plotc_surf(d,u)
  global data vert iint
  uf = intpc(d,u);

  x = [-1:.01:1];
  [X Y] = meshgrid(x,x);
  v = griddata(vert(iint,1),vert(iint,2),uf,X,Y,'cubic');
  surf(x,x,v,'FaceColor','interp','EdgeColor','none');
  axis([-1 1 -1 1]);
  axis square;
  set(gca, 'XTick', []);
  set(gca, 'YTick', []);
  box on;
end
