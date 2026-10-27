function genuniplot(n)
  global vert conn
  unimesh2d(n,1,1);
  clf                                                 
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]), ...
        'FaceColor','none','Marker','none','LineWidth',3);
  axis off;
end
