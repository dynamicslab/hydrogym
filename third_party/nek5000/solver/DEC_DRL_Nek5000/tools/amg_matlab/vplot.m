function vplot(u)
  global conn vert
  clf
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]), ...
        'FaceVertexCData',u,'FaceColor','interp');
end
