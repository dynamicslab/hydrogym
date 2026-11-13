function mplot
  global conn vert
  %clf
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]), ...
        'FaceColor','none','Marker','o');
end
