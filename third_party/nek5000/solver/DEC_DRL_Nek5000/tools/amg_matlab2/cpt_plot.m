function cpt_plot(data,lvl,m)
  global conn vert iint
  C = data.id{lvl+1};
  F = xor(data.id{lvl},C);
  v = vert(iint,:);
  vf = v(F,:);
  vc = v(C,:);
  clf
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]),'FaceColor','none');
  hold on
  plot(vf(:,1),vf(:,2),'o', ...
       'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',m)
  plot(vc(:,1),vc(:,2),'o', ...
       'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',m)
  hold off
end
