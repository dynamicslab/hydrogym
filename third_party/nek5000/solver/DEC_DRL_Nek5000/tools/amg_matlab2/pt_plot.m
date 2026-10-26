function pt_plot(data,lvl,m)
  global conn vert iint
  id = data.id{lvl};
  v = vert(iint,:);
  vc = v(id,:);
  clf
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]),'FaceColor','none');
  hold on
  %plot(vf(:,1),vf(:,2),'o', ...
  %     'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',m/2)
  plot(vc(:,1),vc(:,2),'o', ...
       'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',m)
  hold off
end
