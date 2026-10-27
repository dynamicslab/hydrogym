function cpt_plot(data,lvl,m)
  global conn vert iint
  vf = vert(iint(data.F_id{lvl}),:);
  vc = vert(iint(data.C_id{lvl}),:);
  hold on
  plot(vf(:,1),vf(:,2),'o', ...
       'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',m)
  plot(vc(:,1),vc(:,2),'o', ...
       'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',m)
  hold off
end
