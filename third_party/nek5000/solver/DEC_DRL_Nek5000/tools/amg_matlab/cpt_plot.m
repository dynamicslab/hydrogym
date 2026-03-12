function cpt_plot(data,lvl,m)
  global conn vert
  clf
  patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]),'FaceColor','none');
	hold on
	plot(vert(data.F_id{lvl},1),vert(data.F_id{lvl},2),'o', ...
	     'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',m/2)
	plot(vert(data.C_id{lvl},1),vert(data.C_id{lvl},2),'o', ...
	     'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',m)
	hold off
end
