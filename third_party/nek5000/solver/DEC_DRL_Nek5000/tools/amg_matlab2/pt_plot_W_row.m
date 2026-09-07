function pt_plot_W(lvl,i,rl,m)
  global data conn vert iint
  vf = vert(iint(data.F_id{lvl}),:);
  vc = vert(iint(data.C_id{lvl}),:);
  if rl=='r'
    w = m*full(data.Wr{lvl}(i,:));
  else
    w = m*full(data.Wl{lvl}(i,:));
  end
  j = w>0;
  scatter(vc(j,1),vc(j,2),w(j),'r','filled');
  hold on;
  scatter(vf(i,1),vf(i,2),m,'b','filled');
  hold off;
  axis([-1 1 -1 1]);
  axis square;
  set(gca, 'XTick', []);
  set(gca, 'YTick', []);
  box on;
end
