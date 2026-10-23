function pt_plot_W(lvl,i,rl,m)
  global data conn vert iint
  C = data.id{lvl+1};
  F = xor(data.id{lvl},C);
  v = vert(iint,:);
  vf = vert(F,:);
  vc = vert(C,:);
  if rl=='r'
    w = m*full(data.Wr{lvl}(:,i));
  else
    w = m*full(data.Ws{lvl}(:,i));
  end
  j = w>0;
  scatter(vc(i,1),vc(i,2),m,'r','filled');
  hold on;
  scatter(vf(j,1),vf(j,2),w(j),'b','filled');
  hold off;
  axis([-1 1 -1 1]);
  axis square;
  set(gca, 'XTick', []);
  set(gca, 'YTick', []);
  box on;
end
