function vc = coarsen_plot(method,A,tol)
  global conn vert iint
  n = length(A(:,1));
  fprintf(1,'Computing S = I - D^{-1/2} A D^{-1/2} ...');
  D = spdiag(1./sqrt(diag(A)));
  S = abs(D*A*D);
  S = S - spdiag(diag(S));
  H = max(S,S');
  fprintf(1,' done, nnz(S) = %d.\n',nnz(S));
  
  switch method
      case 'Ostrowski'  % bounds spr S(F,F)
          f = @(vf) vf .* sqrt((S*vf).*(S'*vf));
      case 'norm'  % bounds || S(F,F) ||_2 
          f = @(vf) sqrt(max( vf.*(S'*(vf.*(S*vf))), ...
                              vf.*(S*(vf.*(S'*vf)))));
  end
  
  vc = false(n,1); vf = true(n,1);
  id = (1:n)';
  fprintf(1,'Running coarsening heuristic ...\n');
  k=1;
  while 1
    nc = sum(vc);
    g = f(vf);
    vplot(g); caxis([0 2.3]); hold on;
    vertc = vert(iint,:);
    patch('Vertices',vert,'Faces',conn(:,[1 2 4 3]),'FaceColor','none');
    plot(vertc(vc,1),vertc(vc,2),'o', 'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',16);
    axis equal; axis off
    fname = sprintf('coarsening_anim/crs%02d.png',k); k=k+1;
    print('-dpng','-r300',fname);
    hold off;
    [mg,~] = max(g);
    %fprintf(1,'  ratio = %g, n = %d, max %s radius = %g\n', ...
            %nc/n, nc, method mg);
    if mg<=tol; break; end
      %if nc==0; vc(mi)=true; vf(mi)=false; end
      %break;
    %end
    mask = g>tol;
    m = mat_max(H,vf,mask.*g);
    mask = mask & (g-m>=0);
    m = mat_max(H,vf,mask.*id);
    mask = mask & (id-m>0);
    vc = vc | mask; vf = xor(vf, mask);
  end
  fprintf(1,'  ratio = %g, n = %d, max %s radius = %g\n', ...
          nc/n, nc, method, mg);
end

