function vc = coarsen1(A,tol)
  n = length(A(:,1));
  fprintf(1,'Computing S = I - D^{-1/2} A D^{-1/2} ...');
  D = spdiag(1./sqrt(diag(A)));
  S = abs(D*A*D);
  S = S - spdiag(diag(S));
  fprintf(1,' done, nnz(S) = %d.\n',nnz(S));
  vc = false(n,1); vf = true(n,1);
  id = (1:n)';
  fprintf(1,'Running coarsening heuristic ...\n');
  while 1
    nc = sum(vc);
    g = vf .* (S*vf);
    [mg,mi] = max(g);
    %fprintf(1,'  ratio = %g, n = %d, max %s radius = %g\n', ...
            %nc/n, nc, method mg);
    if mg<=tol; %break; end
      if nc==0; vc(mi)=true; end
      break;
    end
    mask = g>tol;
    m = mat_max(S,vf,mask.*g);
    mask = mask & (g-m>=0);
    m = mat_max(S,vf,mask.*id);
    mask = mask & (id-m>0);
    vc = vc | mask; vf = xor(vf, mask);
  end
  fprintf(1,'  ratio = %g, n = %d, max radius = %g\n', ...
          nc/n, nc, mg);
end

