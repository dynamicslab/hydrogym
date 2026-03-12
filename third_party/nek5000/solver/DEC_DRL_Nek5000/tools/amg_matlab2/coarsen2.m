function vc = coarsen2(A,tol)
  [n,~] = size(A);
  D = spdiag(1./sqrt(diag(A)));
  S = abs(D*A*D);
  S = S - spdiag(diag(S));
  
  vc = false(n,1); vf = true(n,1); id = (1:n)';
  while true
    g = vf.*(S*vf);

    w1 = vf.*(S * g);
    w2 = vf.*(S *(vf.*(S*w1)));
    w = w2./w1; w(w1==0) = 0;
    [w1m,mi] = max(w1);
    b = sqrt(min(w1m,max(w)));
    if b<=tol;
      if ~any(vc); vc(mi)=true; end
      break;
    end

    if b < tol; break; end

    %fprintf(1,'  ratio = %g, n = %d, norm <= %g, max radius = %g\n', ...
    %      sum(vc)/n, sum(vc), b, max(g));

    mask = w > tol^2;
    m = mat_max(S,vf,mask.*g);
    mask = mask & (g-m>=0);
    m = mat_max(S,vf,mask.*id);
    mask = mask & (id-m>0);
    vc = vc | mask; vf = xor(vf, mask);
  end
  fprintf(1,'Coarsening ratio = %g, n = %d, norm bound = %g\n', ...
          sum(vc)/n, sum(vc), b);
end

