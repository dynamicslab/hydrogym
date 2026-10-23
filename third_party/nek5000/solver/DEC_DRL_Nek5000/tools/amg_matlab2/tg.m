function tg(level,m)
  global data
  A = data.A{level};
  Af = data.Af{level}; Ar = data.Ar{level};
  Wr = data.Wr{level}; Wl = data.Wl{level};
  F = data.F{level}; C = data.C{level};
  nf = length(F); nc = length(C); n = nf+nc;
  Al = data.A{level}(C,F)';
  Ach = data.A{level+1};
  
  Pr = sparse(n,nc); Pr(F,:) = Wr; Pr(C,:) = speye(nc);
  Pl = sparse(n,nc); Pl(F,:) = Wl; Pl(C,:) = speye(nc);
  
  Df = data.Df{level};
  EDf = speye(nf) - Df*Af;
  Bf = Df;
  for k=2:m
    Bf = Df + EDf*Bf;
  end
  
  B = sparse(n,n); B(F,F) = Bf;
  
  Etg = (speye(n)-B*A) * (speye(n) - Pr * (Ach \ full(Pl'*A)));
  
  max(abs(eig(Etg)))
  plot(eig(Etg),'.');  
  
  Bfabs = Bf*sqrtm(Bf\full(Bf')); Bfabs = .5*(Bfabs+Bfabs');
  Acabs = Ach*sqrtm(Ach\full(Ach')); Acabs = .5*(Acabs+Acabs');
  
  max(abs(1-eig(full(Af*Bf))))
  rf = gnorm(speye(nf)-Af*Bf,Bfabs,Bfabs)

  cr = gnorm(Af *Wr+Ar,Bfabs,Acabs)
  cl = gnorm(Af'*Wl+Al,Bfabs,Acabs)

  rf + cr*cl
  
end
