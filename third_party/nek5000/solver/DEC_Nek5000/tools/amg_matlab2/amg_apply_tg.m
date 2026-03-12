function x = amg_apply_tg(data,m,level,r)
  if data.n(level)==0
    x = zeros(0,1);
  elseif data.n(level)==1
    %if ns
    %  x = zeros(1,1);
    %else
      x = full(1/data.A{level})*r;
    %end
  else
    Af = data.Af{level}; Ar = data.Ar{level};
    Wr = data.Wr{level}; Wl = data.Wl{level};
    F = data.F{level}; C = data.C{level};
    x = 0*r;
    r(C) = r(C) + Wl'*r(F);
    x(C) = data.A{level+1}\r(C);
    x(F) = Wr*x(C);
    r(F) = r(F) - Af*x(F) - Ar*x(C);
    x(F) = x(F) + iterate(Af,data.Df{level},r(F),m);
  end
end
