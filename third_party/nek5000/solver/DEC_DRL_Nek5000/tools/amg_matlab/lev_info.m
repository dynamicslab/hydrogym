   Aff = data.A{lvl}(data.F{lvl},data.F{lvl});
   Afc = data.A{lvl}(data.F{lvl},data.C{lvl});
   D = diag(sparse(1./sqrt(diag(Aff))));
