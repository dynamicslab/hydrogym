function amg_gamma(data,lvl)
		Aff = data.A{lvl}(data.F{lvl},data.F{lvl});
		Afc = data.A{lvl}(data.F{lvl},data.C{lvl});
		W = data.Wt{lvl}';
		G = W'*(Aff*W + Afc) + Afc'*(W + Aff\Afc);
		G = pinv(full(data.A{lvl+1}))*G;
		gamma = sqrt(max(eig(G)))
end
