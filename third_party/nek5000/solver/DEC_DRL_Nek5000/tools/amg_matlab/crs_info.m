function crs_info(data)
	n = length(data.n);
	for lvl=[1:n-1]
		Aff = data.A{lvl}(data.F{lvl},data.F{lvl});
		Afc = data.A{lvl}(data.F{lvl},data.C{lvl});
		D = diag(sparse(1./sqrt(diag(Aff))));
		lambda = lanczos(D*Aff*D);
		fprintf(1,'%d: ratio = %g, kappa = %g < %g\n', lvl, ...
      data.n(lvl+1)/data.n(lvl), ...
			max(lambda)/min(lambda), (1+data.tolc)/(1-data.tolc));
	end
end
