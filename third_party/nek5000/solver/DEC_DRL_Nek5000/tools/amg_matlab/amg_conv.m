function amg_conv(data,level,ns)
	z = sort(abs(1-feig(amg_full(data,level,ns)*data.A{level})));
	n = length(z);
	z([n-1, n])
end
