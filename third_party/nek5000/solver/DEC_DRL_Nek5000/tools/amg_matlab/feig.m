function z = feig(A)
	z = eig(full(A));
	[g,p] = sort(abs(z));
	z = z(p);
end
