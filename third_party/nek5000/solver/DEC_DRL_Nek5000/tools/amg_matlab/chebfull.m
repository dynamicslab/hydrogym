function B = chebfull(A,B1,rho,m)
	I = speye(length(A(:,1)));
	alpha = .25*rho*rho;
	n = 1; Bp = 0*B1; B = B1;
	gamma = 1;
	while m>n
		n=n+1;
		d = alpha*(1+gamma);
		gamma = d/(1-d);
		Bn = (1+gamma)*(B + B1*(I-A*B)) - gamma*Bp;
		Bp = B; B=Bn;
	end
end
