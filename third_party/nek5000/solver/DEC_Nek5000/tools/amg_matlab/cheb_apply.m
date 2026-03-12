function y = cheb_apply(A,B1,rho,m,r)
	alpha = .25*rho*rho;
	n = 1; yp = 0*r; y = B1*r;
	gamma = 1;
	while m>n
		n=n+1;
		d = alpha*(1+gamma);
		gamma = d/(1-d);
		yn = (1+gamma)*(y + B1*(r-A*y)) - gamma*yp;
		yp = y; y=yn;
	end
end
