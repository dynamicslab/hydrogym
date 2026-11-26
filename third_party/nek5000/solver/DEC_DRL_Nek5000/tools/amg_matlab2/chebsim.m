function [m,c] = chebsim(rho,tol)
	alpha = .25*rho*rho;
	m = 1; cp = 1; c = rho;
	gamma = 1;
	while c>tol
		m=m+1;
		d = alpha*(1+gamma);
		gamma = d/(1-d);
		cn = (1+gamma)*rho*c - gamma*cp;
		cp = c; c=cn;
	end
end
