function x = pcg(Mi,A,b,x0,tol)
	x = x0;
	r = b - A*x;
	z = Mi*r;
	rho = r'*z;
	rho_0 = rho;
	k = 0;
	tol=tol*tol;
	while rho > tol*rho_0 & k<100
		k = k+1;
		if k==1
			p = z;
		else
			beta = rho / rho_old;
			p = z + beta * p;
		end
		w = A*p;
		alpha = rho / (p'*w);
		x = x + alpha*p;
		r = r - alpha*w;
		z = Mi*r;
		rho_old = rho; rho = r'*z;
		sqrt(rho/rho_0);
	end
	fprintf(1, ' %d iterations.\n', k);
end
