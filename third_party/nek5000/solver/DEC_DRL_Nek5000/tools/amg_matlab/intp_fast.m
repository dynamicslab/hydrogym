function W=intp_fast(A,B,u,tol)
	D = full(diag(A));
	fprintf(1,'Determing coarse basis function support (C code) ...');
	[W_skel,bf] = cinterp_skel(A,B,D,u,tol);
	fprintf(1,' done.\nComputing Lagrange multiplier governing matrix skeleton ...');
	T_skel = (+W_skel)*(+W_skel)';
	fprintf(1,' nnz = %d.\n',nnz(T_skel));
	Dp = diag(sparse(1./sum(+W_skel,2)));
	fprintf(1,'Computing Lagrange multiplier governing matrix (C code) ...');
	T = cinterp_lmop(A,u,W_skel,T_skel);
	Tp = cinterp_lmop(A,0*u+1,W_skel,T_skel);
  fprintf(1,'done.\nFinding the ideal interpolant of the constant ... ');
	b = B*u; v=pcg(.5*(Tp*Dp+Dp*Tp),A,b,1*b,1e-14);
	%b = B*u; v=pcg(diag(sparse(diag(A))),A,b,1*b,1e-14);
	fprintf(1,'CG to obtain Lagrange multipliers ...');
	d = diag(T*A); di = 1./d; di(find(abs(d)<1e-11))=0; D=diag(sparse(di));
	lambda = pcg(.5*(A*D+D*A),T,v-bf,0*bf,1.0e-10);
	%norm((1-bf)-T*lambda,2)
	fprintf(1,'Computing interpolation weights (C code) ...');
	W = cinterp(A,B,u,lambda,W_skel);
	fprintf(1,' done.\n');
	%fprintf(1,' done.\n||residual||_{inf} was %g.\n', norm(v-sum(W,2),inf));
	%W = diag(sparse(v./sum(W,2)))*W;
end
