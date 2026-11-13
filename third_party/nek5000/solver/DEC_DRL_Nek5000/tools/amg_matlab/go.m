tol = input('target convergence factor [0.5]: ');
if length(tol) == 0; tol = 0.5; end
fprintf(1,'tol = %g\n',tol);

ctol = input('coarsening aggressiveness [0.9]: ');
if length(ctol) == 0; ctol = 0.9; end
fprintf(1,'ctol = %g\n',ctol);

wtol = input('weights growth parameter [0.5]: ');
if length(wtol) == 0; wtol = 0.5; end
fprintf(1,'wtol = %g\n',wtol);

stol = input('sparsification tolerance [1e-4]: ');
if length(stol) == 0; stol = 1e-4; end
fprintf(1,'stol = %g\n',stol);

global data A nullspace

[id A] = amg_import();
data = amg_setup(A, full(0*A(:,1)+1),ctol, tol, stol, wtol);
Abottom = full(data.A{length(data.n)});
fprintf(1,'A_%d = %g (',length(data.n),Abottom);
if Abottom<1e-9
  nullspace = 1;
else
	nullspace = 0;
	fprintf(1,'no ');
end
fprintf(1,'nullspace)\n');
amg_export(data,id,nullspace);
fprintf(1,'Done!\n\nComputing largest eigenvalues of iteration matrix:\n');
lambda = eigs(@Efun,length(A(:,1)))
if nullspace>0
  rho = abs(lambda(2));
else
  rho = abs(lambda(1));
end
fprintf(1,'Error contraction factor: %g\n', rho);

