function As = simple_sparsify(A,tol,u)
	D = diag(sparse(sqrt(1./diag(A))));
	S = abs(D*A*D)>tol;
	S = S | S';
	As = S .* A;
	As = As + diag(sparse((A*u - As*u) ./ u));
	As = As + diag(sparse((A*u - As*u) ./ u));
	fprintf(1,'simple_sparsify: nnzs %d/%d (%g)\n', ...
	        nnz(As),nnz(A),nnz(As)/nnz(A));
end
