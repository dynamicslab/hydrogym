function S = sym_sparsify(A,tol)
  S = sym_sparsify_U(sparse(A),tol);
	S = S + S' + speye(length(A(:,1)));
end
