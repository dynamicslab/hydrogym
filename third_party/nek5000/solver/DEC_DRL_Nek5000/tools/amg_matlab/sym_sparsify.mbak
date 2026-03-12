function S = sym_sparsify(A,tol)
	n = length(A(:,1));
	S = speye(n);
	while 1
		%spy(S); pause;
		E = abs(A - (A .* S));
		r = find(sum(E,2)>tol);
		if length(r)==0 break; end;
		[y,i] = max(E(r,:),[],2);
		sp = sparse(r,i,1+0*r,n,n);
		%spy(sp); pause;
		S = S | sp | sp';
	end
end
