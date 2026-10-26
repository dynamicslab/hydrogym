function S = sparsify(A,tol)
	[m n] = size(A);
	S = sparse(m,n);
	while 1
		E = abs(A - (A .* S));
		r = find(sum(E,2)>tol);
		if length(r)==0 break; end;
		[y,i] = max(E(r,:),[],2);
		sp = sparse(r,i,1+0*r,m,n);
		S = S | sp;
	end
end
