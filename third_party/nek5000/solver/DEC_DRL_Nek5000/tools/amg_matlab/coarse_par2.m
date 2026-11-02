function [C,F]=coarse_par(A,tol)
	n = length(A(:,1));
	fprintf(1,'Computing S = I - D^{-1/2} A D^{-1/2} ...');
	D = diag(sparse(1./sqrt(diag(A))));
	S = abs(D*A*D);
	S = S - diag(diag(S));
	fprintf(1,' done, nnz(S) = %d.\n',nnz(S));
	vc = sparse(0*S(:,1));
	id = [1:n]';
	fprintf(1,'Running coarsening heuristic ...\n');
	while 1
		nc = ones(n,1)'*vc;
		vf = 1 - vc;
		g = vf .* (S*vf);
		h = diag(sparse(1./(vc + vf.*max(S*diag(sparse(vf)),[],2))))*(g);
		%vplot(vc); pause;
		vtol=h.^(-tol);
		%vplot(vtol); pause;
		%g=h;
		%[mg,mi] = max(g);
		%fprintf(1,'  ratio = %g, n = %d, max Gershgorin radius = %g\n', ...
		%  nc/n, nc, mg);
	  %if(mg<=tol)
		%	if nc==0; vc(mi)=1; end
		%	break;
		%end
		mask = g>vtol;
		fprintf(1,'  ratio = %g, bad points = %g\n', nc/n, sum(mask));
		if sum(mask)==0
			if nc==0; vc(mi)=1; end
			break;
		end
		%ga = mask .* (g + .5*(S*(S*vc)));
		ga = mask .* g;
		m = mxv_max(S,ga);
		mask = mask & (ga-m>=0);
		m = mxv_max(S,mask.*id);
		mask = mask & (id-m>0);
    vc = vc + mask;
	end
	C = find(vc);
	F = setdiff(id,C);
end
