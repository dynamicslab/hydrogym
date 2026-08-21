function [C,F] = coarsefast(A)
	fprintf(1,'Computing S = I - D^{-1/2} A D^{-1/2} ...');
	D = diag(sparse(1./sqrt(diag(A))));
	SC = abs(D*A*D);
	SC = SC - diag(diag(SC));
	fprintf(1,' done, nnz(S) = %d.\n',nnz(SC));
	if nnz(SC) < 3000000
		fprintf(1,'Computing S^2 ...');
		SC2 = SC * SC;
		fprintf(1,' done, nnz(S^2) = %d.\n',nnz(SC2));
	else
		SC2 = 0*SC;
	end
	fprintf(1,'Running coarsening heuristic (C code) ...');
	C = ccoarse(SC,SC2);
	fprintf(1,' done, coarsening ratio = %g.\n',length(C)/length(A(:,1)));
	C = sort(C');
	F = setdiff([1:length(A(:,1))],C);
end
