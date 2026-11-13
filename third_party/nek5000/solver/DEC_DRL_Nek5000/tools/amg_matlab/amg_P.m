function P = amg_P(data,i)
	F = data.F{i};
	C = data.C{i};
	nf = length(F);
	nc = length(C);
	P = sparse(nf+nc,nc);
	P(C,:) = speye(nc);
	P(F,:) = data.Wt{i}';
end
