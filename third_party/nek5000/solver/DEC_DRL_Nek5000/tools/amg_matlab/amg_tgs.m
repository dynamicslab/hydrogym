function amg_tgs(data,level,ns)
	C = data.C{level};
	F = data.F{level};
	A = data.A{level};
	Aff = data.Aff{level};
	W = data.Wt{level}';
	%B = data.B{level};
	%B = chebfull(A(F,F),data.D{level},data.rho{level},data.m{level});
	B = chebfull(Aff,data.D{level},data.rho(level),data.m(level));
	Ac = data.A{level+1};
	Aci = pinv(full(Ac));
	Ahfc = A(F,F)*W + A(F,C);
	Sf = A(F,F) - Ahfc * Aci * Ahfc';
	sb = feig(B*A(F,F));
	sg = feig(A(F,F)\Sf);
	st = feig(B*Sf);
	x = [1:length(F)];
	plot(x,abs(1-sb),'x',x,abs(1-sg),'x',x,abs(1-st),'o');
	
	n = length(F)+length(C);
	I = speye(n);
	P = I(:,C);
	P(F,:) = W;
	Bf = sparse(n,n); Bf(F,F)=B;
	Bc = P*Aci*P';
	%Gc = I - P*Aci*P'*A;
	%Gb = I - Bf*A;
	%Gbs = I - Bfs*A;
	Bmg = Bc + Bf - Bc*A*Bf;
	%norm(Bmg-amg_full(data,level),'fro')
	x = [1:n];
	pause; plot(x,abs(1-feig(Bmg*A)),'x', ...
	            x,abs(1-feig(amg_full(data,level,ns)*A)),'o');
end
