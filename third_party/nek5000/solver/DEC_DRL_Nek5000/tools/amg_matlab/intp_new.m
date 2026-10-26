function W = intp_new(A,C,F,u,tol,wtol)
	nc = length(C);
	nf = length(F);
	Aff = sparse(A(F,F));
	Afc = sparse(A(F,C));
	Acc = sparse(A(C,C));
	
	one = ones(nc,1);

	Dff = diag(sparse(1./sqrt(diag(Aff))));
	W_skel = sparse(nf,nc);
	W = W_skel;

	fprintf(1,'Finding ideal interpolant of near null space ... ');	
	uf = pcg(Dff*Dff,Aff,-Afc*u,zeros(nf,1),1e-13);
	%[min(uf) max(uf)]
	
	fprintf(1,'Determining the coarse basis vector support ...\n');
	
  %v = 0*A(:,1);
	%v(C) = u; v(F) = uf;
	%vplot(v); pause;

	nzf = find(abs(uf)>1e-13);

	Dc = diag(sparse(1./sqrt(diag(Acc))));
	Ra = abs(Dff*Afc*Dc);
	[y j] = max(Ra(nzf,:),[],2);
	
	W_skel = sparse(nzf,j,0*j+1,nf,nc);
	%spy(W_skel); pause;

	W = cinterp(Aff,-Afc,zeros(nc,1),zeros(nf,1),W_skel);
	
	bl = uf - W*u;
	nv = W_skel * (u.*u);
	%plot([1:nf],nv,'x'); pause;
	d = 0*nv;
	d(nzf) = 1 ./ nv(nzf);
	lambda = full(diag(sparse(d))*Aff*bl);
	%plot(lambda); pause;
	
	W = cinterp(Aff,-Afc,u,lambda,W_skel);
	
  %v = 0*A(:,1);
	%v(C) = u; v(F) = W*u;
	%vplot(v); pause;
	
	while 1
		Rd = Aff*W + Afc;
		fprintf(1,'.');
		%Ac = W'*Rd + Afc'*W + Acc;
		%fprintf(1,'.');
		%Dc = diag(sparse(1./sqrt(diag(Ac))));
		Dc = diagmm(W,Rd) + diagmm(Afc,W) + diag(Acc);
		Dc = diag(sparse(1./sqrt(Dc)));
		fprintf(1,'.');
		R = Dff*Rd*Dc;
		fprintf(1,'.');
		%R2 = abs(R'*R);
		R2 = abs(R);
		fprintf(1,'.');

		%c1 = R2 * one;
		%c2 = R2 * c1;
		c1 = R2' * (R2*one);
		fprintf(1,'.');
		c2 = R2' * (R2*c1);
		fprintf(1,'.');
		c = full(sqrt(c2./c1));
		fprintf(1,'.');

		%v = 0*A(:,1);
		%v(C) = sqrt(c2./c1);
		%vplot(v); pause;
		
		Ra = abs(R);
		fprintf(1,'.');
		[y i] = max(Ra,[],1);   % i(j) = row index of largest entry in Ra(:,j)
		fprintf(1,'.');
		p = find(c>(wtol*tol));   % p    = set of deficient cols
                p2 = find(c>tol);
		fprintf(1,'.');
		fprintf(1,'  nnz = %d; %d cols above %g; max = %g\n', ...
			nnz(W_skel),length(p2),tol,max(c));
		if length(p2)==0; break;	end
		
		W_new = sparse(i(p),p,0*p+1,nf,nc);   % W_new := largest entries in deficient cols
		fprintf(1,'.');
		%spy(W_skel&W_new); pause;
		[i j] = find(W_skel & W_new);
		fprintf(1,'.');
		p = union(i,[]); % p := set of rows where largest entry not new
		fprintf(1,'.');
		Ra = Ra .* xor(Ra,W_skel); % mask Ra where W currently has an entry
		fprintf(1,'.');
		[y j] = max(Ra(p,:),[],2);
		fprintf(1,'.');
		q = find(y>1e-13);
		%j = j(find(y>1e-13));
		fprintf(1,'.');
		%length(p)
		%length(j)
		W_new2 = sparse(p(q),j(q),j(q)*0+1,nf,nc);
		fprintf(1,'.');
		%spy(W_new2 | W_skel); pause;
		%spy(W_new & W_new2); pause;
		W_skel = W_skel | W_new | W_new2;
		fprintf(1,'.');
		%spy(W_skel); pause;
		
		W = cinterp(Aff,-Afc,zeros(nc,1),zeros(nf,1),W_skel);
		fprintf(1,'.');
		bl = uf - W*u;
		nv = W_skel * (u.*u);
		%plot([1:nf],nv,'x'); pause;
		d = 0*nv;
		d(nzf) = 1 ./ nv(nzf);
		lambda = full(diag(sparse(d))*Aff*bl);
		fprintf(1,'.');
		%plot(lambda); pause;
	
		W = cinterp(Aff,-Afc,u,lambda,W_skel);
		fprintf(1,'.');
	
	  %v = 0*A(:,1);
		%v(C) = u; v(F) = W*u;
		%vplot(v); pause;

  end

	W = cinterp(Aff,-Afc,zeros(nc,1),zeros(nf,1),W_skel);

	fprintf(1,' done.\nComputing Lagrange multiplier governing matrix skeleton ...');
	T_skel = (+W_skel)*(+W_skel)';
	fprintf(1,' nnz = %d.\n',nnz(T_skel));
	
	Dp = diag(sparse(1./sum(+W_skel,2)));
	fprintf(1,'Computing Lagrange multiplier governing matrix (C code) ...');
	T = cinterp_lmop(Aff,u,W_skel,T_skel);

	fprintf(1,' done.\nCG to obtain Lagrange multipliers ...');
	%d = diag(T*Aff); di = 0*d; di(nzf)=1./d(nzf); D=diag(sparse(di));
	d = diagmm(T,Aff); di = 0*d; di(nzf)=1./d(nzf); D=diag(sparse(di));
	lambda = pcg(.5*(Aff*D+D*Aff),T,uf-W*u,zeros(nf,1),1e-11);

	fprintf(1,'Computing interpolation weights (C code) ...');
	W = cinterp(Aff,-Afc,u,lambda,W_skel);
	fprintf(1,' done.\n');


	%fprintf(1,'Estimated gamma = ');
	%Rd = Aff*W + Afc;
	%Ac = W'*Rd + Afc'*W + Acc;
	%Dc = diag(sparse(1./sqrt(diag(Ac))));
	%R = Dff*Rd*Dc;
	%%R2 = abs(R'*R);
	%R2 = abs(R);

	%%c1 = R2 * one;
	%%c2 = R2 * c1;
	%c1 = R2' * (R2*one);
	%c2 = R2' * (R2*c1);
	%fprintf(1,'%g\n',max(full(sqrt(c2./c1))));

	%fprintf(1,'Actual gamma = ');

	%Z1 = W'*Aff*W + W'*Afc + Afc'*W;
	%Z = pinv(full(Z1+Acc))*full(Z1 + Afc'*(Aff\full(Afc)));
	
  %fprintf(1,'%g\n',max(sqrt(eig(Z))));
	
end
