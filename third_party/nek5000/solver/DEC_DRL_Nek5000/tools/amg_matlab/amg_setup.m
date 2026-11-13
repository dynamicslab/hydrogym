function data = amg_setup(A,nv,tolc,tol,stol,wtol)
  %gamma ~ tol
	%rho_f ~ gamma / (1 + gamma)
	%rho_f * (1-gamma^2) + gamma^2 ~ gamma
	%rho_f = tol / (1 + tol); 
	gamma = sqrt(1-sqrt(1-tol));
	rho_f = gamma*gamma;
	fprintf(1,'gamma = %g, rho_f = %g\n',gamma,rho_f);
	level = 1;
	data.tolc = tolc;
	data.gamma = gamma;
	data.n = zeros(0,1);
	data.nnz = zeros(0,1);
	data.nnzff = zeros(0,1);
	data.nnzfp = zeros(0,1);
	data.m = zeros(0,1);
	data.rho = zeros(0,1);
	id = [1:length(A(:,1))];
	u = nv;
	while 1
		[m,n] = size(A);
		if(m==1 & n==1)	data.A{level} = A; end;
		data.id{level} = id;
		data.n = [data.n; n];
		data.nnz = [data.nnz; nnz(A)];
		fprintf(1,'Level %d, dim(A) = %d\n', level, data.n(level));
		if data.n(level) <= 1; break; end

		%[C F] = coarsefast(A);
		[C F] = coarse_par(A,tolc);
		data.C{level} = C; data.F{level} = F;
		data.C_id{level} = id(C); data.F_id{level} = id(F);
		id = data.C_id{level};
		Aff = sparse(A(F,F)); Afc = sparse(A(F,C)); Acc = sparse(A(C,C));
		u = u(C);

		fprintf(1,'Computing diagonal smoother ...');
		%D = diag(sparse(1./diag(Aff)));
		D = diag(sparse(sai0(Aff)));
		Dh = diag(sparse(sqrt(diag(D))));
		fprintf(1,' done.\n');
		nf = length(Aff(:,1));
		if(nf>=2)
			fprintf(1,'Running Lanczos ...');
			lambda = lanczos(Dh*Aff*Dh); k=length(lambda);
			fprintf(1,' [%g, %g], rho = ',lambda(1),lambda(k));
			a = lambda(1); b = lambda(k);
			D = (2/(a+b))*D; rho = (b-a)/(b+a);
			data.D{level} = D;
			data.rho = [data.rho; rho];
			fprintf(1,'%g\n',rho);
			[m c]=chebsim(rho,rho_f);
			fprintf(1,'Chebyshev smoother iterations: %d\tContraction: %g\n',m,c);
			data.m=[data.m; m];
			
			gap = rho_f-c;
			targ = c + .5*(rho_f-c);
			fprintf(1,'Sparsifying A_{ff}: ');
			Sff = sym_sparsify(Dh*Aff*Dh,(1-rho)*(.5*gap)/(2+.5*gap));
			data.Aff{level} = Aff .* Sff;
			%data.Aff{level} = Aff;
			fprintf(1,'compression = %g\n',nnz(data.Aff{level})/nnz(Aff));
		else
			gap = 0;
			data.D{level} = D;
			data.rho = [data.rho; 0];
			data.m = [data.m; 1];
			data.Aff{level} = Aff;
		end
		data.nnzff = [data.nnzff; nnz(data.Aff{level})];
		
		%W = intp_fast(Aff, -Afc, u, tol);
		W = intp_new(A,C,F,u,gamma,wtol);
		data.Wt{level} = W';
		AfP = Aff*W+Afc;
		
		fprintf(1,'Sparsifying R_f A P: ');
		SAfP = sparsify(diag(sparse(sqrt(diag(D))))*AfP, ...
		         tol * sqrt(1-data.rho(level))*.5*gap);
		data.AfPt{level} = (SAfP .* AfP)';
		%data.AfPt{level} = AfP';
		fprintf(1,'compression = %g\n',nnz(data.AfPt{level})/nnz(AfP));
		data.nnzfp = [data.nnzfp; nnz(data.AfPt{level})];
		
		A = simple_sparsify(W'*AfP + Afc'*W + Acc,stol,u);
		level = level + 1;
	end
end
