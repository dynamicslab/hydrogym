function data = amg_setup(A,methodc,tolc,tol,tols)
    %gamma ~ tol
    %rho_f ~ gamma / (1 + gamma)
	%rho_f * (1-gamma^2) + gamma^2 ~ gamma
	%rho_f = tol / (1 + tol); 
	gamma2 = 1-sqrt(1-tol);
	rho_f = gamma2;
	fprintf(1,'gamma = %g, rho_f = %g\n',sqrt(gamma2),rho_f);
	level = 1;
	data.tolc = tolc;
	data.gamma = sqrt(gamma2);
	data.n = zeros(0,1);
	data.nnz = zeros(0,1);
	data.nnzf = zeros(0,1);
	data.nnzfp = zeros(0,1);
	data.m = zeros(0,1);
	data.rho = zeros(0,1);
    
    [~,n] = size(A);
    u = ones(n,1);
    id = sparse(true(n,1));
    A = sparse(A);
    %w0 = nnz(A)/n;
    if nargin<5; tols=0; end
    while 1
        [~,n] = size(A);
        data.A{level} = A;
        data.id{level} = id;
        data.n = [data.n; n];
        data.nnz = [data.nnz; nnz(A)];
        w = nnz(A)/n;
        fprintf(1,'Level %d, dim(A) = %d, nnz(A)/dim(A) = %g\n', level, n, w);
        if n <= 1; break; end

        %tolc2 = tolc*((w0/max(w,w0))^theta);
        switch lower(methodc)
            case {'mixed','new'}
                C = coarsen2(A,tolc);
            otherwise
                C = coarsen1(A,tolc);
        end
        data.C{level} = C;
        %id(id) = C;
	fid = full(id); fid(id)=full(C); id=sparse(fid);
        F = ~C;
        data.F{level} = F;

		fprintf(1,'Computing diagonal smoother ...');
        Af = A(F,F);
		%D = diag(sparse(1./diag(Aff)));
		D = spdiag(diag(Af)' ./ sum(Af.*conj(Af)));
		Dh = spdiag(sqrt(diag(D)));
		fprintf(1,' done.\n');
		nf = length(Af(:,1));
		if(nf>=2)
			fprintf(1,'Running Lanczos ...');
			lambda = lanczos(Dh*Af*Dh); k=length(lambda);
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
			fprintf(1,'Sparsifying Af: ');
			Sf = sym_sparsify(Dh*Af*Dh,(1-rho)*(.5*gap)/(2+.5*gap));
			data.Af{level} = Af .* Sf;
			%data.Aff{level} = Aff;
			fprintf(1,'compression = %g\n',nnz(data.Af{level})/nnz(Af));
		else
			gap = 0;
			data.D{level} = D;
			data.rho = [data.rho; 0];
			data.m = [data.m; 1];
			data.Af{level} = Af;
		end
		data.nnzf = [data.nnzf; nnz(data.Af{level})];
        
        W = intp.intp(A,C,gamma2,u);
        data.Wt{level} = W';
		AfP = Af*W+A(F,C);
		
		fprintf(1,'Sparsifying R_f A P: ');
		SAfP = sparsify(spdiag(sqrt(diag(D)))*AfP, ...
		         tol * sqrt(1-data.rho(level))*.5*gap);
		data.AfPt{level} = (SAfP .* AfP)';
		%data.AfPt{level} = AfP';
		fprintf(1,'compression = %g\n',nnz(data.AfPt{level})/nnz(AfP));
		data.nnzfp = [data.nnzfp; nnz(data.AfPt{level})];
		
        A = W'*AfP + A(C,F)*W + A(C,C); u = u(C);
        if tols>0 && sum(C)>1; A = simple_sparsify(A,tols,u); end
        level = level + 1;    
    end
end
