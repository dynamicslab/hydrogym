function lambda = lanczos(A)
	n = length(A(:,1));
	r = rand(n,1);
	beta = norm(r,2);
	k = 0; qk = 0*r;
	a = []; b=[];
	if norm(A-speye(n),'fro') < 0.00000000001
		l = [ 1 1]; y = [0 0]; k=2;
		change = 0;
	else
  	change = 1;
	end
	if n==1
	  l = [A(1,1) A(1,1)]; y=[0 0]; k=2;
	  change = 0;
	end
	while k<300 & (change > 0.00001 | y(1) > 0.001 | y(k) > 0.001)
		qkm1 = qk; qk = r/beta; k=k+1;
		Aqk = A*qk;
		alpha = qk'*Aqk;
		a = [a; alpha];
		r = Aqk - alpha*qk - beta*qkm1;
		if k==1
			l = [ alpha ];
			y = [ 1 ];
		else
		  l1 = l(1); lk = l(k-1);
			[l y] = tdeig([0; l; 0],[alpha; beta*y],k-1);
			change = abs(l1-l(1))+abs(lk-l(k));
			%fprintf(1,'Lanczos %d: [%g, %g]\n',k,l(1),l(k));
     	%T=full(spdiags([a [b; 0] [0;b]],[0 -1 1],k,k));
     	%lambda = sort(eig(T));
			%fprintf(1,' [%g, %g]\n',lambda(1),lambda(k));
		end
		beta = norm(r,2);
		b = [b; beta];
		if beta==0 break; end;
	end
	lambda = l(find(y<0.01));
	%T=full(spdiags([a b [0;b(1:k-1)]],[0 -1 1],k,k));
	%lambda = sort(eig(T));
	%[lambda l y]
	%lambda = [lambda(1);lambda(k)];
end

