function Bmg = amg_full(data,level,ns)
	if data.n(level)==0
		Bmg = zeros(0,0);
	elseif data.n(level)==1
		if ns
			Bmg = zeros(1,1);
		else
			Bmg = full(1/data.A{level});
		end
	else
		A = data.A{level};
		W = data.Wt{level}';
		F = data.F{level};
		C = data.C{level};
		%B = data.B{level};
		%B = chebfull(A(F,F),data.D{level},data.rho{level},data.m{level});
		B = chebfull(data.Aff{level},data.D{level},data.rho(level),data.m(level));
		[nf nc] = size(W);
		P = zeros(nf+nc,nc);
		P(C,:) = speye(nc);
		P(F,:) = W;
		Bmg = zeros(nf+nc,nf+nc);
		%Bc = P * pinv(full(data.A{level+1})) * P';
		%Bmg = Bc;
		%Bmg(F,F) = Bmg(F,F) + B;
		%Bmg(:,F) = Bmg(:,F) - Bc*A(:,F)*B;

		%Bmg(C,:) = amg_full(data,level+1,ns)*P';
		%Bmg(F,F) = B;
		%Bmg(F,:) = Bmg(F,:) - B*A(F,C)*Bmg(C,:);
		
		Bc = amg_full(data,level+1,ns)*P';
		Bmg = P*Bc;
		Bmg(F,F) = Bmg(F,F) + B;
		%Bmg(F,:) = Bmg(F,:) - B*A(F,:)*Bc;
		%Bmg(F,:) = Bmg(F,:) - B*data.Afa{level}*Bc;
		Bmg(F,:) = Bmg(F,:) - B*data.AfPt{level}'*Bc;
	end
end
