function x = amg_apply(data,level,ns,r)
	if data.n(level)==0
		x = zeros(0,1);
	elseif data.n(level)==1
		if ns
			x = zeros(1,1);
		else
			x = full(1/data.A{level})*r;
		end
	else
		C = data.C{level}; F = ~C;
		x = zeros(size(r));
		r(C) = r(C) + data.Wt{level}*r(F);
		x(C) = amg_apply(data,level+1,ns,r(C));
		x(F) = (x(C)'*data.Wt{level})';
		r(F) = r(F) - (x(C)'*data.AfPt{level})';
		x(F) = x(F) + cheb_apply(data.Af{level},data.D{level},data.rho(level),data.m(level),r(F));
	end
end
