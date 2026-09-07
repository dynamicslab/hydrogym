function A = toy_a

  global conn vert

  [ne,g] = size(conn);
  [nv,g] = size(vert);

  A = sparse([],[],[],nv,nv,9*nv);
	%S = sparse(nv,nv);
	for e = 1:ne
	  v = conn(e,:);
	  A(v,v) = A(v,v) + stiff(e);
		%S(v,v) = or(1,S(v,v));
  end
	
end

