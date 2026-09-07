function mesh2d(tx,ty)
	global conn vert
	n=length(tx);
	x = repmat(tx,n,1);
	y = reshape(repmat(ty,1,n)',[],1);
	vert = [x y];
	conn = zeros((n-1)*(n-1),4);
	for i = 1:(n-1)
		t = (n-1)*(i-1)+[1:(n-1)];
		conn(t,1) = n*(i-1)+[1:(n-1)];
		conn(t,2) = n*(i-1)+[2:n];
		conn(t,3) = n*i+[1:(n-1)];
		conn(t,4) = n*i+[2:n];
	end
end
