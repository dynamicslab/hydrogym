function chebmesh2d(n)
	t=cos(pi*[1:-1/(n-1):0]');
	mesh2d(t,t);
end
