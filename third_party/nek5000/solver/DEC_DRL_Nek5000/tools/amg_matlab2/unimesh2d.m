function unimesh2d(n,a,b)
	t=[0:1/(n-1):1]';
	mesh2d(a*t,b*t);
end
