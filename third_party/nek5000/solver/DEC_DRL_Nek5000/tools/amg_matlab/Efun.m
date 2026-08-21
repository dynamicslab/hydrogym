function y = Efun(x)
	global data nullspace A
	y = x - amg_apply(data,1,nullspace,A*x);
	fprintf(1,'.');
end
