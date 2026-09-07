function [id A] = amg_import()
	[i_id j_id v] = loadbin;
	id = union(i_id,[]);
	n = length(id);
	perm = zeros(max(id),1);
	perm(id) = [1:n];
	A = sparse(perm(i_id),perm(j_id),v,n,n);
end
