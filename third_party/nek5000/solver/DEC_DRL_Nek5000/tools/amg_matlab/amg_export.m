function amg_export(data,id,nullspace)
	nl = length(data.n);
	n = length(id);
	lvl = 1+0*id;
	dvec = zeros(n,1);
	for i=1:nl-1
		lvl(data.C_id{i})=i+1;
		dvec(data.F_id{i}) = full(diag(data.D{i}));
  end
	if nullspace==0; dvec(data.id{nl}) = 1/full(data.A{nl}); end;



	for i=1:nl-1
                data.C_id{i}=id(data.C_id{i});
                data.F_id{i}=id(data.F_id{i});
                data.id{i}=id(data.id{i});
  end
        data.id{nl}=id(data.id{nl});
	W_len=savemats(lvl,data.C_id,data.Wt,'amg_W.dat');
	AfP_len=savemats(lvl,data.C_id,data.AfPt,'amg_AfP.dat');
	Aff_len=savemats(lvl,data.F_id,data.Aff,'amg_Aff.dat');
	savevec([nl; data.m; data.rho; length(id); reshape( ...
					  [id'; lvl'; W_len'; AfP_len'; Aff_len'; dvec';], ...
				  [],1)],'amg.dat');
end
