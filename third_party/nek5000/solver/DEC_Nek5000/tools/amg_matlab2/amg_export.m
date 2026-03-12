function amg_export(data,id,nullspace)
	nl = length(data.n);
	n = length(id);
	lvl = ones(size(id));
	dvec = zeros(n,1);
    for i=1:nl; lvl(data.id{i})=i; end
    for i=1:nl-1
        dvec(xor(data.id{i},data.id{i+1}),1) = full(diag(data.D{i}));
    end
	if nullspace==0; dvec(data.id{nl}) = 1/full(data.A{nl}); end;

    for i=1:nl-1
        C_id{i}=id(data.id{i+1});
        F_id{i}=id(xor(data.id{i},data.id{i+1}));
        %lid{i}=id(data.id{i});
    end
    %lid{nl}=id(data.id{nl});
	W_len=savemats(lvl,C_id,data.Wt,'amg_W.dat');
	AfP_len=savemats(lvl,C_id,data.AfPt,'amg_AfP.dat');
	Aff_len=savemats(lvl,F_id,data.Af,'amg_Aff.dat');
	savevec([nl; data.m; data.rho; length(id); reshape( ...
        [id'; lvl'; W_len'; AfP_len'; Aff_len'; dvec';], ...
        [],1)],'amg.dat');
end
