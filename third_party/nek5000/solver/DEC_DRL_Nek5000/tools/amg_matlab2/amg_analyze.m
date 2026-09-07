function amg_analyze( data, f, logfile )
%AMG_ANALYZE Summary of this function goes here
%   Detailed explanation goes here

fid=[];
if nargin>2; fid=fopen(logfile,'wt'); end

tee(fid,'Level      n  n/nc       nnz n/nnz\tnnz(Wr)\tnnz(Ws)\tWr/nc\tWs/nc\tWr/nf\tWs/nf\n');

for l=1:length(data.n)
    tee(fid,'%5d',l);
    tee(fid,'%7d',data.n(l));
    if l<length(data.n)
        tee(fid,'  %4.2g',data.n(l+1)/data.n(l));
    else
        tee(fid,'      ');
    end
    tee(fid,'%10d',data.nnz(l));
    tee(fid,' %g',data.nnz(l)/data.n(l));
    if l>length(data.Wr); tee(fid,'\n'); break; end
    tee(fid,'\t%d',nnz(data.Wr{l}));
    tee(fid,'\t%d',nnz(data.Ws{l}));
    tee(fid,'\t%g',nnz(data.Wr{l})/data.n(l+1));
    tee(fid,'\t%g',nnz(data.Ws{l})/data.n(l+1));
    tee(fid,'\t%g',nnz(data.Wr{l})/(data.n(l)-data.n(l+1)));
    tee(fid,'\t%g',nnz(data.Ws{l})/(data.n(l)-data.n(l+1)));
    tee(fid,'\n');
end

for l=1:length(data.n)
    analysis{l} = amg_analyze_level(data,l,f,fid);
    if l>length(data.Wr); break; end
    f = f(data.C{l}) + data.Ws{l}'*f(~data.C{l});
end

for l=1:length(analysis)
    if data.n(l)<=1; break; end
    tee(fid,'Level %d convergence histories:\n',l);
    t = analysis{l};
    [n,~]=size(t.cases);
    for i=1:n
        tee(fid,'  Smoother (%d,%d)   Plain/GMRES/GMRES^2\n',t.cases(i,1),t.cases(i,2));
        len = max(max(length(t.hist{i}), length(t.hist_gmres{i})), length(t.hist_gmres2{i}));
        for j=1:len
            tee(fid,'    ');
            if length(t.hist{i})>=j;        tee(fid,'%g'  ,t.hist{i}(j));        else tee(fid,  '      '); end
            if length(t.hist_gmres{i})>=j;  tee(fid,'\t%g',t.hist_gmres{i}(j));  else tee(fid,'\t      '); end
            if length(t.hist_gmres2{i})>=j; tee(fid,'\t%g',t.hist_gmres2{i}(j)); else tee(fid,'\t      '); end
            tee(fid,'\n');
        end
    end
end

if ~isempty(fid); fclose(fid); end

end

