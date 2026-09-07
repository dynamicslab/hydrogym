function tee( fid, varargin )
%TEE fprintf wrapper that also prints to stdout

fprintf(1,varargin{:});
if ~isempty(fid); fprintf(fid,varargin{:}); end

end

