function M = amg_cycle(data,m,smoother,lvl)

if nargin<2; m=2; end
if nargin<3; smoother='plain'; end
if nargin<4; lvl=1; end
  
if isscalar(m); m=[m m]; end
pre_opts.maxsteps = m(1);
pst_opts.maxsteps = m(2);

A = cell(size(data.A));
Af = cell(size(data.Af));
Bf = cell(size(data.Bf));
for l=1:length(data.A );  A{l} = @(x) data.A{l} *x; end
for l=1:length(data.Af); Af{l} = @(x) data.Af{l}*x; end
for l=1:length(data.Bf); Bf{l} = @(x) data.Bf{l}*x; end
W = data.W;
    
switch smoother
    case 'plain'
        smooth1 = @(l,r) iterate(Af{l},Bf{l},r,m(1));
        smooth2 = @(l,r) iterate(Af{l},Bf{l},r,m(2));
    case 'gmres'
        smooth1 = @(l,r) gmres(Af{l},r,[],Bf{l},data.bfabs{l},pre_opts);
        smooth2 = @(l,r) gmres(Af{l},r,[],Bf{l},data.bfabs{l},pst_opts);
end

function x = apply(l,b)
    n = data.n(l);
    if n==0
        x = zeros(0,1);
    elseif n==1
        x = full(1/A{l}(1))*b;
    else
        C = data.C{l}; F = data.F{l};
        x = zeros(n,1);
        x(F) = smooth1(l,b(F));
        r = b - A{l}(x);
        r(C) = r(C) + (r(F)'*W{l})';
        x(C) = apply(l+1,r(C));
        x(F) = x(F) + W{l}*x(C);
        r = b - A{l}(x);
        x(F) = x(F) + smooth2(l,r(F));
    end
end

M = @(b) apply(lvl,b);

end

