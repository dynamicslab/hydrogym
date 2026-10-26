function out = amg_analyze_level(data, l, f, fid)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
n = data.n(l);
if n==0; out = struct([]); return; end
if nargin<4; fid=[]; end
tee(fid,'Analysis level %d (n=%d):\n', l,n);

A = data.A{l};
C = data.C{l}; F = ~C; nc=sum(C); nf=n-nc;
Af = data.Af{l};
Ar = A(F,C); As = A(C,F)'; Ac = A(C,C);
Wr = data.Wr{l};
Ws = data.Ws{l};
Arhat = Af*Wr+Ar; Ashat = Af'*Ws+As;
Achat = Ws'*Arhat + As'*Wr + Ac;
Bf1 = data.Bf{l};

cutoff=4000;
bm = [1:4, 8, 16];
show_plot = usejava('desktop');
cases = [ 1 0;
          0 1;
          2 0;
          0 2;
          1 1;
          3 0;
          0 3;
          4 0;
          0 4;
          2 2;
          8 0;
          0 8;
          4 4 ];

function r = avg_rate(hist)
    hist = hist(hist>20*min(hist));
    rate = hist(2:end) ./ hist(1:end-1);
    k = length(rate);
    rate = rate(max(1,floor(.5*k)):end);
    r = sum(rate)/length(rate);
end

if n <= cutoff
    absA = absf(A); RabsA = chol(absA);
end

tolopts.tol = 1e-4;

out.cases = cases;
[ncases,~] = size(cases);
opts.maxsteps = 0;
for i=1:ncases
    m = cases(i,:);
    M = amg_cycle(data,m,'plain',l);
    Madj = amg_cycle(data,m,'plain',l,'*');
    [~, hist] = gmres(A,f,[],M);
    out.hist_gmres{i} = hist;
    out.rate_gmres(i) = avg_rate(hist);
    if show_plot; semilogy(hist,'o-'); pause(.01); hold on; end;
    [~, hist] = gmres(A,f,[],amg_cycle(data,m,'gmres',l));
    out.hist_gmres2{i} = hist;
    out.rate_gmres2(i) = avg_rate(hist);
    if show_plot; semilogy(hist,'+-'); pause(.01); hold on; end;
    opts.maxsteps = 2*length(out.hist_gmres{i});
    [~, hist] = iterate_hist(A,f,[],M,[],opts);
    out.hist{i} = hist;
    out.rate(i) = avg_rate(hist);
    if show_plot; semilogy(hist,'x-'); pause(.01); hold on; end;
    E    = @(e) e - M(A*e);
    Eadj = @(r) r - (Madj(r)'*A)';
    if n<=cutoff
        out.E_anorm(i) = sqrt(glanczos(n,@(e) Eadj(absA*E(e)),@(f) RabsA \ (f'/RabsA)'));
    end
    tee(fid,'(%d,%d) ',m(1),m(2));
    
    if n<=cutoff; fprintf(1,' ||E||_|A|: %g, ', out.E_anorm(i)); end
    out.spr(i) = abs(eigs(E,n,1,'LM',tolopts));
    tee(fid,'spr: %g  rate: %g, %g (GMRES), %g (GMRES^2)\n', ...
        out.spr(i),out.rate(i),out.rate_gmres(i),out.rate_gmres2(i));
end
if show_plot; hold off; end

if n>cutoff
    %compute angle
    H = (A+A')/2;
    Hdata = amg_setup(H,'mixed',0.7,0,'restricted abs',0.8,1e-4);
    Hi = @(f) gmres(H,f,[],amg_cycle(Hdata,2,'gmres'));
    out.theta = atan(skew_lanczos(n,A-H,Hi));
    clear Hi Hdata
else
    U = absA\full(A);
    out.theta = max(angle(eig(U)));
end
tee(fid,'  angle = %g = %g deg\n', out.theta, out.theta/pi*180);

%compute || I - D_f \ A_f ||_{|D_f|}
Dfi = spdiag(1./sqrt(diag(Af)));
out.coarse_norm = svds(speye(nf) - Dfi*Af*Dfi,1,'L',tolopts);
tee(fid,'  || I - D_f \\ A_f ||_|D_f| = %g\n', out.coarse_norm);

[Xf Xfi theta_f gamma] = absf_iter(Af);
out.theta_f = theta_f; out.gamma_f = gamma;
tee(fid,'  F angle = %g = %g deg\n', out.theta_f, out.theta_f/pi*180);

dchabsi = 1./abs(full(diag(Achat)));
out.arhat_dnorm_approx = sqrt(glanczos(nc,@(u) (Xfi{2}(Arhat*u)'*Arhat)', dchabsi));
out.ashat_dnorm_approx = sqrt(glanczos(nc,@(u) (Xfi{2}(Ashat*u)'*Ashat)', dchabsi));
tee(fid,'  || F_r ||_|Af|,|D^_c| = %g (%g)\n', out.arhat_dnorm_approx, sqrt(gamma(2))*out.arhat_dnorm_approx);
tee(fid,'  || F_s ||_|Af|,|D^_c| = %g (%g)\n', out.ashat_dnorm_approx, sqrt(gamma(2))*out.ashat_dnorm_approx);

Achati    = @(f) gmres(@(x) Achat*x    ,f,[],amg_cycle(data,2,'gmres',l+1,[] ));
Achatiadj = @(f) gmres(@(x) (x'*Achat)',f,[],amg_cycle(data,2,'gmres',l+1,'*'));
Sf    = @(f) Af*f     - Arhat*(Achati(   (f'*Ashat)'));
Sfadj = @(f) (f'*Af)' - Ashat*(Achatiadj((f'*Arhat)'));
dfi = 1./full(diag(Af)); dfiadj = conj(dfi); dfiabs = abs(dfi);
Afi    = @(f) gmres(@(x) Af*x    ,f,[],dfi   ,dfiabs);
Afiadj = @(f) gmres(@(x) (x'*Af)',f,[],dfiadj,dfiabs);
E    = @(f) f - Afi(Sf(f));
Eadj = @(f) f - Sfadj(Afiadj(f));
out.tl_anorm_inf_approx = sqrt(glanczos(nf,@(f) Eadj(Xf{2}(E(f))), Xfi{2}));
tee(fid,'  || I - Af \\ Sf ||_|Af|  = %g (%g) <=? %g (%g)\n', ...
    out.tl_anorm_inf_approx, sqrt(gamma(2))*out.tl_anorm_inf_approx, ...
    out.arhat_dnorm_approx*out.ashat_dnorm_approx, ...
    gamma(2)*out.arhat_dnorm_approx*out.ashat_dnorm_approx);

out.bm = bm;
nbm = length(bm);

if nf<=cutoff
    %compute angle
    %Hf = (Af+Af')/2; Dfi = spdiag(1./diag(Hf));
    %out.theta_f = atan(skew_lanczos(nf,Af-Hf,@(f) pcg(Hf,f,Dfi,1e-16)));
    %fprintf(1,'  F angle = %g = %g deg\n', out.theta_f, out.theta_f/pi*180);
%else
    absAf = absf(Af); Rf = chol(absAf); % Rf'*Rf = |Af|
    %Uf = Rf\(Rf'\full(Af));
    %out.theta_f = max(angle(eig(Uf)));
    %fprintf(1,'  F angle = %g = %g deg\n', out.theta_f, out.theta_f/pi*180);
    for i = 1:nbm
        m = bm(i);
        Bf{i} = full_matrix(@(f) iterate(Af,Bf1,f,m), nf);
        out.smoother_anorm(i) = norm(Rf*full(eye(nf)-Bf{i}*Af)/Rf,2);
        tee(fid,'  || I - Bf_%d Af ||_|Af|   = %g\n', m,out.smoother_anorm(i));
        absBf{i} = absf(Bf{i});
        RBf{i} = chol(absBf{i});
        UBf = RBf{i}\(RBf{i}'\Bf{i});
        out.theta_Bf(i) = max(angle(eig(UBf)));
        tee(fid,'  Bf_%d angle = %g\n', m,out.theta_Bf(i));
        out.smoother_bnorm(i) = norm(RBf{i}*full(eye(nf)-Af*Bf{i})/RBf{i},2);
        tee(fid,'  || I - Af Bf_%d ||_|Bf_%d| = %g\n', m,m,out.smoother_bnorm(i));
        %eps1 = out.smoother_anorm(m);
        %a = norm(Rf*full(Bf{m}*Af)/Rf,2);
        %fprintf(1,'  || Bf_%d Af ||_|Af|   = %g <= %g \n', m,a,1+eps1);
        %eps2 = norm(Rf*full(eye(nf)-inv(Bf{m}*Af))/Rf,2);
        %fprintf(1,'  || I - Af \\ Bf_%d^-1 ||_|Af|   = %g\n', m,eps2);
        %b = norm(Rf*inv(Bf{m}*Af)/Rf,2);
        %fprintf(1,'  || Af \\ Bf_%d^-1 ||_|Af|   = %g <= %g \n', m,b,1+eps2);
        %out.delta(m) = eps2 + b*(a*eps2 + eps1)/(cos(out.theta_f) + cos(out.theta_Bf{m}));
        %out.ratio(m) = sqrt(1+out.delta(m));
        d = abs(eig(absBf{i}*absAf));
        out.norm_ratio_min(i) = min(d);
        out.norm_ratio_max(i) = max(d);
        tee(fid,'  %g <= ||f||_|Bf_%d|^2 / ||f||_|Af^-1|^2 <= %g \n', min(d),m,max(d));
        %out.ratio(m) = sqrt(max(d));
    end
end

if nf<=cutoff && nc<=cutoff
    absAchat = absf(Achat); Rc = chol(absAchat);
    Sc = Ac - As'*(Af\full(Ar)); absSc = absf(Sc); RSc = chol(absSc);
    Hc = (Sc+Sc')/2; RHc = chol(Hc);
    Sf = Af - Arhat*(Achat\full(Ashat'));
    out.arhat_dnorm = norm(Rf'\full(Arhat*spdiag(sqrt(dchabsi))),2);
    out.ashat_dnorm = norm(Rf'\full(Ashat*spdiag(sqrt(dchabsi))),2);
    tee(fid,'  || F_r ||_|Af|,|D^_c| = %g\n', out.arhat_dnorm);
    tee(fid,'  || F_s ||_|Af|,|D^_c| = %g\n', out.ashat_dnorm);
    out.arhat_hnorm = norm(Rf'\full(Arhat)/RHc,2);
    out.ashat_hnorm = norm(Rf'\full(Ashat)/RHc,2);
    tee(fid,'  || F_r ||_|Af|,H~_c = %g\n', out.arhat_hnorm);
    tee(fid,'  || F_s ||_|Af|,H~_c = %g\n', out.ashat_hnorm);
    out.arhat_snorm = norm(Rf'\full(Arhat)/RSc,2);
    out.ashat_snorm = norm(Rf'\full(Ashat)/RSc,2);
    tee(fid,'  || F_r ||_|Af|,|S_c| = %g\n', out.arhat_snorm);
    tee(fid,'  || F_s ||_|Af|,|S_c| = %g\n', out.ashat_snorm);
    out.arhat_anorm = norm(Rf'\full(Arhat)/Rc,2);
    out.ashat_anorm = norm(Rf'\full(Ashat)/Rc,2);
    tee(fid,'  || F_r ||_|Af|,|A^_c| = %g\n', out.arhat_anorm);
    tee(fid,'  || F_s ||_|Af|,|A^_c| = %g\n', out.ashat_anorm);
    out.tl_anorm_inf = norm(Rf*(eye(nf)-Af\Sf)/Rf,2);
    tee(fid,'  || I - Af \\ Sf ||_|Af|  = %g <= %g\n', out.tl_anorm_inf, out.arhat_anorm*out.ashat_anorm);
    out.tl_spr_inf = max(abs(eig(eye(nf)-Af\Sf)));
    tee(fid,'  rho( I - Af \\ Sf )  = %g\n', out.tl_spr_inf);
    %norm(Rf'\full(Arhat)/Rc,2)
    %norm(Rf'\full(Arhat)/RSc,2)
    for i = 1:nbm
        m = bm(i);
        E = eye(nf)-Bf{i}*Sf;
        out.tl_spr(i) = max(abs(eig(E)));
        tee(fid,'  rho ( I - Bf_%d Sf ) = %g\n', m,out.tl_spr(i));
        out.tl_anorm(i) = norm(Rf*E/Rf,2);
        tee(fid,'  || I - Bf_%d Sf ||_|Af|   = %g <= %g (%g) <= %g\n', m,out.tl_anorm(i), ...
            (1+out.smoother_anorm(i))*(1+out.tl_anorm_inf)-1, ...
            1-(1-out.smoother_anorm(i)^2)*(1-out.tl_anorm_inf), ...
            (1+out.smoother_anorm(i))*(1+out.arhat_anorm*out.ashat_anorm)-1);
        out.arhat_bnorm(i) = norm(RBf{i}*full(Arhat)/Rc,2);
        out.ashat_bnorm(i) = norm(RBf{i}*full(Ashat)/Rc,2);
        tee(fid,'  || A^_r ||_|Bf_%d|,|A^_c| = %g <= %g\n', m,out.arhat_bnorm(i),sqrt(out.norm_ratio_max(i))*out.arhat_anorm);
        tee(fid,'  || A^_s ||_|Bf_%d|,|A^_c| = %g <= %g\n', m,out.ashat_bnorm(i),sqrt(out.norm_ratio_max(i))*out.ashat_anorm);
        out.tl_bnorm_inf(i) = norm(RBf{i}*(Af-Sf)*RBf{i}',2);
        tee(fid,'  || Af - Sf ||_|Bf_%d|,|Bf|^-1 = %g <= %g\n', m,out.tl_bnorm_inf(i), out.arhat_bnorm(i)*out.ashat_bnorm(i));
        out.tl_bnorm(i) = norm(RBf{i}'\E*RBf{i}',2);
        tee(fid,'  || I - Sf Bf_%d ||_|Bf_%d| = %g <= %g <= %g\n', m,m,out.tl_bnorm(i), ...
            out.smoother_bnorm(i)+out.tl_bnorm_inf(i), ...
            out.smoother_bnorm(i)+out.arhat_bnorm(i)*out.ashat_bnorm(i));
        %d = eig(Bf{m}*Sf-Af\Sf);
        %min(d)
        %max(d)
        %d = eig(Af\Sf);
        %min(d)
        %max(d)
        %d = eig(Bf{m}*Sf);
        %min(d)
        %max(d)
    end
end

end

