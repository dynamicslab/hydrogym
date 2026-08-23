function [ V d ] = symeig( A, B )
%SYMEIG eigenvalues, orthonormal eigenvectors for Hermitian pencil
%   A * V = B * V * spdiag(d), V' * B * V = I
A = full((A+A')/2);
if nargin<2
    [V D] = eig(A);
    d = diag(D);
    V = V * spdiag(1./sqrt(sum(conj(V).*V)));
else
    B = full((B+B')/2);
    [V D] = eig(A,B);
    d = diag(D);
    V = V * spdiag(1./sqrt(sum(conj(V).*(B*V))));
end

end

