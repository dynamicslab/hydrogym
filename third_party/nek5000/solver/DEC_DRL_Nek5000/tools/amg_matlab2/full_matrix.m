function A = full_matrix(f,n)
%FULL_MATRIX Matrix representation of linear function f on n-vectors
%   A(:,i) = f(I(:,i))

A=eye(n);
for i=1:n
    A(:,i) = f(A(:,i));
end

end

