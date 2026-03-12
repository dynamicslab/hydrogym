function M = evalm(f,A)

[m n]=size(A);
M=0*A;

for i=1:m
  for j=1:n
    M(i,j)=f(A(i,j))
  end
end

end
