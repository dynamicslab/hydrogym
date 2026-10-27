function x = iterate(A,B,b, m)
  A = make_fun(A);
  B = make_fun(B);
  x = zeros(size(b));
  for i=1:m
    x = x + B(b-A(x));
  end
end
