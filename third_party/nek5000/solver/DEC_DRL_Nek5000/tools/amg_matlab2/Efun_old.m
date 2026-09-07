function y = Efun(x)
  global data m
  l=find(data.n==length(x));
  y = x - amg_apply(data,m,l,data.A{l}*x);
  fprintf(1,'.');
end
