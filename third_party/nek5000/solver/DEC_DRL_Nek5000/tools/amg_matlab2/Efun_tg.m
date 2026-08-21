function y = Efun_tg(x)
  global data A m
  y = x - amg_apply_tg(data,m,1,A*x);
  fprintf(1,'.');
end
