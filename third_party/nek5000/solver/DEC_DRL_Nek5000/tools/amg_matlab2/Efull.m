function E = Efull
  global data A m
  n=data.n(1);
  E = eye(n);
  for i=1:n
    E(:,i) = Efun(E(:,i));
  end
  fprintf(1,'\n');
end
