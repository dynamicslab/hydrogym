function uf = intpc(d,u)
  global data
  if d=='r'; W = data.Wr; elseif d=='s'; W = data.Ws; end
  l=find(data.n==length(u));
  uf = zeros(data.n(1),1);
  uf(data.id{l}) = u;
  %plotint(uf);
  while l>1
    l=l-1;
    C = data.id{l+1};
    F = xor(data.id{l},C);
    uf(F) = W{l}*uf(C);
    %pause; plotint(uf);
  end
end
