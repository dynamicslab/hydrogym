function uf = cbasis(rl,u)
  global data
  if rl=='r'; W = data.Wr; elseif rl=='l'; W = data.Wl; end
  l=find(data.n==length(u));
  uf = zeros(data.n(1),1);
  uf(data.C_id{l-1}) = u;
  while l>1
    l=l-1;
    uf(data.F_id{l}) = W{l}*uf(data.C_id{l});
  end
end
