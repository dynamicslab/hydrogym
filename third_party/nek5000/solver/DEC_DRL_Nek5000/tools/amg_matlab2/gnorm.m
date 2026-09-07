%|| T || _ A,B
function r = gnorm(T,A,B)
  r = sqrt(max(abs(eig(full(T'*A*T),full(B)))));
end
