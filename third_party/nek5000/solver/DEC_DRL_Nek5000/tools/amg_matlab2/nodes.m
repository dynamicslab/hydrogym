function x = nodes( n )
%NODES Chebyshev nodes
%   returns n Chebyshev nodes of the second kind (including +/- 1)
x = cos(pi*(1-[0:n-1]/(n-1)));
end
