function mex_sdop_2d(wind)

if nargin<1
    wind = '{2*y*(1-x*x),-2*x*(1-y*y)}';
    %wind = '{-1,0}';
end

mex('-largeArrayDims', ['-DWIND=\"' wind '\"'], ...
    '-DUSE_NAIVE_BLAS', ... %-DUSE_CBLAS -lcblas ...
    'sd/sdop_2d.c', 'sd/fail.c', 'sd/poly.c', 'sd/sort.c', 'sd/sarray_sort.c', 'sd/tensor.c');

end
