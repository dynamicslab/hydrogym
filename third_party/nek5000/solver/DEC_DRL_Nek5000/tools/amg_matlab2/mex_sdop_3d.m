function mex_sdop_3d(wind)

if nargin<1
    wind = '{2*y*(1-x*x),-2*x*(1-y*y),.05}';
    %wind = '{-1,0}';
end

mex('-largeArrayDims', ['-DWIND=\"' wind '\"'], ...
    '-DUSE_NAIVE_BLAS', ... %-DUSE_CBLAS -lcblas ...
    'sd/sdop_3d.c', 'sd/fail.c', 'sd/poly.c', 'sd/sort.c', 'sd/sarray_sort.c', 'sd/tensor.c');

end
