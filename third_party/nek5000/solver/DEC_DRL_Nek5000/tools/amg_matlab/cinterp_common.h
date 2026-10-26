typedef struct {
  mwSize m,n,nzmax;
  mwIndex *ir, *jc;
  double *pr;
} sp_mat;

typedef struct {
  mwSize m,n;
  const mwIndex *ir, *jc;
  const double *pr;
} sp_mat_c;

typedef struct {
  mwSize m,n,nzmax;
  mwIndex *ir, *jc;
} sp_log_mat;

typedef struct {
  mwSize m,n;
  const mwIndex *ir, *jc;
} sp_log_mat_c;

#ifndef DEBUG_LEVEL
#  define DEBUG_LEVEL 0
#endif

#if DEBUG_LEVEL > 0
  static void *mem_alloc(size_t n)
  { 
    void *ptr;
    mexPrintf("Allocating %u bytes -> ",n);
    ptr = mxMalloc(n);
    mexPrintf("%p\n",ptr);
    return ptr;
  }
  static void *mem_realloc(void *p, size_t n)
  {
    void *ptr;
    mexPrintf("Reallocating %p to %u bytes -> ",p,n);
    ptr = mxRealloc(p,n);
    mexPrintf("%p\n",ptr);
    return ptr;
  }
  static void mem_free(void *p)
  {
    mexPrintf("Freeing %p\n",p);
    mxFree(p);
  }
#else
  static void *mem_alloc(size_t n)
  {
    void *ptr = mxMalloc(n);
    if(!ptr) mexPrintf("Failed to allocate %lu bytes\n",(unsigned long)n);
    return ptr;
  }
  static void *mem_realloc(void *p, size_t n) { return mxRealloc(p,n); }
  static void mem_free(void *p) { mxFree(p); }
#endif

/* Upper triangular transpose matrix vector product
   y[0] = U[0] * x[0]
   y[1] = U[1] * x[0] + U[2] * x[1]
   y[2] = U[3] * x[0] + U[4] * x[1] + U[5] * x[2]
   ... */
static void mv_utt(double *y, mwSize n, const double *U, const double *x)
{
  double *ye = y+n; mwIndex i=1;
  for(;y!=ye;++y,++i) {
    double v=0;
    const double *xp=x, *xe=x+i;
    for(;xp!=xe;++xp) v += (*U++) * (*xp);
    *y=v;
  }
}

/* Upper triangular matrix vector product
   y[0] = U[0] * x[0] + U[1] * x[1] + U[3] * x[2] + ...
   y[1] =               U[2] * x[1] + U[4] * x[2] + ...
   y[2] =                             U[5] * x[2] + ...
   ... */
static void mv_ut(double *y, mwSize n, const double *U, const double *x)
{
  mwIndex i,j;
  for(j=0;j<n;++j) {
    y[j]=0;
    for(i=0;i<=j;++i) y[i] += (*U++) * x[j];
  }
}

/*--------------------------------------------------------------------------
   sparse restriction
   
   y := R * x
   
   the sparse vector x is restricted to y
   R is indicated by map_to_y
   map_to_y[i] == k   <->    e_k^t R == e_i^t I
   map_to_y[i] == -1  <->    row i of I not present in R
--------------------------------------------------------------------------*/
static void sp_restrict_unsorted(double *y, mwSize yn, const mwIndex *map_to_y,
                                 mwSize xn, const mwIndex *xi, const double *x)
{
  const mwIndex *xe = xi+xn; mwIndex i;
  for(i=0;i<yn;++i) y[i]=0;
  for(;xi!=xe;++xi,++x) {
    mwIndex i = map_to_y[*xi];
    if(i>=0) y[i]=*x;
  }
}

/*--------------------------------------------------------------------------
   sparse restriction
   
   y := R * x
   
   the sparse vector x is restricted to y
   Ri[k] == i   <->   e_k^t R == e_i^t I
   Ri must be sorted
--------------------------------------------------------------------------*/
static void sp_restrict_sorted(double *y, mwSize Rn, const mwIndex *Ri,
                               mwSize xn, const mwIndex *xi, const double *x)
{
  const mwIndex *xe = xi+xn;
  double *ye = y+Rn;
  mwIndex iy;
  if(y==ye) return; iy = *Ri;
  for(;xi!=xe;++xi,++x) {
    mwIndex ix = *xi;
    while(iy<ix) { *y++ = 0; if(y==ye) return; iy = *(++Ri); }
    if(iy==ix) { *y++ = *x; if(y==ye) return; iy = *(++Ri); }
  }
  while(y!=ye) *y++ = 0;
}
