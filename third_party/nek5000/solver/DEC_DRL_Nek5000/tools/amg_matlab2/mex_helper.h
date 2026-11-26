typedef struct {
  mwSize m,n;
  const mwIndex *ir, *jc;
  const double *pr;
} sp_mat_c;

typedef struct {
  mwSize m,n;
  const double *pr;
} mat_c;

typedef struct {
  mwSize n;
  const double *pr;
} vec_c;

typedef struct {
  mwSize n;
  const mxLogical *v;
} vec_log_c;

static int load_sp_mat_log_c(sp_mat_c *mat, const mxArray *arr)
{
  if(!mxIsSparse(arr)) return 0;
  mat->m = mxGetM(arr), mat->n = mxGetN(arr);
  mat->ir = mxGetIr(arr), mat->jc = mxGetJc(arr);
  mat->pr = 0;
  return 1;
}

static int load_sp_mat_c(sp_mat_c *mat, const mxArray *arr)
{
  if(!load_sp_mat_log_c(mat,arr)) return 0;
  if(!mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  mat->pr = mxGetPr(arr);
  return 1;
}

static int load_mat_c(mat_c *mat, const mxArray *arr)
{
  mwSize m,n;
  if(mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  mat->m = mxGetM(arr), mat->n = mxGetN(arr);
  mat->pr = mxGetPr(arr);
  return 1;
}

static int load_vec_c(vec_c *vec, const mxArray *arr)
{
  mwSize m,n;
  if(mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  m = mxGetM(arr), n = mxGetN(arr);
  if(m!=1 && n!=1) return 0;
  vec->n = m+n-1;
  vec->pr = mxGetPr(arr);
  return 1;
}

static int load_vec_log_c(vec_log_c *vec, const mxArray *arr)
{
  mwSize m,n;
  if(mxIsSparse(arr) || !mxIsLogical(arr)) return 0;
  m = mxGetM(arr), n = mxGetN(arr);
  if(m!=1 && n!=1) return 0;
  vec->n = m+n-1;
  vec->v = mxGetLogicals(arr);
  return 1;
}

static int load_scalar(double *s, const mxArray *arr)
{
  mwSize m,n;
  if(mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  m = mxGetM(arr), n = mxGetN(arr);
  if(m!=1 || n!=1) return 0;
  *s = *mxGetPr(arr);
  return 1;
}
