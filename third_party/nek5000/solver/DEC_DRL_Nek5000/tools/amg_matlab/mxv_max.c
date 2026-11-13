#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"

typedef struct {
  mwSize m,n;
  const mwIndex *ir, *jc;
  const double *pr;
} sp_mat_c;

typedef struct {
  mwSize n;
  const double *pr;
} vec_c;

static int load_sp_mat_c(sp_mat_c *mat, const mxArray *arr)
{
  if(!mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  mat->m = mxGetM(arr), mat->n = mxGetN(arr);
  mat->ir = mxGetIr(arr), mat->jc = mxGetJc(arr);
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

static void mxv_max(double *y, const sp_mat_c *A, const double *x, double tol)
{
  mwIndex j, m = A->m, n = A->n;
  for(j=0;j<m;++j) y[j] = -DBL_MAX;
  for(j=0;j<n;++j) {
    double xj = x[j];
    mwIndex k, kb = A->jc[j], ke = A->jc[j+1];
    for(k=kb;k<ke;++k) {
      mwIndex i = A->ir[k];
      if(fabs(A->pr[k])<tol) continue;
      if(xj>y[i]) y[i]=xj;
    }
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double tol = 128*DBL_EPSILON;
  vec_c x;
  sp_mat_c mat;
  if(nrhs<2) { mexWarnMsgTxt("Two inputs required."); return; }
  if(!load_sp_mat_c(&mat,prhs[0])) {
    mexWarnMsgTxt("First input not a full, double, real vector.");
    return;
  }
  if(!load_vec_c(&x,prhs[1])) {
    mexWarnMsgTxt("Second input not a full, double, real vector.");
    return;
  }
  if(mat.n != x.n) {
    mexWarnMsgTxt("Dimensions disagree.");
    return;
  }
  plhs[0] = mxCreateDoubleMatrix(mat.m,1,mxREAL);
  mxv_max(mxGetPr(plhs[0]),&mat,x.pr,tol);
}
