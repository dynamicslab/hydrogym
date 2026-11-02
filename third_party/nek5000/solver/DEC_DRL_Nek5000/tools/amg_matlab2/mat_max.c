#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"
#include "mex_helper.h"

static void mat_max(double *y, const sp_mat_c *A, const mxLogical *f,
                    const double *x, double tol)
{
  mwIndex j, m = A->m, n = A->n;
  for(j=0;j<m;++j) y[j] = -DBL_MAX;
  for(j=0;j<n;++j) {
    double xj = x[j];
    mwIndex k, kb = A->jc[j], ke = A->jc[j+1];
    double Amax = 0;
    for(k=kb;k<ke;++k)
      if(f[A->ir[k]] && fabs(A->pr[k])>Amax)
        Amax=fabs(A->pr[k]);
    Amax *= tol;
    for(k=kb;k<ke;++k) {
      mwIndex i = A->ir[k];
      if(!f[i] || fabs(A->pr[k]) < Amax) continue;
      if(xj>y[i]) y[i]=xj;
    }
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  sp_mat_c A;
  vec_log_c f;
  vec_c x;
  double tol = 0.1;
  if(nrhs<3) { mexWarnMsgTxt("Usage: mat_max(A, f, x [, tol = 0.1])"); return; }
  if(!load_sp_mat_c(&A,prhs[0])) {
    mexWarnMsgTxt("First input not a sparse, double, real matrix.");
    return;
  }
  if(!load_vec_log_c(&f,prhs[1])) {
    mexWarnMsgTxt("Second input not a full, logical vector.");
    return;
  }
  if(!load_vec_c(&x,prhs[2])) {
    mexWarnMsgTxt("Third input not a full, double, real vector.");
    return;
  }
  if(nrhs==4) {
    if(!load_scalar(&tol,prhs[3])) {
      mexWarnMsgTxt("Fourth input not a real scalar.");
      return;
    }
  }
  if(A.n != x.n) {
    mexWarnMsgTxt("Dimensions disagree.");
    return;
  }
  plhs[0] = mxCreateDoubleMatrix(A.m,1,mxREAL);
  mat_max(mxGetPr(plhs[0]),&A,f.v,x.pr,tol);
}
