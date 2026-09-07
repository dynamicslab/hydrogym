#include "mex.h"
#include "matrix.h"

static void sai0(double *M, mwIndex n, const mwIndex *jc, const mwIndex *ir,
                 const double *A)
{
  mwIndex j;
  for(j=0;j<n;++j) {
    double Aii = 0;
    double x = 0;
    mwIndex k,ke;
    for(k=jc[j],ke=jc[j+1];k!=ke;++k) {
      double Aij = A[k];
      if(ir[k]==j) Aii=Aij;
      x += Aij * Aij;
    }
    M[j] = Aii / x;
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwIndex n;
  if(nrhs!=1) { mexWarnMsgTxt("One input required."); return; }
  if(nlhs!=1) { mexWarnMsgTxt("One output required."); return; }
  if(!mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]))
    { mexWarnMsgTxt("Sparse real dboule input required."); return; }
  n = mxGetN(prhs[0]);
  if(mxGetM(prhs[0])!=n) { mexWarnMsgTxt("Square input required."); return; }
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
  sai0(mxGetPr(plhs[0]),n,mxGetJc(prhs[0]),mxGetIr(prhs[0]),mxGetPr(prhs[0]));
}
