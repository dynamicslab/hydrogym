#include <stdio.h>
#include "mex.h"
#include "matrix.h"

static int writemat(const char *name, int m, int n,
                    const int *ir, const int *jc, const double *pr,
                    const double *i_id, const double *j_id)
{
  const double magic = 3.14159;
  double t; double *buf; int i;
  int nnz = jc[n];
  int max = 0;
  FILE *f = fopen(name,"w");
  fwrite(&magic,sizeof(double),1,f);
  /*t = n,   fwrite(&t,sizeof(double),1,f);
  fwrite(j_id,sizeof(double),n,f);*/
  
  buf = mxMalloc((n+1>nnz?n+1:nnz)*sizeof(double));

  buf[0]=jc[0];
  for(i=1;i<=n;++i) { 
    int nz=jc[i]-jc[i-1];
    if(nz>max) max=nz;
    buf[i]=jc[i];
  }
  fwrite(buf,sizeof(double),n+1,f);
  for(i=0;i<nnz;++i) buf[i] = i_id[ir[i]];
  fwrite(buf,sizeof(double),nnz,f);

  mxFree(buf);
  
  fwrite(pr,sizeof(double),nnz,f);

  fclose(f);
  
  return max;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char *filename;
  int m,n,max;
  if(nrhs!=4) { mexWarnMsgTxt("Three inputs required."); return; }
  if(mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
    mexWarnMsgTxt("First input not a full, double, real array."); return;
  }
  if(mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
    mexWarnMsgTxt("Second input not a full, double, real array."); return;
  }
  if(!mxIsSparse(prhs[2]) || !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
    mexWarnMsgTxt("Third input not a sparse, double, real matrix."); return;
  }
  if(!mxIsChar(prhs[3]) || mxGetM(prhs[3])!=1) {
    mexWarnMsgTxt("Fourth input not a string."); return;
  }
  m = mxGetM(prhs[2]), n = mxGetN(prhs[2]);
  if(mxGetM(prhs[0])!=m || mxGetN(prhs[0])!=1) {
    mexWarnMsgTxt("First input wrongly sized."); return;
  }
  if(mxGetM(prhs[1])!=n || mxGetN(prhs[1])!=1) {
    mexWarnMsgTxt("Second input wrongly sized."); return;
  }
  filename = mxArrayToString(prhs[3]);
  max=writemat(filename,m,n,
               mxGetIr(prhs[2]),mxGetJc(prhs[2]),mxGetPr(prhs[2]),
               mxGetPr(prhs[0]),mxGetPr(prhs[1]));
  mexPrintf("Wrote %d x %d matrix to %s, max row size = %d\n",m,n,filename,max);
  mxFree(filename);
  plhs[0] = mxCreateDoubleScalar((double)max);
}
