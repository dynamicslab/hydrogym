#include <stdio.h>
#include "mex.h"
#include "matrix.h"

static void writevec(const char *name, mwSize n, const double *v)
{
  const double magic = 3.14159;
  const double stamp = 2.01;
  FILE *f = fopen(name,"w");
  fwrite(&magic,sizeof(double),1,f);
  fwrite(&stamp,sizeof(double),1,f);
  fwrite(v,sizeof(double),n,f);
  fclose(f);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char *filename;
  mwSize m,n;
  if(nrhs!=2) { mexWarnMsgTxt("Two inputs required."); return; }
  if(mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
    mexWarnMsgTxt("First input not a full, double, real array."); return;
  }
  if(!mxIsChar(prhs[1]) || mxGetM(prhs[1])!=1) {
    mexWarnMsgTxt("Second input not a string."); return;
  }
  m = mxGetM(prhs[0]), n = mxGetN(prhs[0]);
  if(n!=1) {
    mexWarnMsgTxt("First input wrongly sized."); return;
  }
  filename = mxArrayToString(prhs[1]);
  mexPrintf("Writing %d vector to %s\n",m,filename);
  writevec(filename,m,mxGetPr(prhs[0]));
  mxFree(filename);
}
