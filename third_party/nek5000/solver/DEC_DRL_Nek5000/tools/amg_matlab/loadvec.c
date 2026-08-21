#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char *filename;
  FILE *f;
  long n;
  if(nlhs!=1) { mexWarnMsgTxt("One output required."); return; }
  if(nrhs!=1) { mexWarnMsgTxt("One input required."); return; }
  if(!mxIsChar(prhs[0]) || mxGetM(prhs[0])!=1) {
    mexWarnMsgTxt("Input not a string."); return;
  }
  filename = mxArrayToString(prhs[0]);
  f = fopen(filename,"r");
  fseek(f,0,SEEK_END);
  n = ftell(f)/sizeof(double);
  fseek(f,0,SEEK_SET);
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
  fread(mxGetPr(plhs[0]),sizeof(double),n,f);
  fclose(f);
  mxFree(filename);
}
