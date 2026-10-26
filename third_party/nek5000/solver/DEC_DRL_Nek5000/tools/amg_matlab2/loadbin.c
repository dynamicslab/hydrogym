#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"

#define N sizeof(double)
static double byteswap(double x)
{
  char buf[N]; char t;
  memcpy(buf,&x,N);
#define SWAP(i) if(N>2*(i)+1) t=buf[i],buf[i]=buf[N-1-(i)],buf[N-1-(i)]=t
#define SWAP2(i) SWAP(i); SWAP((i)+1)
#define SWAP4(i) SWAP2(i); SWAP2((i)+2)
#define SWAP8(i) SWAP4(i); SWAP4((i)+4)
#define SWAP16(i) SWAP8(i); SWAP8((i)+8)
  SWAP16(0);
#undef SWAP
#undef SWAP2
#undef SWAP4
#undef SWAP8
#undef SWAP16
  memcpy(&x,buf,N);
  return x;
}
#undef N

static long filesize(const char *name) {
  long n;
  FILE *f = fopen(name,"r");
  fseek(f,0,SEEK_END);
  n = ftell(f)/sizeof(double);
  fclose(f);
  return n;
}

static long readfile(double *data, long max, const char *name)
{
  const double magic = 3.14159;
  long n;
  FILE *f = fopen(name,"r");
  fseek(f,0,SEEK_END);
  n = ftell(f)/sizeof(double);
  if(n>max) mexWarnMsgTxt("file longer than expected"),n=max;
  fseek(f,0,SEEK_SET);
  fread(data,sizeof(double),n,f);
  fclose(f);
  if(n>0 && fabs(data[0]-magic)>0.000001) {
    long i;
    mexWarnMsgTxt("swapping byte order");
    if(fabs(byteswap(data[0])-magic)>0.000001) {
      mexWarnMsgTxt("magic number for endian test not found");
    } else
      for(i=0;i<n;++i) data[i]=byteswap(data[i]);
  }
  return n;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  long n; double *v;
  if(nlhs!=3) { mexWarnMsgTxt("Three outputs required."); return; }
  n = filesize("amgdmp_i.dat");
  v = mxMalloc(n*sizeof(double));
  plhs[0] = mxCreateDoubleMatrix(n-1,1,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(n-1,1,mxREAL);
  plhs[2] = mxCreateDoubleMatrix(n-1,1,mxREAL);
  mexPrintf("loading i\n"); readfile(v,n,"amgdmp_i.dat");
  memcpy(mxGetPr(plhs[0]),v+1,(n-1)*sizeof(double));
  mexPrintf("loading j\n"); readfile(v,n,"amgdmp_j.dat");
  memcpy(mxGetPr(plhs[1]),v+1,(n-1)*sizeof(double));
  mexPrintf("loading p\n"); readfile(v,n,"amgdmp_p.dat");
  memcpy(mxGetPr(plhs[2]),v+1,(n-1)*sizeof(double));
  mxFree(v);
}
