#include "mex.h"
#include "matrix.h"

/* compute diag(A'*B) */

static void diagmm(double *d, mwIndex n,
                   const mwIndex *Ajc, const mwIndex *Air, const double *A,
                   const mwIndex *Bjc, const mwIndex *Bir, const double *B)
{
  mwIndex j;
  for(j=0;j<n;++j) {
    double x = 0;
          mwIndex ak =Ajc[j  ], bk =Bjc[j  ];
    const mwIndex ake=Ajc[j+1], bke=Bjc[j+1];
    if(ak==ake || bk==bke) continue;
    for(;;) {
      mwIndex i=Bir[bk]; while(Air[ak]<i) if(++ak==ake) goto diagmm_coldone;
              i=Air[ak]; while(Bir[bk]<i) if(++bk==bke) goto diagmm_coldone;
      if(Bir[bk]!=i) continue;
      x += A[ak]*B[bk];
      if(++ak==ake || ++bk==bke) goto diagmm_coldone;
    }
diagmm_coldone: d[j]=x;
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwIndex n1, n2, n;
  if(nrhs!=2) { mexWarnMsgTxt("Two inputs required."); return; }
  if(nlhs>1) { mexWarnMsgTxt("One output expected."); return; }
  if(!mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]))
    { mexWarnMsgTxt("Sparse real double inputs required."); return; }
  if(!mxIsSparse(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsDouble(prhs[1]))
    { mexWarnMsgTxt("Sparse real double inputs required."); return; }
  if(mxGetM(prhs[0])!=mxGetM(prhs[1]))
    { mexWarnMsgTxt("Inputs must have equal column lengths."); return; }
  n1 = mxGetN(prhs[0]), n2 = mxGetN(prhs[1]);
  n = n2<n1 ? n2:n1;
  if(!(plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL)))
    { mexWarnMsgTxt("Out of memory."); return; }
  diagmm(mxGetPr(plhs[0]),n,
         mxGetJc(prhs[0]),mxGetIr(prhs[0]),mxGetPr(prhs[0]),
         mxGetJc(prhs[1]),mxGetIr(prhs[1]),mxGetPr(prhs[1]));
}
