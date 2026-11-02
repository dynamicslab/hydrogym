#include <math.h>
#include "mex.h"
#include "matrix.h"

static void *mem_alloc(size_t n)
{
  void *ptr = mxMalloc(n);
  if(!ptr && n) mexPrintf("Failed to allocate %lu bytes\n",(unsigned long)n);
  return ptr;
}

typedef struct { mwIndex i,j; double v; } matent;

#define DEF_HEAP_SORT(name)                                   \
static void name(T *A, mwSize n)                              \
{                                                             \
  mwIndex i;                                                  \
  if(!n) return;                                              \
  /* build heap */                                            \
  for(i=1;i<n;++i) {                                          \
    T item = A[i];                                            \
    mwIndex h=i, p = (h-1)>>1;                                \
    if(!LT(A[p],item)) continue;                              \
    do A[h]=A[p], h=p, p=(p-1)>>1; while(h && LT(A[p],item)); \
    A[h] = item;                                              \
  }                                                           \
  /* extract */                                               \
  for(i=n-1;i;--i) {                                          \
    T item = A[i];                                            \
    mwIndex h = 0;                                            \
    A[i] = A[0];                                              \
    for(;;) {                                                 \
      mwIndex ch = 1+(h<<1), r = ch+1;                        \
      if(r<i && LT(A[ch],A[r])) ch=r;                         \
      if(ch>=i || !LT(item,A[ch])) break;                     \
      A[h]=A[ch], h=ch;                                       \
    }                                                         \
    A[h] = item;                                              \
  }                                                           \
}

#define T matent
#define LT(a,b) (fabs((a).v)<fabs((b).v))
DEF_HEAP_SORT(heap_sortv)
#undef LT
#define LT(a,b) ((a).j<(b).j || ((a).j==(b).j && (a).i<(b).i))
DEF_HEAP_SORT(heap_sortij)
#undef LT
#undef T

static mxArray *sym_sparsify(
  mwSize n, const mwIndex *jc, const mwIndex *ir, const double *A,
  double tol)
{
  mwIndex j, k; mwSize count=0;
  matent *S, *p, *end; double *E;
  mxArray *U;
  for(j=0;j<n;++j) for(k=jc[j];k<jc[j+1];++k) if(A[k]!=0 && ir[k]<j) ++count;
  if(!(S=mem_alloc(count*sizeof(matent))) && count) return 0;
  if(!(E=mem_alloc(n*sizeof(double)))) { mxFree(S); return 0; }
  p=S;
  for(j=0;j<n;++j) for(k=jc[j];k<jc[j+1];++k) if(A[k]!=0 && ir[k]<j)
    p->j=j,p->i=ir[k],p->v=A[k],++p;
  heap_sortv(S,count);
  for(j=0;j<n;++j) E[j]=0;
  if(count && fabs(S->v)<tol) {
    matent *o = S;
    E[S->i]+=fabs(S->v), E[S->j]+=fabs(S->v);
    for(p=S+1,end=S+count;p!=end;++p) {
      double ei=E[p->i]+fabs(p->v), ej=E[p->j]+fabs(p->v);
      if(ei<tol && ej<tol) E[p->i]=ei,E[p->j]=ej;
      else                 *(o++) = *p;
    }
    count = o-S;
  }
  heap_sortij(S,count);
  U = mxCreateSparse(n,n,count,mxREAL);
  if(U) {
    mwIndex *Ujc=mxGetJc(U), *Uir=mxGetIr(U);
    double *Upr=mxGetPr(U);
    mwIndex lj=0;
    Ujc[0]=0;
    for(k=0;k<count;++k) {
      j=S[k].j;
      while(lj<j) Ujc[++lj]=k;
      Uir[k]=S[k].i;
      Upr[k]=1;/*S[k].v;*/
    }
    while(lj<n) Ujc[++lj]=count;
  } else mexWarnMsgTxt("Out of memory.");
  mxFree(E),mxFree(S);
  return U;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize n;
  if(nrhs!=2) { mexWarnMsgTxt("Two inputs required."); return; }
  if(nlhs>1) { mexWarnMsgTxt("One output expected."); return; }
  if(!mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]) || !mxIsDouble(prhs[0]))
    { mexWarnMsgTxt("First input not a sparse real double matrix."); return; }
  if(mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])
   ||mxGetM(prhs[1])!=1  || mxGetN(prhs[1])!=1) {
    mexWarnMsgTxt("Second input not a double real scalar."); return;
  }
  if((n=mxGetN(prhs[0]))!=mxGetM(prhs[0])) 
    { mexWarnMsgTxt("First input not a square matrix."); return; }
  plhs[0]=sym_sparsify(n,mxGetJc(prhs[0]),mxGetIr(prhs[0]),mxGetPr(prhs[0]),
                       mxGetPr(prhs[1])[0]);
}
