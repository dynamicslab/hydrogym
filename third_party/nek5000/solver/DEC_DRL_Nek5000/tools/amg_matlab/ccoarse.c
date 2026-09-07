#include "mex.h"
#include "matrix.h"

#define TOL 0.9

typedef struct {
  int nzmax;
  const int *ir, *jc;
  const double *pr;
} spmat;

static int n;
static spmat S[2];
static double *g, *q, *h;
static int *heap, *heapinv, heap_n;
static int *C, C_n;
static int max_initial=0;

static void init_metrics(void)
{
  int i,j=0,imax=S[0].jc[n];
  double gmax = 0;
  for(i=0;i<n;++i) g[i]=0;
  for(i=0;i<imax;++i) {
    if(i>=S[0].jc[j+1]) ++j;
    g[j]+=S[0].pr[i];
  }
  for(i=0;i<n;++i) q[i]=0;
  for(i=0;i<n;++i) h[i]=2*g[i];
  for(i=0;i<n;++i) if(g[i]>gmax) gmax=g[i],max_initial=i;
}

static void heap_sift_up(int hole, int i, double hi)
{
  while(hole>1) {
    int parent = hole>>1, ip = heap[parent];
    if(hi<=h[ip]) break;
    heap[hole]=ip, heapinv[ip]=hole;
    hole=parent;
  }
  heap[hole]=i, heapinv[i]=hole;
}

static void heap_sift_down(int hole, int i, double hi)
{
  for(;;) {
    int child=hole<<1, r=child+1, ic;
    if(r<=heap_n && h[heap[r]]>h[heap[child]]) child=r;
    if(child>heap_n || (ic=heap[child],hi>=h[ic])) break;
    heap[hole]=ic, heapinv[ic]=hole;
    hole=child;
  }
  heap[hole]=i, heapinv[i]=hole;
}

static int heap_delete(int p)
{
  int i=heap[p],ia; double hia;
  heapinv[i]=0;
  ia = heap[heap_n--], hia = h[ia];
  if(p==heap_n+1) return i;
  if(p>1 && hia>h[heap[p>>1]]) heap_sift_up(p,ia,hia);
  else heap_sift_down(p,ia,hia);
  return i;
}

static void heap_init(void)
{
  int i;
  heap_n=0;
  for(i=0;i<n;++i) {
    if(g[i]<=TOL) { heapinv[i]=0; continue; }
    ++heap_n, heap_sift_up(heap_n,i,h[i]);
  }
}

static void coarsen(void)
{
  init_metrics();
  heap_init();
  C_n=0;
  while(heap_n>0) {
    int c = heap_delete(1), k,ke, p;
    C[C_n++]=c;
    for(k=S[0].jc[c],ke=S[0].jc[c+1];k!=ke;++k) {
      int i=S[0].ir[k];
      p = heapinv[i]; if(!p) continue;
      g[i]-=S[0].pr[k], h[i]=2*g[i]+q[i];
      if(g[i]<=TOL) heap_delete(p); else heap_sift_down(p,i,h[i]);
    }
    for(k=S[1].jc[c],ke=S[1].jc[c+1];k!=ke;++k) {
      int i=S[1].ir[k];
      p = heapinv[i]; if(!p) continue;
      q[i]+=S[1].pr[k], h[i]=2*g[i]+q[i];
      heap_sift_up(p,i,h[i]);
    }
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int i; double *Cout;
  if(nrhs!=2) { mexWarnMsgTxt("Two inputs required."); return; }
  if(!mxIsSparse(prhs[0]) || !mxIsSparse(prhs[1]))
    { mexWarnMsgTxt("Sparse inputs required."); return; }
  n = mxGetM(prhs[0]);
  if(mxGetN(prhs[0])!=n) { mexWarnMsgTxt("Square inputs required."); return; }
  if(mxGetM(prhs[1])!=n || mxGetN(prhs[1])!=n)
    { mexWarnMsgTxt("Same dimension inputs required."); return; }
  if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    { mexWarnMsgTxt("Double precision inputs required."); return; }
  for(i=0;i<2;++i) {
    S[i].nzmax = mxGetNzmax(prhs[i]);
    S[i].ir = mxGetIr(prhs[i]);
    S[i].jc = mxGetJc(prhs[i]);
    S[i].pr = mxGetPr(prhs[i]);
  }
  g = mxMalloc(3*n*sizeof(double));
  q = g+n, h = q+n;
  heap = mxMalloc(((n+1)+2*n)*sizeof(int));
  heapinv = heap+(n+1), C=heapinv+n;
  coarsen();
  if(C_n==0 && n>0) C_n=1, C[0]=max_initial;
  plhs[0] = mxCreateDoubleMatrix(C_n,1,mxREAL);
  Cout = mxGetPr(plhs[0]);
  for(i=0;i<C_n;++i) Cout[i]=C[i]+1;
  mxFree(g); mxFree(heap);
}
