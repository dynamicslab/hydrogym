#include <stdio.h>
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

static int load_sp_mat(sp_mat_c *mat, const mxArray *arr)
{
  if(!mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  mat->m = mxGetM(arr), mat->n = mxGetN(arr);
  mat->ir = mxGetIr(arr), mat->jc = mxGetJc(arr);
  mat->pr = mxGetPr(arr);
  return 1;
}

static int load_vec(vec_c *vec, const mxArray *arr)
{
  mwSize m,n;
  if(mxIsSparse(arr) || !mxIsDouble(arr) || mxIsComplex(arr)) return 0;
  m = mxGetM(arr), n = mxGetN(arr);
  if(m!=1 && n!=1) return 0;
  vec->n = m+n-1;
  vec->pr = mxGetPr(arr);
  return 1;
}

static char *load_string(const mxArray *arr)
{
  if(!mxIsChar(arr) || mxGetM(arr)!=1) return 0;
  return mxArrayToString(arr);
}

static mwSize max_col_nnz(const sp_mat_c *mat)
{
  mwSize n=mat->n,max=0;
  mwIndex i;
  for(i=0;i<n;++i) {
    mwSize l=mat->jc[i+1]-mat->jc[i];
    if(l>max) max=l;
  }
  return max;
}

static void savemats(double *len, const vec_c *lvl,
                     mwSize nl, const vec_c *id, const sp_mat_c *mat,
                     const char *filename)
{
  const double magic = 3.14159;
  FILE *f = fopen(filename,"w");
  mwSize max=0,n=lvl->n;
  mwIndex i;
  mwIndex *col;
  double *buf;
  fwrite(&magic,sizeof(double),1,f);
  for(i=0;i<nl;++i) { mwSize l=max_col_nnz(&mat[i]); if(l>max) max=l; }
  mexPrintf("maximum col size = %d\n",(int)max);
  buf = mxMalloc(2*max*sizeof(double));
  col = mxMalloc(nl*sizeof(mwIndex));
  for(i=0;i<nl;++i) col[i]=0;
  for(i=0;i<n;++i) {
    mwIndex l = lvl->pr[i]-1;
    const sp_mat_c *M;
    mwIndex j,k,kb,ke;
    double *p;
    if(l<0 || l>nl) { mexWarnMsgTxt("level out of bounds"); continue; }
    if(l==nl) { len[i]=0; continue; }
    M = &mat[l];
    j = col[l]++;
    if(j>=M->n) { mexWarnMsgTxt("column out of bounds"); continue; }
    kb=M->jc[j],ke=M->jc[j+1];
    p = buf;
    for(k=kb;k!=ke;++k) *p++ = id[l].pr[M->ir[k]], *p++ = M->pr[k];
    len[i] = ke-kb;
    fwrite(buf,sizeof(double),2*(ke-kb),f);
  }
  for(i=0;i<nl;++i) {
    if(col[i]!=mat[i].n) mexWarnMsgTxt("matrices not exhausted");
  }
  mxFree(col);
  mxFree(buf);
  fclose(f);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwIndex i,nl;
  sp_mat_c *mat;
  vec_c *id, lvl;
  char *filename;
  if(nrhs!=4) { mexWarnMsgTxt("Four inputs required."); return; }
  if(!load_vec(&lvl,prhs[0])) {
    mexWarnMsgTxt("First input not a full, double, real vector.");
    return;
  }
  if(!mxIsCell(prhs[1])) {
    mexWarnMsgTxt("Second input not a cell array."); return;
  }
  if(!mxIsCell(prhs[2])) {
    mexWarnMsgTxt("Third input not a cell array."); return;
  }
  if(!(filename=load_string(prhs[3]))) {
    mexWarnMsgTxt("Fourth input not a string."); return;
  }
  nl = mxGetN(prhs[2]);
  if(mxGetN(prhs[1])!=nl) {
    mexWarnMsgTxt("Second and third inputs disagree on length."); return;
  }
  mexPrintf("savemats called with %d matrices\n",(int)nl);
  id = mxMalloc(nl*sizeof(vec_c));
  mat = mxMalloc(nl*sizeof(sp_mat_c));
  for(i=0;i<nl;++i) {
    if(!load_vec(&id[i],mxGetCell(prhs[1],i))) {
      mexWarnMsgTxt("Second input cell contents not a full, "
                    "double, real vector."); return;
    }
    if(!load_sp_mat(&mat[i],mxGetCell(prhs[2],i))) {
      mexWarnMsgTxt("Third input cell contents not a sparse, "
                    "double, real matrix."); return;
    }
    if(mat[i].m!=id[i].n) {
      mexPrintf("level %ld, %ld x %ld matrix, %ld id vector\n",
                (long)i,(long)mat[i].m,(long)mat[i].n,(long)id[i].n);
      mexWarnMsgTxt("Mismatch of matrix rows and id length"); return;
    }
  }
  plhs[0] = mxCreateDoubleMatrix(lvl.n,1,mxREAL);
  savemats(mxGetPr(plhs[0]),&lvl,nl,id,mat,filename);
  mxFree(mat);
  mxFree(id);
  mxFree(filename);
}
