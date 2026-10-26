#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "sort.h"

#define sparse_cholesky_factor PREFIXED_NAME(sparse_cholesky_factor)
#define sparse_cholesky_solve  PREFIXED_NAME(sparse_cholesky_solve )
#define sparse_cholesky_free   PREFIXED_NAME(sparse_cholesky_free  )

/* factors: L is in CSR format
            D is a diagonal matrix stored as a vector
   actual factorization is:
   
                  -1      T
     A   = (I-L) D   (I-L)
     
      -1        -T        -1
     A   = (I-L)   D (I-L)
     
   (triangular factor is unit diagonal; the diagonal is not stored)
*/
struct sparse_cholesky {
  uint n, *Lrp, *Lj;
  double *L, *D;
};

/*
  symbolic factorization: finds the sparsity structure of L
  
  uses the concept of elimination tree:
    the parent of node j is node i when L(i,j) is the first
      non-zero in column j below the diagonal (i>j)
    L's structure is discovered row-by-row; the first time
      an entry in column j is set, it must be the parent

  the nonzeros in L are the nonzeros in A + paths up the elimination tree  
  
  linear in the number of nonzeros of L
*/
static void factor_symbolic(uint n, const uint *Arp, const uint *Aj,
                            struct sparse_cholesky *out, buffer *buf)
{
  uint *visit = tmalloc(uint,2*n), *parent = visit+n;
  uint *Lrp, *Lj;
  uint i,nz=0;
  
  out->n=n;

  for(i=0;i<n;++i) {
    uint p=Arp[i], pe=Arp[i+1];
    visit[i]=i, parent[i]=n;
    for(;p!=pe;++p) {
      uint j=Aj[p]; if(j>=i) break;
      for(;visit[j]!=i;j=parent[j]) {
        ++nz, visit[j]=i;
        if(parent[j]==n) { parent[j]=i; break; }
      }
    }
  }
  
  Lrp=out->Lrp=tmalloc(uint,n+1+nz);
  Lj =out->Lj =Lrp+n+1;

  Lrp[0]=0;
  for(i=0;i<n;++i) {
    uint p=Arp[i], pe=Arp[i+1], count=0, *Ljr=&Lj[Lrp[i]];
    visit[i]=i;
    for(;p!=pe;++p) {
      uint j=Aj[p]; if(j>=i) break;
      for(;visit[j]!=i;j=parent[j]) Ljr[count++]=j, visit[j]=i;
    }
    sortv(Ljr, Ljr,count,sizeof(uint), buf);
    Lrp[i+1]=Lrp[i]+count;
  }
  free(visit);
}

/*
  numeric factorization:
  
  L is built row-by-row, using:    ( ' indicates transpose )

  
  [ A  r ]  = [ (I-L)   ] [ D^(-1)  ] [ (I-L)' -s ]
  [ r' a ]    [  -s'  1 ] [     1/d ] [         1 ]
            
            = [ A   (I-L) D^(-1) (-s)  ]
              [ r'  s' D^(-1) s + 1/d  ]
              
  so, if r' is the next row of A, up to but excluding the diagonal,
  then the next row of L, s', obeys
  
     r = - (I-L) D^(-1) s
 
  let y = (I-L)^(-1) (-r)
  then s = D y, and d = 1/(s' y)
  
*/
static void factor_numeric(uint n, const uint *Arp, const uint *Aj,
                           const double *A,
                           struct sparse_cholesky *out,
                           uint *visit, double *y)
{
  const uint *Lrp=out->Lrp, *Lj=out->Lj;
  double *D, *L;
  uint i;
  
  D=out->D=tmalloc(double,n+Lrp[n]);
  L=out->L=D+n;
  
  for(i=0;i<n;++i) {
    uint p,pe; double a=0;
    visit[i]=n;
    for(p=Lrp[i],pe=Lrp[i+1];p!=pe;++p) {
      uint j=Lj[p]; y[j]=0, visit[j]=i;
    }
    for(p=Arp[i],pe=Arp[i+1];p!=pe;++p) {
      uint j=Aj[p];
      if(j>=i) { if(j==i) a=A[p]; break; }
      y[j]=-A[p];
    }
    for(p=Lrp[i],pe=Lrp[i+1];p!=pe;++p) {
      uint q,qe,j=Lj[p]; double lij,yj=y[j];
      for(q=Lrp[j],qe=Lrp[j+1];q!=qe;++q) {
        uint k=Lj[q]; if(visit[k]==i) yj+=L[q]*y[k];
      }
      y[j]=yj;
      L[p]=lij=D[j]*yj;
      a-=yj*lij;
    }
    D[i]=1/a;
  }
}

/* x = A^(-1) b;  works when x and b alias */
void sparse_cholesky_solve(
  double *x, const struct sparse_cholesky *fac, double *b)
{
  const uint n=fac->n, *Lrp=fac->Lrp, *Lj=fac->Lj;
  const double *L=fac->L, *D=fac->D;
  uint i, p,pe;
  for(i=0;i<n;++i) {
    double xi = b[i];
    for(p=Lrp[i],pe=Lrp[i+1];p!=pe;++p) xi+=L[p]*x[Lj[p]];
    x[i]=xi;
  }
  for(i=0;i<n;++i) x[i]*=D[i];
  for(i=n;i;) {
    double xi = x[--i];
    for(p=Lrp[i],pe=Lrp[i+1];p!=pe;++p) x[Lj[p]]+=L[p]*xi;
  }
}

void sparse_cholesky_factor(uint n, const uint *Arp, const uint *Aj,
                            const double *A,
                            struct sparse_cholesky *out, buffer *buf)
{
  const uint n_uints_as_dbls = (n*sizeof(uint)+sizeof(double)-1)/sizeof(double);
  buffer_reserve(buf,(n_uints_as_dbls+n)*sizeof(double));
  factor_symbolic(n,Arp,Aj,out,buf);
  factor_numeric(n,Arp,Aj,A,out,buf->ptr,n_uints_as_dbls+(double*)buf->ptr);
}

void sparse_cholesky_free(struct sparse_cholesky *fac)
{
  free(fac->Lrp); fac->Lj=fac->Lrp=0;
  free(fac->D);   fac->L =fac->D  =0;
}

