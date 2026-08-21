#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    /* for cos, fabs */
#include <float.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "poly.h"

#define lob_bnd_setup  PREFIXED_NAME(lob_bnd_setup)
#define lob_bnd_lin_1  PREFIXED_NAME(lob_bnd_lin_1)
#define lob_bnd_lin_2  PREFIXED_NAME(lob_bnd_lin_2)
#define lob_bnd_lin_3  PREFIXED_NAME(lob_bnd_lin_3)
#define lob_bnd_1      PREFIXED_NAME(lob_bnd_1    )
#define lob_bnd_2      PREFIXED_NAME(lob_bnd_2    )
#define lob_bnd_3      PREFIXED_NAME(lob_bnd_3    )

struct dbl_range { double min,max; };

/*--------------------------------------------------------------------------
   Bounds for Polynomials on [-1,1]^d
     given in the Lagrangian basis on
     Gauss-Lobatto-Legendre quadrature nodes

   The main parameters are the number of GLL nodes in each dimension
     unsigned nr = ..., ns = ..., nt = ...;

   The number of points in the constructed piecewise (tri-,bi-)linear bounds
   is a parameter; more points give tighter bounds, and we expect m>n.
   
     unsigned mr = 4*nr, ms = 4*ns, mt = 4*nt;
   
   The necessary setup is accomplished via:
     double *data_r = tmalloc(double, lob_bnd_size(nr,mr));
     double *data_s = tmalloc(double, lob_bnd_size(ns,ms));
     double *data_t = tmalloc(double, lob_bnd_size(nt,mt));
     lob_bnd_setup(data_r, nr,mr);
     lob_bnd_setup(data_s, ns,ms);
     lob_bnd_setup(data_t, nt,mt);
 
   Bounds may then be computed via:
     double work1r[2*mr], work1s[2*ms];
     double work2[2*mr*(ns+ms+1)];
     double work3[2*mr*ms*(nt+mt+1)];
     double ur[nr], us[ns];    // 1-d polynomials on the zr[] and zs[] nodes
     double u2[ns][nr];        // 2-d polynomial on zr[] (x) zs[]
     double u3[nt][ns][nr];    // 3-d polynomial on zr[] (x) zs[] (x) zt[]
     struct dbl_range bound;
     
     bound = lob_bnd_1(data_r,nr,mr, ur, work1r); // compute bounds on ur
     bound = lob_bnd_1(data_s,ns,ms, us, work1s); // compute bounds on us
     bound = lob_bnd_2(data_r,nr,mr, data_s,ns,ms,
               (const double*)&u2[0][0], work2); // compute bounds on u2
     bound = lob_bnd_3(data_r,nr,mr, data_s,ns,ms, data_t,nt,mt,
               (const double*)&u3[0][0], work3); // compute bounds on u3

    free(data_r), free(data_s), free(data_t);

   The functions lob_bnd_lin_d compute the piecewise d-linear bounds.
   Nodes for these are Chebyshev-Lobatto:
     h[0] = -1, h[m-1] = 1;
     for(j=1;j<m-1;++j) h[j] = cos((m-1-j)*PI/(m-1));
   The functions lob_bnd_d simply call these and return the min and max
   over all nodes.
    
  --------------------------------------------------------------------------*/

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

void lob_bnd_setup(double *restrict data, unsigned n, unsigned m)
{{
  unsigned nm = n*m, i,j;
  double *restrict z=data,
         *restrict Q=z+n, *restrict h=Q+2*n,
         *restrict lb=h+m, *restrict lbnp=lb+2*nm;
  double *restrict pl = tmalloc(double,5*n + gll_lag_size(n)),
         *restrict dl = pl+n, *restrict pr=dl+n, *restrict dr=pr+n,
         *restrict p=dr+n, *restrict gll_data=p+n;
  lagrange_fun *lag = gll_lag_setup(gll_data,n);
  
  /* set z and Q to Lobatto nodes, weights */
  lobatto_quad(z,Q,n);
  
  /* Q0, Q1 : linear functionals on the GLL nodal basis
              for the zeroth and first Legendre coefficient */
  for(i=n;i;) --i, Q[2*i]=Q[i]/2, Q[2*i+1] = 3*Q[2*i]*z[i];
  /*for(i=0;i<n;++i) Q0[i]=Q0[i]/2, Q1[i] = 3*Q0[i]*z[i];*/
  
  /* h : m Chebyshev nodes */
  h[0] = -1, h[m-1] = 1;
  for(j=1;j<m-1;++j) h[j] = cos((m-1-j)*PI/(m-1));

  /* lv, uv : lower, upper piecewise linear bounds (nodes h) of
              Lagrangian basis functions for Gauss-Lobatto nodes */
  for(i=0;i<n;++i)
    lb[(i*m+  0)*2+1]=lb[(i*m+  0)*2+0]=(i==  0?1:0),
    lb[(i*m+m-1)*2+1]=lb[(i*m+m-1)*2+0]=(i==n-1?1:0);

  lag(pl,gll_data,n,1,(h[0]+h[1])/2);
  for(j=1;j<m-1;++j) {
    double x = h[j], xl = (x+h[j-1])/2, xr = (x+h[j+1])/2;
    lag(pr,gll_data,n,1,xr);
    lag(p ,gll_data,n,0,x );
    for(i=0;i<n;++i) {
      double lo,up, cl = pl[i] + (x-xl)*dl[i], cr = pr[i] + (x-xr)*dr[i];
      if(cl<cr) lo=cl,up=cr; else lo=cr,up=cl;
      if(p[i]<lo) lo=p[i]; if(up<p[i]) up=p[i];
      lb[(i*m+j)*2+0] = lo, lb[(i*m+j)*2+1] = up;
    }
    memcpy(pl,pr,2*n*sizeof(double));
  }

  /* lbnp : lb split into negative and positive parts */
  for(i=0;i<nm;++i) {
    double f;
    lbnp[4*i+0]=lbnp[4*i+1]=lbnp[4*i+2]=lbnp[4*i+3]=0;
    if((f=lb[2*i+0])<0) lbnp[4*i+0]=f; else lbnp[4*i+1]=f;
    if((f=lb[2*i+1])<0) lbnp[4*i+2]=f; else lbnp[4*i+3]=f;
  }

  free(pl);
}}

static void lob_bnd_fst(
  double *restrict b,
  const double *restrict z, const double *restrict Q, const double *restrict h,
  const double *restrict lb, unsigned n, unsigned m,
  const double *restrict u)
{
  unsigned i,j;
  double a0=0, a1=0;
  for(i=0;i<n;++i) a0 += Q[2*i]*u[i], a1 += Q[2*i+1]*u[i];
  for(j=0;j<m;++j) b[2*j+1] = b[2*j+0] = a0 + a1*h[j];
  for(i=0;i<n;++i) {
    double w = u[i] - (a0 + a1*z[i]);
    if(w>=0)
      for(j=0;j<m;++j) b[2*j+0]+=w*lb[0], b[2*j+1]+=w*lb[1], lb+=2;
    else
      for(j=0;j<m;++j) b[2*j+0]+=w*lb[1], b[2*j+1]+=w*lb[0], lb+=2;
  }
}

static void lob_bnd_ext(
  double *restrict b_,
  const double *restrict z, const double *restrict Q, const double *restrict h,
  const double *restrict lbnp, unsigned n, unsigned m,
  const double *restrict br_, unsigned mr,
  double *restrict a)
{
  const double *restrict br = br_;
  double *restrict b = b_;
  unsigned i,j,k;
  for(i=0;i<mr;++i) a[2*i+1]=a[2*i+0]=0;
  for(j=0;j<n;++j) {
    double t, q0 = Q[2*j], q1 = Q[2*j+1];
    for(i=0;i<mr;++i) t=(br[0]+br[1])/2, br+=2, a[2*i]+=q0*t, a[2*i+1]+=q1*t;
  }
  for(i=0;i<mr;++i) {
    double a0=a[2*i],a1=a[2*i+1];
    for(k=0;k<m;++k) b[1]=b[0]=a0+a1*h[k], b+=2;
  }
  br=br_;
  for(j=0;j<n;++j,lbnp+=4*m) {
    double zj = z[j];
    b = b_;
    for(i=0;i<mr;++i) {
      double t = a[2*i] + a[2*i+1]*zj;
      double w0 = *br++ - t;
      double w1 = *br++ - t;
      if(w0>=0)      /* 0  <= w0 <= w1 */
        for(k=0;k<m;++k)
          *b++ += w0 * lbnp[4*k+1] + w1 * lbnp[4*k+0],
          *b++ += w1 * lbnp[4*k+3] + w0 * lbnp[4*k+2];
      else if(w1<=0) /* w0 <= w1 <= 0  */
        for(k=0;k<m;++k)
          *b++ += w0 * lbnp[4*k+3] + w1 * lbnp[4*k+2],
          *b++ += w1 * lbnp[4*k+1] + w0 * lbnp[4*k+0];
      else           /* w0 <  0  <  w1 */
        for(k=0;k<m;++k)
          *b++ += w0 * lbnp[4*k+3] + w1 * lbnp[4*k+0],
          *b++ += w1 * lbnp[4*k+3] + w0 * lbnp[4*k+0];
    }
  }
}

void lob_bnd_lin_1(double *restrict b,
                   const double *restrict lob_bnd_data, unsigned n, unsigned m,
                   const double *restrict u, uint un)
{
  const double *z=lob_bnd_data, *Q=z+n, *h=Q+2*n, *lb=h+m;
  for(;un;--un, u+=n, b+=2*m) lob_bnd_fst(b, z,Q,h,lb,n,m, u);
}

/* work holds 2*mr + 2*ns*mr doubles */
void lob_bnd_lin_2(
  double *restrict b,
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *restrict u, uint un, double *restrict work)
{
  unsigned mrs = mr*ms;
  const double *zr=lob_bnd_data_r,*Qr=zr+nr,*hr=Qr+2*nr,*lb_r=hr+mr;
  const double *zs=lob_bnd_data_s,*Qs=zs+ns,*hs=Qs+2*ns,*lbnp_s=hs+ms+2*ns*ms;
  double *a = work, *br = a+2*mr;
  for(;un;--un, b+=2*mrs) {
    double *br_; unsigned i;
    for(i=0,br_=br;i<ns;++i,br_+=2*mr,u+=nr)
      lob_bnd_fst(br_, zr,Qr,hr,lb_r,nr,mr, u);
    lob_bnd_ext(b, zs,Qs,hs,lbnp_s,ns,ms, br,mr, a);
  }
}

/* work holds 2*mr*ms + 2*nt*ms*mr doubles */
void lob_bnd_lin_3(
  double *restrict b,
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *lob_bnd_data_t, unsigned nt, unsigned mt,
  const double *restrict u, uint un, double *restrict work)
{
  unsigned nst=ns*nt, mrst=mr*ms*mt, mrs=mr*ms, mr_ns=mr*ns;
  const double *zr=lob_bnd_data_r,*Qr=zr+nr,*hr=Qr+2*nr,*lb_r=hr+mr;
  const double *zs=lob_bnd_data_s,*Qs=zs+ns,*hs=Qs+2*ns,*lbnp_s=hs+ms+2*ns*ms;
  const double *zt=lob_bnd_data_t,*Qt=zt+nt,*ht=Qt+2*nt,*lbnp_t=ht+mt+2*nt*mt;
  double *a = work, *bs = a+2*mr*ms;
  for(;un;--un, b+=2*mrst) {
    double *br_, *bs_; unsigned i;
    for(i=0,br_=b;i<nst;++i,br_+=2*mr,u+=nr)
      lob_bnd_fst(br_, zr,Qr,hr,lb_r,nr,mr, u);
    for(i=0,br_=b,bs_=bs;i<nt;++i,br_+=2*mr_ns,bs_+=2*mrs)
      lob_bnd_ext(bs_, zs,Qs,hs,lbnp_s,ns,ms, br_,mr, a);
    lob_bnd_ext(b, zt,Qt,ht,lbnp_t,nt,mt, bs,mrs, a);
  }
}

static struct dbl_range minmax(const double *restrict b, unsigned m)
{
  struct dbl_range bnd;
  bnd.min = b[0], bnd.max = b[1];
  for(--m,b+=2;m;--m,b+=2)
    bnd.min = b[0]<bnd.min?b[0]:bnd.min,
    bnd.max = b[1]>bnd.max?b[1]:bnd.max;
  return bnd;
}

/* work holds 2*m doubles */
struct dbl_range lob_bnd_1(
  const double *restrict lob_bnd_data, unsigned n, unsigned m,
  const double *restrict u, double *restrict work)
{
  lob_bnd_lin_1(work, lob_bnd_data,n,m, u,1);
  return minmax(work,m);
}

/* work holds 2*mr*ms + 2*mr + 2*mr*ns
             =2*mr*(ms+1+ns) doubles */
struct dbl_range lob_bnd_2(
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *restrict u, double *restrict work)
{
  unsigned m = mr*ms;
  lob_bnd_lin_2(work, lob_bnd_data_r,nr,mr,
                      lob_bnd_data_s,ns,ms, u,1, work+2*m);
  return minmax(work,m);
}

/* work holds 2*mr*ms*mt + 2*mr*ms + 2*nt*ms*mr
             =2*mr*ms*(nt+mt+1) doubles */
struct dbl_range lob_bnd_3(
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *lob_bnd_data_t, unsigned nt, unsigned mt,
  const double *restrict u, double *restrict work)
{
  unsigned m = mr*ms*mt;
  lob_bnd_lin_3(work, lob_bnd_data_r,nr,mr,
                      lob_bnd_data_s,ns,ms,
                      lob_bnd_data_t,nt,mt, u,1, work+2*m);
  return minmax(work,m);
}
