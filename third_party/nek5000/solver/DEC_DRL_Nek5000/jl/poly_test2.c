#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "poly.h"
#include "rdtsc.h"

#define N 32
#define REPEAT 1000000

#define USE_HW_COUNTER 1

#if USE_HW_COUNTER
DEFINE_HW_COUNTER()
#endif


#define EPS (128*DBL_EPSILON)

static int not_same(double a, double b) {
  return fabs(b-a)/(fabs(b)+fabs(a)) > 16*EPS;
}

/* OLD CODE (reference implemenatoin ) =======================================*/

typedef double real;
#define cosr cos
#define fabsr fabs
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

/* precondition: n >= 0 */
static real legendre(int n, real x)
{
  real p[2];
  int i;
  p[0]=1, p[1]=x;
  for(i=1; i<n; i+=2) {
    p[0] = ((2*i+1)*x*p[1]- i   *p[0])/(i+1);
    p[1] = ((2*i+3)*x*p[0]-(i+1)*p[1])/(i+2);
  }
  return p[n&1];
}

/* precondition: n > 0 */
static real legendre_d1(int n, real x)
{
  real p[2];
  int i;
  p[0]=3*x, p[1]=1;
  for(i=2; i<n; i+=2) {
    p[1] = ((2*i+1)*x*p[0]-(i+1)*p[1])/i;
    p[0] = ((2*i+3)*x*p[1]-(i+2)*p[0])/(i+1);
  }
  return p[n&1];
}

/* precondition: n > 1 */
static real legendre_d2(int n, real x)
{
  real p[2];
  int i;
  p[0]=3, p[1]=15*x;
  for(i=3; i<n; i+=2) {
    p[0] = ((2*i+1)*x*p[1]-(i+2)*p[0])/(i-1);
    p[1] = ((2*i+3)*x*p[0]-(i+3)*p[1])/i;
  }
  return p[n&1];
}

/* n nodes */
static void ref_gauss_nodes(real *z, int n)
{
  int i,j;
  for(i=0; i<=n/2-1; ++i) {
    real ox, x = cosr( (2*n-2*i-1)*(PI/2)/n );
    do {
      ox = x;
      x -= legendre(n,x)/legendre_d1(n,x);
    } while(fabsr(x-ox)>-x*EPS);
    z[i] = x - legendre(n,x)/legendre_d1(n,x);
  }
  if(n&1) z[n/2]=0;
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) z[j]=-z[i];
}

/* n inner lobatto nodes (excluding -1,1) */
static void lobatto_nodes_aux(real *z, int n)
{
  int i,j,np=n+1;
  for(i=0; i<=n/2-1; ++i) {
    real ox, x = cosr( (n-i)*PI/np );
    do {
      ox = x;
      x -= legendre_d1(np,x)/legendre_d2(np,x);
    } while(fabsr(x-ox)>-x*EPS);
    z[i] = x - legendre_d1(np,x)/legendre_d2(np,x);
  }
  if(n&1) z[n/2]=0;
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) z[j]=-z[i];
}

/* n lobatto nodes */
static void ref_lobatto_nodes(real *z, int n)
{
  z[0] = -1, z[n-1] = 1;
  lobatto_nodes_aux(&z[1],n-2);
}

static void ref_gauss_weights(const real *z, real *w, int n)
{
  int i,j;
  for(i=0; i<=(n-1)/2; ++i) {
    real d = (n+1)*legendre(n+1,z[i]);
    w[i] = 2*(1-z[i]*z[i])/(d*d);
  }
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) w[j]=w[i];
}

static void ref_lobatto_weights(const real *z, real *w, int n)
{
  int i,j;
  for(i=0; i<=(n-1)/2; ++i) {
    real d = legendre(n-1,z[i]);
    w[i] = 2/((n-1)*n*d*d);
  }
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) w[j]=w[i];
}

typedef struct {
  unsigned n;                /* number of Lagrange nodes            */
  const real *z;             /* Lagrange nodes (user-supplied)      */
  real *J, *D, *D2;          /* weights for 0th,1st,2nd derivatives */
  real *J_z0, *D_z0, *D2_z0; /* ditto at z[0]   (computed at setup) */
  real *J_zn, *D_zn, *D2_zn; /* ditto at z[n-1] (computed at setup) */
  real *w, *d, *u0, *v0, *u1, *v1, *u2, *v2; /* work data            */
} ref_lagrange_data;

static void ref_lagrange_0(ref_lagrange_data *p, real x)
{
  unsigned i, n=p->n;
  for(i=0  ; i<n  ; ++i) p->d[i] = x-p->z[i];
  for(i=0  ; i<n-1; ++i) p->u0[i+1] = p->d[i]*p->u0[i];
  for(i=n-1; i    ; --i) p->v0[i-1] = p->d[i]*p->v0[i];
  for(i=0  ; i<n  ; ++i) p->J[i] = p->w[i]*p->u0[i]*p->v0[i];
}

static void ref_lagrange_1(ref_lagrange_data *p, real x)
{
  unsigned i, n=p->n;
  for(i=0  ; i<n  ; ++i) p->d[i] = x-p->z[i];
  for(i=0  ; i<n-1; ++i)
    p->u0[i+1] = p->d[i]*p->u0[i],
    p->u1[i+1] = p->d[i]*p->u1[i] + p->u0[i];
  for(i=n-1; i    ; --i)
    p->v0[i-1] = p->d[i]*p->v0[i],
    p->v1[i-1] = p->d[i]*p->v1[i] + p->v0[i];
  for(i=0  ; i<n  ; ++i)
    p->J[i] = p->w[i]*p->u0[i]*p->v0[i],
    p->D[i] = p->w[i]*(p->u1[i]*p->v0[i]+p->u0[i]*p->v1[i]);
}

static void ref_lagrange_2(ref_lagrange_data *p, real x)
{
  unsigned i,n=p->n;
  for(i=0  ; i<n  ; ++i) p->d[i] = x-p->z[i];
  for(i=0  ; i<n-1; ++i)
    p->u0[i+1]=p->d[i]*p->u0[i],
    p->u1[i+1]=p->d[i]*p->u1[i]+p->u0[i],
    p->u2[i+1]=p->d[i]*p->u2[i]+2*p->u1[i];
  for(i=n-1; i    ; --i)
    p->v0[i-1]=p->d[i]*p->v0[i],
    p->v1[i-1]=p->d[i]*p->v1[i]+p->v0[i],
    p->v2[i-1]=p->d[i]*p->v2[i]+2*p->v1[i];
  for(i=0  ; i<n  ; ++i)
    p->J [i]=p->w[i]*p->u0[i]*p->v0[i],
    p->D [i]=p->w[i]*(p->u1[i]*p->v0[i]+p->u0[i]*p->v1[i]),
    p->D2[i]=p->w[i]*(p->u2[i]*p->v0[i]+2*p->u1[i]*p->v1[i]+p->u0[i]*p->v2[i]);
}

static void ref_lagrange_2u(ref_lagrange_data *p)
{
  unsigned i,n=p->n;
  for(i=0  ; i<n-1; ++i)
    p->u2[i+1]=p->d[i]*p->u2[i]+2*p->u1[i];
  for(i=n-1; i    ; --i)
    p->v2[i-1]=p->d[i]*p->v2[i]+2*p->v1[i];
  for(i=0  ; i<n  ; ++i)
    p->D2[i]=p->w[i]*(p->u2[i]*p->v0[i]+2*p->u1[i]*p->v1[i]+p->u0[i]*p->v2[i]);
}

static void ref_lagrange_setup(ref_lagrange_data *p, const real *z, unsigned n)
{
  unsigned i,j;
  p->n=n, p->z=z;
  p->w = tmalloc(real, 17*n);
  p->d = p->w+n;
  p->J = p->d+n, p->D = p->J+n, p->D2 = p->D+n;
  p->u0=p->D2+n, p->v0=p->u0+n;
  p->u1=p->v0+n, p->v1=p->u1+n;
  p->u2=p->v1+n, p->v2=p->u2+n;
  p->J_z0=p->v2+n, p->D_z0=p->J_z0+n, p->D2_z0=p->D_z0+n;
  p->J_zn=p->D2_z0+n, p->D_zn=p->J_zn+n, p->D2_zn=p->D_zn+n;
  for(i=0; i<n; ++i) {
    real ww = 1, zi = z[i];
    for(j=0; j<i; ++j) ww *= zi-z[j];
    for(++j; j<n; ++j) ww *= zi-z[j];
    p->w[i] = 1/ww;
  }
  p->u0[0] = p->v0[n-1] = 1;
  p->u1[0] = p->v1[n-1] = 0;
  p->u2[0] = p->v2[n-1] = 0;
  ref_lagrange_2(p,z[0  ]); memcpy(p->J_z0,p->J,3*n*sizeof(real));
  ref_lagrange_2(p,z[n-1]); memcpy(p->J_zn,p->J,3*n*sizeof(real));
}

void ref_lagrange_free(ref_lagrange_data *p)
{
  free(p->w);
}

/* TEST CODE (compare new against reference) =================================*/

int main()
{
  uint i,n;
  
#if USE_HW_COUNTER
  {
    int d;
    unsigned long long tic, toc;
    unsigned r;
    #define TIME(t, repeat, what) do { \
      for(r=repeat;r;--r) { what; } \
      tic = getticks(); \
      for(r=repeat;r;--r) { what; } \
      toc = getticks(); \
      t = toc-tic; \
    } while(0)
    
    for(d=0;d<3;++d) for(n=1;n<N;++n) {
      unsigned rep = REPEAT/(n+1)/(d+1);
      unsigned long long tr=0,tt=0;

      double *p, x = rand()/(double)RAND_MAX;
      double z[N], zr[N];
      lagrange_fun *lag;
      ref_lagrange_data rld;
      gauss_nodes(z,n);
      p = tmalloc(double, 3*n+lagrange_size(n));
      lag = lagrange_setup(p+3*n,z,n);
    
      ref_gauss_nodes(zr,n);
      ref_lagrange_setup(&rld,zr,n);

      TIME(tt,rep,lag(p,p+3*n,n,d,x));
      switch(d) {
      case 0: TIME(tr,rep,ref_lagrange_0(&rld,x)); break;
      case 1: TIME(tr,rep,ref_lagrange_1(&rld,x)); break;
      case 2: TIME(tr,rep,(ref_lagrange_1(&rld,x),ref_lagrange_2u(&rld)));
      }
      printf("lagrange_%d_%02d cycles per call: "
             "ref=%g\timp=%g\tspup=%g\n", d, (int)n,
        tr/(double)rep,tt/(double)rep,tr/(double)tt);

      ref_lagrange_free(&rld);
      free(p);
    }

    for(d=0;d<3;++d) for(n=2;n<N;++n) {
      unsigned rep = REPEAT/(n+1)/(d+1);
      unsigned long long tr=0,tt=0;

      double *p, x = rand()/(double)RAND_MAX;
      double zr[N];
      lagrange_fun *lag;
      ref_lagrange_data rld;
      p = tmalloc(double, 3*n+gll_lag_size(n));
      lag = gll_lag_setup(p+3*n,n);
    
      ref_lobatto_nodes(zr,n);
      ref_lagrange_setup(&rld,zr,n);

      TIME(tt,rep,lag(p,p+3*n,n,d,x));
      switch(d) {
      case 0: TIME(tr,rep,ref_lagrange_0(&rld,x)); break;
      case 1: TIME(tr,rep,ref_lagrange_1(&rld,x)); break;
      case 2: TIME(tr,rep,(ref_lagrange_1(&rld,x),ref_lagrange_2u(&rld)));
      }
      printf("gll_lag_%d_%02d cycles per call: "
             "ref=%g\timp=%g\tspup=%g\n", d, (int)n,
        tr/(double)rep,tt/(double)rep,tr/(double)tt);

      ref_lagrange_free(&rld);
      free(p);
    }
  }
#endif

  for(n=1;n<N;++n) {
    double z[N], w[N], zr[N], wr[N];
    gauss_quad(z,w,n);
    ref_gauss_nodes(zr,n);
    ref_gauss_weights(zr,wr,n);
    for(i=0;i<n;++i) if(not_same(z[i],zr[i]) || not_same(w[i],wr[i])) break;
    if(i!=n) break;
  }
  printf("Gauss quadrature tests: %s\n", n==N?"passed":"failed");

  for(n=2;n<N;++n) {
    double z[N], w[N], zr[N], wr[N];
    lobatto_quad(z,w,n);
    ref_lobatto_nodes(zr,n);
    ref_lobatto_weights(zr,wr,n);
    for(i=0;i<n;++i) if(not_same(z[i],zr[i]) || not_same(w[i],wr[i])) break;
    if(i!=n) break;
  }
  printf("Gauss-Lobatto quadrature tests: %s\n", n==N?"passed":"failed");

  for(n=1;n<N;++n) {
    double *p, x = rand()/(double)RAND_MAX;
    double z[N], zr[N];
    lagrange_fun *lag;
    ref_lagrange_data rld;
    gauss_nodes(z,n);
    p = tmalloc(double, 3*n+lagrange_size(n));
    lag = lagrange_setup(p+3*n,z,n);
    lag(p,p+3*n,n,2,x);
    
    ref_gauss_nodes(zr,n);
    ref_lagrange_setup(&rld,zr,n);
    ref_lagrange_1(&rld,x);
    ref_lagrange_2u(&rld);

    for(i=0;i<n;++i) if(not_same(p[    i],rld.J [i])) break; if(i!=n) break;
    for(i=0;i<n;++i) if(not_same(p[  n+i],rld.D [i])) break; if(i!=n) break;
    for(i=0;i<n;++i) if(not_same(p[2*n+i],rld.D2[i])) break; if(i!=n) break;
    
    ref_lagrange_free(&rld);
    free(p);
  }
  printf("Gauss nodal Lagrange basis tests: %s\n", n==N?"passed":"failed");

  for(n=2;n<N;++n) {
    double *p, x = rand()/(double)RAND_MAX;
    double zr[N];
    lagrange_fun *lag;
    ref_lagrange_data rld;
    p = tmalloc(double, 3*n+gll_lag_size(n));
    lag = gll_lag_setup(p+3*n,n);
    lag(p,p+3*n,n,2,x);
    
    ref_lobatto_nodes(zr,n);
    ref_lagrange_setup(&rld,zr,n);
    ref_lagrange_1(&rld,x);
    ref_lagrange_2u(&rld);

    for(i=0;i<n;++i) if(not_same(p[    i],rld.J [i])) break; if(i!=n) break;
    for(i=0;i<n;++i) if(not_same(p[  n+i],rld.D [i])) break; if(i!=n) break;
    for(i=0;i<n;++i) if(not_same(p[2*n+i],rld.D2[i])) break; if(i!=n) break;

    ref_lagrange_free(&rld);
    free(p);
  }
  printf("Gauss-Lobatto nodal Lagrange basis tests: %s\n",
         n==N?"passed":"failed");

  return 0;
}
