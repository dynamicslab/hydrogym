#include <stddef.h>
#include <stdlib.h>
#include <math.h>    /* for cos, fabs */
#include <float.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "mem.h"

#define lagrange_size  PREFIXED_NAME(lagrange_size )
#define lagrange_setup PREFIXED_NAME(lagrange_setup)
#define gauss_nodes    PREFIXED_NAME(gauss_nodes   )
#define gauss_quad     PREFIXED_NAME(gauss_quad    )
#define lobatto_nodes  PREFIXED_NAME(lobatto_nodes )
#define lobatto_quad   PREFIXED_NAME(lobatto_quad  )
#define gll_lag_size   PREFIXED_NAME(gll_lag_size  )
#define gll_lag_setup  PREFIXED_NAME(gll_lag_setup )

typedef void lagrange_fun(double *restrict p,
  double *restrict data, unsigned n, int d, double x);

#include "poly_imp.h"

static void lagrange_eval(double *restrict p,
                          double *restrict data, unsigned n, int der, double x)
{{
  unsigned i;
  const double *restrict z=data, *restrict w=z+n;
  double *restrict d=data+2*n, *restrict u0=d+n, *restrict v0=u0+n;
  for(i=0;i<n;++i) d[i]=2*(x-z[i]);
  u0[0  ]=1; for(i=0  ;i<n-1;++i) u0[i+1]=u0[i]*d[i];
  v0[n-1]=1; for(i=n-1;i    ;--i) v0[i-1]=d[i]*v0[i];
  for(i=0;i<n;++i) p[i]=w[i]*u0[i]*v0[i];
  if(der>0) {
    double *restrict p1 = p+n, *restrict u1=v0+n, *restrict v1=u1+n;
    u1[0  ]=0; for(i=0  ;i<n-1;++i) u1[i+1]=u1[i]*d[i]+u0[i];
    v1[n-1]=0; for(i=n-1;i    ;--i) v1[i-1]=d[i]*v1[i]+v0[i];
    for(i=0;i<n;++i) p1[i]=2*w[i]*(u1[i]*v0[i]+u0[i]*v1[i]);
    if(der>1) {
      double *restrict p2 = p1+n, *restrict u2=v1+n, *restrict v2=u2+n;
      u2[0  ]=0; for(i=0  ;i<n-1;++i) u2[i+1]=u2[i]*d[i]+2*u1[i];
      v2[n-1]=0; for(i=n-1;i    ;--i) v2[i-1]=d[i]*v2[i]+2*v1[i];
      for(i=0;i<n;++i)
        p2[i]=4*w[i]*(u2[i]*v0[i]+2*u1[i]*v1[i]+u0[i]*v2[i]);
    }
  }
}}

static void lagrange_coef(
  double *restrict p, double *data, double *w, const double *z,
  unsigned n, lagrange_fun *lag_eval)
{
  unsigned i;
  for(i=0;i<n;++i) w[i]=1;
  for(i=0;i<n;++i) lag_eval(p,data,n,0,z[i]), w[i]=1/p[i];
}

unsigned lagrange_size(unsigned n)
{
  return 9*n;
}

lagrange_fun *lagrange_setup(
  double *restrict data, const double *restrict z, unsigned n)
{
  double *restrict p = tmalloc(double,n);
  memcpy(data,z,n*sizeof(double));
  lagrange_coef(p,data,data+n,z,n,&lagrange_eval);
  free(p);
  return &lagrange_eval;
}

#define EPS   (128*DBL_EPSILON)
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

/* 
  For brevity's sake, some names have been shortened
  Quadrature rules
    Gauss   -> Gauss-Legendre quadrature (open)
    Lobatto -> Gauss-Lobatto-Legendre quadrature (closed at both ends)
  Polynomial bases
    Legendre -> Legendre basis
    Gauss    -> Lagrangian basis using Gauss   quadrature nodes
    Lobatto  -> Lagrangian basis using Lobatto quadrature nodes
*/

/*--------------------------------------------------------------------------
   Legendre Polynomial Computation
   compute P_n(x) or P_n'(x) or P_n''(x)
  --------------------------------------------------------------------------*/

/* precondition: n >= 0 */
static double legendre(int n, double x)
{
  double p[2];
  double i, nn=n-0.5; /* avoid int -> double conversions */
  p[0]=1.,p[1]=x;
  for(i=1; i<nn; i+=2) {
    p[0] = ((2*i+1)*x*p[1]- i   *p[0])/(i+1);
    p[1] = ((2*i+3)*x*p[0]-(i+1)*p[1])/(i+2);
  }
  return p[n&1];
}

/* precondition: n > 0 */
static double legendre_d1(int n, double x)
{
  double p[2];
  double i, nn=n-0.5; /* avoid int -> double conversions */
  p[0]=3*x,p[1]=1;
  for(i=2; i<nn; i+=2) {
    p[1] = ((2*i+1)*x*p[0]-(i+1)*p[1])/i;
    p[0] = ((2*i+3)*x*p[1]-(i+2)*p[0])/(i+1);
  }
  return p[n&1];
}

/* precondition: n > 1 */
static double legendre_d2(int n, double x)
{
  double p[2];
  double i, nn=n-0.5; /* avoid int -> double conversions */
  p[0]=3,p[1]=15*x;
  for(i=3; i<nn; i+=2) {
    p[0] = ((2*i+1)*x*p[1]-(i+2)*p[0])/(i-1);
    p[1] = ((2*i+3)*x*p[0]-(i+3)*p[1])/i;
  }
  return p[n&1];
}

/*--------------------------------------------------------------------------
   Quadrature Nodes and Weights Calculation
   compute the n Gauss-Legendre nodes and weights or
           the n Gauss-Lobatto-Legendre nodes and weights
  --------------------------------------------------------------------------*/

/* n nodes */
void gauss_nodes(double *restrict z, int n)
{
  int i,j;
  for(i=0; i<=n/2-1; ++i) {
    double ox, x = cos( (2*n-2*i-1)*(PI/2)/n );
    do {
      ox = x;
      x -= legendre(n,x)/legendre_d1(n,x);
    } while(fabs(x-ox)>-x*EPS);
    z[i] = x - legendre(n,x)/legendre_d1(n,x);
  }
  if(n&1) z[n/2]=0;
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) z[j]=-z[i];
}

/* n inner lobatto nodes (excluding -1,1) */
static void lobatto_nodes_aux(double *restrict z, int n)
{
  int i,j,np=n+1;
  for(i=0; i<=n/2-1; ++i) {
    double ox, x = cos( (n-i)*PI/np );
    do {
      ox = x;
      x -= legendre_d1(np,x)/legendre_d2(np,x);
    } while(fabs(x-ox)>-x*EPS);
    z[i] = x - legendre_d1(np,x)/legendre_d2(np,x);
  }
  if(n&1) z[n/2]=0;
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) z[j]=-z[i];
}

/* n lobatto nodes */
static void lobatto_nodes_n(double *restrict z, int n)
{
  z[0] = -1, z[n-1] = 1;
  lobatto_nodes_aux(&z[1],n-2);
}

static void lobatto_nodes_fix(double *restrict z, int n)
{
  z[0] = -1, z[n-1] = 1;
  if(n&1) z[n/2]=0;
  if(n>=4) {
    const double *restrict gllz = gllz_table[n-4];
    int i,j;
    for(i=1;i<=n/2-1;++i) z[i] = -gllz[i-1];
    for(j=(n+1)/2,i=n/2-1; j<n-1; ++j,--i) z[j] = gllz[i-1];
  }
}

void lobatto_nodes(double *restrict z, int n)
{
  if(n>GLL_LAG_FIX_MAX) lobatto_nodes_n(z,n);
  else if(n>=2) lobatto_nodes_fix(z,n);
}

void gauss_quad(double *restrict z, double *restrict w, int n)
{
  int i,j;
  gauss_nodes(z,n);
  for(i=0; i<=(n-1)/2; ++i) {
    double d = (n+1)*legendre(n+1,z[i]);
    w[i] = 2*(1-z[i]*z[i])/(d*d);
  }
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) w[j]=w[i];
}

void lobatto_quad(double *restrict z, double *restrict w, int n)
{
  int i,j;
  lobatto_nodes(z,n);
  for(i=0; i<=(n-1)/2; ++i) {
    double d = legendre(n-1,z[i]);
    w[i] = 2/((n-1)*n*d*d);
  }
  for(j=(n+1)/2,i=n/2-1; j<n; ++j,--i) w[j]=w[i];
}

unsigned gll_lag_size(unsigned n)
{
  return (n<=GLL_LAG_FIX_MAX?1:9)*n;
}

lagrange_fun *gll_lag_setup(double *restrict data, int n)
{
  double *z, *w, *p;
  lagrange_fun *f;
  if(n<2) return 0;
  p = tmalloc(double,2*n);
  if(n<=GLL_LAG_FIX_MAX)
    f=gll_lag_table[n-2], z=p+n, w=data;
  else
    f=&lagrange_eval, z=data, w=z+n;
  lobatto_nodes(z,n);
  lagrange_coef(p,data,w,z,n,f);
  free(p);
  return f;
}
