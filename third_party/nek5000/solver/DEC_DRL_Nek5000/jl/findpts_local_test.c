#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "types.h"
#include "poly.h"
#include "obbox.h"
#include "findpts_el.h"
#include "findpts_local.h"
#include "rand_elt_test.h"

#define D 3

#if D==3
#define INITD(a,b,c) {a,b,c}
#define MULD(a,b,c) ((a)*(b)*(c))
#define INDEXD(a,na, b,nb, c) (((c)*(nb)+(b))*(na)+(a))
#define findpts_local_data  findpts_local_data_3
#define findpts_local_setup findpts_local_setup_3
#define findpts_local_free  findpts_local_free_3
#define findpts_local       findpts_local_3
#elif D==2
#define INITD(a,b,c) {a,b}
#define MULD(a,b,c) ((a)*(b))
#define INDEXD(a,na, b,nb, c) ((b)*(na)+(a))
#define findpts_local_data  findpts_local_data_2
#define findpts_local_setup findpts_local_setup_2
#define findpts_local_free  findpts_local_free_2
#define findpts_local       findpts_local_2
#endif

#define NR 5
#define NS 8
#define NT 6
#define K 4
#define NEL MULD(K,K,K)
#define TN 4

#define NPT_MAX 256
#define BBOX_TOL 0.01
#define NEWT_TOL 1024*DBL_EPSILON
#define MAX_HASH_SIZE NEL*MULD(NR,NS,NT)

/*
#define NPT_MAX 256
#define BBOX_TOL 1.00
#define NEWT_TOL 1024*DBL_EPSILON
#define MAX_HASH_SIZE NEL*3
*/

static const unsigned nr[D] = INITD(NR,NS,NT);
static const unsigned mr[D] = INITD(4*NR,4*NS,4*NT);
static double zr[NR], zs[NS], zt[NT];
static double x3[D][MULD(3,3,3)];
static double mesh[D][NEL*MULD(NR,NS,NT)];
static const double *const elx[D] = INITD(mesh[0],mesh[1],mesh[2]);

static double testx[NEL*MULD(TN,TN,TN)*D];
struct pt_data { double r[D], dist2; uint code, el; };
static struct pt_data testp[NEL*MULD(TN,TN,TN)];

static double quad_eval(const double coef[MULD(3,3,3)], const double r[D])
{
  double lr0[D], lr1[D], lr2[D];
  unsigned d;
  for(d=0;d<D;++d) lr0[d]=r[d]*(r[d]-1)/2,
                   lr1[d]=(1+r[d])*(1-r[d]),
                   lr2[d]=r[d]*(r[d]+1)/2;
  #define EVALR(base) ( coef [base   ]*lr0[0] \
                       +coef [base+ 1]*lr1[0] \
                       +coef [base+ 2]*lr2[0] )
  #define EVALS(base) ( EVALR(base   )*lr0[1] \
                       +EVALR(base+ 3)*lr1[1] \
                       +EVALR(base+ 6)*lr2[1] )
  #define EVALT(base) ( EVALS(base   )*lr0[2] \
                       +EVALS(base+ 9)*lr1[2] \
                       +EVALS(base+18)*lr2[2] )
  #if D==2
  #  define EVAL() EVALS(0)
  #elif D==3
  #  define EVAL() EVALT(0)
  #endif
  
  return EVAL();
  
  #undef EVAL
  #undef EVALT
  #undef EVALS
  #undef EVALR
}

static void rand_mesh(void)
{
  const double fac = 1.0/K;
  const double z3[3] = {-1,0,1};
  unsigned ki,kj;
  #if D==3
  unsigned kk;
  rand_elt_3(x3[0],x3[1],x3[2], z3,3,z3,3,z3,3);
  #elif D==2
  rand_elt_2(x3[0],x3[1],       z3,3,z3,3);
  #endif
  #if D==3
  for(kk=0;kk<K;++kk)
  #endif
  for(kj=0;kj<K;++kj) for(ki=0;ki<K;++ki) {
    unsigned off = INDEXD(ki,K, kj,K, kk)*MULD(NR,NS,NT);
    unsigned i,j;
    double r[D], base[D] = INITD(-1+2*fac*ki,-1+2*fac*kj,-1+2*fac*kk);
    #if D==3
    unsigned k;
    for(k=0;k<NT;++k) { r[2] = base[2]+fac*(1+zt[k]);
    #endif
    for(j=0;j<NS;++j) { r[1] = base[1]+fac*(1+zs[j]);
    for(i=0;i<NR;++i) { r[0] = base[0]+fac*(1+zr[i]);
      mesh[0][off+INDEXD(i,NR, j,NS, k)] = quad_eval(x3[0],r);
      mesh[1][off+INDEXD(i,NR, j,NS, k)] = quad_eval(x3[1],r);
      #if D==3
      mesh[2][off+INDEXD(i,NR, j,NS, k)] = quad_eval(x3[2],r);
    }
    #endif
    }}
  }
}

static void test_mesh(void)
{
  const double fac = 1.0/K, step = 2.0/(K*(TN-1));
  unsigned ki,kj;
  #if D==3
  unsigned kk;
  for(kk=0;kk<K;++kk)
  #endif
  for(kj=0;kj<K;++kj) for(ki=0;ki<K;++ki) {
    unsigned off = INDEXD(ki,K, kj,K, kk)*MULD(TN,TN,TN);
    unsigned i,j;
    double r[D], base[D] = INITD(-1+2*fac*ki,-1+2*fac*kj,-1+2*fac*kk);
    #if D==3
    unsigned k;
    for(k=0;k<TN;++k) { r[2] = base[2]+step*k;
    #endif
    for(j=0;j<TN;++j) { r[1] = base[1]+step*j;
    for(i=0;i<TN;++i) { r[0] = base[0]+step*i;
      testx[(off+INDEXD(i,TN, j,TN, k))*D+0] = quad_eval(x3[0],r);
      testx[(off+INDEXD(i,TN, j,TN, k))*D+1] = quad_eval(x3[1],r);
      #if D==3
      testx[(off+INDEXD(i,TN, j,TN, k))*D+2] = quad_eval(x3[2],r);
    }
    #endif
    }}
  }
}

static void print_ptdata(void)
{
  uint i,notfound=0;
  double dist2max=0;
  for(i=0;i<NEL*MULD(TN,TN,TN);++i) {
    printf("code=%u, el=%u, dist2=%g, r=(%.17g,%.17g"
           #if D==3
           ",%.17g"
           #endif
           ")\n",
      testp[i].code,testp[i].el,testp[i].dist2,
      testp[i].r[0],testp[i].r[1]
      #if D==3
      ,testp[i].r[2]
      #endif
      );
    dist2max=testp[i].dist2>dist2max?testp[i].dist2:dist2max;
    if(testp[i].code==2) ++notfound;
  }
  printf("Maximum distance = %g\n%u points not found\n",
         sqrt(dist2max), (unsigned)notfound);
}

static void test(buffer *buf)
{
  const double *const x_base[D]=INITD(testx,testx+1,testx+2);
  const unsigned x_stride[D]=
    INITD(D*sizeof(double),D*sizeof(double),D*sizeof(double));
  struct findpts_local_data fld;
  rand_mesh();
  test_mesh();
  findpts_local_setup(&fld,elx,nr,NEL,mr,BBOX_TOL,MAX_HASH_SIZE,
                      NPT_MAX,NEWT_TOL);
  findpts_local(&testp[0].code , sizeof(struct pt_data),
                &testp[0].el   , sizeof(struct pt_data),
                 testp[0].r    , sizeof(struct pt_data),
                &testp[0].dist2, sizeof(struct pt_data),
                x_base, x_stride,
                NEL*MULD(TN,TN,TN), &fld, buf);
  findpts_local_free(&fld);
  print_ptdata();
}

int main()
{
  buffer buf = null_buffer;
  lobatto_nodes(zr,NR),lobatto_nodes(zs,NS),lobatto_nodes(zt,NT);
  test(&buf);
  buffer_free(&buf);
  return 0;
}
