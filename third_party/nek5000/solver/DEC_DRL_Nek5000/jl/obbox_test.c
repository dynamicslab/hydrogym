#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "poly.h"
#include "lob_bnd.h"
#include "obbox.h"
#include "rand_elt_test.h"

#define REPEAT 20

#define N 100
#define NR 7
#define MR (4*NR)
#define NS 8
#define MS (4*NS)
#define NT 9
#define MT (4*NT)

#define TOL 0.00001

static const unsigned nr[3]={NR,NS,NT}, mr[3]={MR,MS,MT};

static double zr[NR], zs[NS], zt[NT];
static double x[NR*NS*NT*N], y[NR*NS*NT*N], z[NR*NS*NT*N];
static double tx[3][NR*NS*NT];
static const double *const elx[3]={x,y,z};

static struct obbox_2 ob2[N*NT];
static struct obbox_3 ob3[N];

static struct dbl_range dbl_range_expand(struct dbl_range b, double tol)
{
  double a = (b.min+b.max)/2, l = (b.max-b.min)*(1+tol)/2;
  struct dbl_range m;
  m.min = a-l, m.max = a+l;
  return m;
}

int main()
{
  int failure=0;
  unsigned i;
  
  double *lob_bnd_data_r = tmalloc(double,
      lob_bnd_size(NR,MR)+lob_bnd_size(NS,MS)+lob_bnd_size(NT,MT)),
    *lob_bnd_data_s = lob_bnd_data_r + lob_bnd_size(NR,MR),
    *lob_bnd_data_t = lob_bnd_data_s + lob_bnd_size(NS,MS);

  lobatto_nodes(zr,NR); lob_bnd_setup(lob_bnd_data_r,NR,MR);
  lobatto_nodes(zs,NS); lob_bnd_setup(lob_bnd_data_s,NS,MS);
  lobatto_nodes(zt,NT); lob_bnd_setup(lob_bnd_data_t,NT,MT);

  /* 2-D */
  for(i=0;i<REPEAT;++i) {
    unsigned n; double *x_ = x, *y_ = y;
    for(n=0;n<N && n<6;++n, x_+=NR*NS*NT, y_+=NR*NS*NT)
      bubble_elt(x_,y_,z, zr,NR, zs,NS, zt,NT, n);
    for(n=N-6;n;--n, x_+=NR*NS*NT, y_+=NR*NS*NT)
      rand_elt_2(x_,y_, zr,NR, zs,NS);
    obbox_calc_2(ob2, elx, nr,NT*N, mr, TOL);
    x_=x, y_=y;
    for(n=0;n<N*NT;++n, x_+=NR*NS, y_+=NR*NS) {
      const struct obbox_2 *ob = &ob2[n];
      struct dbl_range xr,yr, tr[2];
      static double work[2*MR*(NS+MS+1)];
      unsigned j;
      for(j=0;j<NR*NS;++j) {
        const double dx=x_[j]-ob->c0[0], dy=y_[j]-ob->c0[1];
        tx[0][j] = ob->A[0]*dx+ob->A[1]*dy;
        tx[1][j] = ob->A[2]*dx+ob->A[3]*dy;
        if(   (x_[j]-ob->x[0].min)*(ob->x[0].max-x_[j]) < 0
           || (y_[j]-ob->x[1].min)*(ob->x[1].max-y_[j]) < 0 )
          failure=1,
          printf("%d %d (%g,%g) not in [%g,%g] x [%g,%g]\n", n, j,
            x_[j],y_[j], ob->x[0].min,ob->x[0].max, ob->x[1].min,ob->x[1].max);
        if(   (tx[0][j]+1)*(1-tx[0][j]) < 0
           || (tx[1][j]+1)*(1-tx[1][j]) < 0 )
          failure=1,
          printf("%d %d (%g,%g) not in [-1,1]^2\n", n, j,
            tx[0][j],tx[1][j]);
        if(failure) break;
      }
      
      xr = dbl_range_expand(lob_bnd_2(lob_bnd_data_r,NR,MR,
                                      lob_bnd_data_s,NS,MS, x_, work), TOL);
      yr = dbl_range_expand(lob_bnd_2(lob_bnd_data_r,NR,MR,
                                      lob_bnd_data_s,NS,MS, y_, work), TOL);

      for(j=0;j<2;++j) tr[j] = dbl_range_expand(
          lob_bnd_2(lob_bnd_data_r,NR,MR, lob_bnd_data_s,NS,MS, tx[j], work)
        , TOL);
        
      if(   ob->x[0].min < xr.min - DBL_EPSILON*128
         || ob->x[0].max > xr.max + DBL_EPSILON*128 ) failure = 1;
      if(   ob->x[1].min < yr.min - DBL_EPSILON*128
         || ob->x[1].max > yr.max + DBL_EPSILON*128 ) failure = 1;
      
      for(j=0;j<2;++j)
        if(   tr[j].min > -1 + DBL_EPSILON*128
           || tr[j].max <  1 - DBL_EPSILON*128 ) failure = 1;
           
      if((i==0&&n==0) || failure) {
        printf("x: [%g,%g] in [%g,%g]\n", ob->x[0].min, ob->x[0].max,
                                          xr.min, xr.max);
        printf("y: [%g,%g] in [%g,%g]\n", ob->x[1].min, ob->x[1].max,
                                          yr.min, yr.max);
        for(j=0;j<2;++j)
          printf("r %d: [%g,%g]\n", j, tr[j].min, tr[j].max);
      }
      if(failure) break;
    }
    if(failure) break;
    printf("."); fflush(stdout);
  }
  printf("\n");

  /* 3-D */
  for(i=0;!failure && i<REPEAT;++i) {
    unsigned n; double *x_ = x, *y_ = y, *z_ = z;
    for(n=0;n<N && n<6;++n, x_+=NR*NS*NT, y_+=NR*NS*NT, z_+=NR*NS*NT)
      bubble_elt(x_,y_,z_, zr,NR, zs,NS, zt,NT, n);
    for(n=N-6;n;--n, x_+=NR*NS*NT, y_+=NR*NS*NT, z_+=NR*NS*NT)
      rand_elt_3(x_,y_,z_, zr,NR, zs,NS, zt,NT);
    obbox_calc_3(ob3, elx, nr,N, mr, TOL);
    x_=x, y_=y, z_=z;
    for(n=0;n<N;++n, x_+=NR*NS*NT, y_+=NR*NS*NT, z_+=NR*NS*NT) {
      const struct obbox_3 *ob = &ob3[n];
      struct dbl_range xr,yr,zr, tr[3];
      static double work[2*MR*MS*(NT+MT+1)];
      unsigned j;
      for(j=0;j<NR*NS*NT;++j) {
        const double dx=x_[j]-ob->c0[0], dy=y_[j]-ob->c0[1], dz=z_[j]-ob->c0[2];
        tx[0][j] = ob->A[0]*dx+ob->A[1]*dy+ob->A[2]*dz;
        tx[1][j] = ob->A[3]*dx+ob->A[4]*dy+ob->A[5]*dz;
        tx[2][j] = ob->A[6]*dx+ob->A[7]*dy+ob->A[8]*dz;
        if(   (x_[j]-ob->x[0].min)*(ob->x[0].max-x_[j]) < 0
           || (y_[j]-ob->x[1].min)*(ob->x[1].max-y_[j]) < 0
           || (z_[j]-ob->x[2].min)*(ob->x[2].max-z_[j]) < 0 )
          failure=1,
          printf("%d %d (%g,%g,%g) not in [%g,%g] x [%g,%g] x [%g,%g]\n", n, j,
            x_[j],y_[j],z_[j], ob->x[0].min,ob->x[0].max,
            ob->x[1].min,ob->x[1].max, ob->x[2].min,ob->x[2].max);
        if(   (tx[0][j]+1)*(1-tx[0][j]) < 0
           || (tx[1][j]+1)*(1-tx[1][j]) < 0
           || (tx[2][j]+1)*(1-tx[2][j]) < 0 )
          failure=1,
          printf("%d %d (%g,%g,%g) not in [-1,1]^3\n", n, j,
            tx[0][j],tx[1][j],tx[2][j]);
        if(failure) break;
      }
      
      xr = dbl_range_expand(lob_bnd_3(lob_bnd_data_r,NR,MR,
                                      lob_bnd_data_s,NS,MS,
                                      lob_bnd_data_t,NT,MT, x_, work), TOL);
      yr = dbl_range_expand(lob_bnd_3(lob_bnd_data_r,NR,MR,
                                      lob_bnd_data_s,NS,MS,
                                      lob_bnd_data_t,NT,MT, y_, work), TOL);
      zr = dbl_range_expand(lob_bnd_3(lob_bnd_data_r,NR,MR,
                                      lob_bnd_data_s,NS,MS,
                                      lob_bnd_data_t,NT,MT, z_, work), TOL);

      for(j=0;j<3;++j) tr[j] = dbl_range_expand(
          lob_bnd_3(lob_bnd_data_r,NR,MR, lob_bnd_data_s,NS,MS,
                    lob_bnd_data_t,NT,MT, tx[j], work)
        , TOL);
        
      if(   ob->x[0].min < xr.min - DBL_EPSILON*128
         || ob->x[0].max > xr.max + DBL_EPSILON*128 ) failure = 1;
      if(   ob->x[1].min < yr.min - DBL_EPSILON*128
         || ob->x[1].max > yr.max + DBL_EPSILON*128 ) failure = 1;
      if(   ob->x[2].min < zr.min - DBL_EPSILON*128
         || ob->x[2].max > zr.max + DBL_EPSILON*128 ) failure = 1;
      
      for(j=0;j<3;++j)
        if(   tr[j].min > -1 + DBL_EPSILON*128
           || tr[j].max <  1 - DBL_EPSILON*128 ) failure = 1;
           
      if((i==0&&n==0) || failure) {
        printf("x: [%g,%g] in [%g,%g]\n", ob->x[0].min, ob->x[0].max,
                                          xr.min, xr.max);
        printf("y: [%g,%g] in [%g,%g]\n", ob->x[1].min, ob->x[1].max,
                                          yr.min, yr.max);
        printf("z: [%g,%g] in [%g,%g]\n", ob->x[2].min, ob->x[2].max,
                                          zr.min, zr.max);
        for(j=0;j<3;++j)
          printf("r %d: [%g,%g]\n", j, tr[j].min, tr[j].max);
      }
      if(failure) break;
    }
    if(failure) break;
    printf("."); fflush(stdout);
  }
  printf("\n");
        
  free(lob_bnd_data_r);
 
  printf("Tests %s\n", failure?"failed":"successful");

  return failure;
}
