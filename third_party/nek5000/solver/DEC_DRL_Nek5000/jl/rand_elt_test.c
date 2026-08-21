#include <stdlib.h>
#include <math.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "poly.h"
#include "lob_bnd.h"

static double det_2(const double A[4]) { return A[0]*A[3]-A[1]*A[2]; }

static double quad_2(const double x0, const double g[2], const double H[3],
                     const double r[2])
{
  return x0 + (g[0]*r[0]+g[1]*r[1])
            + (  r[0] * (H[0]*r[0]+H[1]*r[1])
               + r[1] * (H[1]*r[0]+H[2]*r[1]) )/2;
}

static void quad_2_grad(double grad[2], const double g[2], const double H[3],
                        const double r[2])
{
  grad[0] = g[0] + (H[0]*r[0]+H[1]*r[1]);
  grad[1] = g[1] + (H[1]*r[0]+H[2]*r[1]);
}

static double quad_2_jac(const double g[4], const double H[6],
                         const double r[2])
{
  double J[4];
  quad_2_grad(J  ,g  ,H  ,r);
  quad_2_grad(J+2,g+2,H+3,r);
  return det_2(J);
}

static double det_3(const double A[9])
{
  const double a = A[4]*A[8]-A[5]*A[7],
               b = A[5]*A[6]-A[3]*A[8],
               c = A[3]*A[7]-A[4]*A[6];
  return A[0]*a+A[1]*b+A[2]*c;
}

static double quad_3(const double x0, const double g[3], const double H[6],
                     const double r[3])
{
  return x0 + (g[0]*r[0]+g[1]*r[1]+g[2]*r[2])
            + (  r[0] * (H[0]*r[0]+H[1]*r[1]+H[2]*r[2])
               + r[1] * (H[1]*r[0]+H[3]*r[1]+H[4]*r[2])
               + r[2] * (H[2]*r[0]+H[4]*r[1]+H[5]*r[2]) )/2;
}

static void quad_3_grad(double grad[3], const double g[3], const double H[6],
                        const double r[3])
{
  grad[0] = g[0] + (H[0]*r[0]+H[1]*r[1]+H[2]*r[2]);
  grad[1] = g[1] + (H[1]*r[0]+H[3]*r[1]+H[4]*r[2]);
  grad[2] = g[2] + (H[2]*r[0]+H[4]*r[1]+H[5]*r[2]);
}

static double quad_3_jac(const double g[9], const double H[18],
                         const double r[3])
{
  double J[9];
  quad_3_grad(J  ,g  ,H   ,r);
  quad_3_grad(J+3,g+3,H+ 6,r);
  quad_3_grad(J+6,g+6,H+12,r);
  return det_3(J);
}

void rand_elt_2(double *x, double *y,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns)
{
  static int init=0;
  static double z4[4], lob_bnd_data[16+3*4*(2*16+1)],
                work[2*16*(4+16+1)];
  unsigned i,j;
  double x0[2], g[4], H[6], jac[4*4], r[2];
  struct dbl_range jr;
  if(!init) {
    init=1;
    lobatto_nodes(z4,4);
    lob_bnd_setup(lob_bnd_data,4,16);
  }
  do {
    for(i=0;i<4;++i) g[i] = -1+2*(rand()/(double)RAND_MAX);
    for(i=0;i<6;++i) H[i] =.5*(-1+2*(rand()/(double)RAND_MAX));
    for(j=0;j<4;++j) { r[1] = z4[j];
      for(i=0;i<4;++i) { r[0] = z4[i];
        jac[j*4+i] = quad_2_jac(g,H,r);
      }
    }
    jr = lob_bnd_2(lob_bnd_data,4,16, lob_bnd_data,4,16, jac, work);
    /*printf("Jacobian range %g, %g\n", jr.min, jr.max);*/
  } while(jr.max*jr.min<=0);
  for(i=0;i< 2;++i) x0[i] = -1+2*(rand()/(double)RAND_MAX);
  for(j=0;j<ns;++j) {   r[1] = zs[j];
    for(i=0;i<nr;++i) { r[0] = zr[i];
      x[j*nr+i] = quad_2(x0[0],g  ,H  ,r);
      y[j*nr+i] = quad_2(x0[1],g+2,H+3,r);
    }
  }
}

void rand_elt_3(double *x, double *y, double *z,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns,
                const double *zt, unsigned nt)
{
  static int init=0;
  static double z4[4], lob_bnd_data[16+3*4*(2*16+1)],
                work[2*16*16*(4+16+1)];
  unsigned i,j,k;
  double x0[3], g[9], H[18], jac[4*4*4], r[3];
  struct dbl_range jr;
  if(!init) {
    init=1;
    lobatto_nodes(z4,4);
    lob_bnd_setup(lob_bnd_data,4,16);
  }
  do {
    for(i=0;i< 9;++i) g[i] = -1+2*(rand()/(double)RAND_MAX);
    for(i=0;i<18;++i) H[i] =.5*(-1+2*(rand()/(double)RAND_MAX));
    for(k=0;k<4;++k) { r[2] = z4[k];
      for(j=0;j<4;++j) { r[1] = z4[j];
        for(i=0;i<4;++i) { r[0] = z4[i];
          jac[(k*4+j)*4+i] = quad_3_jac(g,H,r);
        }
      }
    }
    jr = lob_bnd_3(lob_bnd_data,4,16, lob_bnd_data,4,16, lob_bnd_data,4,16,
                   jac, work);
    /*printf("Jacobian range %g, %g\n", jr.min, jr.max);*/
  } while(jr.max*jr.min<=0);
  for(i=0;i< 3;++i) x0[i] = -1+2*(rand()/(double)RAND_MAX);
  for(k=0;k<nt;++k) {     r[2] = zt[k];
    for(j=0;j<ns;++j) {   r[1] = zs[j];
      for(i=0;i<nr;++i) { r[0] = zr[i];
        x[(k*ns+j)*nr+i] = quad_3(x0[0],g  ,H   ,r);
        y[(k*ns+j)*nr+i] = quad_3(x0[1],g+3,H+ 6,r);
        z[(k*ns+j)*nr+i] = quad_3(x0[2],g+6,H+12,r);
      }
    }
  }
}

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

void bubble_elt(double *x, double *y, double *z,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns,
                const double *zt, unsigned nt, int type)
{
  unsigned i,j,k;
  for(k=0;k<nt;++k) for(j=0;j<ns;++j) for(i=0;i<nr;++i) {
    double dx=0,dy=0,dz=0;
    switch(type) {
      case 0: dx =  cos(PI*zs[j]/2)*cos(PI*zt[k]/2); break;
      case 1: dx = -cos(PI*zs[j]/2)*cos(PI*zt[k]/2); break;
      case 2: dy =  cos(PI*zt[k]/2)*cos(PI*zr[i]/2); break;
      case 3: dy = -cos(PI*zt[k]/2)*cos(PI*zr[i]/2); break;
      case 4: dz =  cos(PI*zr[i]/2)*cos(PI*zs[j]/2); break;
      case 5: dz = -cos(PI*zr[i]/2)*cos(PI*zs[j]/2); break;
    }
    x[(k*ns+j)*nr+i] = zr[i] + dx;
    y[(k*ns+j)*nr+i] = zs[j] + dy;
    z[(k*ns+j)*nr+i] = zt[k] + dz;
  }
}
