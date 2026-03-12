#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "tensor.h"
#include "poly.h"
#include "lob_bnd.h"


#define RESFAC 4
#define N  12
#define NY 9
#define NZ 4
#define REPEAT 1000000

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923


int main()
{
  int failure=0;
  uint i,r;
  double p[NZ*NY*N];
  double lb[2*(RESFAC*NZ)*(RESFAC*NY)*(RESFAC*N)];
  double work[2*(RESFAC*N)*(RESFAC*NY)*(NZ+1)];

  double *ld_N = tmalloc(double,N+gll_lag_size(N));
  lagrange_fun *lag_N = gll_lag_setup(ld_N+N,N);

  double *ld_NY= tmalloc(double,NY+gll_lag_size(NY));
  lagrange_fun *lag_NY = gll_lag_setup(ld_NY+NY,NY);

  double *ld_NZ= tmalloc(double,NZ+gll_lag_size(NZ));
  lagrange_fun *lag_NZ = gll_lag_setup(ld_NZ+NZ,NZ);
  
  double *lb_N  = tmalloc(double,lob_bnd_size(N ,RESFAC*N ));
  double *lb_NY = tmalloc(double,lob_bnd_size(NY,RESFAC*NY));
  double *lb_NZ = tmalloc(double,lob_bnd_size(NZ,RESFAC*NZ));
  lob_bnd_setup(lb_N , N ,RESFAC*N );
  lob_bnd_setup(lb_NY, NY,RESFAC*NY );
  lob_bnd_setup(lb_NZ, NZ,RESFAC*NZ );
  /*for(i=0;i<NY*N;++i) p[i]=rand()/(double)RAND_MAX;
  
  lob_bnd_lin_1(lb, lb_N,N,RESFAC*N, p,NY); */
  
  /* 1D */
  for(r=0;r<REPEAT;++r) {
    int m = RESFAC*N;
    double x = (rand()/(double)RAND_MAX)*2-1;
    /* x = cos((m-1-j)*PI/(m-1)) */
    int j = -1 + m - 1 - (int) (acos(x) * (m-1) / PI);
    double f = (x - cos((m-1-j)*PI/(m-1))) /
               (cos((m-1-(j+1))*PI/(m-1)) - cos((m-1-j)*PI/(m-1)));

    if(r%256==0) {
      for(i=0;i<NY*N;++i) p[i]=rand()/(double)RAND_MAX;
  
      lob_bnd_lin_1(lb, lb_N,N,RESFAC*N, p,NY);
    }

    if(r<3)
      printf("%g <= %g <= %g,   f = %g\n",
        cos((m-1-j)*PI/(m-1)), x, cos((m-1-(j+1))*PI/(m-1)), f);
    lag_N(ld_N,ld_N+N,N,0,x);
    for(i=0;i<NY;++i) {
      double lo = (1-f)*lb[(i*m+j)*2  ] + f*lb[(i*m+j+1)*2  ],
             up = (1-f)*lb[(i*m+j)*2+1] + f*lb[(i*m+j+1)*2+1],
             px = tensor_dot(ld_N,p+i*N,N);
      if(r<3 || px < lo || up < px)
        printf("p_%02d(%g) = %g in [%g,%g]\n",i,x,px,lo,up);
      if(px<lo || up<px) {failure=1; break;}
    }
    if(i!=NY) break;
  }

  /* x = cos((m-1-j)*PI/(m-1)) */
  #define GET_JF(x) \
    int j##x = -1 + m##x - 1 - (int) (acos(x) * (m##x-1) / PI); \
    double f##x = (x - cos((m##x-1-j##x)*PI/(m##x-1))) / \
                (cos((m##x-1-(j##x+1))*PI/(m##x-1)) \
                 - cos((m##x-1-j##x)*PI/(m##x-1)))

  /* 2D */
  for(r=0;r<REPEAT;++r) {
    int mx = RESFAC*N, my = RESFAC*NY;
    double x = (rand()/(double)RAND_MAX)*2-1,
           y = (rand()/(double)RAND_MAX)*2-1;
    GET_JF(x); GET_JF(y);

    if(r%256==0) {
      for(i=0;i<NZ*NY*N;++i) p[i]=rand()/(double)RAND_MAX;
  
      lob_bnd_lin_2(lb, lb_N,N,mx, lb_NY,NY,my, p,NZ, work);
    }

    if(r<3)
      printf("x: %g <= %g <= %g,   f = %g\n",
        cos((mx-1-jx)*PI/(mx-1)), x, cos((mx-1-(jx+1))*PI/(mx-1)), fx),
      printf("y: %g <= %g <= %g,   f = %g\n",
        cos((my-1-jy)*PI/(my-1)), y, cos((my-1-(jy+1))*PI/(my-1)), fy);
    lag_N (ld_N ,ld_N +N ,N ,0,x);
    lag_NY(ld_NY,ld_NY+NY,NY,0,y);

    for(i=0;i<NZ;++i) {
      double lo = (1-fx)*(1-fy)*lb[((i*mx+jx  )*my+jy  )*2  ]
                +    fx *(1-fy)*lb[((i*mx+jx+1)*my+jy  )*2  ]
                + (1-fx)*   fy *lb[((i*mx+jx  )*my+jy+1)*2  ]
                +    fx *   fy *lb[((i*mx+jx+1)*my+jy+1)*2  ],
             up = (1-fx)*(1-fy)*lb[((i*mx+jx  )*my+jy  )*2+1]
                +    fx *(1-fy)*lb[((i*mx+jx+1)*my+jy  )*2+1]
                + (1-fx)*   fy *lb[((i*mx+jx  )*my+jy+1)*2+1]
                +    fx *   fy *lb[((i*mx+jx+1)*my+jy+1)*2+1],
             pxy = tensor_i2(ld_N,N, ld_NY,NY, p+i*N*NY, work);
      if(r<3 || pxy < lo || up < pxy)
        printf("p_%02d(%g,%g) = %g in [%g,%g]\n",i,x,y,pxy,lo,up);
      if(pxy<lo || up<pxy) {failure=1; break;}
    }
    if(i!=NZ) break;

  }

  /* 3D */
  for(r=0;r<REPEAT;++r) {
    int mx = RESFAC*N, my = RESFAC*NY, mz = RESFAC*NZ;
    double x = (rand()/(double)RAND_MAX)*2-1,
           y = (rand()/(double)RAND_MAX)*2-1,
           z = (rand()/(double)RAND_MAX)*2-1;
    GET_JF(x); GET_JF(y); GET_JF(z);
    if(r%256==0) {
      for(i=0;i<NZ*NY*N;++i) p[i]=rand()/(double)RAND_MAX;
  
      lob_bnd_lin_3(lb, lb_N,N,mx, lb_NY,NY,my, lb_NZ,NZ,mz, p,1, work);
    }

    if(r<3)
      printf("x: %g <= %g <= %g,   f = %g\n",
        cos((mx-1-jx)*PI/(mx-1)), x, cos((mx-1-(jx+1))*PI/(mx-1)), fx),
      printf("y: %g <= %g <= %g,   f = %g\n",
        cos((my-1-jy)*PI/(my-1)), y, cos((my-1-(jy+1))*PI/(my-1)), fy),
      printf("z: %g <= %g <= %g,   f = %g\n",
        cos((mz-1-jz)*PI/(mz-1)), z, cos((mz-1-(jz+1))*PI/(mz-1)), fz);
    lag_N (ld_N ,ld_N +N ,N ,0,x);
    lag_NY(ld_NY,ld_NY+NY,NY,0,y);
    lag_NZ(ld_NZ,ld_NZ+NZ,NZ,0,z);

    {
      double lo = 
                + (1-fx)*(1-fy)*(1-fz)*lb[(((jx  )*my+jy  )*mz+jz  )*2  ]
                +    fx *(1-fy)*(1-fz)*lb[(((jx+1)*my+jy  )*mz+jz  )*2  ]
                + (1-fx)*   fy *(1-fz)*lb[(((jx  )*my+jy+1)*mz+jz  )*2  ]
                +    fx *   fy *(1-fz)*lb[(((jx+1)*my+jy+1)*mz+jz  )*2  ]
                + (1-fx)*(1-fy)*   fz *lb[(((jx  )*my+jy  )*mz+jz+1)*2  ]
                +    fx *(1-fy)*   fz *lb[(((jx+1)*my+jy  )*mz+jz+1)*2  ]
                + (1-fx)*   fy *   fz *lb[(((jx  )*my+jy+1)*mz+jz+1)*2  ]
                +    fx *   fy *   fz *lb[(((jx+1)*my+jy+1)*mz+jz+1)*2  ],
             up =
                + (1-fx)*(1-fy)*(1-fz)*lb[(((jx  )*my+jy  )*mz+jz  )*2+1]
                +    fx *(1-fy)*(1-fz)*lb[(((jx+1)*my+jy  )*mz+jz  )*2+1]
                + (1-fx)*   fy *(1-fz)*lb[(((jx  )*my+jy+1)*mz+jz  )*2+1]
                +    fx *   fy *(1-fz)*lb[(((jx+1)*my+jy+1)*mz+jz  )*2+1]
                + (1-fx)*(1-fy)*   fz *lb[(((jx  )*my+jy  )*mz+jz+1)*2+1]
                +    fx *(1-fy)*   fz *lb[(((jx+1)*my+jy  )*mz+jz+1)*2+1]
                + (1-fx)*   fy *   fz *lb[(((jx  )*my+jy+1)*mz+jz+1)*2+1]
                +    fx *   fy *   fz *lb[(((jx+1)*my+jy+1)*mz+jz+1)*2+1],
             pxyz = tensor_i3(ld_N,N, ld_NY,NY, ld_NZ,NZ, p, work);
      if(r<3 || pxyz < lo || up < pxyz)
        printf("p(%g,%g,%g) = %g in [%g,%g]\n",x,y,z,pxyz,lo,up);
      if(pxyz<lo || up<pxyz) failure=1;
    }
    if(failure) break;

  }
  
  free(lb_NZ), free(lb_NY), free(lb_N), free(ld_NZ), free(ld_NY), free(ld_N);
  
  printf("Tests %s\n", failure?"failed":"successful");

  return failure;
}
