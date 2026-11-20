#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "tensor.h"
#include "poly.h"
#include "lob_bnd.h"
#include "obbox.h"
#include "findpts_el.h"
#include "rand_elt_test.h"
#include "rdtsc.h"

#define USE_HW_COUNTER 1

#if USE_HW_COUNTER
DEFINE_HW_COUNTER()
#endif

#define REPEAT 100

#define  NR 7
#define TNR 8
#define  NS 8
#define TNS 9
#define  NT 9
#define TNT 7
#define TNTOT (TNR*TNS*TNT)
#define  MR (4*NR)
#define  MS (4*NS)
#define  MT (4*NT)

static const unsigned nr[3] = {NR,NS,NT};

/* #define NPT 1 */
#define NPT 256
/* #define NPT TNR*TNS*TNT */

#define TOL 1024*DBL_EPSILON

static double zr[NR], zs[NS], zt[NT];
static double tzr[TNR], tzs[TNS], tzt[TNT];
static double Jr[NR*TNR],Js[NS*TNS],Jt[NT*TNT];
static double elx[NR*NS*NT], ely[NR*NS*NT], elz[NR*NS*NT];
static const double *const elxyz[3] = {elx,ely,elz};
static double telx[3][TNR*TNS*TNT];
static double work[TNR*(NS+TNS)*NT];

int main()
{
  int failure=0;
  unsigned n,i,ie;

#if USE_HW_COUNTER
  unsigned long long tic,toc, tot=0;
#else
  int unconv=0;
#endif

  struct findpts_el_data_3 fd;
  struct findpts_el_pt_3 *pt;
  findpts_el_setup_3(&fd,nr,NPT);
  pt = findpts_el_points_3(&fd);
  
  lobatto_nodes(tzr,TNR), lobatto_nodes(tzs,TNS), lobatto_nodes(tzt,TNT);
  lobatto_nodes(zr,NR), lobatto_nodes(zs,NS), lobatto_nodes(zt,NT);

  for(i=0;i<TNR;++i) fd.lag[0](Jr+i*NR, fd.lag_data[0], NR, 0, tzr[i]);
  for(i=0;i<TNS;++i) fd.lag[1](Js+i*NS, fd.lag_data[1], NS, 0, tzs[i]);
  for(i=0;i<TNT;++i) fd.lag[2](Jt+i*NT, fd.lag_data[2], NT, 0, tzt[i]);
  for(n=0;n<6+REPEAT;++n) {
    if(n<6)
      bubble_elt(elx,ely,elz, zr,NR, zs,NS, zt,NT, n);
    else
      rand_elt_3(elx,ely,elz, zr,NR, zs,NS, zt,NT);
    tensor_3t(telx[0], Jr,TNR,NR, Js,TNS,NS, Jt,TNT,NT, elx, work);
    tensor_3t(telx[1], Jr,TNR,NR, Js,TNS,NS, Jt,TNT,NT, ely, work);
    tensor_3t(telx[2], Jr,TNR,NR, Js,TNS,NS, Jt,TNT,NT, elz, work);
#if USE_HW_COUNTER
    tic = getticks();
#endif
    findpts_el_start_3(&fd, elxyz);
    for(i=0;i<TNTOT;) {
      unsigned i0=i;
      ie = i+NPT, ie = ie>TNTOT ? TNTOT : ie;
      for(;i!=ie;++i) {
        struct findpts_el_pt_3 *p = pt+(i-i0);
        const double x=telx[0][i],y=telx[1][i],z=telx[2][i];
        p->x[0]=x,p->x[1]=y,p->x[2]=z;
        p->flags = 0;
      }
      findpts_el_3(&fd, ie-i0, 1024*DBL_EPSILON);
#if !(USE_HW_COUNTER)
      for(i=i0;i!=ie;++i) {
        struct findpts_el_pt_3 *p = pt+(i-i0);
        const double r=tzr[i%TNR], s=tzs[(i/TNR)%TNS], t=tzt[i/(TNR*TNS)];
        if((p->flags&(1u<<6))==0) ++unconv;
        if(fabs(p->r[0]-r)+fabs(p->r[1]-s)+fabs(p->r[2]-t)>1024*DBL_EPSILON) {
          printf("found (%g,%g,%g) for (%g,%g,%g) ; error (%g,%g,%g)\n",
            p->r[0],p->r[1],p->r[2], r,s,t, p->r[0]-r,p->r[1]-s,p->r[2]-t);
          printf("(%g,%g,%g) for (%.15g,%.15g,%.15g) ; dist2 = %g\n",
            p->x[0],p->x[1],p->x[2],
            telx[0][i],telx[1][i],telx[2][i],p->dist2);
          ++failure;
        }
      }
#endif
    }
#if USE_HW_COUNTER
    toc = getticks();
    printf("element took %llu cycles\n",toc-tic);
    tot+=toc-tic;
#endif
  }

  findpts_el_free_3(&fd);

#if !(USE_HW_COUNTER)
  printf("%u failed points (out of %u)\n", failure, (6+REPEAT)*TNTOT);
  printf("%u unconverged points\n", unconv);
#else
  printf("average cycles = %g\n", tot/(double)(6+REPEAT));
#endif

  return failure;
}
