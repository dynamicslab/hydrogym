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

#define USE_HW_COUNTER 0

#if USE_HW_COUNTER
DEFINE_HW_COUNTER()
#endif

#define REPEAT 10000

#define  NR 7
#define TNR 8
#define  NS 8
#define TNS 9
#define TNTOT (TNR*TNS)
#define  MR (4*NR)
#define  MS (4*NS)

static const unsigned nr[2] = {NR,NS};

/* #define NPT 1 */
#define NPT 256
/* #define NPT TNR*TNS */

#define TOL 1024*DBL_EPSILON

static double zr[NR], zs[NS];
static double tzr[TNR], tzs[TNS];
static double Jr[NR*TNR],Js[NS*TNS];
static double elx[NR*NS], ely[NR*NS];
static const double *const elxy[2] = {elx,ely};
static double telx[2][TNR*TNS];
static double work[TNR*NS];

int main()
{
  int failure=0, unconv=0;
  unsigned n,i,ie;

#if USE_HW_COUNTER
  unsigned long long tic,toc, tot=0;
#endif

  struct findpts_el_data_2 fd;
  struct findpts_el_pt_2 *pt;
  findpts_el_setup_2(&fd,nr,NPT);
  pt = findpts_el_points_2(&fd);
  
  lobatto_nodes(tzr,TNR), lobatto_nodes(tzs,TNS);
  lobatto_nodes(zr,NR), lobatto_nodes(zs,NS);

  for(i=0;i<TNR;++i) fd.lag[0](Jr+i*NR, fd.lag_data[0], NR, 0, tzr[i]);
  for(i=0;i<TNS;++i) fd.lag[1](Js+i*NS, fd.lag_data[1], NS, 0, tzs[i]);
  for(n=0;n<REPEAT;++n) {
    rand_elt_2(elx,ely, zr,NR, zs,NS);
    tensor_2t(telx[0], Jr,TNR,NR, Js,TNS,NS, elx, work);
    tensor_2t(telx[1], Jr,TNR,NR, Js,TNS,NS, ely, work);
#if USE_HW_COUNTER
    tic = getticks();
#endif
    findpts_el_start_2(&fd, elxy);
    for(i=0;i<TNTOT;) {
      unsigned i0=i;
      ie = i+NPT, ie = ie>TNTOT ? TNTOT : ie;
      for(;i!=ie;++i) {
        struct findpts_el_pt_2 *p = pt+(i-i0);
        const double x=telx[0][i],y=telx[1][i];
        p->x[0]=x,p->x[1]=y;
        p->flags = 0;
      }
      findpts_el_2(&fd, ie-i0, 1024*DBL_EPSILON);
#if !(USE_HW_COUNTER)
      for(i=i0;i!=ie;++i) {
        struct findpts_el_pt_2 *p = pt+(i-i0);
        const double r=tzr[i%TNR], s=tzs[i/TNR];
        if((p->flags&(1u<<4))==0) ++unconv;
        if(fabs(p->r[0]-r)+fabs(p->r[1]-s)>1024*DBL_EPSILON) {
          printf("found (%g,%g) for (%g,%g) ; error (%g,%g)\n",
            p->r[0],p->r[1], r,s, p->r[0]-r,p->r[1]-s);
          printf("(%g,%g) for (%.15g,%.15g) ; dist2 = %g\n",
            p->x[0],p->x[1],
            telx[0][i],telx[1][i],p->dist2);
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

  findpts_el_free_2(&fd);

#if !(USE_HW_COUNTER)
  printf("%u failed points (out of %u)\n", failure, REPEAT*TNTOT);
  printf("%u unconverged points\n", unconv);
#else
  printf("average cycles = %g\n", tot/(double)REPEAT);
#endif

  return failure;
}
