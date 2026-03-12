#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "poly.h"
#include "findpts_el.h"

#define NR 14
#define NS 7

static const unsigned nr[3]={NR,NS};

static double elx[NR*NS], ely[NR*NS];
static const double *const elx2[2] = {elx,ely};

int main()
{
  int pass=1;
  unsigned i,j;
  double zr[NR], zs[NS];
  struct findpts_el_data_2 fd;
  struct findpts_el_pt_2 *pt;
  findpts_el_setup_2(&fd,nr,NR*NS);
  pt = findpts_el_points_2(&fd);
  
  lobatto_nodes(zr,NR);
  lobatto_nodes(zs,NS);

  for(j=0;j<NS;++j) for(i=0;i<NR;++i)
    elx[j*NR+i] = zr[i],
    ely[j*NR+i] = zs[j];

  findpts_el_start_2(&fd, elx2);

  for(j=0;j<NS;++j) for(i=0;i<NR;++i) {
    struct findpts_el_pt_2 *p = pt + j*NR+i;
    p->x[0] = zr[i]*2, p->x[1] = zs[j]*2;
    p->r[0] = 0, p->r[1] = 0;
    p->flags = 0;
  }

  findpts_el_2(&fd, NR*NS, 1024*DBL_EPSILON);

  for(j=0;j<NS;++j) for(i=0;i<NR;++i) {
    double r,s;
    struct findpts_el_pt_2 *p = pt + j*NR+i;
    printf("x = (%g,%g), r = (%g,%g), flags = %x, dist2 = %g\n",
      p->x[0],p->x[1], p->r[0],p->r[1],
      p->flags, p->dist2);
    #define CLAMP(x,r) \
      do { double temp=r; x = temp<-1?-1:(temp>1?1:temp); } while(0)
    CLAMP(r,zr[i]*2); CLAMP(s,zs[j]*2);
    #undef CLAMP
    if( fabs(r-p->r[0])+fabs(s-p->r[1]) > 1024*DBL_EPSILON )
      { printf("off by %g\n", fabs(r-p->r[0])+fabs(s-p->r[1]));
        pass=0; goto fin; }
  }

fin:
  
  findpts_el_free_2(&fd);

  printf("Tests %s\n", pass?"passed":"failed");

  return 0;
}
