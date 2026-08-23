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
#define NT 25

static const unsigned nr[3]={NR,NS,NT};

static double elx[NR*NS*NT], ely[NR*NS*NT], elz[NR*NS*NT];
static const double *const elx3[3] = {elx,ely,elz};

int main()
{
  int pass=1;
  unsigned i,j,k;
  double zr[NR], zs[NS], zt[NT];
  struct findpts_el_data_3 fd;
  struct findpts_el_pt_3 *pt;
  findpts_el_setup_3(&fd,nr,NR*NS*NT);
  pt = findpts_el_points_3(&fd);
  
  lobatto_nodes(zr,NR);
  lobatto_nodes(zs,NS);
  lobatto_nodes(zt,NT);

  for(k=0;k<NT;++k) for(j=0;j<NS;++j) for(i=0;i<NR;++i)
    elx[(k*NS+j)*NR+i] = zr[i],
    ely[(k*NS+j)*NR+i] = zs[j],
    elz[(k*NS+j)*NR+i] = zt[k];

  findpts_el_start_3(&fd, elx3);

  for(k=0;k<NT;++k) for(j=0;j<NS;++j) for(i=0;i<NR;++i) {
    struct findpts_el_pt_3 *p = pt + (k*NS+j)*NR+i;
    p->x[0] = zr[i]*2, p->x[1] = zs[j]*2, p->x[2] = zt[k]*2;
    p->r[0] = 0, p->r[1] = 0, p->r[2] = 0;
    p->flags = 0;
  }

  findpts_el_3(&fd, NR*NS*NT, 1024*DBL_EPSILON);
  /* sort_points(pt,NR*NS*NT); */

  for(k=0;k<NT;++k) for(j=0;j<NS;++j) for(i=0;i<NR;++i) {
    double r,s,t;
    struct findpts_el_pt_3 *p = pt + (k*NS+j)*NR+i;
    printf("x = (%g,%g,%g), r = (%g,%g,%g), flags = %x, dist2 = %g\n",
      p->x[0],p->x[1],p->x[2], p->r[0],p->r[1],p->r[2],
      p->flags, p->dist2);
    #define CLAMP(x,r) \
      do { double temp=r; x = temp<-1?-1:(temp>1?1:temp); } while(0)
    CLAMP(r,zr[i]*2); CLAMP(s,zs[j]*2); CLAMP(t,zt[k]*2);
    #undef CLAMP
    if( fabs(r-p->r[0])+fabs(s-p->r[1])+fabs(t-p->r[2]) > 1024*DBL_EPSILON )
      { printf("off by %g\n", fabs(r-p->r[0])+fabs(s-p->r[1])+fabs(t-p->r[2]));
        pass=0; goto fin; }
  }

fin:
  
  findpts_el_free_3(&fd);

  printf("Tests %s\n", pass?"passed":"failed");

  return 0;
}
