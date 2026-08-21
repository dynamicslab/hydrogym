#include <stdio.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "tensor.h"
#include "poly.h"

#define findpts_el_setup_2   PREFIXED_NAME(findpts_el_setup_2)
#define findpts_el_free_2    PREFIXED_NAME(findpts_el_free_2 )
#define findpts_el_2         PREFIXED_NAME(findpts_el_2      )
#define findpts_el_eval_2    PREFIXED_NAME(findpts_el_eval_2 )
/*
#define DIAGNOSTICS_1
#define DIAGNOSTICS_2
*/
#define DIAGNOSTICS_ITERATIONS 0

#if defined(DIAGNOSTICS_1) || defined(DIAGNOSTICS_2) \
    || DIAGNOSTICS_ITERATIONS > 0
#include <stdio.h>
#endif

/* A is row-major */
static void lin_solve_2(double x[2], const double A[4], const double y[2])
{
  const double idet = 1/(A[0]*A[3] - A[1]*A[2]);
  x[0] = idet*(A[3]*y[0] - A[1]*y[1]);
  x[1] = idet*(A[0]*y[1] - A[2]*y[0]);
}

struct findpts_el_pt_2 {
  double x[2],r[2],oldr[2],dist2,dist2p,tr;
  unsigned index,flags;
};

/* the bit structure of flags is CSSRR
   the C bit --- 1<<4 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS is similarly for s constraints
*/

#define CONVERGED_FLAG (1u<<4)
#define FLAG_MASK 0x1fu

static unsigned num_constrained(const unsigned flags)
{
  const unsigned y = flags | flags>>1;
  return (y&1u) + (y>>2 & 1u);
}

static unsigned pt_flags_to_bin_noC(const unsigned flags)
{
  return (flags>>2 & 3u)*3 + (flags & 3u);
}

/* map flags to 9 if the C bit is set,
   else to [0,8] --- the 9 valid configs of SSRR */
static unsigned pt_flags_to_bin(const unsigned flags)
{
  const unsigned mask = 0u - (flags>>4); /* 0 or 0xfff... when converged */
  return (mask & 9u) | (~mask & pt_flags_to_bin_noC(flags));
}

/* assumes x = 0, or 1  */
static unsigned plus_1_mod_2(const unsigned x) { return x^1u; } 

/* assumes x = 1 << i, with i < 4, returns i+1 */
static unsigned which_bit(const unsigned x)
{
  const unsigned y = x&7u;
  return (y-(y>>2)) | ((x-1)&4u);
}

static unsigned edge_index(const unsigned x) { return which_bit(x)-1; }

static unsigned point_index(const unsigned x)
{
  return ((x>>1)&1u) | ((x>>2)&2u);
}

/* extra data

  we need x, dx/dn for each edge
    r: x at 0, nrs - nr,
      4*nr extra for dx/dn
    s: 8*ns extra

*/

struct findpts_el_gedge_2 { const double *x[2], *dxdn[2]; };
struct findpts_el_gpt_2   { double x[2], jac[4], hes[4]; };

struct findpts_el_data_2 {
  unsigned npt_max;
  struct findpts_el_pt_2 *p;

  unsigned n[2];
  double *z[2];
  lagrange_fun *lag[2];
  double *lag_data[2];
  double *wtend[2];
  
  const double *x[2];
  
  unsigned side_init;
  double *sides;
  struct findpts_el_gedge_2 edge[4]; /* R=-1 S; R=1 S; ... */
  struct findpts_el_gpt_2 pt[4];

  double *work;
};

/* work[2*(nr+ns)] */
/* work[4*(nr+ns)] */
/* work[6*(nr+6)] */
/* work[(6+2*(2*nr+ns))*pn] */
/* work[(10+3*n)*pn] */
static unsigned work_size(
  const unsigned nr, const unsigned ns, const unsigned npt_max)
{
  const unsigned n = ns>nr?ns:nr;
  unsigned wsize;
  #define DO_MAX(x) do { const unsigned temp=(x); \
                         wsize=temp>wsize?temp:wsize; } while(0)
  wsize = (6 + 2*(2*nr+ns)) * npt_max;
  DO_MAX(4*(nr+ns));
  DO_MAX(6*(nr+6));
  DO_MAX(npt_max*(10+3*n));
  #undef DO_MAX
  return wsize;
}

void findpts_el_setup_2(struct findpts_el_data_2 *const fd,
                        const unsigned n[2],
                        const unsigned npt_max)
{
  const unsigned nr=n[0], ns=n[1];
  const unsigned tot = 8*ns + 4*nr;
  unsigned d,i, lag_size[2];

  fd->npt_max = npt_max;
  fd->p = tmalloc(struct findpts_el_pt_2, npt_max*2);

  fd->n[0]=nr, fd->n[1]=ns;
  for(d=0;d<2;++d) lag_size[d] = gll_lag_size(fd->n[d]);

  fd->z[0]        = tmalloc(double,lag_size[0]+lag_size[1]
                                   +7*(nr+ns) + tot +
                                   work_size(nr,ns,npt_max));
  fd->z[1]        = fd->z[0]+nr;
  fd->lag_data[0] = fd->z[1]+ns;
  fd->lag_data[1] = fd->lag_data[0]+lag_size[0];
  fd->wtend[0]    = fd->lag_data[1]+lag_size[1];
  fd->wtend[1]    = fd->wtend[0]+6*nr;
  fd->sides       = fd->wtend[1]+6*ns;
  fd->work        = fd->sides + tot;

  fd->side_init = 0;
  
  for(d=0;d<2;++d) {
    double *wt=fd->wtend[d]; unsigned n=fd->n[d];
    lobatto_nodes(fd->z[d],n);
    fd->lag[d] = gll_lag_setup(fd->lag_data[d],n);
    fd->lag[d](wt    , fd->lag_data[d],n,2,-1);
    fd->lag[d](wt+3*n, fd->lag_data[d],n,2, 1);
    
    wt[0]=1; for(i=1;i<n;++i) wt[i]=0;
    wt+=3*n; { for(i=0;i<n-1;++i) wt[i]=0; } wt[i]=1;
  }

  for(d=0;d<2;++d)
    fd->edge[0].x[d]    = fd->sides +    d *ns, \
    fd->edge[0].dxdn[d] = fd->sides + (2+d)*ns, \
    fd->edge[1].x[d]    = fd->sides + (4+d)*ns, \
    fd->edge[1].dxdn[d] = fd->sides + (6+d)*ns; \

  for(d=0;d<2;++d)
    fd->edge[2].x[d] = 0, /* will point to user data */
    fd->edge[2].dxdn[d] = fd->sides + 8*ns +    d *nr,
    fd->edge[3].x[d] = 0, /* will point to user data */
    fd->edge[3].dxdn[d] = fd->sides + 8*ns + (2+d)*nr;
}

void findpts_el_free_2(struct findpts_el_data_2 *const fd)
{
  free(fd->p);
  free(fd->z[0]);
}

typedef void compute_edge_data_fun(struct findpts_el_data_2 *fd);

/* work[2*(nr+ns)] */
static void compute_edge_data_r(struct findpts_el_data_2 *fd)
{
  const unsigned nr = fd->n[0], ns=fd->n[1], nrsm1 = nr*(ns-1);
  unsigned d;
  double *work = fd->work, *out = fd->sides + 8*ns;
  memcpy(work   , fd->wtend[1]+  ns, ns*sizeof(double));
  memcpy(work+ns, fd->wtend[1]+4*ns, ns*sizeof(double));
  for(d=0;d<2;++d) {
    tensor_mxm(work+2*ns,nr, fd->x[d],ns, work,2);
    memcpy(out+   d *nr, work+2*ns      , nr*sizeof(double));
    memcpy(out+(2+d)*nr, work+2*ns+nr   , nr*sizeof(double));
    fd->edge[2].x[d] = fd->x[d];
    fd->edge[3].x[d] = fd->x[d] + nrsm1;
  }
}

/* work[4*(nr+ns)] */
static void compute_edge_data_s(struct findpts_el_data_2 *fd)
{
  const unsigned nr = fd->n[0], ns=fd->n[1];
  unsigned d;
  double *work = fd->work, *out = fd->sides;
  memcpy(work     , fd->wtend[0]     , 2*nr*sizeof(double));
  memcpy(work+2*nr, fd->wtend[0]+3*nr, 2*nr*sizeof(double));
  for(d=0;d<2;++d) {
    tensor_mtxm(work+4*nr,ns, fd->x[d],nr, work,4);
    memcpy(out+   d *ns, work+4*nr     , ns*sizeof(double));
    memcpy(out+(2+d)*ns, work+4*nr+  ns, ns*sizeof(double));
    memcpy(out+(4+d)*ns, work+4*nr+2*ns, ns*sizeof(double));
    memcpy(out+(6+d)*ns, work+4*nr+3*ns, ns*sizeof(double));
  }
}

static const struct findpts_el_gedge_2 *get_edge(
  struct findpts_el_data_2 *fd, unsigned ei)
{
  const unsigned mask = 1u<<(ei/2);
  if((fd->side_init&mask)==0) {
    compute_edge_data_fun *const fun[2] = {
      compute_edge_data_s,
      compute_edge_data_r
    };
    fun[ei/2](fd);
    fd->side_init |= mask;
  }
  return &fd->edge[ei];
}

/* work[6*(nr+6)] */
static void compute_pt_data(struct findpts_el_data_2 *fd)
{
  const unsigned nr = fd->n[0], ns = fd->n[1];
  double *work = fd->work, *work2 = work+6*nr;
  unsigned d,i,j;
  for(d=0;d<2;++d) {
    tensor_mxm(work,nr, fd->x[d],ns, fd->wtend[1],6);
    tensor_mtxm(work2,6, fd->wtend[0],nr, work,6);
    for(j=0;j<2;++j) for(i=0;i<2;++i) {
      fd->pt[2*j+i].x[d]       = work2[6*(3*j+0)+(3*i+0)];
      fd->pt[2*j+i].jac[2*d+0] = work2[6*(3*j+0)+(3*i+1)];
      fd->pt[2*j+i].jac[2*d+1] = work2[6*(3*j+1)+(3*i+0)];
      fd->pt[2*j+i].hes[2*d+0] = work2[6*(3*j+0)+(3*i+2)];
      fd->pt[2*j+i].hes[2*d+1] = work2[6*(3*j+2)+(3*i+0)];
    }
  }
}

static const struct findpts_el_gpt_2 *get_pt(
  struct findpts_el_data_2 *fd, unsigned pi)
{
  if((fd->side_init&4u)==0)
    compute_pt_data(fd), fd->side_init |= 4u;
  return &fd->pt[pi];
}

/* check reduction in objective against prediction, and adjust
   trust region radius (p->tr) accordingly;
   may reject the prior step, returning 1; otherwise returns 0
   sets out->dist2, out->index, out->x, out->oldr in any event,
   leaving out->r, out->dr, out->flags to be set when returning 0 */
static int reject_prior_step_q(struct findpts_el_pt_2 *const out,
                               const double resid[2],
                               const struct findpts_el_pt_2 *const p,
                               const double tol)
{
  const double old_dist2 = p->dist2;
  const double dist2 = resid[0]*resid[0]+resid[1]*resid[1];
  const double decr = old_dist2-dist2;
  const double pred = p->dist2p;
  out->x[0]=p->x[0],out->x[1]=p->x[1];
  out->oldr[0]=p->r[0],out->oldr[1]=p->r[1];
  out->index=p->index;
  out->dist2=dist2;
#ifdef DIAGNOSTICS_2
  printf("Checking prior step:\n"
         "       old r = (%.17g,%.17g), old flags = %x\n"
         "   old_dist2 = %.17g\n"
         "           r = (%.17g,%.17g),     flags = %x\n"
         "       dist2 = %.17g\n"
         "  difference = %.17g\n"
         "   predicted = %.17g\n"
         "         rho = %.17g\n",
         p->oldr[0],p->oldr[1],(p->flags>>5)&FLAG_MASK,old_dist2,
         p->r[0],p->r[1],p->flags&FLAG_MASK,dist2,
         decr, pred, decr/pred);
#endif
  if(decr>= 0.01 * pred) {
    if(decr>= 0.9 * pred) {
      out->tr = p->tr*2;
#ifdef DIAGNOSTICS_2
      printf("  very good iteration; tr -> %g\n", out->tr);
#endif
    } else {
#ifdef DIAGNOSTICS_2
      printf("  good iteration; tr = %g\n", p->tr);
#endif
      out->tr = p->tr;
    }
    return 0;
  } else {
    /* reject step; note: the point will pass through this routine
       again, and we set things up here so it gets classed as a
       "very good iteration" --- this doubles the trust radius,
       which is why we divide by 4 below */
    double v0 = fabs(p->r[0]-p->oldr[0]),
           v1 = fabs(p->r[1]-p->oldr[1]);
    out->tr = (v0>v1?v0:v1)/4;
#ifdef DIAGNOSTICS_2
    printf("  bad iteration; tr -> %g\n", out->tr);
#endif
    out->dist2=old_dist2;
    out->r[0]=p->oldr[0],out->r[1]=p->oldr[1];
    out->flags=p->flags>>5;
    out->dist2p=-DBL_MAX;
    if(pred < dist2*tol) out->flags|=CONVERGED_FLAG;
    return 1;
  }
}

/* minimize ||resid - jac * dr||_2, with |dr| <= tr, |r0+dr|<=1
   (exact solution of trust region problem) */
static void newton_area(struct findpts_el_pt_2 *const out,
                        const double jac[4], const double resid[2],
                        const struct findpts_el_pt_2 *const p, const double tol)
{
  const double tr = p->tr;
  double bnd[4] = { -1,1, -1,1 };
  double r0[2];
  double dr[2], fac;
  unsigned d, mask, flags;
  
  r0[0] = p->r[0], r0[1] = p->r[1];
  
#ifdef DIAGNOSTICS_1
  printf("newton_area:\n");
  printf("  resid = (%g,%g); r^T r / 2 = %g\n",resid[0],resid[1],
         (resid[0]*resid[0]+resid[1]*resid[1])/2);
  printf("  jac = %g\t%g\n"
         "        %g\t%g\n",
         jac[0],jac[1],jac[2],jac[3]);
  printf("  r = (%.17g,%.17g)\n",r0[0],r0[1]);
#endif

  mask = 0xfu;
  for(d=0;d<2;++d) {
    if(r0[d]-tr>-1) bnd[2*d  ]=r0[d]-tr, mask^=1u<<(2*d);
    if(r0[d]+tr< 1) bnd[2*d+1]=r0[d]+tr, mask^=2u<<(2*d);
  }
  
  lin_solve_2(dr, jac,resid);

#ifdef DIAGNOSTICS_1
  printf("  min at r = (%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1]);
#endif

  fac = 1, flags = 0;
  for(d=0;d<2;++d) {
    double nr = r0[d]+dr[d];
    if((nr-bnd[2*d])*(bnd[2*d+1]-nr)>=0) continue;
    if(nr<bnd[2*d]) {
      double f = (bnd[2*d  ]-r0[d])/dr[d];
      if(f<fac) fac=f, flags = 1u<<(2*d);
    } else {
      double f = (bnd[2*d+1]-r0[d])/dr[d];
      if(f<fac) fac=f, flags = 2u<<(2*d);
    }
  }

#ifdef DIAGNOSTICS_1
  printf("  flags = %x, fac = %.15g\n",flags,fac);
#endif
  
  if(flags==0) goto newton_area_fin;
  
  for(d=0;d<2;++d) dr[d]*=fac;

  newton_area_edge: {
    const unsigned ei = edge_index(flags);
    const unsigned dn = ei>>1, de = plus_1_mod_2(dn);
    const double res0 = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]),
                 res1 = resid[1]-(jac[2]*dr[0]+jac[3]*dr[1]);
    /* y = J_u^T res */
    const double y = jac[de]*res0+jac[2+de]*res1;
    /* JtJ = J_u^T J_u */
    const double JtJ = jac[  de]*jac[  de]
                      +jac[2+de]*jac[2+de];
    const double drc = y/JtJ;
    double fac = 1;
    unsigned new_flags = 0;
#ifdef DIAGNOSTICS_1
    printf("  edge %u, de=%u\n",ei,de);
    printf("    r=(%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1]);
    printf("    resid = (%g,%g); r^T r / 2 = %g\n",res0,res1,
           (res[0]*res[0]+res[1]*res[1])/2);
    printf("    min at %.17g\n", r0[de]+dr[de]+drc);
#endif
    {
      const double rz = r0[de]+dr[de], lb=bnd[2*de],ub=bnd[2*de+1];
      const double nr = r0[de]+(dr[de]+drc);
      if((nr-lb)*(ub-nr)<0) {
        if(nr<lb) {
          double f = (lb-rz)/drc;
          if(f<fac) fac=f, new_flags = 1u<<(2*de);
        } else {
          double f = (ub-rz)/drc;
          if(f<fac) fac=f, new_flags = 2u<<(2*de);
        }
      }
    }
#ifdef DIAGNOSTICS_1
    printf("    new_flags = %x, fac = %.17g\n",new_flags,fac);
#endif
    dr[de] += fac*drc;
    flags |= new_flags;
    goto newton_area_relax;
  }

  /* check and possibly relax constraints */
  newton_area_relax: {
    const unsigned old_flags = flags;
    /* res := res_0 - J dr */
    const double res0 = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]),
                 res1 = resid[1]-(jac[2]*dr[0]+jac[3]*dr[1]);
    /* y := J^T res */
    double y[2]; y[0] = jac[0]*res0+jac[2]*res1,
                 y[1] = jac[1]*res0+jac[3]*res1;
    #define SETDR(d) do { \
      unsigned f = flags>>(2*d) & 3u; \
      if(f) dr[d] = bnd[2*d+(f-1)] - r0[d]; \
    } while(0)
    SETDR(0); SETDR(1);
    #undef SETDR
    for(d=0;d<2;++d) {
      unsigned c = flags>>(2*d) & 3u;
      if(c==0) continue;
      else if(dr[d]*y[d]<0) flags &= ~(3u<<(2*d));
#ifdef DIAGNOSTICS_1
      if( (c==1&&dr[d]>0) || (c==2&&dr[d]<0) )
        printf("FAIL! c=%u, dr[d]=%g\n",c,dr[d]);
#endif
    }
#ifdef DIAGNOSTICS_1
    printf("  checking constraints (%x)\n",old_flags);
    printf("    r=(%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1]);
    printf("    resid = (%g,%g); r^T r / 2 = %g\n",res[0],res[1],
           (res[0]*res[0]+res[1]*res[1])/2);
    printf("    relaxed %x -> %x\n",old_flags,flags);
#endif
    if(flags==old_flags) goto newton_area_fin;
    switch(num_constrained(flags)) {
      case 1: goto newton_area_edge;
    }
  }

newton_area_fin:
#ifdef DIAGNOSTICS_1
  {
    const double res[2]={ resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]),
                          resid[1]-(jac[2]*dr[0]+jac[3]*dr[1]) };
    printf("  r=(%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1]);
    printf("  resid = (%g,%g); r^T r / 2 = %g\n",res[0],res[1],
           (res[0]*res[0]+res[1]*res[1])/2);
  }
#endif
  flags &= mask;
  if(fabs(dr[0])+fabs(dr[1]) < tol) flags |= CONVERGED_FLAG;
  {
    const double res0 = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]),
                 res1 = resid[1]-(jac[2]*dr[0]+jac[3]*dr[1]);
    out->dist2p=resid[0]*resid[0]+resid[1]*resid[1]
                -(res0*res0+res1*res1);
  }
  #define SETR(d) do { \
    unsigned f = flags>>(2*d) & 3u; \
    out->r[d] = f==0 ? r0[d]+dr[d] : ( f==1 ? -1 : 1 ); \
  } while(0)
  SETR(0); SETR(1);
  #undef SETR
  out->flags = flags | (p->flags<<5);
}

static void newton_edge(struct findpts_el_pt_2 *const out,
  const double jac[4], const double rhes, const double resid[2],
  const unsigned de, const unsigned dn,
  unsigned flags,
  const struct findpts_el_pt_2 *const p, const double tol)
{
  const double tr = p->tr;
  /* A = J^T J - resid_d H_d */
  const double A = jac[  de]*jac[  de]
                  +jac[2+de]*jac[2+de] - rhes;
  /* y = J^T r */
  const double y = jac[  de]*resid[0]
                  +jac[2+de]*resid[1];

  const double oldr = p->r[de];
  double dr,nr,tdr,tnr;
  double v,tv; unsigned new_flags=0, tnew_flags=0;

#ifdef DIAGNOSTICS_1
  printf("Newton edge %u (dn=%u) flags=%x\n",de,dn,flags);
  printf("  A=%g, y=%g\n",A,y);
  if(A<=0) printf("  A not positive\n");
  printf("  r=(%.17g,%.17g)\n",p->r[0],p->r[1]);
#endif

  #define EVAL(dr) (dr*A-2*y)*dr

  /* if A is not SPD, quadratic model has no minimum */
  if(A>0) {
    dr = y/A, nr = oldr+dr;
    if(fabs(dr)<tr && fabs(nr)<1) { v=EVAL(dr); goto newton_edge_fin; }
  }

  if(( nr=oldr-tr)>-1)  dr=-tr;
  else                  nr=-1,  dr=-1-oldr,  new_flags = flags | 1u<<(2*de);
  v =EVAL( dr);

  if((tnr=oldr+tr)< 1) tdr=tr;
  else                 tnr= 1, tdr= 1-oldr, tnew_flags = flags | 2u<<(2*de);
  tv=EVAL(tdr);
  
  if(tv<v) nr=tnr, dr=tdr, v=tv, new_flags=tnew_flags;

newton_edge_fin:
  /* check convergence */
  if(fabs(dr) < tol) new_flags |= CONVERGED_FLAG;
  out->r[de]=nr;
  out->r[dn]=p->r[dn];
  out->dist2p = -v;
  out->flags = flags | new_flags | (p->flags<<5);
#ifdef DIAGNOSTICS_1
  printf("  new r = (%.17g,%.17g)\n",out->r[0],out->r[1]);
#endif
}

typedef void findpt_fun(
  struct findpts_el_pt_2 *const out,
  struct findpts_el_data_2 *const fd,
  const struct findpts_el_pt_2 *const p, const unsigned pn, const double tol);

/* work[(6+2*(2*nr+ns))*pn] */
static void findpt_area(
  struct findpts_el_pt_2 *const out,
  struct findpts_el_data_2 *const fd,
  const struct findpts_el_pt_2 *const p, const unsigned pn, const double tol)
{
  const unsigned nr=fd->n[0],ns=fd->n[1];
  double *const resid = fd->work, *const jac = resid + 2*pn,
         *const wtr = jac+4*pn, *const wts = wtr+2*nr*pn,
         *const slice = wts+2*ns*pn;
  unsigned i; unsigned d;
  /* evaluate x(r) and jacobian */
  for(i=0;i<pn;++i)
    fd->lag[0](wtr+2*i*nr, fd->lag_data[0], nr, 1, p[i].r[0]);
  for(i=0;i<pn;++i)
    fd->lag[1](wts+2*i*ns, fd->lag_data[1], ns, 1, p[i].r[1]);
  for(d=0;d<2;++d) {
    tensor_mxm(slice,nr, fd->x[d],ns, wts,2*pn);
    for(i=0;i<pn;++i) {
      const double *const wtr_i = wtr+2*i*nr, *const slice_i = slice+2*i*nr;
      double *const jac_i = jac+4*i+2*d;
      resid[2*i+d] = p[i].x[d] - tensor_ig1(jac_i,
        wtr_i,nr, slice_i);
      jac_i[1] = tensor_i1(wtr_i,nr, slice_i+nr);
    }
  }
  /* perform Newton step */
  for(i=0;i<pn;++i) {
    if(reject_prior_step_q(out+i,resid+2*i,p+i,tol)) continue;
    else newton_area(out+i, jac+4*i, resid+2*i, p+i, tol);
  }
}

/* work[(10+3*n)*pn] */
static void findpt_edge(
  struct findpts_el_pt_2 *const out,
  struct findpts_el_data_2 *const fd,
  const struct findpts_el_pt_2 *const p, const unsigned pn, const double tol)
{
  const unsigned pflag = p->flags & FLAG_MASK;
  const unsigned ei = edge_index(pflag);
  const unsigned dn = ei>>1, de = plus_1_mod_2(dn);
  const unsigned n = fd->n[de];
  double *const resid=fd->work, *const jac=resid+2*pn, *const hes=jac+4*pn,
         *const wt = hes+pn, *const slice = wt+3*n*pn;
  const struct findpts_el_gedge_2 *const edge = get_edge(fd,ei);
  unsigned i; unsigned d;

#ifdef DIAGNOSTICS_1
  printf("Edge %u\n",ei);
  printf("  pflag = %u\n",pflag);
  printf("  ei = %u\n",ei);
  printf("  dn, de = %u, %u\n",dn,de);
  printf("  n = %u \n", n);
#endif

  /* evaluate x(r), jacobian, hessian */
  for(i=0;i<pn;++i)
    fd->lag[de](wt+3*i*n, fd->lag_data[de], n, 2, p[i].r[de]);
  for(i=0;i<pn;++i) hes[i]=0;
  for(d=0;d<2;++d) {
    tensor_mtxv(slice,3*pn, wt, edge->x[d],n);
    for(i=0;i<pn;++i) {
      const double *const slice_i = slice+3*i;
      double r;
      resid[2*i+d] = r = p[i].x[d] - slice_i[0];
      jac[4*i+2*d+de] = slice_i[1];
      hes[i] += r * slice_i[2];
    }
  }
  for(i=1;i<pn;++i) memcpy(wt+i*n, wt+3*i*n, n*sizeof(double));
  for(d=0;d<2;++d) {
    tensor_mtxv(slice,pn, wt, edge->dxdn[d],n);
    for(i=0;i<pn;++i) jac[4*i+2*d+dn] = slice[i];
  }
  /* perform Newton step */
  for(i=0;i<pn;++i) {
    double *const resid_i=resid+2*i, *const jac_i=jac+4*i, *const hes_i=hes+i;
    /* check prior step */
    if(!reject_prior_step_q(out+i,resid_i,p+i,tol)) {
      /* check constraint */
      const double steep = resid_i[0] * jac_i[  dn]
                          +resid_i[1] * jac_i[2+dn];
#ifdef DIAGNOSTICS_1
      printf("jacobian = %g\t%g\n"
             "           %g\t%g\n",jac_i[0],jac_i[1],jac_i[2],jac_i[3]);
      printf("resid_i = (%g,%g)\n", resid_i[0],resid_i[1]);
      printf("steep = %g (%s)\n", steep, steep * p[i].r[dn] < 0 ? "in" : "out");
#endif
      if(steep * p[i].r[dn] < 0) /* relax constraint */
        newton_area(out+i, jac_i, resid_i, p+i, tol);
      else
        newton_edge(out+i, jac_i, *hes_i, resid_i, de,dn,pflag, p+i, tol);
    }
  }
}

static void findpt_pt(
  struct findpts_el_pt_2 *const out,
  struct findpts_el_data_2 *const fd,
  const struct findpts_el_pt_2 *const p, const unsigned pn, const double tol)
{
  const unsigned pflag = p->flags & FLAG_MASK;
  const unsigned pi = point_index(pflag);
  const struct findpts_el_gpt_2 *gpt = get_pt(fd,pi);
  const double *const x = gpt->x, *const jac = gpt->jac, *const hes = gpt->hes;
  unsigned i;

#ifdef DIAGNOSTICS_1
  printf("Point %u\n",pi);
  printf("  pflag = %u\n",pflag);
  printf("  pi = %u\n",pi);
#endif

  for(i=0;i<pn;++i) {
    double resid[2], steep[2], sr[2];
    unsigned dn,de;
    resid[0] = p[i].x[0]-x[0],
    resid[1] = p[i].x[1]-x[1];
    steep[0] = jac[0]*resid[0] + jac[2]*resid[1],
    steep[1] = jac[1]*resid[0] + jac[3]*resid[1];
    sr[0] = steep[0]*p[i].r[0],
    sr[1] = steep[1]*p[i].r[1];
    /* check prior step */
    if(reject_prior_step_q(out+i,resid,p+i,tol)) continue;
    /* check constraints */
    if(sr[0]<0) {
      if(sr[1]<0) goto findpt_pt_area;
      else { de=0,dn=1; goto findpt_pt_edge; }
    }
    else if(sr[1]<0) { de=1,dn=0; goto findpt_pt_edge; }
    out[i].r[0]=p[i].r[0],out[i].r[1]=p[i].r[1];
    out[i].dist2p=0;
    out[i].flags = pflag | CONVERGED_FLAG;
    continue;
    findpt_pt_area:
      newton_area(out+i, jac,resid, p+i, tol);
      continue;
    findpt_pt_edge: {
      const double rh = resid[0]*hes[de]+resid[1]*hes[2+de];
      newton_edge(out+i, jac,rh,resid, de,dn,
                  pflag&(3u<<(2*dn)), p+i, tol);
    } continue;
  }
}

static void seed(struct findpts_el_data_2 *const fd,
                 struct findpts_el_pt_2 *const pt, const unsigned npt)
{
  struct findpts_el_pt_2 *p, *const pe = pt+npt;
  const unsigned nr=fd->n[0], ns=fd->n[1];
  unsigned i,j, ii=0;
  for(p=pt;p!=pe;++p) p->dist2=DBL_MAX;
  for(j=0;j<ns;++j) {
    const double zs=fd->z[1][j];
    for(i=0;i<nr;++i) {
      const double zr=fd->z[0][i];
      const double x=fd->x[0][ii], y=fd->x[1][ii];
      ++ii;
      for(p=pt;p!=pe;++p) {
        const double dx=p->x[0]-x,dy=p->x[1]-y;
        const double dist2 = dx*dx+dy*dy;
        if(p->dist2<=dist2) continue;
        p->dist2=dist2;
        p->r[0]=zr, p->r[1]=zs;
      }
    }
  }
}

void findpts_el_2(struct findpts_el_data_2 *const fd, const unsigned npt,
                  const double tol)
{
  findpt_fun *const fun[3] = 
    { &findpt_area, &findpt_edge, &findpt_pt };
  struct findpts_el_pt_2 *const pbuf = fd->p, *const pstart = fd->p + npt;
  unsigned nconv = npt;
  unsigned step = 0;
  unsigned count[9] = { 0,0,0, 0,0,0, 0,0,0 } ;
  count[0]=npt;
  seed(fd,pbuf,npt);
  { unsigned i;
    for(i=0;i<npt;++i) {
      pstart[i].x[0]=pbuf[i].x[0];
      pstart[i].x[1]=pbuf[i].x[1];
      pstart[i].r[0]=pbuf[i].r[0];
      pstart[i].r[1]=pbuf[i].r[1];
      pstart[i].index=i,pstart[i].flags=0;
      pstart[i].dist2=DBL_MAX,pstart[i].dist2p=0,pstart[i].tr=1;
    }
  }
  while(nconv && step++ < 50) {
    /* advance each group of points */
    struct findpts_el_pt_2 *p, *const pe=pstart+nconv, *pout; unsigned pn;
    
#if DIAGNOSTICS_ITERATIONS>1
    { unsigned i; 
      printf("findpts_el_2 Newton step (%u), %u unconverged:\n ", step,nconv);
      for(i=0;i<9;++i) printf(" %u",count[i]);
      printf("\n");
    }
#endif

    for(p=pstart,pout=pbuf; p!=pe; p+=pn,pout+=pn) {
      const unsigned pflags = p->flags & FLAG_MASK;
      pn = count[pt_flags_to_bin_noC(pflags)];
      fun[num_constrained(pflags)](pout, fd, p,pn, tol);
    }
    /* group points by contsraints */
    {
      unsigned offset[10] = { 0,0,0, 0,0,0, 0,0,0, 0 };
      struct findpts_el_pt_2 *const pe = pbuf+nconv;
      for(pout=pbuf; pout!=pe; ++pout)
        ++offset[pt_flags_to_bin(pout->flags & FLAG_MASK)];
      {
        unsigned i; unsigned sum=0;
        for(i=0;i<9;++i) {
          unsigned ci=offset[i]; count[i]=ci, offset[i]=sum, sum+=ci;
        }
        nconv = offset[9] = sum; /* last bin is converged; forget it */
      }
      for(pout=pbuf; pout!=pe; ++pout)
        pstart[offset[pt_flags_to_bin(pout->flags & FLAG_MASK)]++] = *pout;
    }
  }
  { struct findpts_el_pt_2 *p, *const pe=pstart+npt;
    for(p=pstart;p!=pe;++p)
      pbuf[p->index]=*p, pbuf[p->index].flags&=FLAG_MASK;
  }
#if DIAGNOSTICS_ITERATIONS
  printf("findpts_el_2 took %u steps\n ", step);
#endif
}

void findpts_el_eval_2(
        double *const out_base, const unsigned out_stride,
  const double *const   r_base, const unsigned   r_stride, const unsigned pn,
  const double *const in, struct findpts_el_data_2 *const fd)
{
  const unsigned nr=fd->n[0],ns=fd->n[1];
  double *const wtr = fd->work, *const wts = wtr+nr*pn,
         *const slice = wts+ns*pn;
  unsigned i; const double *r; double *out;
  for(i=0,r=r_base;i<pn;++i) {
    fd->lag[0](wtr+i*nr, fd->lag_data[0], nr, 0, r[0]);
    fd->lag[1](wts+i*ns, fd->lag_data[1], ns, 0, r[1]);
    r = (const double*)((const char*)r + r_stride);
  }
  
  tensor_mxm(slice,nr, in,ns, wts,pn);
  for(i=0,out=out_base;i<pn;++i) {
    const double *const wtr_i = wtr+i*nr, *const slice_i = slice+i*nr;
    *out = tensor_i1(wtr_i,nr, slice_i);
    out = (double*)((char*)out + out_stride);
  }
}
