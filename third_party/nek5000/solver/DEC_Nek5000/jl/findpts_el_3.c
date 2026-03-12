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

#define findpts_el_setup_3   PREFIXED_NAME(findpts_el_setup_3)
#define findpts_el_free_3    PREFIXED_NAME(findpts_el_free_3 )
#define findpts_el_3         PREFIXED_NAME(findpts_el_3      )
#define findpts_el_eval_3    PREFIXED_NAME(findpts_el_eval_3 )
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
static void lin_solve_3(double x[3], const double A[9], const double y[3])
{
  const double a = A[4]*A[8]-A[5]*A[7],
               b = A[5]*A[6]-A[3]*A[8],
               c = A[3]*A[7]-A[4]*A[6],
            idet = 1/(A[0]*a+A[1]*b+A[2]*c);
  const double
    inv0 = a,
    inv1 = A[2]*A[7]-A[1]*A[8],
    inv2 = A[1]*A[5]-A[2]*A[4],
    inv3 = b,
    inv4 = A[0]*A[8]-A[2]*A[6],
    inv5 = A[2]*A[3]-A[0]*A[5],
    inv6 = c,
    inv7 = A[1]*A[6]-A[0]*A[7],
    inv8 = A[0]*A[4]-A[1]*A[3];
  x[0] = idet*(inv0*y[0] + inv1*y[1] + inv2*y[2]);
  x[1] = idet*(inv3*y[0] + inv4*y[1] + inv5*y[2]);
  x[2] = idet*(inv6*y[0] + inv7*y[1] + inv8*y[2]);
}

static void lin_solve_sym_2(double x[2], const double A[3], const double y[2])
{
  const double idet = 1/(A[0]*A[2] - A[1]*A[1]);
  x[0] = idet*(A[2]*y[0] - A[1]*y[1]);
  x[1] = idet*(A[0]*y[1] - A[1]*y[0]);
}


struct findpts_el_pt_3 {
  double x[3],r[3],oldr[3],dist2,dist2p,tr;
  unsigned index,flags;
};

/* the bit structure of flags is CTTSSRR
   the C bit --- 1<<6 --- is set when the point is converged
   RR is 0 = 00b if r is unconstrained,
         1 = 01b if r is constrained at -1
         2 = 10b if r is constrained at +1
   SS, TT are similarly for s and t constraints
*/

#define CONVERGED_FLAG (1u<<6)
#define FLAG_MASK 0x7fu

static unsigned num_constrained(const unsigned flags)
{
  const unsigned y = flags | flags>>1;
  return (y&1u) + (y>>2 & 1u) + (y>>4 & 1u);
}

static unsigned pt_flags_to_bin_noC(const unsigned flags)
{
  return ((flags>>4 & 3u)*3 + (flags>>2 & 3u))*3 + (flags & 3u);
}

/* map flags to 27 if the C bit is set,
   else to [0,26] --- the 27 valid configs of TTSSRR */
static unsigned pt_flags_to_bin(const unsigned flags)
{
  const unsigned mask = 0u - (flags>>6); /* 0 or 0xfff... when converged */
  return (mask & 27u) | (~mask & pt_flags_to_bin_noC(flags));
}

/* assumes x = 0, 1, or 2 */
static unsigned plus_1_mod_3(const unsigned x) { return ((x | x>>1)+1) & 3u; } 
static unsigned plus_2_mod_3(const unsigned x)
{
  const unsigned y = (x-1) & 3u;
  return y ^ (y>>1);
}

/* assumes x = 1 << i, with i < 6, returns i+1 */
static unsigned which_bit(const unsigned x)
{
  const unsigned y = x&7u;
  return (y-(y>>2)) | ((x-1)&4u) | (x>>4);
}

static unsigned face_index(const unsigned x) { return which_bit(x)-1; }

static unsigned edge_index(const unsigned x)
{
  const unsigned y = ~((x>>1) | x);
  const unsigned RTSR = ((x>>1)&1u) | ((x>>2)&2u) | ((x>>3)&4u) | ((x<<2)&8u);
  const unsigned re = RTSR>>1;
  const unsigned se = 4u | RTSR>>2;
  const unsigned te = 8u | (RTSR&3u);
  return   ( (0u - ( y    &1u)) & re )
         | ( (0u - ((y>>2)&1u)) & se )
         | ( (0u - ((y>>4)&1u)) & te );
}

static unsigned point_index(const unsigned x)
{
  return ((x>>1)&1u) | ((x>>2)&2u) | ((x>>3)&4u);
}

/* extra data

  we need x, dx/dn for each face
    rs: x at 0, nrst - nrs,
      6*nrs extra for dx/dn
    st: 12*nst extra
    tr: 12*ntr extra
      (transposed order for embedded t-edges)

  for each edge,
    have x, dx/dn2 already as part of face data
    need dx/dn1 (strided in face data)
    need d^2x/dn1^2, d^2x/dn2^2 possibly, if constraints relax
      thats 3*4*(nr+ns+nt) extra

*/

struct findpts_el_gface_3 { const double *x[3], *dxdn[3]; };
struct findpts_el_gedge_3 { const double *x[3], *dxdn1[3], *dxdn2[3],
                                         *d2xdn1[3], *d2xdn2[3]; };
struct findpts_el_gpt_3   { double x[3], jac[9], hes[18]; };

struct findpts_el_data_3 {
  unsigned npt_max;
  struct findpts_el_pt_3 *p;

  unsigned n[3];
  double *z[3];
  lagrange_fun *lag[3];
  double *lag_data[3];
  double *wtend[3];
  
  const double *x[3];
  
  unsigned side_init;
  double *sides;
  struct findpts_el_gface_3 face[6]; /* ST R=-1,R=+1; TR S=-1,S=+1; ... */
  struct findpts_el_gedge_3 edge[12]; /* R S=-1,T=-1; R S=1,T=-1; ... */
  struct findpts_el_gpt_3 pt[8];

  double *work;
};

/* work[2*nt+2*nrs] */
/* work[4*nr+4*nst] */
/* work[4*ns+4*nr] */
/* work[4*n1+4*n], work[2*n2+2*n] */
/* work[4*nr+4], work[2*nt+2] */
/* work[(3+9+2*(nr+ns+nt+nrs))*pn + max(2*nr,ns) ] */
/* work[(3+9+3+3*(n1+n2+n1))*pn ] */
/* work[ 3*n ] */
static unsigned work_size(
  const unsigned nr, const unsigned ns, const unsigned nt,
  const unsigned npt_max)
{
  unsigned n1, n2, wsize;
  if(nr>ns) {
    if(nr>nt) n1=nr, n2 = (ns>nt ? ns : nt);
    else      n1=nt, n2 = nr;
  } else {
    if(ns>nt) n1=ns, n2 = (nr>nt ? nr : nt);
    else      n1=nt, n2 = ns;
  }
  #define DO_MAX(x) do { const unsigned temp=(x); \
                         wsize=temp>wsize?temp:wsize; } while(0)
  wsize = (12 + 2*(nr+ns+nt+nr*ns)) * npt_max + (2*nr>ns?2*nr:ns);
  DO_MAX(2*(nt+nr*ns));
  DO_MAX(4*(nr+ns*nt));
  DO_MAX(4*(n1+n2));
  DO_MAX(npt_max*(15+3*(2*n1+n2)));
  #undef DO_MAX
  return wsize;
}

void findpts_el_setup_3(struct findpts_el_data_3 *const fd,
                        const unsigned n[3],
                        const unsigned npt_max)
{
  const unsigned nr=n[0], ns=n[1], nt=n[2];
  const unsigned nrs = nr*ns, nst=ns*nt, ntr=nt*nr;
  const unsigned face_size = 12*nst + 12*ntr + 6*nrs;
  const unsigned off_es = face_size + 36*nr, off_et = off_es + 36*ns,
                 tot = off_et + 36*nt;
  unsigned d,i, lag_size[3];

  fd->npt_max = npt_max;
  fd->p = tmalloc(struct findpts_el_pt_3, npt_max*2);

  fd->n[0]=nr, fd->n[1]=ns, fd->n[2]=nt;
  for(d=0;d<3;++d) lag_size[d] = gll_lag_size(fd->n[d]);

  fd->z[0]        = tmalloc(double,lag_size[0]+lag_size[1]+lag_size[2]
                                   +7*(nr+ns+nt) + tot +
                                   work_size(nr,ns,nt,npt_max));
  fd->z[1]        = fd->z[0]+nr;
  fd->z[2]        = fd->z[1]+ns;
  fd->lag_data[0] = fd->z[2]+nt;
  fd->lag_data[1] = fd->lag_data[0]+lag_size[0];
  fd->lag_data[2] = fd->lag_data[1]+lag_size[1];
  fd->wtend[0]    = fd->lag_data[2]+lag_size[2];
  fd->wtend[1]    = fd->wtend[0]+6*nr;
  fd->wtend[2]    = fd->wtend[1]+6*ns;
  fd->sides       = fd->wtend[2]+6*nt;
  fd->work        = fd->sides + tot;

  fd->side_init = 0;
  
  for(d=0;d<3;++d) {
    double *wt=fd->wtend[d]; unsigned n=fd->n[d];
    lobatto_nodes(fd->z[d],n);
    fd->lag[d] = gll_lag_setup(fd->lag_data[d],n);
    fd->lag[d](wt    , fd->lag_data[d],n,2,-1);
    fd->lag[d](wt+3*n, fd->lag_data[d],n,2, 1);
    
    wt[0]=1; for(i=1;i<n;++i) wt[i]=0;
    wt+=3*n; { for(i=0;i<n-1;++i) wt[i]=0; } wt[i]=1;
  }

  #define SET_FACE(i,base,n) do { for(d=0;d<3;++d) \
    fd->face[2*i  ].x[d]    = fd->sides + base +    d *n, \
    fd->face[2*i  ].dxdn[d] = fd->sides + base + (3+d)*n, \
    fd->face[2*i+1].x[d]    = fd->sides + base + (6+d)*n, \
    fd->face[2*i+1].dxdn[d] = fd->sides + base + (9+d)*n; \
  } while(0)
  SET_FACE(0,0,nst);
  SET_FACE(1,12*nst,ntr);
  #undef SET_FACE

  for(d=0;d<3;++d)
    fd->face[4].x[d] = 0, /* will point to user data */
    fd->face[4].dxdn[d] = fd->sides + 12*(nst+ntr) + d*nrs,
    fd->face[5].x[d] = 0, /* will point to user data */
    fd->face[5].dxdn[d] = fd->sides + 12*(nst+ntr) + (3+d)*nrs;

  #define SET_EDGE1(j,k,d,rd,rn,base) \
    for(i=0;i<2;++i) \
      fd->edge[4*j+2*i+0].dxdn2[d] = fd->face[2*k+i].dxdn[d], \
      fd->edge[4*j+2*i+1].dxdn2[d] = fd->face[2*k+i].dxdn[d]+n##rd##rn-n##rd;
  #define SET_EDGE2(j,d,rd,rn,base) \
    for(i=0;i<4;++i) \
      fd->edge[4*j+i].dxdn1 [d] = fd->sides + base + (9*i  +d)*n##rd, \
      fd->edge[4*j+i].d2xdn1[d] = fd->sides + base + (9*i+3+d)*n##rd, \
      fd->edge[4*j+i].d2xdn2[d] = fd->sides + base + (9*i+6+d)*n##rd;
  #define SET_EDGE(j,rd,rn,base) do { \
    for(d=0;d<3;++d) { SET_EDGE1(j,plus_2_mod_3(j),d,rd,rn,base); \
                       SET_EDGE2(j,d,rd,rn,base); } \
  } while(0)
  SET_EDGE(0,r,s,face_size);
  SET_EDGE(1,s,t,off_es);
  SET_EDGE(2,t,r,off_et);
  #undef SET_EDGE
  #undef SET_EDGE2
  #undef SET_EDGE1
}

void findpts_el_free_3(struct findpts_el_data_3 *const fd)
{
  free(fd->p);
  free(fd->z[0]);
}

typedef void compute_face_data_fun(struct findpts_el_data_3 *fd);

/* work[2*nt+2*nrs] */
static void compute_face_data_rs(struct findpts_el_data_3 *fd)
{
  const unsigned nr = fd->n[0], ns=fd->n[1], nt=fd->n[2],
                 nrs = nr*ns, nst=ns*nt, ntr = nt*nr, nrstm1 = nrs*(nt-1);
  unsigned d;
  double *work = fd->work, *out = fd->sides + 12*(nst+ntr);
  memcpy(work   , fd->wtend[2]+  nt, nt*sizeof(double));
  memcpy(work+nt, fd->wtend[2]+4*nt, nt*sizeof(double));
  for(d=0;d<3;++d) {
    tensor_mxm(work+2*nt,nrs, fd->x[d],nt, work,2);
    memcpy(out+   d *nrs, work+2*nt       , nrs*sizeof(double));
    memcpy(out+(3+d)*nrs, work+2*nt+nrs   , nrs*sizeof(double));
    fd->face[4].x[d] = fd->x[d];
    fd->face[5].x[d] = fd->x[d] + nrstm1;
  }
}

/* work[4*nr+4*nst] */
static void compute_face_data_st(struct findpts_el_data_3 *fd)
{
  const unsigned nr = fd->n[0], ns=fd->n[1], nt=fd->n[2], nst=ns*nt;
  unsigned i;
  double *work = fd->work, *out = fd->sides;
  memcpy(work     , fd->wtend[0]     , 2*nr*sizeof(double));
  memcpy(work+2*nr, fd->wtend[0]+3*nr, 2*nr*sizeof(double));
  for(i=0;i<3;++i) {
    tensor_mtxm(work+4*nr,nst, fd->x[i],nr, work,4);
    memcpy(out+   i *nst, work+4*nr      , nst*sizeof(double));
    memcpy(out+(3+i)*nst, work+4*nr+  nst, nst*sizeof(double));
    memcpy(out+(6+i)*nst, work+4*nr+2*nst, nst*sizeof(double));
    memcpy(out+(9+i)*nst, work+4*nr+3*nst, nst*sizeof(double));
  }
}

/* work[4*ns+4*nr] */
static void compute_face_data_tr(struct findpts_el_data_3 *fd)
{
  const unsigned nr = fd->n[0], ns=fd->n[1], nt=fd->n[2],
                 nrs = nr*ns, nst=ns*nt, ntr=nt*nr;
  unsigned i,k,d;
  double *work = fd->work, *out = fd->sides + 12*nst;
  memcpy(work     , fd->wtend[1]     , 2*ns*sizeof(double));
  memcpy(work+2*ns, fd->wtend[1]+3*ns, 2*ns*sizeof(double));
  for(d=0;d<3;++d) {
    for(k=0;k<nt;++k) {
      double *outk; double *in = work+4*ns;
      tensor_mxm(in,nr, fd->x[d]+k*nrs,ns, work,4);
      for(outk=out+   d *ntr+k,i=0;i<nr;++i,outk+=nt) *outk=*in++;
      for(outk=out+(3+d)*ntr+k,i=0;i<nr;++i,outk+=nt) *outk=*in++;
      for(outk=out+(6+d)*ntr+k,i=0;i<nr;++i,outk+=nt) *outk=*in++;
      for(outk=out+(9+d)*ntr+k,i=0;i<nr;++i,outk+=nt) *outk=*in++;
    }
  }
}

static const struct findpts_el_gface_3 *get_face(
  struct findpts_el_data_3 *fd, unsigned fi)
{
  const unsigned mask = 1u<<(fi/2);
  if((fd->side_init&mask)==0) {
    compute_face_data_fun *const fun[3] = {
      compute_face_data_st,
      compute_face_data_tr,
      compute_face_data_rs
    };
    fun[fi/2](fd);
    fd->side_init |= mask;
  }
  return &fd->face[fi];
}

/* work[4*n1+4*n], work[2*n2+2*n] */
static void compute_edge_data(struct findpts_el_data_3 *fd, unsigned d)
{
  const unsigned dn1 = plus_1_mod_3(d), dn2 = plus_2_mod_3(d);
  const unsigned n = fd->n[d], n1 = fd->n[dn1], n2 = fd->n[dn2];
  const unsigned nr=fd->n[0],ns=fd->n[1],nt=fd->n[2],
                 nrs=nr*ns,nst=ns*nt,ntr=nt*nr;
  const unsigned base = 6*nrs + 12*nst + 12*ntr
                        + (d>0 ? 36*nr : 0) + (d>1 ? 36*ns : 0);
  #define DXDN1(i,d)  (fd->sides+base+(9*(i)  +(d))*n)
  #define D2XDN1(i,d) (fd->sides+base+(9*(i)+3+(d))*n)
  #define D2XDN2(i,d) (fd->sides+base+(9*(i)+6+(d))*n)
  const struct findpts_el_gface_3 *face_d_n1 = get_face(fd,2*dn2),
                                  *face_n2_d = get_face(fd,2*dn1);
  struct findpts_el_gedge_3 *e = fd->edge + 4*d;
  unsigned i,xd;
  double *work = fd->work;
  for(xd=0;xd<3;++xd) for(i=0;i<2;++i)
    e[2*i  ].x[xd] = face_d_n1[i].x[xd],
    e[2*i+1].x[xd] = face_d_n1[i].x[xd]+n*(n1-1);
  memcpy(work     , fd->wtend[dn1]+  n1,2*n1*sizeof(double));
  memcpy(work+2*n1, fd->wtend[dn1]+4*n1,2*n1*sizeof(double));
  for(i=0;i<2;++i) for(xd=0;xd<3;++xd) {
    tensor_mxm(work+4*n1,n, face_d_n1[i].x[xd],n1, work,4);
    memcpy( DXDN1(2*i+0,xd), work+4*n1    , n*sizeof(double));
    memcpy(D2XDN1(2*i+0,xd), work+4*n1+  n, n*sizeof(double));
    memcpy( DXDN1(2*i+1,xd), work+4*n1+2*n, n*sizeof(double));
    memcpy(D2XDN1(2*i+1,xd), work+4*n1+3*n, n*sizeof(double));
  }
  memcpy(work   , fd->wtend[dn2]+2*n2,n2*sizeof(double));
  memcpy(work+n2, fd->wtend[dn2]+5*n2,n2*sizeof(double));
  for(i=0;i<2;++i) for(xd=0;xd<3;++xd) {
    tensor_mtxm(work+2*n2,n, face_n2_d[i].x[xd],n2, work,2);
    memcpy(D2XDN2(  i,xd), work+2*n2  , n*sizeof(double));
    memcpy(D2XDN2(2+i,xd), work+2*n2+n, n*sizeof(double));
  }
  #undef D2XDN2
  #undef D2XDN1
  #undef DXDN1
}

static const struct findpts_el_gedge_3 *get_edge(
  struct findpts_el_data_3 *fd, unsigned ei)
{
  const unsigned mask = 8u<<(ei/4);
  if((fd->side_init&mask)==0)
    compute_edge_data(fd,ei/4), fd->side_init |= mask;
  return &fd->edge[ei];
}

/* work[4*nr+4], work[2*nt+2] */
static void compute_pt_data(struct findpts_el_data_3 *fd)
{
  const unsigned nr = fd->n[0], nt = fd->n[2];
  const struct findpts_el_gedge_3 *e = get_edge(fd,0);
  unsigned d,i;
  double *work = fd->work;
  for(i=0;i<4;++i) for(d=0;d<3;++d)
    fd->pt[2*i  ].x[d] = e[i].x[d][0],
    fd->pt[2*i  ].jac[3*d+1] = e[i].dxdn1[d][0],
    fd->pt[2*i  ].jac[3*d+2] = e[i].dxdn2[d][0],
    fd->pt[2*i  ].hes[6*d+3] = e[i].d2xdn1[d][0],
    fd->pt[2*i  ].hes[6*d+5] = e[i].d2xdn2[d][0],
    fd->pt[2*i+1].x[d] = e[i].x[d][nr-1],
    fd->pt[2*i+1].jac[3*d+1] = e[i].dxdn1[d][nr-1],
    fd->pt[2*i+1].jac[3*d+2] = e[i].dxdn2[d][nr-1],
    fd->pt[2*i+1].hes[6*d+3] = e[i].d2xdn1[d][nr-1],
    fd->pt[2*i+1].hes[6*d+5] = e[i].d2xdn2[d][nr-1];
  memcpy(work     , fd->wtend[0]+  nr, 2*nr*sizeof(double));
  memcpy(work+2*nr, fd->wtend[0]+4*nr, 2*nr*sizeof(double));
  for(i=0;i<4;++i) for(d=0;d<3;++d) {
    tensor_mtxv(work+4*nr,4, work, e[i].x[d],nr);
    fd->pt[2*i  ].jac[3*d  ] = work[4*nr  ];
    fd->pt[2*i  ].hes[6*d  ] = work[4*nr+1];
    fd->pt[2*i+1].jac[3*d  ] = work[4*nr+2];
    fd->pt[2*i+1].hes[6*d  ] = work[4*nr+3];
  }
  memcpy(work+nr,work+2*nr,nr*sizeof(double));
  for(i=0;i<4;++i) for(d=0;d<3;++d) {
    tensor_mtxv(work+2*nr,2, work, e[i].dxdn1[d],nr);
    fd->pt[2*i  ].hes[6*d+1] = work[2*nr  ];
    fd->pt[2*i+1].hes[6*d+1] = work[2*nr+1];
    tensor_mtxv(work+2*nr,2, work, e[i].dxdn2[d],nr);
    fd->pt[2*i  ].hes[6*d+2] = work[2*nr  ];
    fd->pt[2*i+1].hes[6*d+2] = work[2*nr+1];
  }
  e = get_edge(fd,8);
  memcpy(work   , fd->wtend[2]+  nt, nt*sizeof(double));
  memcpy(work+nt, fd->wtend[2]+4*nt, nt*sizeof(double));
  for(i=0;i<4;++i) for(d=0;d<3;++d) {
    tensor_mtxv(work+2*nt,2, work, e[i].dxdn2[d],nt);
    fd->pt[  i].hes[6*d+4] = work[2*nt  ];
    fd->pt[4+i].hes[6*d+4] = work[2*nt+1];
  }
}

static const struct findpts_el_gpt_3 *get_pt(
  struct findpts_el_data_3 *fd, unsigned pi)
{
  if((fd->side_init&0x40u)==0)
    compute_pt_data(fd), fd->side_init |= 0x40u;
  return &fd->pt[pi];
}

/* check reduction in objective against prediction, and adjust
   trust region radius (p->tr) accordingly;
   may reject the prior step, returning 1; otherwise returns 0
   sets out->dist2, out->index, out->x, out->oldr in any event,
   leaving out->r, out->dr, out->flags to be set when returning 0 */
static int reject_prior_step_q(struct findpts_el_pt_3 *const out,
                               const double resid[3],
                               const struct findpts_el_pt_3 *const p,
                               const double tol)
{
  const double old_dist2 = p->dist2;
  const double dist2 = resid[0]*resid[0]+resid[1]*resid[1]+resid[2]*resid[2];
  const double decr = old_dist2-dist2;
  const double pred = p->dist2p;
  out->x[0]=p->x[0],out->x[1]=p->x[1],out->x[2]=p->x[2];
  out->oldr[0]=p->r[0],out->oldr[1]=p->r[1],out->oldr[2]=p->r[2];
  out->index=p->index;
  out->dist2=dist2;
#ifdef DIAGNOSTICS_2
  printf("Checking prior step:\n"
         "       old r = (%.17g,%.17g,%.17g), old flags = %x\n"
         "   old_dist2 = %.17g\n"
         "           r = (%.17g,%.17g,%.17g),     flags = %x\n"
         "       dist2 = %.17g\n"
         "  difference = %.17g\n"
         "   predicted = %.17g\n"
         "         rho = %.17g\n",
         p->oldr[0],p->oldr[1],p->oldr[2],(p->flags>>7)&FLAG_MASK,old_dist2,
         p->r[0],p->r[1],p->r[2],p->flags&FLAG_MASK,dist2,
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
           v1 = fabs(p->r[1]-p->oldr[1]),
           v2 = fabs(p->r[2]-p->oldr[2]);
    out->tr = (v1>v2?(v0>v1?v0:v1):(v0>v2?v0:v2))/4;
#ifdef DIAGNOSTICS_2
    printf("  bad iteration; tr -> %g\n", out->tr);
#endif
    out->dist2=old_dist2;
    out->r[0]=p->oldr[0],out->r[1]=p->oldr[1],out->r[2]=p->oldr[2];
    out->flags=p->flags>>7;
    out->dist2p=-DBL_MAX;
    if(pred < dist2*tol) out->flags|=CONVERGED_FLAG;
    return 1;
  }
}

/* minimize ||resid - jac * dr||_2, with |dr| <= tr, |r0+dr|<=1
   (exact solution of trust region problem) */
static void newton_vol(struct findpts_el_pt_3 *const out,
                       const double jac[9], const double resid[3],
                       const struct findpts_el_pt_3 *const p, const double tol)
{
  const double tr = p->tr;
  double bnd[6] = { -1,1, -1,1, -1,1 };
  double r0[3];
  double dr[3], fac;
  unsigned d, mask, flags;
  r0[0]=p->r[0],r0[1]=p->r[1],r0[2]=p->r[2];
#ifdef DIAGNOSTICS_1
  printf("newton_vol:\n");
  printf("  resid = (%g,%g,%g); r^T r / 2 = %g\n",resid[0],resid[1],resid[2],
         (resid[0]*resid[0]+resid[1]*resid[1]+resid[2]*resid[2])/2);
  printf("  jac = %g\t%g\t%g\n"
         "        %g\t%g\t%g\n"
         "        %g\t%g\t%g\n",
         jac[0],jac[1],jac[2],jac[3],jac[4],jac[5],jac[6],jac[7],jac[8]);
  printf("  r = (%.15g,%.15g,%.15g)\n",r0[0],r0[1],r0[2]);
#endif

  mask = 0x3fu;
  for(d=0;d<3;++d) {
    if(r0[d]-tr>-1) bnd[2*d  ]=r0[d]-tr, mask^=1u<<(2*d);
    if(r0[d]+tr< 1) bnd[2*d+1]=r0[d]+tr, mask^=2u<<(2*d);
  }

  lin_solve_3(dr, jac,resid);

#ifdef DIAGNOSTICS_1
  printf("  min at r = (%.17g,%.17g,%.17g)\n",
         r0[0]+dr[0],r0[1]+dr[1],r0[2]+dr[2]);
#endif

  fac = 1, flags = 0;
  for(d=0;d<3;++d) {
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
  
  if(flags==0) goto newton_vol_fin;
  
  for(d=0;d<3;++d) dr[d]*=fac;

  newton_vol_face: {
    const unsigned fi = face_index(flags);
    const unsigned dn = fi>>1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
    double drc[2], fac=1;
    unsigned new_flags=0;
    double res[3], y[2], JtJ[3];
    res[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]),
    res[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]),
    res[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
    /* y = J_u^T res */
    y[0] = jac[d1]*res[0]+jac[3+d1]*res[1]+jac[6+d1]*res[2],
    y[1] = jac[d2]*res[0]+jac[3+d2]*res[1]+jac[6+d2]*res[2];
    /* JtJ = J_u^T J_u */
    JtJ[0] = jac[  d1]*jac[  d1]
            +jac[3+d1]*jac[3+d1]
            +jac[6+d1]*jac[6+d1],
    JtJ[1] = jac[  d1]*jac[  d2]
            +jac[3+d1]*jac[3+d2]
            +jac[6+d1]*jac[6+d2],
    JtJ[2] = jac[  d2]*jac[  d2]
            +jac[3+d2]*jac[3+d2]
            +jac[6+d2]*jac[6+d2];
    lin_solve_sym_2(drc, JtJ,y);
#ifdef DIAGNOSTICS_1
    printf("  face %u, dn=%u, (d1,d2)=(%u,%u)\n",fi,dn,d1,d2);
    printf("    r=(%.17g,%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1],r0[2]+dr[2]);
    printf("    resid = (%g,%g,%g); r^T r / 2 = %g\n",res[0],res[1],res[2],
           (res[0]*res[0]+res[1]*res[1]+res[2]*res[2])/2);
    printf("    min at (%.17g,%.17g)\n",
           r0[d1]+dr[d1]+drc[0],r0[d2]+dr[d2]+drc[1]);
#endif
    #define CHECK_CONSTRAINT(drcd,d3) do { \
      const double rz = r0[d3]+dr[d3], lb=bnd[2*d3],ub=bnd[2*d3+1]; \
      const double delta=drcd, nr = r0[d3]+(dr[d3]+delta); \
      if((nr-lb)*(ub-nr)<0) { \
        if(nr<lb) { \
          double f = (lb-rz)/delta; \
          if(f<fac) fac=f, new_flags = 1u<<(2*d3); \
        } else { \
          double f = (ub-rz)/delta; \
          if(f<fac) fac=f, new_flags = 2u<<(2*d3); \
        } \
      } \
    } while(0)
    CHECK_CONSTRAINT(drc[0],d1); CHECK_CONSTRAINT(drc[1],d2);
#ifdef DIAGNOSTICS_1
    printf("    new_flags = %x, fac = %.17g\n",new_flags,fac);
#endif
    dr[d1] += fac*drc[0], dr[d2] += fac*drc[1];
    if(new_flags==0) goto newton_vol_fin;
    flags |= new_flags;
  }

  newton_vol_edge: {
    const unsigned ei = edge_index(flags);
    const unsigned de = ei>>2;
    double fac = 1;
    unsigned new_flags = 0;
    double res[3],y,JtJ,drc;
    res[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]),
    res[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]),
    res[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
    /* y = J_u^T res */
    y = jac[de]*res[0]+jac[3+de]*res[1]+jac[6+de]*res[2];
    /* JtJ = J_u^T J_u */
    JtJ = jac[  de]*jac[  de]
         +jac[3+de]*jac[3+de]
         +jac[6+de]*jac[6+de];
    drc = y/JtJ;
#ifdef DIAGNOSTICS_1
    printf("  edge %u, de=%u\n",ei,de);
    printf("    r=(%.17g,%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1],r0[2]+dr[2]);
    printf("    resid = (%g,%g,%g); r^T r / 2 = %g\n",res[0],res[1],res[2],
           (res[0]*res[0]+res[1]*res[1]+res[2]*res[2])/2);
    printf("    min at %.17g\n", r0[de]+dr[de]+drc);
#endif
    CHECK_CONSTRAINT(drc,de);
    #undef CHECK_CONSTRAINT
#ifdef DIAGNOSTICS_1
    printf("    new_flags = %x, fac = %.17g\n",new_flags,fac);
#endif
    dr[de] += fac*drc;
    flags |= new_flags;
    goto newton_vol_relax;
  }

  /* check and possibly relax constraints */
  newton_vol_relax: {
    const unsigned old_flags = flags;
    double res[3], y[3];
    /* res := res_0 - J dr */
    res[0] = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]),
    res[1] = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]),
    res[2] = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
    /* y := J^T res */
    y[0] = jac[0]*res[0]+jac[3]*res[1]+jac[6]*res[2],
    y[1] = jac[1]*res[0]+jac[4]*res[1]+jac[7]*res[2],
    y[2] = jac[2]*res[0]+jac[5]*res[1]+jac[8]*res[2];
    #define SETDR(d) do { \
      unsigned f = flags>>(2*d) & 3u; \
      if(f) dr[d] = bnd[2*d+(f-1)] - r0[d]; \
    } while(0)
    SETDR(0); SETDR(1); SETDR(2);
    #undef SETDR
    for(d=0;d<3;++d) {
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
    printf("    r=(%.17g,%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1],r0[2]+dr[2]);
    printf("    resid = (%g,%g,%g); r^T r / 2 = %g\n",res[0],res[1],res[2],
           (res[0]*res[0]+res[1]*res[1]+res[2]*res[2])/2);
    printf("    relaxed %x -> %x\n",old_flags,flags);
#endif
    if(flags==old_flags) goto newton_vol_fin;
    switch(num_constrained(flags)) {
      case 1: goto newton_vol_face;
      case 2: goto newton_vol_edge;
    }
  }

newton_vol_fin:
#ifdef DIAGNOSTICS_1
  {
    const double res[3]={ resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]),
                          resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]),
                          resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]) };
    printf("  r=(%.17g,%.17g,%.17g)\n", r0[0]+dr[0],r0[1]+dr[1],r0[2]+dr[2]);
    printf("  resid = (%g,%g,%g); r^T r / 2 = %g\n",res[0],res[1],res[2],
           (res[0]*res[0]+res[1]*res[1]+res[2]*res[2])/2);
  }
#endif
  flags &= mask;
  if(fabs(dr[0])+fabs(dr[1])+fabs(dr[2]) < tol) flags |= CONVERGED_FLAG;
  {
    const double res0 = resid[0]-(jac[0]*dr[0]+jac[1]*dr[1]+jac[2]*dr[2]),
                 res1 = resid[1]-(jac[3]*dr[0]+jac[4]*dr[1]+jac[5]*dr[2]),
                 res2 = resid[2]-(jac[6]*dr[0]+jac[7]*dr[1]+jac[8]*dr[2]);
    out->dist2p=resid[0]*resid[0]+resid[1]*resid[1]+resid[2]*resid[2]
                -(res0*res0+res1*res1+res2*res2);
  }
  #define SETR(d) do { \
    unsigned f = flags>>(2*d) & 3u; \
    out->r[d] = f==0 ? r0[d]+dr[d] : ( f==1 ? -1 : 1 ); \
  } while(0)
  SETR(0); SETR(1); SETR(2);
  #undef SETR
  out->flags = flags | (p->flags<<7);
}

static void newton_face(struct findpts_el_pt_3 *const out,
                        const double jac[9], const double rhes[3],
                        const double resid[3],
                        const unsigned d1, const unsigned d2, const unsigned dn,
                        const unsigned flags,
                        const struct findpts_el_pt_3 *const p, const double tol)
{
  const double tr = p->tr;
  double bnd[4];
  double r[2], dr[2]={0,0};
  unsigned mask, new_flags;
  double v, tv; unsigned i;
  double A[3], y[2], r0[2];
  /* A = J^T J - resid_d H_d */
  A[0] = jac[  d1]*jac[  d1]
        +jac[3+d1]*jac[3+d1]
        +jac[6+d1]*jac[6+d1] - rhes[0],
  A[1] = jac[  d1]*jac[  d2]
        +jac[3+d1]*jac[3+d2]
        +jac[6+d1]*jac[6+d2] - rhes[1],
  A[2] = jac[  d2]*jac[  d2]
        +jac[3+d2]*jac[3+d2]
        +jac[6+d2]*jac[6+d2] - rhes[2];
  /* y = J^T r */
  y[0] = jac[  d1]*resid[0]
        +jac[3+d1]*resid[1]
        +jac[6+d1]*resid[2],
  y[1] = jac[  d2]*resid[0]
        +jac[3+d2]*resid[1]
        +jac[6+d2]*resid[2];
  r0[0] = p->r[d1], r0[1] = p->r[d2];

#ifdef DIAGNOSTICS_1
  printf("newton_face, dn=%u, (d1,d2)=%u,%u:\n", dn,d1,d2);
  printf("  J^T r = (%g,%g)\n", y[0],y[1]);
  printf("  A = %g\t%g\n"
         "      %g\t%g\n", A[0],A[1],A[1],A[2]);
  printf("  r = (%.15g,%.15g)\n", r0[0],r0[1]);
#endif

  new_flags=flags;
  mask=0x3fu;
  if(r0[0]-tr>-1) bnd[0]=-tr, mask^=1u;    else bnd[0]=-1-r0[0];
  if(r0[0]+tr< 1) bnd[1]= tr, mask^=2u;    else bnd[1]= 1-r0[0];
  if(r0[1]-tr>-1) bnd[2]=-tr, mask^=1u<<2; else bnd[2]=-1-r0[1];
  if(r0[1]+tr< 1) bnd[3]= tr, mask^=2u<<2; else bnd[3]= 1-r0[1];

#ifdef DIAGNOSTICS_1
  printf("  bounds = ([%.15g,%.15g],[%.15g,%.15g])\n",
         r0[0]+bnd[0],r0[0]+bnd[1],r0[1]+bnd[2],r0[1]+bnd[3]);
#endif

  if(A[0]+A[2]<=0 || A[0]*A[2]<=A[1]*A[1]) goto newton_face_constrained;
  lin_solve_sym_2(dr, A,y);

#ifdef DIAGNOSTICS_1
  printf("  min at r = (%.15g,%.15g)\n", r0[0]+dr[0],r0[1]+dr[1]);
#endif
  
  #define EVAL(r,s) -(y[0]*r+y[1]*s)+(r*A[0]*r+(2*r*A[1]+s*A[2])*s)/2
  if(   (dr[0]-bnd[0])*(bnd[1]-dr[0])>=0
     && (dr[1]-bnd[2])*(bnd[3]-dr[1])>=0) {
    r[0] = r0[0]+dr[0], r[1] = r0[1]+dr[1];
    v = EVAL(dr[0],dr[1]);
    goto newton_face_fin;
  }
newton_face_constrained:
  v  = EVAL(bnd[0],bnd[2]); i=1u|(1u<<2);
  tv = EVAL(bnd[1],bnd[2]); if(tv<v) v=tv, i=2u|(1u<<2);
  tv = EVAL(bnd[0],bnd[3]); if(tv<v) v=tv, i=1u|(2u<<2);
  tv = EVAL(bnd[1],bnd[3]); if(tv<v) v=tv, i=2u|(2u<<2);
  if(A[0]>0) {
    double drc;
    drc = (y[0] - A[1]*bnd[2])/A[0];
    if((drc-bnd[0])*(bnd[1]-drc)>=0 && (tv=EVAL(drc,bnd[2]))<v)
      v=tv,i=1u<<2,dr[0]=drc;
    drc = (y[0] - A[1]*bnd[3])/A[0];
    if((drc-bnd[0])*(bnd[1]-drc)>=0 && (tv=EVAL(drc,bnd[3]))<v)
      v=tv,i=2u<<2,dr[0]=drc;
  }
  if(A[2]>0) {
    double drc;
    drc = (y[1] - A[1]*bnd[0])/A[2];
    if((drc-bnd[2])*(bnd[3]-drc)>=0 && (tv=EVAL(bnd[0],drc))<v)
      v=tv,i=1u,dr[1]=drc;
    drc = (y[1] - A[1]*bnd[1])/A[2];
    if((drc-bnd[2])*(bnd[3]-drc)>=0 && (tv=EVAL(bnd[1],drc))<v)
      v=tv,i=2u,dr[1]=drc;
  }
  #undef EVAL

  #define SETR(d,d3) do { \
    const unsigned f = i>>(2*d) & 3u; \
    if(f==0) r[d]=r0[d]+dr[d]; \
    else { \
      if((f&(mask>>(2*d)))==0) r[d]=r0[d]+(f==1?-tr:tr); \
      else                     r[d]=(f==1?-1:1), new_flags |= f<<(2*d3); \
    } \
  } while(0)
  SETR(0,d1); SETR(1,d2);
#ifdef DIAGNOSTICS_1
  printf("  constrained min at r = (%.15g,%.15g)\n", r[0],r[1]);
#endif
newton_face_fin:
  out->dist2p = -2*v;
  dr[0]=r[0]-p->r[d1];
  dr[1]=r[1]-p->r[d2];
  if(fabs(dr[0])+fabs(dr[1]) < tol) new_flags |= CONVERGED_FLAG;
  out->r[dn]=p->r[dn], out->r[d1]=r[0],out->r[d2]=r[1];
  out->flags = new_flags | (p->flags<<7);
}

static void newton_edge(struct findpts_el_pt_3 *const out,
  const double jac[9], const double rhes, const double resid[3],
  const unsigned de, const unsigned dn1, const unsigned dn2,
  unsigned flags,
  const struct findpts_el_pt_3 *const p, const double tol)
{
  const double tr = p->tr;
  /* A = J^T J - resid_d H_d */
  const double A = jac[  de]*jac[  de]
                  +jac[3+de]*jac[3+de]
                  +jac[6+de]*jac[6+de] - rhes;
  /* y = J^T r */
  const double y = jac[  de]*resid[0]
                  +jac[3+de]*resid[1]
                  +jac[6+de]*resid[2];

  const double oldr = p->r[de];
  double dr,nr,tdr,tnr;
  double v,tv; unsigned new_flags=0, tnew_flags=0;

#ifdef DIAGNOSTICS_1
  printf("Newton edge %u (dn1=%u,dn2=%u) flags=%x\n",de,dn1,dn2,flags);
  printf("  A=%g, y=%g\n",A,y);
  if(A<=0) printf("  A not positive\n");
  printf("  r=(%g,%g,%g)\n",p->r[0],p->r[1],p->r[2]);
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
  out->r[dn1]=p->r[dn1];
  out->r[dn2]=p->r[dn2];
  out->dist2p = -v;
  out->flags = flags | new_flags | (p->flags<<7);
#ifdef DIAGNOSTICS_1
  printf("  new r = (%g,%g,%g)\n",out->r[0],out->r[1],out->r[2]);
#endif
}

typedef void findpt_fun(
  struct findpts_el_pt_3 *const out,
  struct findpts_el_data_3 *const fd,
  const struct findpts_el_pt_3 *const p, const unsigned pn, const double tol);

/* work[(3+9+2*(nr+ns+nt+nrs))*pn + max(2*nr,ns) ] */
static void findpt_vol(
  struct findpts_el_pt_3 *const out,
  struct findpts_el_data_3 *const fd,
  const struct findpts_el_pt_3 *const p, const unsigned pn, const double tol)
{
  const unsigned nr=fd->n[0],ns=fd->n[1],nt=fd->n[2],
                 nrs=nr*ns;
  double *const resid = fd->work, *const jac = resid + 3*pn,
         *const wtrs = jac+9*pn, *const wtt = wtrs+2*(nr+ns)*pn,
         *const slice = wtt+2*nt*pn, *const temp = slice + 2*pn*nrs;
  unsigned i; unsigned d;
  /* evaluate x(r) and jacobian */
  for(i=0;i<pn;++i)
    fd->lag[0](wtrs+2*i*(nr+ns)     , fd->lag_data[0], nr, 1, p[i].r[0]);
  for(i=0;i<pn;++i)
    fd->lag[1](wtrs+2*i*(nr+ns)+2*nr, fd->lag_data[1], ns, 1, p[i].r[1]);
  for(i=0;i<pn;++i)
    fd->lag[2](wtt+2*i*nt           , fd->lag_data[2], nt, 1, p[i].r[2]);
  for(d=0;d<3;++d) {
    tensor_mxm(slice,nrs, fd->x[d],nt, wtt,2*pn);
    for(i=0;i<pn;++i) {
      const double *const wtrs_i = wtrs+2*i*(nr+ns),
                   *const slice_i = slice+2*i*nrs;
      double *const jac_i = jac+9*i+3*d;
      resid[3*i+d] = p[i].x[d] - tensor_ig2(jac_i,
        wtrs_i,nr, wtrs_i+2*nr,ns, slice_i, temp);
      jac_i[2] = tensor_i2(wtrs_i,nr, wtrs_i+2*nr,ns, slice_i+nrs, temp);
    }
  }
  /* perform Newton step */
  for(i=0;i<pn;++i) {
    if(reject_prior_step_q(out+i,resid+3*i,p+i,tol)) continue;
    else newton_vol(out+i, jac+9*i, resid+3*i, p+i, tol);
  }
}

/* work[(3+9+3+3*(n1+n2+n1))*pn ] */
static void findpt_face(
  struct findpts_el_pt_3 *const out,
  struct findpts_el_data_3 *const fd,
  const struct findpts_el_pt_3 *const p, const unsigned pn, const double tol)
{
  const unsigned pflag = p->flags & FLAG_MASK;
  const unsigned fi = face_index(pflag);
  const unsigned dn = fi>>1, d1 = plus_1_mod_3(dn), d2 = plus_2_mod_3(dn);
  const unsigned n1 = fd->n[d1], n2 = fd->n[d2];
  double *const resid=fd->work, *const jac=resid+3*pn, *const hes=jac+9*pn,
         *const wt1 = hes+3*pn, *const wt2 = wt1+3*n1*pn,
         *const slice = wt2+3*n2*pn;
  const struct findpts_el_gface_3 *const face = get_face(fd,fi);
  unsigned i; unsigned d;

#ifdef DIAGNOSTICS_1
  printf("Face %u\n",fi);
  printf("  pflag = %u\n",pflag);
  printf("  fi = %u\n",fi);
  printf("  dn, d1, d2 = %u, %u, %u\n",dn,d1,d2);
  printf("  n1, n2 = %u, %u \n", n1,n2);
#endif

  /* evaluate x(r), jacobian, hessian */
  for(i=0;i<pn;++i)
    fd->lag[d1](wt1+3*i*n1, fd->lag_data[d1], n1, 2, p[i].r[d1]);
  for(i=0;i<pn;++i)
    fd->lag[d2](wt2+3*i*n2, fd->lag_data[d2], n2, 2, p[i].r[d2]);
  for(i=0;i<3*pn;++i) hes[i]=0;
  for(d=0;d<3;++d) {
    tensor_mxm(slice,n1, face->x[d],n2, wt2,3*pn);
    for(i=0;i<pn;++i) {
      const double *const wt1_i = wt1+3*i*n1, *const slice_i = slice+3*i*n1;
      double v[9], r;
      tensor_mtxm(v,3, wt1_i,n1, slice_i,3);
      /* v[3*j + i] = d^i/dr1^i d^j/dr2^j x_d */
      resid[3*i+d] = r = p[i].x[d] - v[0];
      jac[9*i+3*d+d1] = v[1];
      jac[9*i+3*d+d2] = v[3];
      hes[3*i  ] += r * v[2];
      hes[3*i+1] += r * v[4];
      hes[3*i+2] += r * v[6];
    }
  }
  for(i=1;i<pn;++i) memcpy(wt2+i*n2, wt2+3*i*n2, n2*sizeof(double));
  for(d=0;d<3;++d) {
    tensor_mxm(slice,n1, face->dxdn[d],n2, wt2,pn);
    for(i=0;i<pn;++i)
      jac[9*i+3*d+dn] = tensor_dot(wt1+3*i*n1, slice+i*n1, n1);
  }
  /* perform Newton step */
  for(i=0;i<pn;++i) {
    double *const resid_i=resid+3*i, *const jac_i=jac+9*i, *const hes_i=hes+3*i;
    /* check prior step */
    if(!reject_prior_step_q(out+i,resid_i,p+i,tol)) {
      /* check constraint */
      const double steep = resid_i[0] * jac_i[  dn]
                          +resid_i[1] * jac_i[3+dn]
                          +resid_i[2] * jac_i[6+dn];
#ifdef DIAGNOSTICS_1
      printf("jacobian = %g\t%g\t%g\n"
             "           %g\t%g\t%g\n"
             "           %g\t%g\t%g\n",jac_i[0],jac_i[1],jac_i[2],
             jac_i[3],jac_i[4],jac_i[5],jac_i[6],jac_i[7],jac_i[8]);
      printf("resid_i = (%g,%g,%g)\n", resid_i[0],resid_i[1],resid_i[2]);
      printf("steep = %g (%s)\n", steep, steep * p[i].r[dn] < 0 ? "in" : "out");
#endif
      if(steep * p[i].r[dn] < 0) /* relax constraint */
        newton_vol(out+i, jac_i, resid_i, p+i, tol);
      else
        newton_face(out+i, jac_i, hes_i, resid_i, d1,d2,dn,pflag, p+i, tol);
    }
  }
}

/* work[ 3*n ] */
static void findpt_edge(
  struct findpts_el_pt_3 *const out,
  struct findpts_el_data_3 *const fd,
  const struct findpts_el_pt_3 *const p, const unsigned pn, const double tol)
{
  const unsigned pflag = p->flags & FLAG_MASK;
  const unsigned ei = edge_index(pflag);
  const unsigned de = ei>>2, dn1 = plus_1_mod_3(de), dn2 = plus_2_mod_3(de);
  const unsigned n = fd->n[de];
  double *wt = fd->work;
  const struct findpts_el_gedge_3 *edge = get_edge(fd,ei);
  unsigned i; unsigned d;

#ifdef DIAGNOSTICS_1
  printf("Edge %u\n",ei);
  printf("  pflag = %u\n",pflag);
  printf("  ei = %u\n",ei);
  printf("  de, dn1, dn2 = %u, %u, %u\n",de,dn1,dn2);
  printf("  n = %u \n", n);
#endif

  for(i=0;i<pn;++i) {
    double dxi[3], resid[3], jac[9];
    double hes[5] = {0,0,0,0,0};
    /* evaluate x(r), jacobian, hessian */
    fd->lag[de](wt, fd->lag_data[de], n, 2, p[i].r[de]);
    for(d=0;d<3;++d) {
      double r;
      tensor_mtxv(dxi,3, wt, edge->x[d],n);
      resid[d] = r = p[i].x[d] - dxi[0];
      jac[3*d+de] = dxi[1];
      hes[0] += r * dxi[2];
      tensor_mtxv(dxi,2, wt, edge->dxdn1[d],n);
      jac[3*d+dn1] = dxi[0];
      hes[1] += r * dxi[1];
      tensor_mtxv(dxi,2, wt, edge->dxdn2[d],n);
      jac[3*d+dn2] = dxi[0];
      hes[2] += r * dxi[1];
      hes[3] += r * tensor_dot(wt, edge->d2xdn1[d], n);
      hes[4] += r * tensor_dot(wt, edge->d2xdn2[d], n);
    }
    /* check prior step */
    if(reject_prior_step_q(out+i,resid,p+i,tol)) continue;
    /* check constraint */
    {
      double steep[3], sr1, sr2;
      steep[0] = jac[0]*resid[0] + jac[3]*resid[1] + jac[6]*resid[2],
      steep[1] = jac[1]*resid[0] + jac[4]*resid[1] + jac[7]*resid[2],
      steep[2] = jac[2]*resid[0] + jac[5]*resid[1] + jac[8]*resid[2];
      sr1 = steep[dn1]*p[i].r[dn1],
      sr2 = steep[dn2]*p[i].r[dn2];
#ifdef DIAGNOSTICS_1
    printf("jacobian = %g\t%g\t%g\n"
           "           %g\t%g\t%g\n"
           "           %g\t%g\t%g\n",jac[0],jac[1],jac[2],
           jac[3],jac[4],jac[5],jac[6],jac[7],jac[8]);
    printf("hessian = %g\t%g\t%g\n"
           "            \t%g    \n"
           "            \t  \t%g\n", hes[0],hes[1],hes[2],hes[3],hes[4]);
    printf("resid = (%g,%g,%g)\n", resid[0],resid[1],resid[2]);
    printf("steep1 = %g (%s)\n", steep[dn1], sr1 < 0 ? "in" : "out");
    printf("steep2 = %g (%s)\n", steep[dn2], sr2 < 0 ? "in" : "out");
#endif
      if(sr1<0) {
        if(sr2<0)
          newton_vol(out+i, jac,resid, p+i, tol);
        else {
          double rh[3]; rh[0]=hes[0], rh[1]=hes[1], rh[2]=hes[3];
          newton_face(out+i, jac,rh,resid, de,dn1,dn2,
                      pflag & (3u<<(dn2*2)), p+i, tol);
        }
      } else if(sr2<0) {
          double rh[3]; rh[0]=hes[4], rh[1]=hes[2], rh[2]=hes[0];
          newton_face(out+i, jac,rh,resid, dn2,de,dn1,
                      pflag & (3u<<(dn1*2)), p+i, tol);
      } else
        newton_edge(out+i, jac,hes[0],resid, de,dn1,dn2, pflag, p+i, tol);
    }
  }
}

static void findpt_pt(
  struct findpts_el_pt_3 *const out,
  struct findpts_el_data_3 *const fd,
  const struct findpts_el_pt_3 *const p, const unsigned pn, const double tol)
{
  const unsigned pflag = p->flags & FLAG_MASK;
  const unsigned pi = point_index(pflag);
  const struct findpts_el_gpt_3 *gpt = get_pt(fd,pi);
  const double *const x = gpt->x, *const jac = gpt->jac, *const hes = gpt->hes;
  unsigned i;

#ifdef DIAGNOSTICS_1
  printf("Point %u\n",pi);
  printf("  pflag = %u\n",pflag);
  printf("  pi = %u\n",pi);
#endif

  for(i=0;i<pn;++i) {
    unsigned d1,d2,dn, de,dn1,dn2, hi0,hi1,hi2;
    double resid[3], steep[3], sr[3];
    resid[0] = p[i].x[0]-x[0],
    resid[1] = p[i].x[1]-x[1],
    resid[2] = p[i].x[2]-x[2];
    steep[0] = jac[0]*resid[0] + jac[3]*resid[1] + jac[6]*resid[2],
    steep[1] = jac[1]*resid[0] + jac[4]*resid[1] + jac[7]*resid[2],
    steep[2] = jac[2]*resid[0] + jac[5]*resid[1] + jac[8]*resid[2];
    sr[0] = steep[0]*p[i].r[0],
    sr[1] = steep[1]*p[i].r[1], 
    sr[2] = steep[2]*p[i].r[2];
    /* check prior step */
    if(reject_prior_step_q(out+i,resid,p+i,tol)) continue;
    /* check constraints */
    if(sr[0]<0) {
      if(sr[1]<0) {
        if(sr[2]<0) goto findpt_pt_vol;
        else { d1=0,d2=1,dn=2, hi0=0,hi1=1,hi2=3; goto findpt_pt_face; }
      }
      else if(sr[2]<0) {d1=2,d2=0,dn=1, hi0=5,hi1=2,hi2=0; goto findpt_pt_face;}
      else { de=0,dn1=1,dn2=2, hi0=0; goto findpt_pt_edge; }
    }
    else if(sr[1]<0) {
      if(sr[2]<0) { d1=1,d2=2,dn=0, hi0=3,hi1=4,hi2=5; goto findpt_pt_face; }
      else { de=1,dn1=2,dn2=0, hi0=3; goto findpt_pt_edge; }
    }
    else if(sr[2]<0) { de=2,dn1=0,dn2=1, hi0=5; goto findpt_pt_edge; }
    out[i].r[0]=p[i].r[0],out[i].r[1]=p[i].r[1],out[i].r[2]=p[i].r[2];
    out[i].dist2p=0;
    out[i].flags = pflag | CONVERGED_FLAG;
    continue;
    findpt_pt_vol:
      newton_vol(out+i, jac,resid, p+i, tol);
      continue;
    findpt_pt_face: {
      double rh[3];
      rh[0] = resid[0]*hes[hi0]+resid[1]*hes[6+hi0]+resid[2]*hes[12+hi0],
      rh[1] = resid[0]*hes[hi1]+resid[1]*hes[6+hi1]+resid[2]*hes[12+hi1],
      rh[2] = resid[0]*hes[hi2]+resid[1]*hes[6+hi2]+resid[2]*hes[12+hi2];
      newton_face(out+i, jac,rh,resid, d1,d2,dn,
                  pflag&(3u<<(2*dn)), p+i, tol);
    } continue;
    findpt_pt_edge: {
      const double rh =
        resid[0]*hes[hi0]+resid[1]*hes[6+hi0]+resid[2]*hes[12+hi0];
      newton_edge(out+i, jac,rh,resid, de,dn1,dn2,
                  pflag&~(3u<<(2*de)), p+i, tol);
    } continue;
  }
}

static void seed(struct findpts_el_data_3 *const fd,
                 struct findpts_el_pt_3 *const pt, const unsigned npt)
{
  struct findpts_el_pt_3 *p, *const pe = pt+npt;
  const unsigned nr=fd->n[0], ns=fd->n[1], nt=fd->n[2];
  unsigned i,j,k, ii=0;
  for(p=pt;p!=pe;++p) p->dist2=DBL_MAX;
  for(k=0;k<nt;++k) {
    const double zt=fd->z[2][k];
    for(j=0;j<ns;++j) {
      const double zs=fd->z[1][j];
      for(i=0;i<nr;++i) {
        const double zr=fd->z[0][i];
        const double x=fd->x[0][ii], y=fd->x[1][ii], z=fd->x[2][ii];
        ++ii;
        for(p=pt;p!=pe;++p) {
          const double dx=p->x[0]-x,dy=p->x[1]-y,dz=p->x[2]-z;
          const double dist2 = dx*dx+dy*dy+dz*dz;
          if(p->dist2<=dist2) continue;
          p->dist2=dist2;
          p->r[0]=zr, p->r[1]=zs, p->r[2]=zt;
        }
      }
    }
  }
}

void findpts_el_3(struct findpts_el_data_3 *const fd, const unsigned npt,
                  const double tol)
{
  findpt_fun *const fun[4] = 
    { &findpt_vol, &findpt_face, &findpt_edge, &findpt_pt };
  struct findpts_el_pt_3 *const pbuf = fd->p, *const pstart = fd->p + npt;
  unsigned nconv = npt;
  unsigned step = 0;
  unsigned count[27] = { 0,0,0, 0,0,0, 0,0,0,
                         0,0,0, 0,0,0, 0,0,0,
                         0,0,0, 0,0,0, 0,0,0 } ;
  count[0] = npt;
  seed(fd,pbuf,npt);
  { unsigned i;
    for(i=0;i<npt;++i) {
      pstart[i].x[0]=pbuf[i].x[0];
      pstart[i].x[1]=pbuf[i].x[1];
      pstart[i].x[2]=pbuf[i].x[2];
      pstart[i].r[0]=pbuf[i].r[0];
      pstart[i].r[1]=pbuf[i].r[1];
      pstart[i].r[2]=pbuf[i].r[2];
      pstart[i].index=i,pstart[i].flags=0;
      pstart[i].dist2=DBL_MAX,pstart[i].dist2p=0,pstart[i].tr=1;
    }
  }
  while(nconv && step++ < 50) {
    /* advance each group of points */
    struct findpts_el_pt_3 *p, *const pe=pstart+nconv, *pout; unsigned pn;
    
#if DIAGNOSTICS_ITERATIONS>1
    { unsigned i; 
      printf("findpts_el_3 Newton step (%u), %u unconverged:\n ", step,nconv);
      for(i=0;i<27;++i) printf(" %u",count[i]);
      printf("\n");
    }
#endif
#ifdef DIAGNOSTICS_3
    if(step==50) {
      unsigned d, i, n=fd->n[0]*fd->n[1]*fd->n[2];
      printf("geometry:\n{\n");
      for(d=0;d<3;++d) {
        printf(" {\n");
        for(i=0;i<n;++i)
          printf("  %.15g%s\n",fd->x[d][i],i==n-1?"":",");
        printf(" }%s\n",d==3-1?"":",");
      }
      printf("}\n");
    }
#endif

    for(p=pstart,pout=pbuf; p!=pe; p+=pn,pout+=pn) {
      const unsigned pflags = p->flags & FLAG_MASK;
      pn = count[pt_flags_to_bin_noC(pflags)];
      fun[num_constrained(pflags)](pout, fd, p,pn, tol);
    }
    /* group points by contsraints */
    {
      unsigned offset[28] = { 0,0,0, 0,0,0, 0,0,0,
                              0,0,0, 0,0,0, 0,0,0,
                              0,0,0, 0,0,0, 0,0,0, 0 };
      struct findpts_el_pt_3 *const pe = pbuf+nconv;
      for(pout=pbuf; pout!=pe; ++pout)
        ++offset[pt_flags_to_bin(pout->flags & FLAG_MASK)];
      {
        unsigned i; unsigned sum=0;
        for(i=0;i<27;++i) {
          unsigned ci=offset[i]; count[i]=ci, offset[i]=sum, sum+=ci;
        }
        nconv = offset[27] = sum; /* last bin is converged; forget it */
      }
      for(pout=pbuf; pout!=pe; ++pout)
        pstart[offset[pt_flags_to_bin(pout->flags & FLAG_MASK)]++] = *pout;
    }
  }
  { struct findpts_el_pt_3 *p, *const pe=pstart+npt;
    for(p=pstart;p!=pe;++p)
      pbuf[p->index]=*p, pbuf[p->index].flags&=FLAG_MASK;
  }
#if DIAGNOSTICS_ITERATIONS
  printf("findpts_el_3 took %u steps\n ", step);
#endif
}

void findpts_el_eval_3(
        double *const out_base, const unsigned out_stride,
  const double *const   r_base, const unsigned   r_stride, const unsigned pn,
  const double *const in, struct findpts_el_data_3 *const fd)
{
  const unsigned nr=fd->n[0],ns=fd->n[1],nt=fd->n[2],
                 nrs=nr*ns;
  double *const wtrs = fd->work, *const wtt = wtrs+(nr+ns)*pn,
         *const slice = wtt+nt*pn, *const temp = slice + pn*nrs;
  unsigned i; const double *r; double *out;
  for(i=0,r=r_base;i<pn;++i) {
    fd->lag[0](wtrs+i*(nr+ns)   , fd->lag_data[0], nr, 0, r[0]);
    fd->lag[1](wtrs+i*(nr+ns)+nr, fd->lag_data[1], ns, 0, r[1]);
    fd->lag[2](wtt +i*nt        , fd->lag_data[2], nt, 0, r[2]);
    r = (const double*)((const char*)r + r_stride);
  }
  
  tensor_mxm(slice,nrs, in,nt, wtt,pn);
  for(i=0,out=out_base;i<pn;++i) {
    const double *const wtrs_i = wtrs+i*(nr+ns), *const slice_i = slice+i*nrs;
    *out = tensor_i2(wtrs_i,nr, wtrs_i+nr,ns, slice_i, temp);
    out = (double*)((char*)out + out_stride);
  }
}

