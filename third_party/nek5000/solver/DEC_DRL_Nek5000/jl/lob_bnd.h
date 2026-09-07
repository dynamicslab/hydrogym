#ifndef LOB_BND_H
#define LOB_BND_H

#if !defined(TYPES_H) || !defined(NAME_H)
#warning "lob_bnd.h" requires "types.h" and "name.h"
#endif

#define lob_bnd_setup  PREFIXED_NAME(lob_bnd_setup)
#define lob_bnd_lin_1  PREFIXED_NAME(lob_bnd_lin_1)
#define lob_bnd_lin_2  PREFIXED_NAME(lob_bnd_lin_2)
#define lob_bnd_lin_3  PREFIXED_NAME(lob_bnd_lin_3)
#define lob_bnd_1      PREFIXED_NAME(lob_bnd_1    )
#define lob_bnd_2      PREFIXED_NAME(lob_bnd_2    )
#define lob_bnd_3      PREFIXED_NAME(lob_bnd_3    )

/*--------------------------------------------------------------------------
   Bounds for Polynomials on [-1,1]^d
     given in the Lagrangian basis on
     Gauss-Lobatto-Legendre quadrature nodes

   The main parameters are the number of GLL nodes in each dimension
     unsigned nr = ..., ns = ..., nt = ...;

   The number of points in the constructed piecewise (tri-,bi-)linear bounds
   is a parameter; more points give tighter bounds, and we expect m>n.
   
     unsigned mr = 4*nr, ms = 4*ns, mt = 4*nt;
   
   The necessary setup is accomplished via:
     double *data_r = tmalloc(double, lob_bnd_size(nr,mr));
     double *data_s = tmalloc(double, lob_bnd_size(ns,ms));
     double *data_t = tmalloc(double, lob_bnd_size(nt,mt));
     lob_bnd_setup(data_r, nr,mr);
     lob_bnd_setup(data_s, ns,ms);
     lob_bnd_setup(data_t, nt,mt);
 
   Bounds may then be computed via:
     double work1r[2*mr], work1s[2*ms];
     double work2[2*mr*(ns+ms+1)];
     double work3[2*mr*ms*(nt+mt+1)];
     double ur[nr], us[ns];    // 1-d polynomials on the zr[] and zs[] nodes
     double u2[ns][nr];        // 2-d polynomial on zr[] (x) zs[]
     double u3[nt][ns][nr];    // 3-d polynomial on zr[] (x) zs[] (x) zt[]
     struct dbl_range bound;
     
     bound = lob_bnd_1(data_r,nr,mr, ur, work1r); // compute bounds on ur
     bound = lob_bnd_1(data_s,ns,ms, us, work1s); // compute bounds on us
     bound = lob_bnd_2(data_r,nr,mr, data_s,ns,ms,
               (const double*)&u2[0][0], work2); // compute bounds on u2
     bound = lob_bnd_3(data_r,nr,mr, data_s,ns,ms, data_t,nt,mt,
               (const double*)&u3[0][0], work3); // compute bounds on u3

    free(data_r), free(data_s), free(data_t);

   The functions lob_bnd_lin_d compute the piecewise d-linear bounds.
   Nodes for these are Chebyshev-Lobatto:
     h[0] = -1, h[m-1] = 1;
     for(j=1;j<m-1;++j) h[j] = cos((m-1-j)*PI/(m-1));
   The functions lob_bnd_d simply call these and return the min and max
   over all nodes.
    
  --------------------------------------------------------------------------*/

static unsigned lob_bnd_size(unsigned n, unsigned m)
{ return m+3*n*(2*m+1); }

void lob_bnd_setup(double *restrict data, unsigned n, unsigned m);

void lob_bnd_lin_1(double *restrict b,
                   const double *restrict lob_bnd_data, unsigned n, unsigned m,
                   const double *restrict u, uint un);

/* work holds 2*mr + 2*ns*mr doubles */
void lob_bnd_lin_2(
  double *restrict b,
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *restrict u, uint un, double *restrict work);

/* work holds 2*mr*ms + 2*nt*ms*mr doubles */
void lob_bnd_lin_3(
  double *restrict b,
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *lob_bnd_data_t, unsigned nt, unsigned mt,
  const double *restrict u, uint un, double *restrict work);

struct dbl_range { double min, max; };

/* work holds 2*m doubles */
struct dbl_range lob_bnd_1(
  const double *restrict lob_bnd_data, unsigned n, unsigned m,
  const double *restrict u, double *restrict work);

/* work holds 2*mr*ms + 2*mr + 2*mr*ns
             =2*mr*(ns+ms+1) doubles */
struct dbl_range lob_bnd_2(
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *restrict u, double *restrict work);

/* work holds 2*mr*ms*mt + 2*mr*ms + 2*nt*ms*mr
             =2*mr*ms*(nt+mt+1) doubles */
struct dbl_range lob_bnd_3(
  const double *lob_bnd_data_r, unsigned nr, unsigned mr,
  const double *lob_bnd_data_s, unsigned ns, unsigned ms,
  const double *lob_bnd_data_t, unsigned nt, unsigned mt,
  const double *restrict u, double *restrict work);

#endif

