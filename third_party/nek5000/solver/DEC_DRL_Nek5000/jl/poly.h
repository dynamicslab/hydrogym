#ifndef POLY_H
#define POLY_H

#if !defined(NAME_H)
#warning "poly.h" requires "name.h"
#endif

#define lagrange_size  PREFIXED_NAME(lagrange_size )
#define lagrange_setup PREFIXED_NAME(lagrange_setup)
#define gauss_nodes    PREFIXED_NAME(gauss_nodes   )
#define gauss_quad     PREFIXED_NAME(gauss_quad    )
#define lobatto_nodes  PREFIXED_NAME(lobatto_nodes )
#define lobatto_quad   PREFIXED_NAME(lobatto_quad  )
#define gll_lag_size   PREFIXED_NAME(gll_lag_size  )
#define gll_lag_setup  PREFIXED_NAME(gll_lag_setup )

/*--------------------------------------------------------------------------
   Quadrature Nodes and Weights Calculation

    Gauss   -> Gauss-Legendre quadrature (open)
    Lobatto -> Gauss-Lobatto-Legendre quadrature (closed at both ends)
   
   the _quad functions compute both nodes and weights
  --------------------------------------------------------------------------*/

void   gauss_nodes(double *restrict z, int n); /* n nodes (order = 2n-1) */
void lobatto_nodes(double *restrict z, int n); /* n nodes (order = 2n-3) */

void   gauss_quad(double *restrict z, double *restrict w, int n);
void lobatto_quad(double *restrict z, double *restrict w, int n);

/*--------------------------------------------------------------------------
   Lagrangian basis function evaluation
   
   Usage:
   
   double z[N] = ..., x = ...; // nodes and evaluation point
   double p[3*N];
   double *data = tmalloc(double, lagrange_size(N));
   lagrange_fun *const lag = lagrange_setup(data, z, N);
   
   int d = ...; // 0, 1, or 2  --- the highest derivative to compute
   lag(p, data,N,d, x);
   // now p[i] = h_i(x), 0 <= i < N 
   // if d>=1, p[N+i] = h_i'(x)
   // if d>=2, p[2*N+i] = h_i''(x)
   free(data);
   
   gll_lag_* are similar, but are specialized  for GLL nodes, and faster,
   and also don't need to be given the nodal locations
  --------------------------------------------------------------------------*/

typedef void lagrange_fun(double *restrict p,
  double *restrict data, unsigned n, int d, double x);

unsigned lagrange_size(unsigned n);
lagrange_fun *lagrange_setup(
  double *restrict data, const double *restrict z, unsigned n);

unsigned gll_lag_size(unsigned n);
lagrange_fun *gll_lag_setup(double *restrict data, int n);


#endif

