#ifndef SPARSE_CHOLESKY_H
#define SPARSE_CHOLESKY_H

#if !defined(TYPES_H) || !defined(MEM_H)
#warning "sparse_cholesky.h" requires "types.h" and "mem.h"
#endif

#define sparse_cholesky_factor PREFIXED_NAME(sparse_cholesky_factor)
#define sparse_cholesky_solve  PREFIXED_NAME(sparse_cholesky_solve )
#define sparse_cholesky_free   PREFIXED_NAME(sparse_cholesky_free  )

struct sparse_cholesky {
  uint n, *Lrp, *Lj;
  double *L, *D;
};

/* input data is the usual CSR
   matrix is n by n
   Arp has n+1 elements
   elements of row i are A [Arp[i]], ..., A [Arp[i+1]-1]
              in columns Aj[Arp[i]], ..., Aj[Arp[i+1]-1]
*/
void sparse_cholesky_factor(uint n, const uint *Arp, const uint *Aj,
                            const double *A,
                            struct sparse_cholesky *out, buffer *buf);
                            
/* x = A^(-1) b;  works when x and b alias */
void sparse_cholesky_solve(
  double *x, const struct sparse_cholesky *fac, double *b);

void sparse_cholesky_free(struct sparse_cholesky *fac);

#endif

