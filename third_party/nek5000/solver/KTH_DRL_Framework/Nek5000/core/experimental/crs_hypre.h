#ifndef CRS_HYPRE_H
#define CRS_HYPRE_H

#ifdef HYPRE

#include "_hypre_utilities.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE.h"

struct hypre_crs_data {
  HYPRE_Solver solver;
  HYPRE_IJMatrix A;
  HYPRE_IJVector b;
  HYPRE_IJVector x;
  HYPRE_Int ilower;
  uint un; 
  struct comm comm;
  struct gs_data *gs_top;
  uint *umap;
  uint nullspace;
  double tni;
};
#else
  struct hypre_crs_data;
#endif

/* remote id - designates the i-th uknown owned by proc p */
struct rid { uint p,i; };
#define rid_equal(a,b) ((a).p==(b).p && (a).i==(b).i)
#define nz_pos_equal(a,b) \
  (rid_equal((a).i,(b).i) && rid_equal((a).j,(b).j))
struct labelled_rid {
  struct rid rid; ulong id;
};

/* rnz is a mnemonic for remote non zero */
struct rnz {
  double v; struct rid i,j;
};

/* global non-zero (matrix entry) */
struct gnz { ulong i,j; double a; };

enum mat_order { row_major, col_major };
enum distr { row_distr, col_distr };

void build_hypre_matrix(struct hypre_crs_data *hypre_data, uint n, 
  const ulong *id, uint nz_unassembled, const uint *Ai, const uint *Aj, const double *Av);

struct hypre_crs_data *ccrs_hypre_setup( uint n, const ulong *id,
                 uint nz, const uint *Ai, const uint *Aj, const double *Av,
                 const uint null_space, const struct comm *comm,
                 const double *param);

void ccrs_hypre_solve(double *x, struct hypre_crs_data *data, double *b);

void ccrs_hypre_free(struct hypre_crs_data *data);

static uint *assign_dofs_hypre(struct array *const uid,
                         struct rid *const rid_map,
                         const ulong *const id, const uint n,
                         const uint p,
			 struct gs_data *gs_top, buffer *const buf,
			 struct comm *comm);

static void mat_distribute(
			   struct array *const mat, const enum distr d, const enum mat_order o,
			   struct crystal *const cr);

static void mat_list_nonlocal_sorted(
				      struct array *const nonlocal_id,
				      const struct array *const mat, const enum distr d,
				      const ulong *uid, struct crystal *const cr);

#endif
