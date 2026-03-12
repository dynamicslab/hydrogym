#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "gslib.h"
#include "crs_hypre.h"

#ifdef HYPRE

#define NPARAM 10
double hypre_param[NPARAM];

static struct hypre_crs_data **handle_array = 0;
static int handle_max = 0;
static int handle_n = 0;

struct hypre_crs_data *ccrs_hypre_setup( uint n, const ulong *id,
		 uint nz, const uint *Ai, const uint *Aj, const double *Av,
  		 const uint null_space, const struct comm *comm, 
                 const double *param)
{
  struct hypre_crs_data *hypre_data = tmalloc(struct hypre_crs_data, 1);

  hypre_data->nullspace = null_space;
  
  comm_dup(&hypre_data->comm,comm);
  hypre_data->gs_top = gs_setup((const slong*)id,n,&hypre_data->comm, 1,
    gs_auto, 1);

  // Build CRS matrix in hypre format and initialize stuff
  build_hypre_matrix(hypre_data, n, id, nz, Ai, Aj, Av);

  // Compute total number of unique IDs and tni
  double tmp = (double)hypre_data->un;
  double buf;
  comm_allreduce(&hypre_data->comm,gs_double,gs_add,&tmp,1,&buf);       
  hypre_data->tni = 1./tmp;

  // Create AMG solver
  HYPRE_BoomerAMGCreate(&hypre_data->solver);
  HYPRE_Solver solver = hypre_data->solver;

  int uparam = (int) param[0];
 
  // Set AMG parameters
  if (uparam) {
      int i;
      for (i = 0; i < NPARAM; i++) {
          hypre_param[i] = param[i+1]; 
           if (comm->id == 0)
              printf("Custom HYPREsettings[%d]: %.2f\n", i+1, hypre_param[i]);
      }
  } else {
      hypre_param[0] = 10;    /* HMIS                       */
      hypre_param[1] = 6;     /* Extended+i                 */
      hypre_param[2] = 0;     /* not used                   */
      hypre_param[3] = 3;     /* SOR smoother for crs level */
      hypre_param[4] = 1;
      hypre_param[5] = 0.25;
      hypre_param[6] = 0.1;
      hypre_param[7] = 0.0;
      hypre_param[8] = 0.01;
      hypre_param[9] = 0.05;
  }

  HYPRE_BoomerAMGSetCoarsenType(solver,hypre_param[0]);
  HYPRE_BoomerAMGSetInterpType(solver,hypre_param[1]);
  if (null_space == 1 || uparam) {
//     HYPRE_BoomerAMGSetRelaxType(solver, hypre_param[2]);
     HYPRE_BoomerAMGSetCycleRelaxType(solver, hypre_param[3], 3);
     HYPRE_BoomerAMGSetCycleNumSweeps(solver, hypre_param[4], 3);
  }
  HYPRE_BoomerAMGSetStrongThreshold(solver,hypre_param[5]);
  HYPRE_BoomerAMGSetNonGalerkinTol(solver,hypre_param[6]);
  HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver,hypre_param[7], 0);
  HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver,hypre_param[8], 1);
  HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver,hypre_param[9], 2);

  HYPRE_BoomerAMGSetMinCoarseSize(solver,2);
  HYPRE_BoomerAMGSetMaxIter(solver,1);
  HYPRE_BoomerAMGSetPrintLevel(solver,1);
  HYPRE_BoomerAMGSetTol(solver,0.0);

  // Create and initialize rhs and solution vectors
  HYPRE_Int ilower = hypre_data->ilower;
  HYPRE_Int iupper = ilower + (HYPRE_Int)(hypre_data->un) - 1;

  HYPRE_IJVectorCreate(hypre_data->comm.c,ilower,iupper,&hypre_data->b);
  HYPRE_IJVector b = hypre_data->b;
  HYPRE_IJVectorCreate(hypre_data->comm.c,ilower,iupper,&hypre_data->x);
  HYPRE_IJVector x = hypre_data->x;
  HYPRE_IJVectorSetObjectType(b,HYPRE_PARCSR);
  HYPRE_IJVectorSetObjectType(x,HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(b);
  HYPRE_IJVectorInitialize(x);
  HYPRE_IJVectorAssemble(b);
  HYPRE_IJVectorAssemble(x);

  // Perform AMG setup
  HYPRE_ParVector par_b;
  HYPRE_ParVector par_x;
  HYPRE_IJVectorGetObject(b,(void**) &par_b);
  HYPRE_IJVectorGetObject(x,(void**) &par_x);

  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJMatrixGetObject(hypre_data->A,(void**) &par_A);

  HYPRE_BoomerAMGSetup(solver,par_A,par_b,par_x);

  return hypre_data;
}

sint set_BoomerAMG_parameters(HYPRE_Solver *solver_ptr, const double *param)
{
  HYPRE_Solver solver = *solver_ptr;

  return 0;
}

void build_hypre_matrix(struct hypre_crs_data *hypre_data, uint n, const ulong
		      *id, uint nz_unassembled, const uint *Ai, const uint* Aj, const double *Av)
{
  struct crystal cr;
  struct comm comm;
  struct gs_data* gs_top;
  struct array uid; 
  ulong *uid_p = uid.ptr;
  struct rid *rid_map = tmalloc(struct rid,n);
  struct array mat;
  uint k; 
  struct rnz *out;
  uint *umap;

  comm = hypre_data->comm;
  gs_top = hypre_data->gs_top;

  // Initialize crystal router
  crystal_init(&cr, &comm);

  // Assign degrees of freedom
  assign_dofs_hypre(&uid,rid_map,id,n,comm.id,gs_top,&cr.data,&comm);

  // Build unassembled matrix
  array_init(struct rnz,&mat,nz_unassembled);
  for(out=mat.ptr,k=0;k<nz_unassembled;++k) {
    uint i=Ai[k], j=Aj[k]; double a=Av[k];
    if(id[i]==0 || id[j]==0 || fabs(a)==0) continue;
    out->v = a, out->i=rid_map[i], out->j=rid_map[j];
    ++out;
  }
  mat.n = out-(struct rnz*)mat.ptr;
  free(rid_map);

  // Assemble the matrix
  const ulong *const uid_ptr = uid.ptr;
  const uint pid = comm.id, np = comm.np;
  struct array nonlocal_id;
  uint nnz;
  const struct rnz *nz, *enz;
  const struct labelled_rid *rlbl;

  mat_distribute(&mat,row_distr,col_major,&cr);
  nnz = mat.n;

  mat_list_nonlocal_sorted(&nonlocal_id,&mat,row_distr,uid_ptr,&cr);
  rlbl = nonlocal_id.ptr;
  
  // Build Hypre matrix
  hypre_data->un = uid.n; 

  HYPRE_Int ilower = (HYPRE_Int)(uid_ptr[0]);
  hypre_data->ilower = ilower;
  HYPRE_Int iupper = ilower + (HYPRE_Int)(uid.n) - 1; 

  HYPRE_IJMatrixCreate(comm.c,ilower,iupper,ilower,iupper,&hypre_data->A);
  HYPRE_IJMatrix A_ij = hypre_data->A;
  HYPRE_IJMatrixSetObjectType(A_ij,HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(A_ij);

  HYPRE_Int mati, matj;
  double matv;
  HYPRE_Int nrows = 1, ncols = 1;
  
  for(nz=mat.ptr,enz=nz+nnz;nz!=enz;++nz) 
    {
      mati = (HYPRE_Int)(uid_ptr[nz->i.i]);
      matv = nz->v; 
      if(nz->j.p==pid)
        {
	  matj = (HYPRE_Int)(uid_ptr[nz->j.i]);
        }
      else 
        {
	  const uint jp = nz->j.p, ji = nz->j.i;
	  while(rlbl->rid.p<jp) ++rlbl;
	  if(rlbl->rid.p!=jp) printf("Error when assembling matrix\n");
	  while(rlbl->rid.i<ji) ++rlbl;
	  if(rlbl->rid.i!=ji) printf("Error when assembling matrix\n");
	  matj = (HYPRE_Int)(rlbl->id);
        }
      HYPRE_IJMatrixSetValues(A_ij, nrows, &ncols, &mati, &matj, &matv);
    }

  // Free data not necessary anymore
  array_free(&uid);
  array_free(&mat);
  array_free(&nonlocal_id);

  // Get the mapping
  hypre_data->umap=assign_dofs_hypre(&uid,0, id,n,pid,gs_top,&cr.data,&comm);
  array_free(&uid);
  crystal_free(&cr);

  HYPRE_IJMatrixAssemble(A_ij);
}

void ccrs_hypre_solve(double *x, struct hypre_crs_data *data, double *b)
{
  uint i; const uint un = data->un; const uint *const umap = data->umap;
  HYPRE_Int ilower = data->ilower;
  HYPRE_IJVector ij_x = data->x;
  HYPRE_IJVector ij_b = data->b;
  HYPRE_IJMatrix ij_A = data->A;
  HYPRE_Solver solver = data->solver;

  HYPRE_ParVector par_x;
  HYPRE_ParVector par_b;
  HYPRE_ParCSRMatrix par_A;
 
  gs(b, gs_double,gs_add, 1, data->gs_top, 0);
  for(i=0;i<un;++i) 
  {
    HYPRE_Int ii = ilower + (HYPRE_Int)i;
    double bb = b[umap[i]];
    HYPRE_IJVectorSetValues(ij_b,1,&ii,&bb);
  }

  HYPRE_IJVectorAssemble(ij_b);
  HYPRE_IJVectorGetObject(ij_b,(void**) &par_b);

  HYPRE_IJVectorAssemble(ij_x);
  HYPRE_IJVectorGetObject(ij_x,(void **) &par_x);

  HYPRE_IJMatrixGetObject(ij_A,(void**) &par_A);

  HYPRE_BoomerAMGSolve(solver,par_A,par_b,par_x);

  double avg = 0.0;
  for(i=0;i<un;++i) 
  {
    HYPRE_Int ii = ilower + (HYPRE_Int)i;
    double xx;
    HYPRE_IJVectorGetValues(ij_x,1,&ii,&xx);
    x[umap[i]] = xx;
    if (data->nullspace) avg += xx;
  }

  if (data->nullspace)
  {
    double buf;
    comm_allreduce(&data->comm,gs_double,gs_add,&avg,1,&buf);
    avg = avg*data->tni;
    for(i=0;i<un;++i) x[umap[i]] -= avg;
  }
  
  gs(x, gs_double,gs_add, 0, data->gs_top, 0);
}

void ccrs_hypre_free(struct hypre_crs_data *data)
{
  HYPRE_BoomerAMGDestroy(data->solver);
  HYPRE_IJMatrixDestroy(data->A);
  HYPRE_IJVectorDestroy(data->x);
  HYPRE_IJVectorDestroy(data->b);
  comm_free(&data->comm);
  gs_free(data->gs_top);
  free(data->umap);
  free(data);
}

// Functions borrowed from amg.c
/*
  The user ids   id[n]   are assigned to unique procs.
  
  Output
    uid --- ulong array; uid[0 ... uid.n-1]  are the ids owned by this proc
    rid_map --- id[i] is uid[ rid_map[i].i ] on proc rid_map[i].p
    map = assign_dofs_hypre(...) --- uid[i] is id[map[i]]

    map is returned only when rid_map is null, and vice versa
*/
static uint *assign_dofs_hypre(struct array *const uid,
                         struct rid *const rid_map,
                         const ulong *const id, const uint n,
                         const uint p,
			       struct gs_data *gs_top, buffer *const buf,
			       struct comm *comm)
{
  uint i,count, *map=0;
  struct rid *const rid = rid_map?rid_map:tmalloc(struct rid,n);
  for(i=0;i<n;++i) rid[i].p=p, rid[i].i=i;
  gs_vec(n?&rid[0].p:0,2, gs_sint,gs_add,0, gs_top,buf);
  for(count=0,i=0;i<n;++i) {
    if(rid[i].i==i && id[i]!=0 && rid[i].p==p) ++count; }
  array_init(ulong,uid,count); uid->n=count;
  if(rid_map==0) {
    map=tmalloc(uint,count); 
    for(count=0,i=0;i<n;++i) {
      if(rid[i].i==i && id[i]!=0 && rid[i].p==p) map[count++]=i; }
  }
  { // HYPRE NUMBERING
    ulong scan_out[2],scan_buf[2],uln;
    uln = (ulong) uid->n;
    comm_scan(scan_out,comm,gs_long,gs_add,&uln,1,scan_buf);
    ulong uilow = scan_out[0];
    ulong *const uid_p = uid->ptr;
    for(count=0,i=0;i<n;++i) if(rid[i].i==i && id[i]!=0 && rid[i].p==p)
			       uid_p[count]=uilow+(ulong)count, rid[i].i=count, ++count;
    //
  }
  if(rid_map) gs_vec(n?&rid[0].p:0,2, gs_sint,gs_add,0, gs_top,buf);
  else free(rid);
  return map;
}

static void mat_sort(struct array *const mat,
                     const enum mat_order order, buffer *const buf)
{
  switch(order) {
  case col_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                j.p,0,j.i,0, i.p,0,i.i,0, buf); break;
  case row_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                i.p,0,i.i,0, j.p,0,j.i,0, buf); break;
  }
}

/* assumes matrix is already sorted */
static void mat_condense_sorted(struct array *const mat)
{
  struct rnz *dst,*src, *const end=(struct rnz*)mat->ptr+mat->n;
  if(mat->n<=1) return;
  for(dst=mat->ptr;;++dst) {
    if(dst+1==end) return;
    if(nz_pos_equal(dst[0],dst[1])) break;
  }
  for(src=dst+1;src!=end;++src) {
    if(nz_pos_equal(*dst,*src))
      dst->v += src->v;
    else
      *(++dst) = *src;
  }
  mat->n = (dst+1)-(struct rnz*)mat->ptr;
}

static void mat_condense(
			 struct array *const mat, const enum mat_order order, buffer *const buf)
{
  mat_sort(mat,order,buf); mat_condense_sorted(mat);
}

static void mat_distribute(
			   struct array *const mat, const enum distr d, const enum mat_order o,
			   struct crystal *const cr)
{
  switch(d) {
  case row_distr: mat_condense(mat,row_major,&cr->data);
    sarray_transfer(struct rnz,mat, i.p,0, cr); break;
  case col_distr: mat_condense(mat,col_major,&cr->data);
    sarray_transfer(struct rnz,mat, j.p,0, cr); break;
  }
  mat_condense(mat,o,&cr->data);
}

static void mat_list_nonlocal_sorted(
				      struct array *const nonlocal_id,
				      const struct array *const mat, const enum distr d,
				      const ulong *uid, struct crystal *const cr)
 {
   const uint pid = cr->comm.id;
   struct rid last_rid;
   const struct rnz *p, *const e=(const struct rnz*)mat->ptr+mat->n;
   uint count; struct labelled_rid *out, *end;
#define BUILD_LIST(k) do { \
     last_rid.p=UINT_MAX,last_rid.i=UINT_MAX; \
     for(count=0,p=mat->ptr;p!=e;++p) { \
       if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
       last_rid=p->k; ++count; \
     } \
     array_init(struct labelled_rid, nonlocal_id, count); \
     nonlocal_id->n=count; out=nonlocal_id->ptr; \
     last_rid.p=UINT_MAX,last_rid.i=UINT_MAX; \
     for(p=mat->ptr;p!=e;++p) { \
       if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
       (out++)->rid=last_rid=p->k; \
     } \
 } while(0)
   switch(d) {
   case row_distr: BUILD_LIST(j); break;
   case col_distr: BUILD_LIST(i); break;
   }
  #undef BUILD_LIST
   sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
   for(out=nonlocal_id->ptr,end=out+nonlocal_id->n;out!=end;++out)
     out->id=uid[out->rid.i];
   sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
   sarray_sort_2(struct labelled_rid,nonlocal_id->ptr,nonlocal_id->n,
		 rid.p,0, rid.i,0, &cr->data);
 }

 static void mat_list_nonlocal(
			       struct array *const nonlocal_id,
			       struct array *const mat, const enum distr d,
			       const ulong *uid, struct crystal *const cr)
 {
   switch(d) {
   case row_distr: mat_sort(mat,col_major,&cr->data); break;
   case col_distr: mat_sort(mat,row_major,&cr->data); break;
   }
   mat_list_nonlocal_sorted(nonlocal_id,mat,d,uid,cr);
 }

#else

struct hypre_crs_data* ccrs_hypre_setup( uint n, const ulong *id,
                  uint nz, const uint *Ai, const uint *Aj, const double *Av,
                  const uint null_space, const struct comm *comm,
                  const double *param)
{
  fail(1,__FILE__,__LINE__,"recompile with HYPRE support.");
  exit(EXIT_FAILURE);
  return NULL;
}

void ccrs_hypre_solve(double *x, struct hypre_crs_data *data, double *b)
{
  fail(1,__FILE__,__LINE__,"recompile with HYPRE support.");
  exit(EXIT_FAILURE);
  while(1);
}


void ccrs_hypre_free(struct hypre_crs_data *data)
{
  fail(1,__FILE__,__LINE__,"recompile with HYPRE support.");
  exit(EXIT_FAILURE);
  while(1);
}

#endif
