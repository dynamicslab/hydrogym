#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "gslib.h"
#include "crs.h"

/*--------------------------------------------------------------------------
   FORTRAN wrapper interface to coarse solver
  --------------------------------------------------------------------------*/

#undef crs_xxt_setup
#undef crs_xxt_solve
#undef crs_xxt_stats
#undef crs_xxt_free
#define ccrs_xxt_setup   PREFIXED_NAME(crs_xxt_setup)
#define ccrs_xxt_solve   PREFIXED_NAME(crs_xxt_solve)
#define ccrs_xxt_stats   PREFIXED_NAME(crs_xxt_stats)
#define ccrs_xxt_free    PREFIXED_NAME(crs_xxt_free )

#undef crs_amg_setup
#undef crs_amg_solve
#undef crs_amg_stats
#undef crs_amg_free
#define ccrs_amg_setup   PREFIXED_NAME(crs_amg_setup)
#define ccrs_amg_solve   PREFIXED_NAME(crs_amg_solve)
#define ccrs_amg_stats   PREFIXED_NAME(crs_amg_stats)
#define ccrs_amg_free    PREFIXED_NAME(crs_amg_free )

#define fcrs_setup   FORTRAN_NAME(crs_setup,CRS_SETUP)
#define fcrs_solve   FORTRAN_NAME(crs_solve,CRS_SOLVE)
#define fcrs_stats   FORTRAN_NAME(crs_stats,CRS_STATS)
#define fcrs_free    FORTRAN_NAME(crs_free ,CRS_FREE)

static struct crs_data **handle_array = 0;
static int handle_max = 0;
static int handle_n = 0;
static int *sid_array; 

#define CHECK_HANDLE(func) do \
  if(*handle<0 || *handle>=handle_n || !handle_array[*handle]) \
    fail(1,__FILE__,__LINE__,func ": invalid handle"); \
while(0)

void fcrs_setup(sint *handle, const sint *sid, const MPI_Fint *comm, const sint *np,
                const sint *n, const slong id[], const sint *nz,
                const sint Ai[], const sint Aj[], const double A[],
                const sint *null_space, const double *param,
                const char *datafname, uint *ierr)
{
  struct comm c;
  if(handle_n==handle_max)
    handle_max+=handle_max/2+1,
    handle_array=trealloc(struct crs_data*,handle_array,handle_max),
    sid_array=trealloc(int,sid_array,handle_max);
  comm_init_check(&c, *comm, *np);

  sid_array[handle_n]=*sid;

  switch(sid_array[handle_n]) {
    case 0: handle_array[handle_n]=ccrs_xxt_setup(*n,(const ulong*)id,
                                                  *nz,(const uint*)Ai,(const uint*)Aj,A,
                                                  *null_space,&c); break;
    case 1: handle_array[handle_n]=ccrs_amg_setup(*n,(const ulong*)id,
                                                  *nz,(const uint*)Ai,(const uint*)Aj,A,
                                                  *null_space,&c,
                                                  datafname,ierr); break;
    case 2: handle_array[handle_n]=ccrs_hypre_setup(*n,(const ulong*)id,
                                                  *nz,(const uint*)Ai,(const uint*)Aj,A,
                                                  *null_space,&c,param); break;
  }

  comm_free(&c);
  *handle = handle_n++;
}

void fcrs_solve(const sint *handle, double x[], double b[])
{
  CHECK_HANDLE("crs_solve");
  switch(sid_array[*handle]) {
    case 0: ccrs_xxt_solve(x,handle_array[*handle],b); break;
    case 1: ccrs_amg_solve(x,handle_array[*handle],b); break;
    case 2: ccrs_hypre_solve(x,handle_array[*handle],b); break;
  }
}

void fcrs_free(sint *handle)
{
  CHECK_HANDLE("crs_free");
  switch(sid_array[*handle]) {
    case 0: ccrs_xxt_free(handle_array[*handle]); break;
    case 1: ccrs_amg_free(handle_array[*handle]); break;
    case 2: ccrs_hypre_free(handle_array[*handle]); break;
  }
  handle_array[*handle] = 0;
}
