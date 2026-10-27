#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "comm.h"
#include "crs.h"

/*--------------------------------------------------------------------------
   FORTRAN Interface to coarse solver
  --------------------------------------------------------------------------*/

#undef crs_setup
#undef crs_solve
#undef crs_stats
#undef crs_free
#define ccrs_setup   PREFIXED_NAME(crs_setup)
#define ccrs_solve   PREFIXED_NAME(crs_solve)
#define ccrs_stats   PREFIXED_NAME(crs_stats)
#define ccrs_free    PREFIXED_NAME(crs_free )

#define fcrs_setup   FORTRAN_NAME(crs_setup,CRS_SETUP)
#define fcrs_solve   FORTRAN_NAME(crs_solve,CRS_SOLVE)
#define fcrs_stats   FORTRAN_NAME(crs_stats,CRS_STATS)
#define fcrs_free    FORTRAN_NAME(crs_free ,CRS_FREE)

static struct crs_data **handle_array = 0;
static int handle_max = 0;
static int handle_n = 0;

void fcrs_setup(sint *handle, const MPI_Fint *comm, const sint *np,
                const sint *n, const slong id[], const sint *nz,
                const sint Ai[], const sint Aj[], const double A[],
                const sint *null_space)
{
  struct comm c;
  if(handle_n==handle_max)
    handle_max+=handle_max/2+1,
    handle_array=trealloc(struct crs_data*,handle_array,handle_max);
  comm_init_check(&c, *comm, *np);
  handle_array[handle_n]=ccrs_setup(*n,(const ulong*)id,
                                    *nz,(const uint*)Ai,(const uint*)Aj,A,
                                    *null_space,&c);
  comm_free(&c);
  *handle = handle_n++;
}

#define CHECK_HANDLE(func) do \
  if(*handle<0 || *handle>=handle_n || !handle_array[*handle]) \
    fail(1,__FILE__,__LINE__,func ": invalid handle"); \
while(0)

void fcrs_solve(const sint *handle, double x[], double b[])
{
  CHECK_HANDLE("crs_solve");
  ccrs_solve(x,handle_array[*handle],b);
}

void fcrs_stats(const sint *handle)
{
  CHECK_HANDLE("crs_stats");
  ccrs_stats(handle_array[*handle]);
}

void fcrs_free(sint *handle)
{
  CHECK_HANDLE("crs_free");
  ccrs_free(handle_array[*handle]);
  handle_array[*handle] = 0;
}


