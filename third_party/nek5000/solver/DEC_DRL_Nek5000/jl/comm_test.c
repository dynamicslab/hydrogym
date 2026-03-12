#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "name.h"
#include "fail.h"
#include "types.h"
#include "gs_defs.h"
#include "comm.h"

int main(int narg, char *arg[])
{
  comm_ext world; int np;
  struct comm comm;
  ulong sum[2],r[2],v, test;
#ifdef MPI
  MPI_Init(&narg,&arg);
  world = MPI_COMM_WORLD;
  MPI_Comm_size(world,&np);
#else
  world=0, np=1;
#endif

  comm_init(&comm,world);
  
  v = comm.id+1;
  test = comm_reduce_slong(&comm,gs_add,(slong*)&v,1);
  comm_scan(sum, &comm,gs_slong,gs_add, &v,1, r);
  printf("%02d: %d %d %d\n",(int)comm.id,(int)sum[0],(int)sum[1],(int)test);

  comm_free(&comm);
  
#ifdef MPI
  MPI_Finalize();
#endif

  return 0;
}
