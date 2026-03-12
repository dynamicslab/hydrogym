#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "gs_defs.h"
#include "comm.h"
#include "gs.h"
#include "crs.h"

void test(const struct comm *const comm)
{
  const double A[16] = {  2, -1, -1,  0,
                         -1,  2,  0, -1,
                         -1,  0,  2, -1,
                          0, -1, -1,  2 };
  const uint Ai[16]  = { 0, 0, 0, 0,
                         1, 1, 1, 1,
                         2, 2, 2, 2,
                         3, 3, 3, 3 },
             Aj[16]  = { 0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 1, 2, 3 };
  ulong xid[4]; slong uid[4];
  double x[4]={1,1,1,1}, b[4], bmean;
  uint i, w, gn, px, py;
  
  slong *xgid=0; double *xg=0; struct gs_data *gsh;
  
  struct crs_data *crs;

  w = ceil(sqrt(comm->np)); gn = (w+1)*(w+1);
  
  if(comm->id==0) printf("arranging procs in a %u x %u square\n", w, w);
  
  px = comm->id%w, py = comm->id/w;
  b[0] = xid[0] = (w+1)*py    +px+1;
  b[1] = xid[1] = (w+1)*py    +px+2;
  b[2] = xid[2] = (w+1)*(py+1)+px+1;
  b[3] = xid[3] = (w+1)*(py+1)+px+2;

  gn = comm_reduce_slong(comm, gs_max, (const slong*)&xid[3],1);
  bmean = comm_reduce_double(comm, gs_add, b,4)/gn;

  gsh = gs_setup((const slong*)xid,4, comm,0,gs_crystal_router,0);
  gs(x,gs_double,gs_add,0,gsh,0);
  gs(b,gs_double,gs_add,0,gsh,0);
  for(i=0;i<4;++i) b[i]=xid[i]-bmean/x[i];
  gs(b,gs_double,gs_add,0,gsh,0);
  gs_free(gsh);
  
  gsh = gs_setup((const slong*)xid,4, comm,1,gs_crystal_router,0);
  for(i=0;i<4;++i) uid[i]=comm->id;
  gs(uid,gs_slong,gs_min,0,gsh,0);
  gs_free(gsh);
  for(i=0;i<4;++i) uid[i] = (uid[i]==comm->id?(slong)xid[i]:-(slong)xid[i]);

  if(comm->id==0) {
    xgid = tmalloc(slong, gn);
    xg   = tmalloc(double,gn);
    for(i=0;i<gn;++i) xgid[i] = -(slong)(i+1);
    for(i=0;i<4;++i) xgid[xid[i]-1] = uid[i];
  }
  gsh = gs_setup(comm->id?uid:xgid,comm->id?4:gn, comm,0,gs_crystal_router,0);


  if(comm->id==0) for(i=0;i<4;++i) xg[xid[i]-1]=b[i];
  gs(comm->id?b:xg,gs_double,gs_add, 0, gsh, 0);
  if(comm->id==0) for(i=0;i<gn;++i) printf("b[%u] = %g\n",i,xg[i]);
  for(i=0;i<4;++i) b[i]=xid[i]-bmean/x[i];

  crs = crs_setup(4,xid, 16,Ai,Aj,A, 1, comm);

  crs_solve(x,crs,b);

  crs_stats(crs);

  crs_free(crs);

  if(comm->id==0) for(i=0;i<4;++i) xg[xid[i]-1]=x[i];
  gs(comm->id?x:xg,gs_double,gs_add, 0, gsh, 0);
  if(comm->id==0) for(i=0;i<gn;++i) printf("x[%u] = %g\n",i,xg[i]);

  gs_free(gsh);  
  if(comm->id==0) free(xg), free(xgid);
}

int main(int narg, char* arg[])
{
  comm_ext world; int np;
  struct comm comm;
#ifdef MPI
  MPI_Init(&narg,&arg);
  world = MPI_COMM_WORLD;
  MPI_Comm_size(world,&np);
#else
  world=0, np=1;
#endif

  comm_init(&comm,world);
  test(&comm);
  comm_free(&comm);
  
#ifdef MPI
  MPI_Finalize();
#endif

  return 0;
}

