#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "comm.h"
#include "mem.h"
#include "gs_defs.h"
#include "gs.h"

static void test(const struct comm *comm)
{
  uint i,np=comm->np,id=comm->id;
  slong *glindex = tmalloc(slong,np*2);
  char *out, *buf = tmalloc(char,80+np*2*30);
  struct gs_data *gsh;
  
  for(i=0;i<np;++i) glindex[2*i+1]=glindex[2*i]=i+1;
  
  out = buf+sprintf(buf, "%03d bgn : [", (int)comm->id);
  for(i=0;i<np*2;++i) out += sprintf(out, " %+d", (int)glindex[i]);
  sprintf(out," ]"), puts(buf);
  
  gs_unique(glindex,np*2,comm);

  out = buf+sprintf(buf, "%03d end : [", (int)comm->id);
  for(i=0;i<np*2;++i) out += sprintf(out, " %+d", (int)glindex[i]);
  sprintf(out," ]"), puts(buf);


  for(i=0;i<np;++i) glindex[2*i+1]=glindex[2*i]=i+1;
  gsh=gs_setup(glindex,np*2,comm,1,gs_auto,1);
  for(i=0;i<np;++i) glindex[2*i+1]=glindex[2*i]=id;
  gs(glindex,gs_slong,gs_add,0,gsh,0);
  gs_free(gsh);

  out = buf+sprintf(buf, "%03d own : [", (int)comm->id);
  for(i=0;i<np*2;++i) out += sprintf(out, " %+d", (int)glindex[i]);
  sprintf(out," ]"), puts(buf);

  free(buf);
  free(glindex);
}

int main(int narg, char *arg[])
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
