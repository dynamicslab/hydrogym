#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "comm.h"
#include "mem.h"
#include "crs.h"

#define M 3

int main(int narg, char* arg[])
{
  uint n; ulong *xid;
  uint nz; uint *Ai, *Aj; double *A;
  uint i;
  double *x, *b, *x2;

  struct crs_data *crs;
  comm_ext world; int id,np;
  struct comm comm;
#ifdef MPI
  MPI_Init(&narg,&arg);
  world = MPI_COMM_WORLD;
  MPI_Comm_size(world,&np);
#else
  world=0, np=1;
#endif

  comm_init(&comm,world);
  id = comm.id;

  n=M+1; if(id==np-1) --n;
  xid = tmalloc(ulong,n); x=tmalloc(double,3*n), b=x+n, x2=b+n;
  for(i=0;i<n;++i) xid[i]=1+id*M+i;
  nz=2*M; if(id==np-1) --nz;
  Ai=tmalloc(uint,2*nz), Aj=Ai+nz;
  A =tmalloc(double,nz);
  for(i=0;i<M;++i) Ai[i]=i,Aj[i]=i,A[i]=2;
  if(id==0) A[0]=1;
  if(id==np-1) A[n-1]=1;
  for(i=M;i<nz;++i) Ai[i]=i-M,Aj[i]=i-M+1,A[i]=-1;

  crs = crs_setup(n,xid, nz,Ai,Aj,A, 1, &comm);
  crs_stats(crs);
  
  {
    double tn = M*np, mean = 1*(tn+1)/2;
    double avg = 1*(tn-1)*(tn+1)/12;
    for(i=0;i<n;++i) x[i]=(xid[i]-mean)*(xid[i]-mean)-avg;
    for(i=0;i<n;++i) b[i]=A[i]*x[i]
                - (i>0   || id!=0    ? (xid[i]-1-mean)*(xid[i]-1-mean)-avg : 0)
                - (i+1<n || id!=np-1 ? (xid[i]+1-mean)*(xid[i]+1-mean)-avg : 0);
    if(id!=np-1) b[n-1]=0;
  }
  crs_solve(x2,crs,b);
  crs_free(crs);
  comm_free(&comm);
  
  { double dif=0;
    for(i=0;i<n;++i) {
      double d=fabs(x[i]-x2[i])/(.1+fabs(x[i]));
      if(d>dif) dif=d;
    }
    printf("%d : max dif = %g\n",id,dif);
  }
  free(A); free(Ai); free(x); free(xid);
  
#ifdef MPI
  MPI_Finalize();
#endif

  if(id==0) printf("test successful\n");
  
  return 0;
}

