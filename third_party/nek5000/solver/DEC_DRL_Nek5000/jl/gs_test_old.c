/* simple stand-alone test for parallel gather-scatter routines
   assumes gather-scatter routines were compiled with default names
   can compile to sequential version if MPI is not defined
   
   the test is as follows, where N is the number of procs:
     there are N physical nodes (vertices)
     each proc has 2 local/virtual nodes mapping to each physical node,
       for a total of 2*N*N virtual nodes
     virtual nodes are given values that correspond to a sequential ordering
       (so that they range from 0 to 2*N*N-1)
       the addition operation is performed and the result is checked,
       the correct result being known a priori
     the addition operation is also checked, in a similar manner, for
       both the cpgs_op_vec and cpgs_op_many routines with vector dimension 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef MPI
#  include <mpi.h>
#else
   typedef void MPI_Comm;
#endif
#include "name.h"
#include "types.h"

typedef long real;
sint datatype = 4;

#define fgs_setup     FORTRAN_NAME(gs_setup    ,GS_SETUP    )
#define fgs_op        FORTRAN_NAME(gs_op       ,GS_OP       )
#define fgs_op_vec    FORTRAN_NAME(gs_op_vec   ,GS_OP_VEC   )
#define fgs_op_many   FORTRAN_NAME(gs_op_many  ,GS_OP_MANY  )
#define fgs_op_fields FORTRAN_NAME(gs_op_fields,GS_OP_FIELDS)
#define fgs_free      FORTRAN_NAME(gs_free     ,GS_FREE     )

void fgs_setup(sint *handle, const slong id[], const sint *n,
               const MPI_Comm *comm, const sint *np);
void fgs_op(const sint *handle, void *u, const sint *dom, const sint *op,
            const sint *transpose);
void fgs_op_vec(const sint *handle, void *u, const sint *n,
                const sint *dom, const sint *op, const sint *transpose);
void fgs_op_many(const sint *handle, void *u1, void *u2, void *u3,
                 void *u4, void *u5, void *u6, const sint *n,
                 const sint *dom, const sint *op, const sint *transpose);
void fgs_free(const sint *handle);

void assert_is_zero(real v)
{
  if(fabs(v) < 1e-20) return;
  printf("test failed\n");
  exit(1);
}

int main(int narg, char* arg[])
{
  sint transpose=0;
  sint id=0,np=1;
  sint i,handle,maxv=3;
  real *u;
  slong *glindex;
#ifndef MPI
  int comm;
#else
  MPI_Comm comm;
  MPI_Init(&narg,&arg);
  MPI_Comm_dup(MPI_COMM_WORLD,&comm);
  { int i;
    MPI_Comm_rank(comm,&i); id=i;
    MPI_Comm_size(comm,&i); np=i;
  }
#endif

  glindex = malloc(np*2*sizeof(slong));
  for(i=0;i<np;++i) glindex[2*i+1] = glindex[2*i] = i+1;
  i=np*2;
  fgs_setup(&handle,glindex,&i,&comm,&np);
  free(glindex);
  
  u = malloc(np*2*sizeof(real));
  for(i=0;i<np;++i) u[2*i  ] = (real)( 2*np*id + 2*i ),
                    u[2*i+1] = (real)( 2*np*id + 2*i+1 );
  /*for(i=0;i<np;++i) printf(" (%g %g)", u[2*i], u[2*i+1]); printf("\n");*/
  i=1, fgs_op(&handle,u,&datatype,&i,&transpose);
  /*for(i=0;i<np;++i) printf(" (%g %g)", u[2*i], u[2*i+1]); printf("\n");*/
  for(i=0;i<np;++i) assert_is_zero( np*(2*np*(np-1)+4*i+1) - u[2*i] ),
                    assert_is_zero( np*(2*np*(np-1)+4*i+1) - u[2*i+1]  );
  free(u);

  u = malloc(np*2*3*sizeof(real));
  for(i=0;i<np;++i)
    u[3*(2*i  )+0] = (real)( 3*(2*np*id + 2*i  ) + 0 ),
    u[3*(2*i  )+1] = (real)( 3*(2*np*id + 2*i  ) + 1 ),
    u[3*(2*i  )+2] = (real)( 3*(2*np*id + 2*i  ) + 2 ),
    u[3*(2*i+1)+0] = (real)( 3*(2*np*id + 2*i+1) + 0 ),
    u[3*(2*i+1)+1] = (real)( 3*(2*np*id + 2*i+1) + 1 ),
    u[3*(2*i+1)+2] = (real)( 3*(2*np*id + 2*i+1) + 2 );
  /*for(i=0;i<np;++i) {
    int j;
    printf("%d: ( ", id);
    for(j=3*(2*i);j<=3*(2*i+1)+2;++j) printf("%g ",u[j]);
    printf(")\n");
  }*/
  i=1, maxv=3, fgs_op_vec(&handle,u,&maxv,&datatype,&i,&transpose);
  /*for(i=0;i<np;++i) {
    int j;
    printf("%d: ( ", id);
    for(j=3*(2*i);j<=3*(2*i+1)+2;++j) printf("%g ",u[j]);
    printf(")\n");
  }*/
  for(i=0;i<np;++i)
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*0) - u[3*(2*i  )+0] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*1) - u[3*(2*i  )+1] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*2) - u[3*(2*i  )+2] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*0) - u[3*(2*i+1)+0] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*1) - u[3*(2*i+1)+1] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*2) - u[3*(2*i+1)+2] );
  free(u);

  u = malloc(np*2*3*sizeof(real));
  for(i=0;i<np;++i)
    u[2*np*0+(2*i  )] = (real)( 3*(2*np*id + 2*i  ) + 0 ),
    u[2*np*1+(2*i  )] = (real)( 3*(2*np*id + 2*i  ) + 1 ),
    u[2*np*2+(2*i  )] = (real)( 3*(2*np*id + 2*i  ) + 2 ),
    u[2*np*0+(2*i+1)] = (real)( 3*(2*np*id + 2*i+1) + 0 ),
    u[2*np*1+(2*i+1)] = (real)( 3*(2*np*id + 2*i+1) + 1 ),
    u[2*np*2+(2*i+1)] = (real)( 3*(2*np*id + 2*i+1) + 2 );
  i=1, maxv=3, fgs_op_many(&handle,u,u+2*np,u+4*np,0,0,0,&maxv,
                           &datatype,&i,&transpose);
  for(i=0;i<np;++i)
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*0) - u[2*np*0+(2*i  )] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*1) - u[2*np*1+(2*i  )] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*2) - u[2*np*2+(2*i  )] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*0) - u[2*np*0+(2*i+1)] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*1) - u[2*np*1+(2*i+1)] ),
    assert_is_zero( np*(6*np*(np-1)+12*i+3+2*2) - u[2*np*2+(2*i+1)] );
  free(u);
  
  fgs_free(&handle);
  printf("test on node %d/%d succeeded\n", (int)id+1, (int)np);
#ifdef MPI  
  MPI_Comm_free(&comm);
  MPI_Finalize();
#endif
  return 0;
}
