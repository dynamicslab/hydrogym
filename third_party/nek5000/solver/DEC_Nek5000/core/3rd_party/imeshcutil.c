/* I can't believe I need a function to do this... */
#ifdef MPI
#include "mpi.h"
#else
#define MPI_Comm int
#endif

//void moab_comm_f2c_(int *fcomm, MPI_Comm *ccomm)
#ifdef UPCASE
#  define FORTRAN_NAME(low,up) up
#else
#ifdef UNDERSCORE
#  define FORTRAN_NAME(low,up) low##_
#else
#  define FORTRAN_NAME(low,up) low
#endif
#endif

#define moab_comm_f2c FORTRAN_NAME(moab_comm_f2c, MOAB_COMM_F2C)

void moab_comm_f2c(int *fcomm, MPI_Comm *ccomm)
{
#ifdef MPI
  *ccomm = MPI_Comm_f2c(*fcomm);
#endif
}

