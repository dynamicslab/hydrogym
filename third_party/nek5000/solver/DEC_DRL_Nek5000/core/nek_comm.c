/*
 * MPI wrappers
 *
 * timer:
 * 0 MPI_Allreduce        
 * 1 MPI_Allreduce(sync)  
 * 2 MPI_Waitall     
 * 3 MPI_Barrier
 * 4 MPI_Irecv
 * 5 MPI_Isend
 * 6 MPI_Recv
 * 7 MPI_Send
 *
 */
#include <stdio.h>
#ifdef MPI
#include <mpi.h>
#endif
#include <math.h>

#ifdef UPCASE
#  define FORTRAN_NAME(low,up) up
#else
#ifdef UNDERSCORE
#  define FORTRAN_NAME(low,up) low##_
#else
#  define FORTRAN_NAME(low,up) low
#endif
#endif

#define nek_comm_settings  FORTRAN_NAME(nek_comm_settings, NEK_COMM_SETTINGS)
#define nek_comm_getstat   FORTRAN_NAME(nek_comm_getstat, NEK_COMM_GETSTAT)
#define nek_comm_startstat FORTRAN_NAME(nek_comm_startstat, NEK_COMM_STARTSTAT)

#define NTIMER 8
#define NCOUNTER NTIMER

int SYNC = 0;
int TIMING = 1;
double MPI_TIMERS[NTIMER] = {(double)0.0};
int COUNTER[NCOUNTER] = {0};


#if defined(MPITIMER)

void nek_comm_settings(int *sync,int *timing)
{
     SYNC   = *sync;
     TIMING = *timing;
}

void nek_comm_getstat(double *timers, int *counters)
{
     int i;
     double t0,t1;

     /* estimate timer overhead */
     t0 = PMPI_Wtime();
     t1 = PMPI_Wtime();
     while (t1 == t0) t1 = PMPI_Wtime();
     t1 -= t0; 


     /* substract timer overhead */

 /*
     MPI_TIMERS[0] -= 3*COUNTER[0] * t1; 
     MPI_TIMERS[1] -= 2*COUNTER[1] * t1;
     MPI_TIMERS[2] -= 2*COUNTER[2] * t1;
     MPI_TIMERS[3] -= 2*COUNTER[3] * t1; 
     MPI_TIMERS[4] -= 2*COUNTER[4] * t1; 
     MPI_TIMERS[5] -= 2*COUNTER[5] * t1; 
     MPI_TIMERS[6] -= 2*COUNTER[6] * t1; 
     MPI_TIMERS[7] -= 2*COUNTER[7] * t1; 
 */

     MPI_TIMERS[1] += MPI_TIMERS[0];
     COUNTER[1] = COUNTER[0];
     for (i = 0; i < NTIMER; i++) timers[i] = fmax(MPI_TIMERS[i],(double)0.0);
     for (i = 0; i < NCOUNTER; i++) counters[i] = COUNTER[i];
}

void nek_comm_startstat(void)
{
     int i;
     for (i = 0; i < NTIMER; i++) MPI_TIMERS[i] = 0.0;
     for (i = 0; i < NCOUNTER; i++) COUNTER[i] = 0;
}

/* FORTRAN wrappers */

#pragma weak MPI_ALLREDUCE = mpi_allreduce_f
#pragma weak mpi_allreduce = mpi_allreduce_f
#pragma weak mpi_allreduce_ = mpi_allreduce_f
#pragma weak mpi_allreduce__ = mpi_allreduce_f
void mpi_allreduce_f(char *sendbuf, char *recvbuf, MPI_Fint *count,
                    MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm,
                    MPI_Fint *ierr)
{
    MPI_Comm c_comm = MPI_Comm_f2c(*comm);
    MPI_Datatype c_type = MPI_Type_f2c(*datatype);
    MPI_Op c_op = MPI_Op_f2c(*op);
    double t0,t1;

    COUNTER[0]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    if (SYNC) *ierr = MPI_Barrier(c_comm);
    if (TIMING) t1 = PMPI_Wtime(); 
    *ierr = PMPI_Allreduce(sendbuf, recvbuf, *count, c_type, c_op, c_comm);
    if (TIMING) {MPI_TIMERS[0] += PMPI_Wtime()-t1; MPI_TIMERS[1] += t1-t0;}
}

#pragma weak MPI_BARRIER   = mpi_barrier_f
#pragma weak mpi_barrier   = mpi_barrier_f
#pragma weak mpi_barrier_  = mpi_barrier_f
#pragma weak mpi_barrier__ = mpi_barrier_f
void mpi_barrier_f(MPI_Fint *comm, MPI_Fint *ierr)
{
    MPI_Comm c_comm = MPI_Comm_f2c(*comm);
    double t0;

    COUNTER[3]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    *ierr = PMPI_Barrier(c_comm);
    if (TIMING) MPI_TIMERS[3] += PMPI_Wtime()-t0;
}

#pragma weak MPI_RECV   = mpi_recv_f
#pragma weak mpi_recv   = mpi_recv_f
#pragma weak mpi_recv_  = mpi_recv_f
#pragma weak mpi_recv__ = mpi_recv_f
void mpi_recv_f(char *buf, MPI_Fint *count, MPI_Fint *datatype,
                MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, 
                MPI_Fint *status, MPI_Fint *ierr)
{
    MPI_Comm c_comm = MPI_Comm_f2c(*comm);
    MPI_Datatype c_type = MPI_Type_f2c(*datatype);
    MPI_Status *c_status = (MPI_Status *) status; /* FORTRAN_INTEGER == SIZEOF_INT */
    double t0;

    COUNTER[6]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    *ierr = PMPI_Recv(buf, *count, c_type, *source, *tag, c_comm, c_status);
    if (TIMING) MPI_TIMERS[6] += PMPI_Wtime()-t0;
}

#pragma weak MPI_SEND   = mpi_send_f
#pragma weak mpi_send   = mpi_send_f
#pragma weak mpi_send_  = mpi_send_f
#pragma weak mpi_send__ = mpi_send_f
void mpi_send_f(char *buf, MPI_Fint *count, MPI_Fint *datatype,
                MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierr)
{
    MPI_Comm c_comm = MPI_Comm_f2c(*comm);
    MPI_Datatype c_type = MPI_Type_f2c(*datatype);
    double t0;

    COUNTER[7]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    *ierr = PMPI_Send(buf, *count, c_type, *dest, *tag, c_comm);
    if (TIMING) MPI_TIMERS[7] += PMPI_Wtime()-t0;
}


/* C wrappers */


#pragma weak MPI_Allreduce  = mpi_allreduce_c
#pragma weak MPI_Allreduce_ = mpi_allreduce_c
int mpi_allreduce_c(void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int ierr;
    double t0,t1;

    COUNTER[0]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    if (SYNC) ierr = MPI_Barrier(comm);
    if (TIMING) t1 = PMPI_Wtime(); 
    ierr = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    if (TIMING) {MPI_TIMERS[0] += PMPI_Wtime()-t1; MPI_TIMERS[1] += t1-t0;}
    return ierr;
}

#pragma weak MPI_Waitall  = mpi_waitall_c
#pragma weak MPI_Waitall_ = mpi_waitall_c
int mpi_waitall_c(int count, MPI_Request *request, MPI_Status *status)
{
    int ierr;
    double t0,t1;


    COUNTER[2]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Waitall(count, request, status);
    if (TIMING) MPI_TIMERS[2] += PMPI_Wtime()-t0;
    return ierr;
}

#pragma weak MPI_Barrier  = mpi_barrier_c
#pragma weak MPI_Barrier_ = mpi_barrier_c
int mpi_barrier_c(MPI_Comm comm)
{
    int ierr;
    double t0;

    COUNTER[3]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Barrier(comm);
    if (TIMING) MPI_TIMERS[3] += PMPI_Wtime()-t0;
    return ierr;
}

#pragma weak MPI_Irecv  = mpi_irecv_c
#pragma weak MPI_Irecv_ = mpi_irecv_c
int mpi_irecv_c(void *buf, int count, MPI_Datatype type, int source,
                int tag, MPI_Comm comm, MPI_Request *request)
{
    int ierr;
    double t0;

    COUNTER[4]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Irecv(buf,count,type,source,tag,comm,request);
    if (TIMING) MPI_TIMERS[4] += PMPI_Wtime()-t0;
    return ierr;
}


#pragma weak MPI_Isend  = mpi_isend_c
#pragma weak MPI_Isend_ = mpi_isend_c
int mpi_isend_c(void *buf, int count, MPI_Datatype type, int dest,
                int tag, MPI_Comm comm, MPI_Request *request)
{
    int ierr;
    double t0;

    COUNTER[5]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Isend(buf,count,type,dest,tag,comm,request);
    if (TIMING) MPI_TIMERS[5] += PMPI_Wtime()-t0;
    return ierr;
}

#pragma weak MPI_Recv  = mpi_recv_c
#pragma weak MPI_Recv_ = mpi_recv_c
int mpi_recv_c(void *buf, int count, MPI_Datatype type, int source,
               int tag, MPI_Comm comm, MPI_Status *status)
{
    int ierr;
    double t0;

    COUNTER[6]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Recv(buf,count,type,source,tag,comm,status);
    if (TIMING) MPI_TIMERS[6] += PMPI_Wtime()-t0;
    return ierr;
}

#pragma weak MPI_Send  = mpi_send_c
#pragma weak MPI_Send_ = mpi_send_c
int mpi_send_c(void *buf, int count, MPI_Datatype type, int dest,
               int tag, MPI_Comm comm)
{
    int ierr;
    double t0;

    COUNTER[7]++;

    if (TIMING) t0 = PMPI_Wtime(); 
    ierr = PMPI_Send(buf,count,type,dest,tag,comm);
    if (TIMING) MPI_TIMERS[7] += PMPI_Wtime()-t0;
    return ierr;
}

#else

void nek_comm_settings(int *sync,int *timing){}
void nek_comm_getstat(double *timer, int *counter)
{
     int i;
     for (i = 0; i < NTIMER; i++) timer[i] = MPI_TIMERS[i];
     for (i = 0; i < NCOUNTER; i++) counter[i] = COUNTER[i];
}
void nek_comm_startstat(void){}

#endif

