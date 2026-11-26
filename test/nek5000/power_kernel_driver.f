c Driver of test/test_nek5000_power_kernel.py: the power kernel of the
c NACA0012 wing case (drl_power_kernel, extracted from drl_power.f) on
c the nodes of every rank, the global sum as gop does it, and (rank 0)
c the two projections of drl_state.f (statements extracted from it).
      program pwtest
      implicit none
      include 'mpif.h'
      integer ierr,rank,n,i,k
      real, allocatable :: w(:),vn(:),u2(:),p(:),pr(:),pn(:)
      real, allocatable :: vx(:),vy(:),body_sin(:),body_cos(:),rotV(:)
      real s(6),sg(6),rho
      character*64 fn
      call mpi_init(ierr)
      call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
      write(fn,'(a,i0,a)') 'nodes_',rank,'.txt'
      open(10,file=fn,status='old')
      read(10,*) n,rho
      allocate(w(max(n,1)),vn(max(n,1)),u2(max(n,1)),p(max(n,1)))
      allocate(pr(max(n,1)),pn(max(n,1)))
      do i=1,n
         read(10,*) w(i),vn(i),u2(i),p(i),pr(i)
      enddo
      close(10)
      call drl_power_kernel(n,w,vn,u2,p,pr,rho,s,pn)
      call mpi_allreduce(s,sg,6,MPI_DOUBLE_PRECISION,MPI_SUM,
     $                   MPI_COMM_WORLD,ierr)
      write(fn,'(a,i0,a)') 'out_',rank,'.txt'
      open(11,file=fn)
      write(11,'(6es25.16e3)') (sg(k),k=1,6)
      do i=1,n
         write(11,'(es25.16e3)') pn(i)
      enddo
      close(11)
      if (rank.eq.0) then
         open(12,file='proj.txt',status='old')
         read(12,*) n
         allocate(vx(n),vy(n),body_sin(n),body_cos(n),rotV(n))
         do i=1,n
            read(12,*) vx(i),vy(i),body_sin(i),body_cos(i)
         enddo
         close(12)
         open(13,file='proj_out.txt')
         include 'proj_orig.inc'
         write(13,'(es25.16e3)') (rotV(i),i=1,n)
         include 'proj_wall.inc'
         write(13,'(es25.16e3)') (rotV(i),i=1,n)
         close(13)
      endif
      call mpi_finalize(ierr)
      end
