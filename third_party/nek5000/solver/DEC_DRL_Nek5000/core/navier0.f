      SUBROUTINE ESOLVER (RES,H1,H2,H2INV,INTYPE)
C---------------------------------------------------------------------
C
C     Choose E-solver
C
C--------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'ESOLV'
      INCLUDE 'INPUT'
C
      REAL RES   (LX2,LY2,LZ2,LELV)
      REAL H1    (LX1,LY1,LZ1,LELV)
      REAL H2    (LX1,LY1,LZ1,LELV)
      REAL H2INV (LX1,LY1,LZ1,LELV)
      common /scruz/ wk1(lx2*ly2*lz2*lelv)
     $             , wk2(lx2*ly2*lz2*lelv)
     $             , wk3(lx2*ly2*lz2*lelv)

      include 'CTIMER'
      real kwave2

      if (icalld.eq.0) teslv=0.0
      icalld=icalld+1
      neslv=icalld
      etime1=dnekclock()

c     write(6,*) solver_type,' solver type',iesolv
      if (iesolv.eq.1) then
         if (solver_type.eq.'fdm') then
            ntot2 = nx2*ny2*nz2*nelv
            kwave2 = 0.
            call gfdm_pres_solv  (wk1,res,wk2,wk3,kwave2)
            call copy            (res,wk1,ntot2)
         else
            if (param(42).eq.1.or.solver_type.eq.'pdm') then
               CALL UZAWA (RES,H1,H2,H2INV,INTYPE,ICG)
            else
               call uzawa_gmres(res,h1,h2,h2inv,intype,icg)
            endif
         endif
      else
         WRITE(6,*) 'ERROR: E-solver does not exist',iesolv
         WRITE(6,*) 'Stop in ESOLVER'
         CALL EXITT
      ENDIF

      teslv=teslv+(dnekclock()-etime1)

      RETURN
      END
      SUBROUTINE ESTRAT
C---------------------------------------------------------------------------
C
C     Decide strategy for E-solver
C
C---------------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
C
      IESOLV = 1
      if (ifsplit) iesolv=0

      solver_type='itr'
      if (param(116).ne.0) solver_type='fdm'
c     if (param(90).ne.0)  solver_type='itn'

c     The following change recognizes that geometry is logically 
c     tensor-product, but deformed:  pdm = Preconditioner is fdm

      if (param(59).ne.0.and.solver_type.eq.'fdm') solver_type='pdm'

      if (istep.lt.2.and.nio.eq.0) write(6,10) iesolv,solver_type
   10 format(2X,'E-solver strategy: ',I2,1X,A)



C
      RETURN
      END
      SUBROUTINE EINIT
C-----------------------------------------------------------------------------
C
C     Initialize E-solver
C
C-----------------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'
      INCLUDE 'ESOLV'
      COMMON /SCRHI/  H2INV (LX1,LY1,LZ1,LELV)
      LOGICAL IFNEWP
C
      CALL ESTRAT 
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine dmp_map(imap)
c
c     Dump map file and element center point
c
      include 'SIZE'
      include 'TOTAL'

      common /ivrtx/ vertex ((2**ldim)*lelt)
      common /scruz/ xbar(ldim,lelt),ibar(lelt)
      integer vertex
      integer imap(nelgt)

      integer e,eg

      nxb = (nx1+1)/2
      nyb = (ny1+1)/2
      nzb = (nz1+1)/2
      
      do e=1,nelt
         xbar(ndim,e) = zm1(nxb,nyb,nzb,e)
         xbar(1   ,e) = xm1(nxb,nyb,nzb,e)
         xbar(2   ,e) = ym1(nxb,nyb,nzb,e)
         eg           = lglel(e)
         ibar(e)      = imap(eg)
      enddo
      call p_outvec_ir(ibar,xbar,ndim,'mpxyz.dat')

      return
      end
c-----------------------------------------------------------------------
      subroutine p_outvec_ir(ia,a,lda,name9)
      integer ia(1)
      real    a(lda,1)
      character*9 name9

      include 'SIZE'
      include 'TOTAL'

      parameter (lbuf=50)
      common /scbuf/ buf(lbuf)
      integer ibuf(10),e,eg
      equivalence (buf,ibuf)

      if (nid.eq.0) then
         open(unit=49,file=name9)
         write(6,*) 'Opening ',name9,' in p_outveci. lda=',lda
      endif

      len = wdsize*(lda+1)
      dum = 0.

      do eg=1,nelgt

         mid   = gllnid(eg)
         e     = gllel (eg)
c     tag for sending and receiving changed from global (eg) to 
c     local (e) element number to avoid problems with MPI_TAG_UB on Cray
         mtype = 2000+e

         if (nid.eq.0) then
            if (mid.eq.0) then
               call icopy(buf(1),ia(e),1)
               call  copy(buf(2),a(1,e),lda)
            else
               call csend (mtype,dum,wdsize,mid,nullpid)
               call crecv2 (mtype,buf,len,mid)
            endif
            write(49,49) mid,ibuf(1),(buf(k+1),k=1,lda)
   49       format(2i12,1p3e16.7)
         elseif (nid.eq.mid) then
            call icopy(buf(1),ia(e),1)
            call  copy(buf(2),a(1,e),lda)
            call crecv2 (mtype,dum,wdsize,0)
            call csend (mtype,buf,len,node0,nullpid)
         endif
      enddo

      if (nid.eq.0) then
         close(49)
         write(6,*) 'Done writing to ',name9,' p_outveci.'
      endif

      call nekgsync()

      return
      end
c-----------------------------------------------------------------------
