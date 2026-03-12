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

      call ortho(res) !Ensure that residual is orthogonal to null space

      icalld=icalld+1
      neslv=icalld
      etime1=dnekclock()

      if (.not. ifsplit) then
         if (param(42).eq.1) then
            CALL UZAWA (RES,H1,H2,H2INV,INTYPE,ICG)
         else
            call uzawa_gmres(res,h1,h2,h2inv,intype,icg)
         endif
      else
         WRITE(6,*) 'ERROR: E-solver does not exist PnPn'
         CALL EXITT
      ENDIF

      teslv=teslv+(dnekclock()-etime1)

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

      nxb = (lx1+1)/2
      nyb = (ly1+1)/2
      nzb = (lz1+1)/2
      
      do e=1,nelt
         xbar(ldim,e) = zm1(nxb,nyb,nzb,e)
         xbar(1   ,e) = xm1(nxb,nyb,nzb,e)
         xbar(2   ,e) = ym1(nxb,nyb,nzb,e)
         eg           = lglel(e)
         ibar(e)      = imap(eg)
      enddo
      call p_outvec_ir(ibar,xbar,ldim,'mpxyz.dat')

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
