c-----------------------------------------------------------------------
      subroutine cdscal (igeom)
C
C     Solve the convection-diffusion equation for passive scalar IPSCAL
C
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'MVGEOM'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      LOGICAL          IFCONV
C
      COMMON /SCRNS/ TA(LX1,LY1,LZ1,LELT)
     $              ,TB(LX1,LY1,LZ1,LELT)
      COMMON /SCRVH/ H1(LX1,LY1,LZ1,LELT)
     $              ,H2(LX1,LY1,LZ1,LELT)

      include 'ORTHOT'

      if (ifdg) then
         call cdscal_dg(igeom)
         return
      endif


      napproxt(1) = laxtt  ! Fix this... pff 10/10/15

      nel    = nelfld(ifield)
      n   = nx1*ny1*nz1*nel

      if (igeom.eq.1) then   ! geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! geometry at t^n

         IF (IFPRINT) THEN
         IF (IFMODEL .AND. IFKEPS) THEN
            NFLDT = NFIELD - 1
            IF (IFIELD.EQ.NFLDT.AND.NID.EQ.0) THEN
               WRITE (6,*) ' Turbulence Model - k/epsilon solution'
            ENDIF
         ELSE
            IF (IFIELD.EQ.2.AND.NID.EQ.0) THEN
               WRITE (6,*) ' Temperature/Passive scalar solution'
            ENDIF
         ENDIF
         ENDIF
         if1=ifield-1
         write(name4t,1) if1-1
    1    format('PS',i2)
         if(ifield.eq.2) write(name4t,'(A4)') 'TEMP'

C
C        New geometry
C
         isd = 1
         if (ifaxis.and.ifaziv.and.ifield.eq.2) isd = 2
c        if (ifaxis.and.ifmhd) isd = 2 !This is a problem if T is to be T!

         do 1000 iter=1,nmxnl ! iterate for nonlin. prob. (e.g. radiation b.c.)

         INTYPE = 0
         IF (IFTRAN) INTYPE = -1
         CALL SETHLM  (H1,H2,INTYPE)
         CALL BCNEUSC (TA,-1)
         CALL ADD2    (H2,TA,N)
         CALL BCDIRSC (T(1,1,1,1,IFIELD-1))
         CALL AXHELM  (TA,T(1,1,1,1,IFIELD-1),H1,H2,IMESH,isd)
         CALL SUB3    (TB,BQ(1,1,1,1,IFIELD-1),TA,N)
         CALL BCNEUSC (TA,1)
         CALL ADD2    (TB,TA,N)

c        CALL HMHOLTZ (name4t,TA,TB,H1,H2
c    $                 ,TMASK(1,1,1,1,IFIELD-1)
c    $                 ,TMULT(1,1,1,1,IFIELD-1)
c    $                 ,IMESH,TOLHT(IFIELD),NMXH,isd)

         if(iftmsh(ifield)) then
           call hsolve  (name4t,TA,TB,H1,H2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxh,1
     $                   ,approxt,napproxt,bintm1)
         else
           call hsolve  (name4t,TA,TB,H1,H2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxh,1
     $                   ,approxt,napproxt,binvm1)
         endif 

         call add2    (t(1,1,1,1,ifield-1),ta,n)

         call cvgnlps (ifconv) ! Check convergence for nonlinear problem 
         if (ifconv) goto 2000

C        Radiation case, smooth convergence, avoid flip-flop (ER).
         CALL CMULT (TA,0.5,N)
         CALL SUB2  (T(1,1,1,1,IFIELD-1),TA,N)

 1000    CONTINUE
 2000    CONTINUE
         CALL BCNEUSC (TA,1)
         CALL ADD2 (BQ(1,1,1,1,IFIELD-1),TA,N) ! no idea why... pf

      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine makeuq

c     Fill up user defined forcing function and collocate will the
c     mass matrix on the Gauss-Lobatto mesh.

      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'

      n = nx1*ny1*nz1*nelfld(ifield)

      if (.not.ifcvfld(ifield)) time = time-dt ! Set time to t^n-1 for user function

      call rzero   (bq(1,1,1,1,ifield-1),n)
      call setqvol (bq(1,1,1,1,ifield-1))
      call col2    (bq(1,1,1,1,ifield-1) ,bm1,n)

      if (.not.ifcvfld(ifield)) time = time+dt ! Restore time

      return
      end
c-----------------------------------------------------------------------
      subroutine setqvol(bql)

c     Set user specified volumetric forcing function (e.g. heat source).

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'

      real bql(lx1*ly1*lz1,lelt)

#ifdef MOAB
c     pulling in temperature right now, since we dont have anything else
      call userq2(bql)
      return
#endif

      nel   = nelfld(ifield)
      nxyz1 = nx1*ny1*nz1
      n     = nxyz1*nel

      do iel=1,nel

         call nekuq (bql,iel) ! ONLY SUPPORT USERQ - pff, 3/08/16

c        igrp = igroup(iel)
c        if (matype(igrp,ifield).eq.1) then ! constant source within a group
c           cqvol = cpgrp(igrp,ifield,3)
c           call cfill (bql(1,iel),cqvol,nxyz1)
c        else  !  pff 2/6/96 ............ default is to look at userq
c           call nekuq (bql,iel)
c        endif

      enddo
c
c 101 FORMAT(' Wrong material type (',I3,') for group',I3,', field',I2
c    $    ,/,' Aborting in SETQVOL.')
C   
      return
      end
C
      subroutine nekuq (bql,iel)
C------------------------------------------------------------------
C
C     Generate user-specified volumetric source term (temp./p.s.)
C
C------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'PARALLEL'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
c
      real bql(lx1,ly1,lz1,lelt)
c
      ielg = lglel(iel)
      do 10 k=1,nz1
      do 10 j=1,ny1
      do 10 i=1,nx1
         if (optlevel.le.2) call nekasgn (i,j,k,iel)
         qvol = 0.0
         call userq   (i,j,k,ielg)
         bql(i,j,k,iel) = qvol
 10   continue

      return
      end
c-----------------------------------------------------------------------
      subroutine convab
C---------------------------------------------------------------
C
C     Eulerian scheme, add convection term to forcing function 
C     at current time step.
C
C---------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      COMMON /SCRUZ/ TA (LX1,LY1,LZ1,LELT)
C
      NEL = NELFLD(IFIELD)
      N   = NX1*NY1*NZ1*NEL
      CALL CONVOP  (TA,T(1,1,1,1,IFIELD-1))
      CALL COL2    (TA,VTRANS(1,1,1,1,IFIELD),N)
      CALL SUBCOL3 (BQ(1,1,1,1,IFIELD-1),BM1,TA,N)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine makeabq
C
C     Sum up contributions to 3rd order Adams-Bashforth scheme.
C
      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'
C
      COMMON /SCRUZ/ TA (LX1,LY1,LZ1,LELT)
C
      AB0   = AB(1)
      AB1   = AB(2)
      AB2   = AB(3)
      NEL   = NELFLD(IFIELD)
      N = NX1*NY1*NZ1*NEL
C
      CALL ADD3S2 (TA,VGRADT1(1,1,1,1,IFIELD-1),
     $                VGRADT2(1,1,1,1,IFIELD-1),AB1,AB2,N)
      CALL COPY   (   VGRADT2(1,1,1,1,IFIELD-1),
     $                VGRADT1(1,1,1,1,IFIELD-1),N)
      CALL COPY   (   VGRADT1(1,1,1,1,IFIELD-1),
     $                     BQ(1,1,1,1,IFIELD-1),N)
      CALL ADD2S1 (BQ(1,1,1,1,IFIELD-1),TA,AB0,N)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine makebdq
C-----------------------------------------------------------------------
C
C     Add contributions to F from lagged BD terms.
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'

      parameter (lt=lx1*ly1*lz1*lelt)
      common /scrns/ ta(lt),tb(lt),h2(lt)

      nel   = nelfld(ifield)
      n     = nx1*ny1*nz1*nel

   
      const = 1./dt
c      call cmult2(h2,vtrans(1,1,1,1,ifield),const,n)
      do i=1,n
         h2(i)=const*vtrans(i,1,1,1,ifield)
         tb(i)=bd(2)*bm1(i,1,1,1)*t(i,1,1,1,ifield-1)
      enddo

      DO 100 ILAG=2,NBD
         IF (IFGEOM) THEN
            CALL COL3 (TA,BM1LAG(1,1,1,1,ILAG-1),
     $                    TLAG  (1,1,1,1,ILAG-1,IFIELD-1),N)
         ELSE
            CALL COL3 (TA,BM1,
     $                    TLAG  (1,1,1,1,ILAG-1,IFIELD-1),N)
         ENDIF
         CALL CMULT (TA,BD(ILAG+1),N)
         CALL ADD2  (TB,TA,N)
 100  CONTINUE

c      do i=1,n
c         bq(i,1,1,1,ifield-1) = bq(i,1,1,1,ifield-1) + tb(i)*h2(i)
c      enddo
      call addcol3 (bq(1,1,1,1,ifield-1),tb,h2,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine lagscal   !  Keep old passive scalar field(s) 

      include 'SIZE'
      include 'TOTAL'

      n = nx1*ny1*nz1*nelfld(ifield)

      do ilag=nbdinp-1,2,-1
         call copy (tlag(1,1,1,1,ilag  ,ifield-1),
     $              tlag(1,1,1,1,ilag-1,ifield-1),n)
      enddo

      call copy (tlag(1,1,1,1,1,ifield-1),t(1,1,1,1,ifield-1),n)

      return
      end
c-----------------------------------------------------------------------
      subroutine outfldrq (x,txt10,ichk)
      include 'SIZE'
      include 'TSTEP'
      real x(nx1,ny1,nz1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /-3/

      if (idum.lt.0) return
c
C
      mtot = nx1*ny1*nz1*nelv
      if (nx1.gt.8.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      nell = nelt
      rnel = nell
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nell-ne+1
      k = 1
      do ie=1,1
         ne = 0
         write(6,116) txt10,k,ie,xmin,xmax,istep,time
         do l=0,1
            write(6,117) 
            do j=ny1,1,-1
              if (nx1.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
              if (nx1.eq.8) write(6,118) ((x(i,j,k,e+l),i=1,nx1),e=1,1)
            enddo
         enddo
      enddo
C
  102 FORMAT(4(2f9.5,2x))
  103 FORMAT(4(3f9.5,2x))
  104 FORMAT(4(4f7.3,2x))
  105 FORMAT(5f9.5,10x,5f9.5)
  106 FORMAT(6f9.5,5x,6f9.5)
  107 FORMAT(7f8.4,5x,7f8.4)
  108 FORMAT(8f8.4,4x,8f8.4)
  118 FORMAT(8f12.9)
c
  116 FORMAT(  /,5X,'     ^              ',/,
     $    5X,'   Y |              ',/,
     $    5X,'     |              ',A10,/,
     $    5X,'     +---->         ','Plane = ',I2,'/',I2,2x,2e12.4,/,
     $    5X,'       X            ','Step  =',I9,f15.5)
  117 FORMAT(' ')
c
      if (ichk.eq.1.and.idum.gt.0) call checkit(idum)
      return
      end
c-----------------------------------------------------------------------
      subroutine cdscal_expl (igeom)
C
C     explicit convection-diffusion equation for passive scalar
C
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'MVGEOM'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
      common  /cprint/ ifprint
      logical          ifprint
      logical          ifconv

      common /scrns/ ta(lx1,ly1,lz1,lelt)
     $              ,tb(lx1,ly1,lz1,lelt)
      common /scrvh/ h1(lx1,ly1,lz1,lelt)
     $              ,h2(lx1,ly1,lz1,lelt)


c     QUESTIONABLE support for Robin BC's at this point! (5/15/08)

      nel    = nelfld(ifield)
      n   = nx1*ny1*nz1*nel

      if (igeom.eq.1) then   ! geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! geometry at t^n

         if (.true..and.nio.eq.0) 
     $      write (6,*) istep,ifield,' explicit step'


C        New geometry

         isd = 1
         if (ifaxis.and.ifmhd) isd = 2 !This is a problem if T is to be T!

         intype = 0
         if (iftran) intype = -1
         call sethlm  (h1,h2,intype)

         call bcneusc (ta,-1)       ! Modify diagonal for Robin condition
         call add2    (h2,ta ,n)
         call col2    (h2,BM1,n)

         call bcneusc (tb,1)        ! Modify rhs for flux bc
         call add2    (bq(1,1,1,1,ifield-1),tb,n)

         call dssum   (bq(1,1,1,1,ifield-1),nx1,ny1,nz1)
         call dssum   (h2,nx1,ny1,nz1)

         call invcol3 (t(1,1,1,1,ifield-1),bq(1,1,1,1,ifield-1),h2,n)

         call bcdirsc (t(1,1,1,1,ifield-1)) ! --> no mask needed

      endif                   ! geometry at t^n

      return
      end
c-----------------------------------------------------------------------
      subroutine diffab  ! explicit treatment of diffusion operator
c
c     Eulerian scheme, add diffusion term to forcing function 
c     at current time step.
c

      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
      include 'INPUT'

      common /scruz/ ta(lx1,ly1,lz1,lelt)
     $              ,h2(lx1,ly1,lz1,lelt)

      nel = nelfld(ifield)
      n   = nx1*ny1*nz1*nel

      intype = 0
      if (iftran) intype = -1

      isd = 1
      if (ifaxis.and.ifmhd) isd = 2 !This is a problem if T is to be T!

      imesh = 1
c      if (iftmsh(ifield)) imesh=2

      call rzero   (h2,n)
      call axhelm  (ta,t(1,1,1,1,ifield-1),vdiff(1,1,1,1,ifield)
     $             ,h2,imesh,isd)
      call sub2    (bq(1,1,1,1,ifield-1),ta,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine cdscal_dg (igeom)
C
C     Solve the convection-diffusion equation for passive scalar IPSCAL
C
      include 'SIZE'
      include 'TOTAL'
      common  /cprint/ ifprint
      logical          ifprint,ifconv

      parameter (lt=lx1*ly1*lz1*lelt)
      common /scrns/ ta(lt),tb(lt)
      common /scrvh/ h1(lt),h2(lt)

      include 'ORTHOT'  ! This must be fixed

      nel = nelfld(ifield)
      n   = nx1*ny1*nz1*nel

      if (igeom.eq.1) then   ! old geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! new geometry at t^n

         if (ifprint.and.ifield.eq.2.and.nio.eq.0) 
     $      write (6,*) ' Temperature/Passive scalar solution'

         if1=ifield-1
         write(name4t,1) if1-1
    1    format('PS',i2)
         if(ifield.eq.2) write(name4t,'(A4)') 'TEMP'
 
         isd = 1
         if (ifaxis.and.ifaziv.and.ifield.eq.2) isd = 2
c        if (ifaxis.and.ifmhd) isd = 2 !This is a problem if T is to be T!

         intype = 0
         if (iftran) intype = -1
         call sethlm  (h1,h2,intype)
         call bcneusc (ta,-1)
         call add2    (h2,ta,n)
         call bcdirsc (t(1,1,1,1,ifield-1))
         call axhelm  (ta,t(1,1,1,1,ifield-1),h1,h2,imesh,isd)
         call sub3    (tb,bq(1,1,1,1,ifield-1),ta,n)
         call bcneusc (ta,1)
         call add2    (tb,ta,n)


c        if (ifdg) then
c           do i=1,n                        ! Quick Hack, 10/10/15, pff.
c              s = h2(i)*bm1(i,1,1,1)
c              ta(i)=tb(i)/s
c              ta(i)=tmask(i,1,1,1,ifield-1)*tb(i)/s
c           enddo
c        elseif (iftmsh(ifield)) then

         if (iftmsh(ifield)) then
           call hsolve  (name4t,ta,tb,h1,h2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxh,1
     $                   ,approxt,napproxt,bintm1)
         else
           call hsolve  (name4t,ta,tb,h1,h2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxh,1
     $                   ,approxt,napproxt,binvm1)
         endif 

         call add2    (t(1,1,1,1,ifield-1),ta,n)

      endif  ! End of IGEOM branch.

      return
      end
c-----------------------------------------------------------------------
