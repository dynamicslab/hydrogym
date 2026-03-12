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

      COMMON /SCRNS/ TA(LX1,LY1,LZ1,LELT)
     $              ,TB(LX1,LY1,LZ1,LELT)
      COMMON /SCRVH/ H1(LX1,LY1,LZ1,LELT)
     $              ,H2(LX1,LY1,LZ1,LELT)

      include 'ORTHOT'

      if (ifdgfld(ifield)) then
         call cdscal_dg(igeom)
         return
      endif


      ifld1 = ifield-1
      napproxt(1,ifld1) = laxtt

      nel    = nelfld(ifield)
      n   = lx1*ly1*lz1*nel

      if (igeom.eq.1) then   ! geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! geometry at t^n

         IF (IFPRINT) THEN
            IF (IFIELD.EQ.2.AND.NID.EQ.0)
     $          WRITE (6,*) ' Temperature/Passive scalar solution'
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

         intype = 0
         if (iftran) intype = -1
         call sethlm  (h1,h2,intype)
         call bcneusc (ta,-1)
         call add2    (h2,ta,n)
         call bcdirsc (t(1,1,1,1,ifield-1))
         call axhelm  (ta,t(1,1,1,1,ifield-1),h1,h2,imesh,ISD)
         call sub3    (tb,bq(1,1,1,1,ifield-1),ta,n)
         call bcneusc (ta,1)
         call add2    (tb,ta,n)

         if(iftmsh(ifield)) then
           call hsolve  (name4t,TA,TB,H1,H2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxt(ifield-1),1
     $                   ,approxt(1,0,ifld1),napproxt(1,ifld1),bintm1)
         else
           call hsolve  (name4t,TA,TB,H1,H2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tmult(1,1,1,1,ifield-1)
     $                   ,imesh,tolht(ifield),nmxt(ifield-1),1
     $                   ,approxt(1,0,ifld1),napproxt(1,ifld1),binvm1)
         endif 

         call add2    (t(1,1,1,1,ifield-1),ta,n)

         call cvgnlps (ifconv) ! Check convergence for nonlinear problem 
         if (ifconv) goto 2000

C        Radiation case, smooth convergence, avoid flip-flop (ER).
         call cmult (ta,0.5,n)
         call sub2  (t(1,1,1,1,ifield-1),ta,n)

 1000    continue
 2000    continue

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

      n = lx1*ly1*lz1*nelfld(ifield)

      if (.not.ifcvfld(ifield)) time = time-dt ! Set time to t^n-1 for user function

      if (nio.eq.0.and.loglevel.gt.2) 
     $   write(6,*) 'makeuq', ifield, time
      call setqvol(bq(1,1,1,1,ifield-1))
      call col2   (bq(1,1,1,1,ifield-1) ,bm1,n)

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
      include 'CTIMER'

      real bql(lx1*ly1*lz1,lelt)

#ifdef TIMER
      etime1=dnekclock()
#endif

      nel   = nelfld(ifield)
      nxyz1 = lx1*ly1*lz1
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

#ifdef TIMER
      tusfq=tusfq+(dnekclock()-etime1)
#endif
 
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
      do 10 k=1,lz1
      do 10 j=1,ly1
      do 10 i=1,lx1
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

      common /scruz/ ta (lx1*ly1*lz1*lelt)

      nel = nelfld(ifield)
      n   = lx1*ly1*lz1*nel

      call convop  (ta,t(1,1,1,1,ifield-1))
      do i=1,n
        bq(i,1,1,1,ifield-1) = bq (i,1,1,1,ifield-1)
     $                       - bm1(i,1,1,1)*ta(i)*vtrans(i,1,1,1,ifield)
      enddo

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

      ab0   = ab(1)
      ab1   = ab(2)
      ab2   = ab(3)
      nel   = nelfld(ifield)
      n     = lx1*ly1*lz1*nel

      do i=1,n
         ta=ab1*vgradt1(i,1,1,1,ifield-1)+ab2*vgradt2(i,1,1,1,ifield-1)
         vgradt2(i,1,1,1,ifield-1)=vgradt1(i,1,1,1,ifield-1)
         vgradt1(i,1,1,1,ifield-1)=bq     (i,1,1,1,ifield-1)
         bq     (i,1,1,1,ifield-1)=bq     (i,1,1,1,ifield-1)*ab0+ta
      enddo

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
      common /scrns/ tb(lt),h2(lt)

      nel   = nelfld(ifield)
      n     = lx1*ly1*lz1*nel
   
      const = 1./dt
      do i=1,n
         h2(i)=const*vtrans(i,1,1,1,ifield)
         tb(i)=bd(2)*bm1(i,1,1,1)*t(i,1,1,1,ifield-1)
      enddo

      do ilag=2,nbd
         if (ifgeom) then
            do i=1,n
               ta=bm1lag(i,1,1,1,ilag-1)*tlag(i,1,1,1,ilag-1,ifield-1)
               tb(i)=tb(i)+ta*bd(ilag+1)
            enddo
         else
            do i=1,n
               ta=bm1(i,1,1,1)*tlag(i,1,1,1,ilag-1,ifield-1)
               tb(i)=tb(i)+ta*bd(ilag+1)
            enddo
         endif
      enddo

      call addcol3 (bq(1,1,1,1,ifield-1),tb,h2,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine lagscal   !  Keep old passive scalar field(s) 

      include 'SIZE'
      include 'TOTAL'

      n = lx1*ly1*lz1*nelfld(ifield)

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
      real x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /-3/

      if (idum.lt.0) return
c
C
      mtot = lx1*ly1*lz1*nelv
      if (lx1.gt.8.or.nelv.gt.16) return
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
            do j=ly1,1,-1
              if (lx1.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
              if (lx1.eq.8) write(6,118) ((x(i,j,k,e+l),i=1,lx1),e=1,1)
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
      n   = lx1*ly1*lz1*nel

      if (igeom.eq.1) then   ! geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! geometry at t^n

         if (nio.eq.0) write (6,*) istep,ifield,' explicit step'


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

         call dssum   (bq(1,1,1,1,ifield-1),lx1,ly1,lz1)
         call dssum   (h2,lx1,ly1,lz1)

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
      n   = lx1*ly1*lz1*nel

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
      subroutine set_eta_alpha2

c     Set up required dg terms, e.g., alpha, eta, etc.
c     Face weight: .5 interior, 1. boundary

      include 'SIZE'
      include 'TOTAL'
      common /mysedg/ eta


      integer e,f,pf

      nface = 2*ldim
      call dsset(lx1,ly1,lz1)

      eta =  5         !   Semi-optimized value, single domain

      do e=1,nelfld(ifield)
      do f=1,nface

         pf     = eface1(f)
         js1    = skpdat(1,pf)
         jf1    = skpdat(2,pf)
         jskip1 = skpdat(3,pf)
         js2    = skpdat(4,pf)
         jf2    = skpdat(5,pf)
         jskip2 = skpdat(6,pf)

         i = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1
            i = i+1
            a = area(i,1,f,e)  ! Check Fydkowski notes 
            a = a*a*fw(f,e)    ! For ds_avg used below, plus quad weight
            etalph(i,f,e) = eta*(a/bm1(j1,j2,1,e))
c           write(6,*) i,j1,j2,e,f,a,etalph(i,f,e)
         enddo
         enddo
      enddo
      enddo

      call fgslib_gs_op (dg_hndlx,etalph,1,1,0)  ! 1 ==> +

      return
      end
c-----------------------------------------------------------------------
      subroutine fwght(msk1,mult)
      include 'SIZE'
      include 'TOTAL'
      parameter (lx=lx1*ly1*lz1)
      real msk1(lx1,ly1,lz1),mult(lx1,ly1,lz1,1)
      integer e,f

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /scrdg/uf(lx1*lz1,2*ldim,lelt)



      do i=1,lf
         uf(i,1,1)=1.
      enddo
      call fgslib_gs_op (dg_hndlx,uf,1,1,0)  ! 1 ==> +

      nface = 2*ldim
      do e=1,nelt
      do f=1,nface
         fw(f,e) = 1                        ! Boundary
         if (uf(1,f,e).gt.1.1) fw(f,e)=0.5  ! Interior
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine dg_setup
      include 'SIZE'
      include 'TOTAL'

      common /ivrtx/ vertex ((2**ldim)*lelt)
      common /ctmp1/ qs(lx1*ly1*lz1*lelt)
      integer vertex

      logical ifany
      save    ifany
      data    ifany /.false./

      do ifield=1,ldimt1
         if (ifdgfld(ifield)) ifany=.true.
      enddo

      if (ifany) then

        call setup_dg_gs(dg_hndlx,lx1,ly1,lz1,nelt,nelgt,vertex)
        call dg_set_fc_ptr

        param(59)=1
        call geom_reset(1)
        call set_unr
      endif


      n = lx1*ly1*lz1*nelt
      call invers2(binvdg,bm1,n)

      call set_dg_wgts

      return
      end
c-----------------------------------------------------------------------
      subroutine dg_setup2(mask)
      include 'SIZE'
      include 'TOTAL'

      real mask(1)
      common /ctmp0/ qs(lx1*ly1*lz1*lelt)

      integer ifld_last
      save    ifld_last
      data    ifld_last /0/

      if (ifield.eq.ifld_last) return
      ifld_last = ifield

      call fwght  (qs,mask)  ! qs is work array
      call set_eta_alpha2     ! Set eta/h

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

      call dg_setup2(tmask(1,1,1,1,ifield-1))

      nel = nelfld(ifield)
      n   = lx1*ly1*lz1*nel

      if (igeom.eq.1) then   ! old geometry at t^{n-1}

         call makeq
         call lagscal

      else                   ! new geometry at t^n

         if (ifprint.and.nio.eq.0) 
     $      write (6,*) ' Temperature/Passive scalar solution',ifield

         if1=ifield-1
         write(name4t,1) if1-1
    1    format('PS',i2)
         if (ifield.eq.2) write(name4t,'(A4)') 'TEMP'
 
         isd = 1
         if (ifaxis.and.ifaziv.and.ifield.eq.2) isd = 2
c        if (ifaxis.and.ifmhd) isd = 2 !This is a problem if T is to be T!

         intype = 0
         if (iftran) intype = -1
         call sethlm  (h1,h2,intype)

c        call bcneusc (ta,-1)     !! Not Yet supported for DG
c        call add2    (h2,ta,n)   !! Not Yet supported for DG

         call rzero             (tb,n)
         call bcdirsc           (   t (1,1,1,1,ifield-1))
         call conv_bdry_dg_weak (tb,t (1,1,1,1,ifield-1))
         call hxdg_surfa        (tb,t (1,1,1,1,ifield-1),h1,h2)
         call add2              (tb,bq(1,1,1,1,ifield-1),n)

         call hmholtz_dg(name4t,t(1,1,1,1,ifield-1),tb,h1,h2 
     $                   ,tmask(1,1,1,1,ifield-1)
     $                   ,tolht(ifield),nmxt(ifield-1))

      endif  ! End of IGEOM branch.

      return
      end
c-----------------------------------------------------------------------
