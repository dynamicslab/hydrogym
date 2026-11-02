c-----------------------------------------------------------------------
c
c     TO DO:   Need to monitor dssum when using DG.  (11/29/15, pff)
c
c              Specifically - line 837 in navier4.f
c                           - line 537 in hmholtz.f
c
c-----------------------------------------------------------------------
      subroutine setrhs(p,h1,h2,h2inv)
C
C     Project rhs onto best fit in the "E" norm.
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
C
      REAL             P    (LX2,LY2,LZ2,LELV)
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      logical ifdump
      save    ifdump
      data    ifdump /.false./
C
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev
      REAL ALPHA,WORK
C
C
      integer icalld
      save    icalld
      data    icalld/0/
C
C     First call, we have no vectors to orthogonalize against.
      IF (ICALLD.EQ.0) THEN
         icalld=icalld+1
         Nprev=0
         Mprev=param(93)
         Mprev=min(Mprev,Mxprev)
      ENDIF
C
C     Diag to see how much reduction in the residual is attained.
C
      NTOT2  = NX2*NY2*NZ2*NELV
      ALPHA1 = GLSC3(p,p,bm2inv,NTOT2)
      if (alpha1.gt.0) ALPHA1 = sqrt(alpha1/volvm2)
C
C     Update rhs's if E-matrix has changed
C
      CALL UPDRHSE(P,H1,H2,H2INV,ierr)
      if (ierr.eq.1) Nprev=0
C
C     Perform Gram-Schmidt for previous rhs's.
C
      DO 10 I=1,Nprev
         ALPHA(i) = VLSC2(P,RHS(1,i),NTOT2)
   10 CONTINUE
C
      IF (Nprev.GT.0) CALL gop(alpha,WORK,'+  ',Nprev)
C
      CALL RZERO(Pbar,NTOT2)
      DO 20 I=1,Nprev
         alphas = alpha(i)
         CALL ADD2S2(Pbar,RHS(1,i),alphas,NTOT2)
   20 CONTINUE
C
      if (Nprev.gt.0) then
         INTETYPE = 1
         CALL CDABDTP(Pnew,Pbar,H1,H2,H2INV,INTETYPE)
         CALL SUB2   (P,Pnew,NTOT2)
C    ................................................................
C      Diag.
         ALPHA2 = GLSC3(p,p,bm2inv,NTOT2)
         if (alpha2.gt.0) ALPHA2 = sqrt(alpha2/volvm2)
         ratio  = alpha1/alpha2
         n10=min(10,nprev)
c         IF (NIO.EQ.0) WRITE(6,11)ISTEP,Nprev,(ALPHA(I),I=1,N10)
c         IF (NIO.EQ.0) WRITE(6,12) ISTEP,nprev,ALPHA1,ALPHA2,ratio
   11    FORMAT(2I5,' alpha:',1p10e12.4)
   12    FORMAT(I6,i4,1p3e12.4,' alph12')
C
c        if (alpha1.gt.0.001 .and. .not.ifdump) then
c           IF (NID.EQ.0) WRITE(6,*) 'alph1 large ... '
c           if (istep.gt.10) then 
c              IF (NID.EQ.0) WRITE(6,*) ' ... dumping'
c              call prepost (.true.,'   ')
c              ifdump = .true.
c           else
c              IF (NID.EQ.0) WRITE(6,*) ' ... doing nothing'
c           endif
c        endif
c        if (alpha1.gt.0.1.and.istep.gt.10) then
c           IF (NID.EQ.0) WRITE(6,*) 'alph1 too large ... aborting'
c           call prepost (.true.,'   ')
c           call exitt
c        endif
C    ................................................................
      endif
C
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine gensoln(p,h1,h2,h2inv)
C
C     Reconstruct the solution to the original problem by adding back
C     the previous solutions
C     know the soln.
C
      include 'SIZE'
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev
      REAL ALPHA,WORK

      REAL             P    (LX2,LY2,LZ2,LELV)
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      NTOT2=NX2*NY2*NZ2*NELV
C
C     First, save current solution
C
      CALL COPY (Pnew,P,NTOT2)
C
C     Reconstruct solution
C
      CALL ADD2(P,Pbar,NTOT2)
C
C     Update the set of <p,rhs>
C
      CALL UPDTSET(P,H1,H2,H2INV,ierr)
      if (ierr.eq.1) Nprev = 0
c
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine updtset(p,h1,h2,h2inv,IERR)
C
C     Update the set of rhs's and the corresponding p-set:
C
C        . Standard case is to add P_new, and RHS_new = E*P_new
C
C        . However, when Nprev=Mprev (max. allowed), we throw out
C          the old set, and set P_1 = P, RHS_1=E*P_1
C
C        . Other schemes are possible, e.g., let's save a bunch of
C          old vectors, perhaps chosen wisely via P.O.D.
C
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev

      REAL ALPHA,WORK

      REAL             P    (LX2,LY2,LZ2,LELV)
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      NTOT2=NX2*NY2*NZ2*NELV
C
      IF (Nprev.EQ.Mprev) THEN
         CALL COPY(Pnew,P,NTOT2)
         Nprev=0
      ENDIF
C
C     Increment solution set
      Nprev = Nprev+1
C
      CALL COPY   (RHS(1,Nprev),Pnew,NTOT2)
C
C     Orthogonalize rhs against previous rhs and normalize
C
      CALL ECONJ (Nprev,H1,H2,H2INV,ierr)
c     CALL ECHECK(Nprev,H1,H2,H2INV,INTETYPE)
C
c     Save last sol'n
      CALL COPY(Pnew,P,NTOT2)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine econj(kprev,h1,h2,h2inv,ierr)
C
C     Orthogonalize the rhs wrt previous rhs's for which we already
C     know the soln.
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
C
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2),Pbrr(ltot2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev
      REAL ALPHA,WORK
      real ALPHAd
C
C
      ierr  = 0
      NTOT2 = NX2*NY2*NZ2*NELV
      INTETYPE=1
C
C     Gram Schmidt, w re-orthogonalization
C
      npass=1
      if (abs(param(105)).eq.2) npass=2
      do ipass=1,npass
c
         CALL CDABDTP(Pbrr,RHS(1,Kprev),H1,H2,H2INV,INTETYPE)
C
C        Compute part of the norm
         Alphad = GLSC2(RHS(1,Kprev),Pbrr,NTOT2)
C
C        Gram-Schmidt
         Kprev1=Kprev-1
         DO 10 I=1,Kprev1
            ALPHA(I) = VLSC2(Pbrr,RHS(1,i),NTOT2)
   10    CONTINUE
         IF (Kprev1.GT.0) CALL gop(alpha,WORK,'+  ',Kprev1)
C
         DO 20 I=1,Kprev1
            alpham = -alpha(i)
            CALL ADD2S2(RHS(1,Kprev),RHS (1,i),alpham,NTOT2)
            Alphad = Alphad - alpha(i)**2
   20    CONTINUE
      enddo
C
C    .Normalize new element in P~
C
      if (ALPHAd.le.0.0) then
         write(6,*) 'ERROR:  alphad .le. 0 in ECONJ',alphad,Kprev
         ierr = 1
         return
      endif
      ALPHAd = 1.0/SQRT(ALPHAd)
      ALPHAN = Alphad
      CALL CMULT(RHS (1,Kprev),alphan,NTOT2)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine chkptol
C--------------------------------------------------------------------
C
C     Check pressure tolerance for transient case.
C
C     pff 6/20/92
C     This routine has been modified for diagnostic purposes only.
C     It can be replaced with the standard nekton version.
C 
C--------------------------------------------------------------------
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'INPUT'
      include 'TSTEP'
      COMMON /CTOLPR/ DIVEX
      COMMON /CPRINT/ IFPRINT
      LOGICAL         IFPRINT
C
      COMMON /SCRUZ/  DIVV (LX2,LY2,LZ2,LELV)
     $ ,              BDIVV(LX2,LY2,LZ2,LELV)
C
      if (ifsplit) return
      IF (param(102).eq.0.and.(TOLPDF.NE.0. .OR. ISTEP.LE.5)) RETURN
      five = 5.0
      if (param(102).ne.0.0) five=param(102)
C
      NTOT2 = NX2*NY2*NZ2*NELV
      if (ifield.eq.1) then     ! avo: sub arguments?
         CALL OPDIV (BDIVV,VX,VY,VZ)
      else
         CALL OPDIV (BDIVV,BX,BY,BZ)
      endif
      CALL COL3 (DIVV,BDIVV,BM2INV,NTOT2)
      DNORM = SQRT(GLSC2(DIVV,BDIVV,NTOT2)/VOLVM2) 
C
      if (nio.eq.0) WRITE (6,*) istep,' DNORM, DIVEX',DNORM,DIVEX
C
c     IF (istep.gt.10.and.DNORM.GT.(1.01*DIVEX).AND.DIVEX.GT.0.) then
c        if (DNORM.gt.1e-8) then
c           if (nid.eq.0) WRITE(6,*) 'DNORM-DIVEX div. ... aborting'
c           call prepost (.true.,'   ')
c           call exitt
c        else
c           if (nid.eq.0) WRITE(6,*) 'DNORM-DIVEX div. ... small'
c        endif
c     endif
C
c     IF (DNORM.GT.(1.2*DIVEX).AND.DIVEX.GT.0.) TOLPDF = 5.*DNORM
      IF (istep.gt.5.and.tolpdf.eq.0.0.and.
     $    DNORM.GT.(1.2*DIVEX).AND.DIVEX.GT.0.) 
     $     TOLPDF = FIVE*DNORM
C
      RETURN
      END
      FUNCTION VLSC3(X,Y,B,N)
C
C     local inner product, with weight
C
      DIMENSION X(1),Y(1),B(1)
      REAL DT
C
      include 'OPCTR'
C
#ifdef TIMER
C
      if (isclld.eq.0) then
          isclld=1
          nrout=nrout+1
          myrout=nrout
          rname(myrout) = 'VLSC3 '
      endif
      isbcnt = 3*n
      dct(myrout) = dct(myrout) + dfloat(isbcnt)
      ncall(myrout) = ncall(myrout) + 1
      dcount      =      dcount + dfloat(isbcnt)
#endif
C
      DT = 0.0
      DO 10 I=1,N
         T = X(I)*Y(I)*B(I)
         DT = DT+T
 10   CONTINUE
      T=DT
      VLSC3 = T
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine updrhse(p,h1,h2,h2inv,ierr)
C
C     Update rhs's if E-matrix has changed
C
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
C
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev
      COMMON /ORTHOL/ IFNEWE
      REAL ALPHA,WORK
      LOGICAL IFNEWE
C
C
      REAL             P    (LX2,LY2,LZ2,LELV)
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      integer icalld
      save    icalld
      data    icalld/0/

      ntot2=nx2*ny2*nz2*nelv


C     First, we have to decide if the E matrix has changed.

      if (icalld.eq.0) then
         icalld=1
         dtlast=dt
      endif

      ifnewe=.false.
      if (ifmvbd) then
         ifnewe=.true.
         call invers2(bm2inv,bm2,ntot2)
      endif
      if (dtlast.ne.dt) then
         ifnewe=.true.
         dtlast=dt
      endif
      if (ifnewe.and.nio.eq.0) write(6,*) istep,'reorthogo:',nprev

     
C     
C     Next, we reconstruct a new rhs set.
C     
      if (ifnewe) then
c
c        new idea...
c        if (nprev.gt.0) nprev=1
c        call copy(rhs,pnew,ntot2)
c
         Nprevt = Nprev
         DO 100 Iprev=1,Nprevt
C           Orthogonalize this rhs w.r.t. previous rhs's
            CALL ECONJ (Iprev,H1,H2,H2INV,ierr)
            if (ierr.eq.1) then
               if (nio.eq.0) write(6,*) istep,ierr,' ECONJ error'
               nprev = 0
               return
            endif
  100    CONTINUE
C
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine echeck(kprev,h1,h2,h2inv,intetype)
C
C     Orthogonalize the rhs wrt previous rhs's for which we already
C     know the soln.
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
C
      REAL             H1   (LX1,LY1,LZ1,LELV)
      REAL             H2   (LX1,LY1,LZ1,LELV)
      REAL             H2INV(LX1,LY1,LZ1,LELV)
C
      PARAMETER (LTOT2=LX2*LY2*LZ2*LELV)
      COMMON /ORTHOV/ RHS(LTOT2,MXPREV)
      COMMON /ORTHOX/ Pbar(LTOT2),Pnew(LTOT2)
      COMMON /ORTHOS/ ALPHA(Mxprev), WORK(Mxprev), ALPHAN, DTLAST
      COMMON /ORTHOI/ Nprev,Mprev
      REAL ALPHA,WORK,GLSC2
      REAL Alphad
C
C
      NTOT2=NX2*NY2*NZ2*NELV
C
C     Compute part of the norm
C
      do 20 j=1,kprev
         CALL CDABDTP(Pbar,rhs(1,j),H1,H2,H2INV,INTETYPE)
         do 10 i=1,kprev
            Alphad = GLSC2(RHS(1,i),Pbar,NTOT2)
            Alphas = alphad
            if (nio.eq.0) then
               write(6,5) i,j,alphad,alphas,istep,kprev
    5          format(' E-check:',2i4,e16.8,g12.5,i6,i4)
            endif
   10    continue
   20 continue
      return
      end
c-----------------------------------------------------------------------
c     THE ROUTINES BELOW ARE THE NEW Helmholtz projectors
c-----------------------------------------------------------------------
      subroutine projh(r,h1,h2,bi,vml,vmk,approx,napprox,wl,ws,name4)
C
C     Orthogonalize the rhs wrt previous rhs's for which we already
C     know the soln.
c
c     Input:   r         -- residual
c              h1,h2     -- Helmholtz arrays
c              bi        -- inverse mass matrix
c              vml,vmk   -- multiplicity and mask arrays
c              approx    -- approximation space
c              napprox   -- (1) = max vecs,  (2) = current number of vecs
c              wl        -- large work array of size lx1*ly1*lz1*nelv
c              ws        -- small work array of size 2*max vecs
c
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
c
      parameter(lt=lx1*ly1*lz1*lelt)
C
      real r(1),h1(1),h2(1),vml(1),vmk(1)
      real bi(1)
      real wl(1),ws(1)
      real approx(lt,0:1)
      integer napprox(2)
      character*4 name4
c
      n_max = napprox(1)
      n_sav = napprox(2)
      if (n_sav.eq.0) return
      nel =nelfld(ifield)
      ntot=nx1*ny1*nz1*nel

      vol = voltm1
      if (nel.eq.nelv) vol = volvm1
c
c     Diag to see how much reduction in the residual is attained.
c
      alpha1 = glsc23(r,bi,vml,ntot)
      if (alpha1.gt.0) alpha1 = sqrt(alpha1/vol)
c
c     Update approximation space if dt has changed
      call updrhsh(approx,napprox,h1,h2,vml,vmk,ws,name4)
c
c
c     Perform Gram-Schmidt for previous soln's
c
      do i=1,n_sav
         ws(i) = vlsc3(r,approx(1,i),vml,ntot)
      enddo
      call gop    (ws,ws(n_sav+1),'+  ',n_sav)
c
      call cmult2   (approx(1,0),approx(1,1),ws(1),ntot)
      do i=2,n_sav
         call add2s2(approx(1,0),approx(1,i),ws(i),ntot)
      enddo
c
      call axhelm  (wl,approx(1,0),h1,h2,1,1)
      call col2    (wl,vmk,ntot)
      call dssum   (wl,nx1,ny1,nz1)
      call sub2    (r ,wl,ntot)
c ................................................................
c   Diag.
      alpha2 = glsc23(r,bi,vml,ntot)
      if (alpha2.gt.0) alpha2 = sqrt(alpha2/vol)
      ratio  = alpha1/alpha2
      n10=min(10,n_sav)
c
      if (nio.eq.0) write(6,10) istep,name4,alpha1,alpha2,ratio,n_sav
   10 format(4X,I7,4x,a4,' alph1n',1p3e12.4,i6)
c
      if (nio.eq.0) write(6,11) istep,name4,n_sav,(ws(i),i=1,n10)
   11 format(4X,I7,4x,a4,' halpha',i6,10(1p10e12.4,/,17x))
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gensh(v1,h1,h2,vml,vmk,approx,napprox,wl,ws,name4)
c
c     Reconstruct the solution to the original problem by adding back
c     the previous solutions
c
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
c
      common /iterhm/ niterhm
c
      parameter(lt=lx1*ly1*lz1*lelt)
      real v1(1),h1(1),h2(1),vml(1),vmk(1)
      real wl(1),ws(1)
      real approx(lt,0:1)
      integer napprox(2)
      character*4 name4
c
      n_max = napprox(1)
      n_sav = napprox(2)
      ntot=nx1*ny1*nz1*nelfld(ifield)
c
c     Reconstruct solution and save current du
c
      if (n_sav.lt.n_max) then
c
         if (niterhm.gt.0) then      ! new vector not in space
            n_sav = n_sav+1
            call copy(approx(1,n_sav),v1,ntot)
            call add2(v1,approx(1,0),ntot)
c           orthogonalize rhs against previous rhs and normalize
            call hconj(approx,n_sav,h1,h2,vml,vmk,ws,name4,ierr)

c           if (ierr.ne.0) n_sav = n_sav-1
            if (ierr.ne.0) n_sav = 0

         else

            call add2(v1,approx(1,0),ntot)

         endif
      else
         n_sav = 1
         call add2(v1,approx(1,0),ntot)
         call copy(approx(1,n_sav),v1,ntot)
c        normalize
         call hconj(approx,n_sav,h1,h2,vml,vmk,ws,name4,ierr)
         if (ierr.ne.0) n_sav = 0
      endif

      napprox(2)=n_sav

      return
      end
c-----------------------------------------------------------------------
      subroutine hconj(approx,k,h1,h2,vml,vmk,ws,name4,ierr)
c
c     Orthonormalize the kth vector against vector set
c
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'PARALLEL'
      include 'TSTEP'
c
      parameter  (lt=lx1*ly1*lz1*lelt)
      real approx(lt,0:1),h1(1),h2(1),vml(1),vmk(1),ws(1)
      character*4 name4
c
      ierr=0
      ntot=nx1*ny1*nz1*nelfld(ifield)
c
      call axhelm  (approx(1,0),approx(1,k),h1,h2,1,1)
      call col2    (approx(1,0),vmk,ntot)
c
c     Compute part of the norm   (Note:  a(0) already scaled by vml)
c
      alpha = glsc2(approx(1,0),approx(1,k),ntot)
      alph1 = alpha
c
c     Gram-Schmidt
c
      km1=k-1
      do i=1,km1
         ws(i) = vlsc2(approx(1,0),approx(1,i),ntot)
      enddo
      if (km1.gt.0) call gop(ws,ws(k),'+  ',km1)
c
      do i=1,km1
         alpham = -ws(i)
         call add2s2(approx(1,k),approx(1,i),alpham,ntot)
         alpha = alpha - ws(i)**2
      enddo
c
c    .Normalize new element in approximation space
c
      eps = 1.e-7
      if (wdsize.eq.8) eps = 1.e-15
      ratio = alpha/alph1
c
      if (ratio.le.0) then
         ierr=1
         if (nio.eq.0) write(6,12) istep,name4,k,alpha,alph1
   12    format(I6,1x,a4,' alpha b4 sqrt:',i4,1p2e12.4)
      elseif (ratio.le.eps) then
         ierr=2
         if (nio.eq.0) write(6,12) istep,name4,k,alpha,alph1
      else
         ierr=0
         alpha = 1.0/sqrt(alpha)
         call cmult(approx(1,k),alpha,ntot)
      endif
c
      if (ierr.ne.0) then
         call axhelm  (approx(1,0),approx(1,k),h1,h2,1,1)
         call col2    (approx(1,0),vmk,ntot)
c
c        Compute part of the norm   (Note:  a(0) already scaled by vml)
c
         alpha = glsc2(approx(1,0),approx(1,k),ntot)
         if (nio.eq.0) write(6,12) istep,name4,k,alpha,alph1
         if (alpha.le.0) then
            ierr=3
            if (nio.eq.0) write(6,12) istep,name4,k,alpha,alph1
            return
         endif
         alpha = 1.0/sqrt(alpha)
         call cmult(approx(1,k),alpha,ntot)
         ierr = 0
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine updrhsh(approx,napprox,h1,h2,vml,vmk,ws,name4)
c
c     Reorthogonalize approx if dt has changed
c
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
c
c
      parameter  (lt=lx1*ly1*lz1*lelt)
      real approx(lt,0:1),h1(1),h2(1),vml(1),vmk(1),ws(1)
      integer napprox(2)
      character*4 name4
c
      logical ifupdate
      logical ifnewdt
      save    ifnewdt
      data    ifnewdt /.false./
c
      character*4 name_old
      save    name_old
      data    name_old/'DMIR'/
c
      real    dtold
      save    dtold
      data    dtold/0.0/
c
c     First, we have to decide if the dt has changed.
c
      ifupdate = .false.
      if (dt.ne.dtold) then
         dtold    = dt
         name_old = name4
         ifnewdt  = .true.
         ifupdate = .true.
      elseif (ifnewdt) then
         if (name4.eq.name_old) then
            ifnewdt = .false.
         else
            ifupdate = .true.
         endif
      endif
      if (ifvarp(ifield)) ifupdate = .true.
      if (iflomach)       ifupdate = .true.

      if (ifupdate) then    ! reorthogonalize 
         n_sav = napprox(2)
         l     = 1
         do k=1,n_sav
c           Orthogonalize kth vector against {v_1,...,v_k-1}
            if (k.ne.l) then
               ntot = nx1*ny1*nz1*nelfld(ifield)
               call copy(approx(1,l),approx(1,k),ntot)
            endif
            call hconj(approx,l,h1,h2,vml,vmk,ws,name4,ierr)
            if (ierr.eq.0) l=l+1
         enddo
         napprox(2)=min(l,n_sav)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine hmhzpf(name,u,r,h1,h2,mask,mult,imesh,tli,maxit,isd,bi)
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'FDMH1'
      include 'CTIMER'
c
      CHARACTER*4    NAME
      REAL           U    (LX1,LY1,LZ1,1)
      REAL           R    (LX1,LY1,LZ1,1)
      REAL           H1   (LX1,LY1,LZ1,1)
      REAL           H2   (LX1,LY1,LZ1,1)
      REAL           MASK (LX1,LY1,LZ1,1)
      REAL           MULT (LX1,LY1,LZ1,1)
      REAL           bi   (LX1,LY1,LZ1,1)
      COMMON /CTMP0/ W1   (LX1,LY1,LZ1,LELT)
     $ ,             W2   (LX1,LY1,LZ1,LELT)
c
      etime1=dnekclock()
c
      IF (IMESH.EQ.1) NTOT = NX1*NY1*NZ1*NELV
      IF (IMESH.EQ.2) NTOT = NX1*NY1*NZ1*NELT
c
      tol = tli
      if (param(22).ne.0) tol = abs(param(22))
      CALL CHKTCG1 (TOL,R,H1,H2,MASK,MULT,IMESH,ISD)
c
c
c     Set flags for overlapping Schwarz preconditioner (pff 11/12/98)
c
                          kfldfdm = -1
c     if (name.eq.'TEMP') kfldfdm =  0
c     if (name.eq.'VELX') kfldfdm =  1
c     if (name.eq.'VELY') kfldfdm =  2
c     if (name.eq.'VELZ') kfldfdm =  3
      if (name.eq.'PRES') kfldfdm =  ndim+1

      if (ifdg) then
         call cggo_dg (u,r,h1,h2,bi,mask,name,tol,maxit)
      else
         call cggo
     $      (u,r,h1,h2,mask,mult,imesh,tol,maxit,isd,bi,name)
      endif
      thmhz=thmhz+(dnekclock()-etime1)
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine hsolve(name,u,r,h1,h2,vmk,vml,imsh,tol,maxit,isd
     $                 ,approx,napprox,bi)
c
c     Either std. Helmholtz solve, or a projection + Helmholtz solve
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
c
      CHARACTER*4    NAME
      REAL           U    (LX1,LY1,LZ1,1)
      REAL           R    (LX1,LY1,LZ1,1)
      REAL           H1   (LX1,LY1,LZ1,1)
      REAL           H2   (LX1,LY1,LZ1,1)
      REAL           vmk  (LX1,LY1,LZ1,1)
      REAL           vml  (LX1,LY1,LZ1,1)
      REAL           bi   (LX1,LY1,LZ1,1)
      REAL           approx (1)
      integer        napprox(1)
      common /ctmp2/ w1   (lx1,ly1,lz1,lelt)
      common /ctmp3/ w2   (2+2*mxprev)

      logical ifstdh
      character*4  cname
      character*6  name6

      logical ifwt,ifvec

      call chcopy(cname,name,4)
      call capit (cname,4)


      p945 = param(94)
      if (cname.eq.'PRES') p945 = param(95)

                          ifstdh = .false.
      if (param(93).eq.0) ifstdh = .true.
      if (p945.eq.0)      ifstdh = .true.
      if (istep.lt.p945)  ifstdh = .true.

      if (ifstdh) then
         call hmholtz(name,u,r,h1,h2,vmk,vml,imsh,tol,maxit,isd)
      else

         n = nx1*ny1*nz1*nelfld(ifield)

         call col2   (r,vmk,n)
         call dssum  (r,nx1,ny1,nz1)

         call blank (name6,6)
         call chcopy(name6,name,4)
         ifwt  = .true.
         ifvec = .false.

         call project1
     $       (r,n,approx,napprox,h1,h2,vmk,vml,ifwt,ifvec,name6)

         call hmhzpf (name,u,r,h1,h2,vmk,vml,imsh,tol,maxit,isd,bi)

         call project2
     $       (u,n,approx,napprox,h1,h2,vmk,vml,ifwt,ifvec,name6)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine project1(b,n,rvar,ivar,h1,h2,msk,w,ifwt,ifvec,name6)

c     1. Compute the projection of x onto X

c     2. Re-orthogonalize the X basis set and corresponding B=A*X
c        vectors if A has changed.

c     Output:  b = b - projection of b onto B

c     Input:   n     = length of field (or multifields, when ifvec=true)
c              rvar  = real array of field values, including old h1,h2, etc.
c              ivar  = integer array of pointers, etc.
c              h1    = current h1, for Axhelm(.,.,h1,h2,...)
c              h2    = current h2
c              msk   = mask for Dirichlet BCs
c              w     = weight for inner products (typ. w=vmult, tmult, etc.)
c              ifwt  = use weighted inner products when ifwt=.true.
c              ifvec = are x and b vectors, or scalar fields?
c              name6 = discriminator for action of A*x

c     The idea here is to have one pair of projection routines for 
c     constructing the new rhs (project1) and reconstructing the new
c     solution (x = xbar + dx) plus updating the approximation space.
c     The latter functions are done in project2.
c
c     The approximation space X and corresponding right-hand sides,
c     B := A*X are stored in rvar, as well as h1old and h2old and a
c     couple of other auxiliary arrays.

c     In this new code, we retain both X and B=A*X and we re-orthogonalize
c     at each timestep (with no extra matrix-vector products, but O(n m^2)
c     work.   The idea is to retain fresh vectors by injecting the most 
c     recent solution and pushing the oldest off the stack, hopefully 
c     keeping the number of vectors, m, small.


      include 'SIZE'   ! For nid/nio
      include 'TSTEP'  ! For istep

      real b(n),rvar(n,1),h1(n),h2(n),w(n),msk(n)
      integer ivar(1)
      character*6 name6
      logical ifwt,ifvec

      nn = n
      if (ifvec) nn = n*ndim

      call proj_get_ivar
     $   (m,mmx,ixb,ibb,ix,ib,ih1,ih2,ivar,n,ifvec,name6)

      if (m.le.0) return

      ireset=iproj_chk(rvar(ih1,1),rvar(ih2,1),h1,h2,n) ! Updated matrix?

      bb4 = glsc3(b,w,b,n)
      bb4 = sqrt(bb4)


c     Re-orthogonalize basis set w.r.t. new vectors if space has changed.

      if (ireset.eq.1) then

         do j=0,m-1         ! First, set B := A*X
            jb = ib+j*nn
            jx = ix+j*nn
            call proj_matvec (rvar(jb,1),rvar(jx,1),n,h1,h2,msk,name6)
         enddo

c         if (nio.eq.0) write(6,'(13x,A)') 'Reorthogonalize Basis'

         call proj_ortho    ! Orthogonalize X & B basis sets
     $      (rvar(ix,1),rvar(ib,1),n,m,w,ifwt,ifvec,name6)

      endif

c     ixb is pointer to xbar,  ibb is pointer to bbar := A*xbar

      call project1_a(rvar(ixb,1),rvar(ibb,1),b,rvar(ix,1),rvar(ib,1)
     $               ,n,m,w,ifwt,ifvec)

      baf = glsc3(b,w,b,n)
      baf = sqrt(baf)
      ratio = bb4/baf

c      if (nio.eq.0) write(6,1) istep,bb4,baf,ratio,m,name6
c    1 format(4x,i7,1p3e13.4,i4,1x,a6,' PROJECT')


      if (nio.eq.0) write(6,1) istep,'  Project ' // name6,
     &                         bb4,baf,ratio,m,ireset
    1 format(i11,a,6x,1p3e13.4,i4,i4)



      return
      end
c-----------------------------------------------------------------------
      subroutine project1_a(xbar,bbar,b,xx,bb,n,m,w,ifwt,ifvec)

c     xbar is best fit in xx, bbar = A*xbar
c     b <-- b - bbar

      include 'SIZE'
      real xbar(n),bbar(n),b(n),xx(n,m),bb(n,m),w(n)
      logical ifwt,ifvec

      real alpha(mxprev),work(mxprev)


      if (m.le.0) return

      if (ifwt) then
         do j=1,m
            alpha(j)=vlsc3(xx(1,j),w,b,n)
         enddo
      else
         do j=1,m
            alpha(j)=vlsc2(xx(1,j),b,n)
         enddo
      endif
      call gop(alpha,work,'+  ',m)

      call cmult2(xbar,xx(1,1),alpha(1),n)
      call cmult2(bbar,bb(1,1),alpha(1),n)

      do j=2,m
         call add2s2(xbar,xx(1,j),alpha(j),n)
         call add2s2(bbar,bb(1,j),alpha(j),n)
      enddo

      call sub2(b,bbar,n)

      return
      end
c-----------------------------------------------------------------------
      function iproj_chk(h1old,h2old,h1,h2,n)
      include 'SIZE'
      include 'TOTAL'

c     Matrix has changed if h1/h2 differ from old values

      real h1(n),h2(n),h1old(n),h2old(n)

      iproj_chk = 0

      if (ifmvbd) then
         iproj_chk = 1
         return
      endif

      dh1 = 0.
      dh2 = 0.
      do i=1,n
         dh1 = max(dh1,abs(h1(i)-h1old(i)))
         dh2 = max(dh2,abs(h2(i)-h2old(i)))
      enddo
      dh = max(dh1,dh2)
      dh = glmax(dh,1)  ! Max across all processors

      if (dh.gt.0) then

         call copy(h1old,h1,n)   ! Save old h1 / h2 values
         call copy(h2old,h2,n)

         iproj_chk = 1      ! Force re-orthogonalization of basis

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine proj_matvec(b,x,n,h1,h2,msk,name6)
      include 'SIZE'
      include 'TOTAL'
      real b(n),x(n),h1(n),h2(n),msk(n)
      character*6 name6

c     This is the default matvec for nekcem.

c     The code can later be updated to support different matvec
c     implementations, which would be discriminated by the character
c     string "name6"

      imsh = 1
      isd  = 1
      call axhelm  (b,x,h1,h2,imsh,isd)       ! b = A x
      call dssum   (b,nx1,ny1,nz1)
      call col2    (b,msk,n)

      return
      end
c-----------------------------------------------------------------------
      subroutine proj_ortho(xx,bb,n,m,w,ifwt,ifvec,name6)

      include 'SIZE'      ! nio
      include 'TSTEP'     ! istep
      include 'PARALLEL'  ! wdsize

      real xx(n,1),bb(n,1),w(n)
      character*6 name6
      logical ifwt,ifvec
      integer flag(mxprev)
      real normk,normp

      if (m.le.0) return

      if (      ifwt) alpha = glsc3(xx(1,m),w,bb(1,m),n)
      if (.not. ifwt) alpha = glsc2(xx(1,m),bb(1,m),n)
      if (alpha.eq.0) return

      scale = 1./sqrt(alpha)
      call cmult(xx(1,m),scale,n)
      call cmult(bb(1,m),scale,n)
      flag(m) = 1

      do k=m-1,1,-1  ! Reorthogonalize, starting with latest solution

         if (      ifwt) normk = glsc3(xx(1,k),w,bb(1,k),n)
         if (.not. ifwt) normk = glsc2(xx(1,k),bb(1,k),n)
         normk=sqrt(normk)

         do j=m,k+1,-1   ! Modified GS
            alpha = 0.
            if (ifwt) then
               alpha = alpha + .5*(vlsc3(xx(1,j),w,bb(1,k),n)
     $                       +     vlsc3(bb(1,j),w,xx(1,k),n))
            else
               alpha = alpha + .5*(vlsc2(xx(1,j),bb(1,k),n)
     $                       +     vlsc2(bb(1,j),xx(1,k),n))
            endif
            scale = -glsum(alpha,1)
            call add2s2(xx(1,k),xx(1,j),scale,n)
            call add2s2(bb(1,k),bb(1,j),scale,n)
         enddo
         if (      ifwt) normp = glsc3(xx(1,k),w,bb(1,k),n)
         if (.not. ifwt) normp = glsc2(xx(1,k),bb(1,k),n)
         normp=sqrt(normp)

         tol = 1.e-12
         if (wdsize.eq.4) tol=1.e-6

         if (normp.gt.tol*normk) then ! linearly independent vectors
           scale = 1./normp
           call cmult(xx(1,k),scale,n)
           call cmult(bb(1,k),scale,n)
           flag(k) = 1
c          if (nio.eq.0) write(6,2) istep,k,m,name6,normp,normk
    2      format(i9,'proj_ortho: ',2i4,1x,a6,' project ok.'
     $           ,1p2e12.4)
         else
           flag(k) = 0
           if (nio.eq.0) write(6,1) istep,k,m,name6,normp,normk
    1      format(i9,'proj_ortho: ',2i4,1x,a6,' Detect rank deficiency:'
     $           ,1p2e12.4)
         endif

      enddo

      k=0
      do j=1,m
         if (flag(j).eq.1) then
            k=k+1
            if (k.lt.j) then
               call copy(xx(1,k),xx(1,j),n)
               call copy(bb(1,k),bb(1,j),n)
            endif
         endif
      enddo
      m = k

      return
      end
c-----------------------------------------------------------------------
      subroutine project2(x,n,rvar,ivar,h1,h2,msk,w,ifwt,ifvec,name6)
      real x(n),b(n),rvar(n,1),h1(n),h2(n),w(n),msk(n)
      integer ivar(1)
      character*6 name6
      logical ifwt,ifvec

      call proj_get_ivar(m,mmx,ixb,ibb,ix,ib,ih1,ih2,ivar,n,ifvec,name6)

c     ix  is pointer to X,     ib  is pointer to B
c     ixb is pointer to xbar,  ibb is pointer to bbar := A*xbar

      call project2_a(x,rvar(ixb,1),rvar(ix,1),rvar(ib,1)
     $              ,n,m,mmx,h1,h2,msk,w,ifwt,ifvec,name6)

      ivar(2) = m ! Update number of saved vectors

      return
      end
c-----------------------------------------------------------------------
      subroutine project2_a
     $      (x,xbar,xx,bb,n,m,mmx,h1,h2,msk,w,ifwt,ifvec,name6)

      real x(n),xbar(n),xx(n,1),bb(n,1),h1(n),h2(n),w(n),msk(n)
      character*6 name6
      logical ifwt,ifvec

      nn = n
      if (ifvec) nn=ndim*n

      call add2        (x,xbar,n)      ! Restore desired solution

      if (m.eq.mmx) then ! Push old vector off the stack
         do k=2,mmx
            call copy     (xx(1,k-1),xx(1,k),nn)
            call copy     (bb(1,k-1),bb(1,k),nn)
         enddo
      endif

      m = min(m+1,mmx)
      call copy        (xx(1,m),x,nn)   ! Update (X,B)
      call proj_matvec (bb(1,m),xx(1,m),n,h1,h2,msk,name6)
      call proj_ortho  (xx,bb,n,m,w,ifwt,ifvec,name6) ! w=mult array

      return
      end
c-----------------------------------------------------------------------
      subroutine proj_get_ivar
     $    (m,mmx,ixb,ibb,ix,ib,ih1,ih2,ivar,n,ifvec,name6)

      include 'SIZE'
      include 'TSTEP'

      logical ifvec
      character*6 name6

      integer ivar(10)

      integer icalld
      save    icalld
      data    icalld/0/

      m    = ivar(2)
      mmx  = (mxprev-4)/2 ! ivar=0 --> mxprev array
      ivar(1) = mmx

      nn = n
      if (ifvec) nn = n*ndim  ! Number of entries in a vector


      ih1  = 1
      ih2  = ih1 + n
      ixb  = ih2 + n      ! pointer to xbar
      ibb  = ixb + nn     !    "    to bbar
      ix   = ibb + nn     !    "    to X
      ib   = ix  + nn*mmx !    "    to B

      return
      end
c-----------------------------------------------------------------------
      subroutine laplacep(name,u,mask,mult,ifld,tol,maxi,approx,napprox)
c
c     Solve Laplace's equation, with projection onto previous solutions.
c
c     Boundary condition strategy:
c
c     u = u0 + ub
c
c        u0 = 0 on Dirichlet boundaries
c        ub = u on Dirichlet boundaries
c
c        _
c        A ( u0 + ub ) = 0
c
c        _            _
c        A  u0  =   - A ub
c
c        _             _
c       MAM u0  =   -M A ub,    M is the mask
c
c                      _
c        A  u0  =   -M A ub ,  Helmholtz solve with SPD matrix A
c
c        u = u0+ub
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
c
      character*4 name
      real u(1),mask(1),mult(1),approx (1)
      integer   napprox(1)

      parameter (lt=lx1*ly1*lz1*lelt)
      common /scrvh/ h1(lt),h2(lt)
      common /scruz/ r (lt),ub(lt)

      logical ifstdh
      character*4  cname
      character*6  name6

      logical ifwt,ifvec

      call chcopy(cname,name,4)
      call capit (cname,4)

      call blank (name6,6)
      call chcopy(name6,name,4)
      ifwt  = .true.
      ifvec = .false.
      isd   = 1
      imsh  = 1
      nel   = nelfld(ifld)

      n = nx1*ny1*nz1*nel

      call copy (ub,u,n)             ! ub = u on boundary
      call dsavg(ub)                 ! Make certain ub is in H1
      call rone (h1,n)
      call rzero(h2,n)
                                     !     _
      call axhelm (r,ub,h1,h2,1,1)   ! r = A*ub

      do i=1,n                       !        _
         r(i)=-r(i)*mask(i)          ! r = -M*A*ub
      enddo

      call dssum  (r,nx1,ny1,nz1)    ! dssum rhs

      call project1
     $    (r,n,approx,napprox,h1,h2,mask,mult,ifwt,ifvec,name6)

      if (nel.eq.nelv) then
        call hmhzpf (name,u,r,h1,h2,mask,mult,imsh,tol,maxi,isd,binvm1)
      else
        call hmhzpf (name,u,r,h1,h2,mask,mult,imsh,tol,maxi,isd,bintm1)
      endif

      call project2
     $     (u,n,approx,napprox,h1,h2,mask,mult,ifwt,ifvec,name6)

      call add2(u,ub,n)

      return
      end
c-----------------------------------------------------------------------
