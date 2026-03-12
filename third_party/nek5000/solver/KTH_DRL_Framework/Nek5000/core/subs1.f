c-----------------------------------------------------------------------
      subroutine cggosf (u1,u2,u3,r1,r2,r3,h1,h2,rmult,binv,
     $                   vol,tin,maxit,matmod)

C     Conjugate gradient iteration for solution of coupled 
C     Helmholtz equations 

      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'FDMH1'

      common /screv/  dpc(lx1*ly1*lz1*lelt)
     $     ,          p1 (lx1*ly1*lz1*lelt)
      common /scrch/  p2 (lx1*ly1*lz1*lelt)
     $     ,          p3 (lx1*ly1*lz1*lelt)
      common /scrsl/  qq1(lx1*ly1*lz1*lelt)
     $     ,          qq2(lx1*ly1*lz1*lelt)
     $     ,          qq3(lx1*ly1*lz1*lelt)
      common /scrmg/  pp1(lx1*ly1*lz1*lelt)
     $     ,          pp2(lx1*ly1*lz1*lelt)
     $     ,          pp3(lx1*ly1*lz1*lelt)
     $     ,          wa (lx1*ly1*lz1*lelt)
      real ap1(1),ap2(1),ap3(1)
      equivalence (ap1,pp1),(ap2,pp2),(ap3,pp3)

      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      common /cprint/ ifprint
      logical ifdfrm, iffast, ifh2, ifsolv, ifprint

      real u1(1),u2(1),u3(1),
     $     r1(1),r2(1),r3(1),h1(1),h2(1),rmult(1),binv(1)


      logical iffdm,ifcrsl

      iffdm  = .true.
      iffdm  = .false.
c     ifcrsl = .true.
      ifcrsl = .false.

      nel   = nelfld(ifield)
      nxyz  = lx1*ly1*lz1
      n     = nxyz*nel

      if (istep.le.1.and.iffdm) call set_fdm_prec_h1A

      tol  = tin

c     overrule input tolerance
      if (restol(ifield).ne.0) tol=restol(ifield)

      if (ifcrsl) call set_up_h1_crs_strs(h1,h2,ifield,matmod)

c      if (nio.eq.0.and.istep.eq.1) write(6,6) ifield,tol,tin
c   6  format(i3,1p2e12.4,' ifield, tol, tol_in')

      if ( .not.ifsolv ) then           !     Set logical flags
         call setfast (h1,h2,imesh)
         ifsolv = .true.
      endif

      call opdot (wa,r1,r2,r3,r1,r2,r3,n)
      rbnorm = glsc3(wa,binv,rmult,n)
      rbnorm = sqrt ( rbnorm / vol )
      if (rbnorm .lt. tol**2) then
         iter = 0
         r0 = rbnorm
c        if ( .not.ifprint )  goto 9999
         if (matmod.ge.0.and.nio.eq.0) write (6,3000) 
     $                                 istep,iter,rbnorm,r0,tol
         if (matmod.lt.0.and.nio.eq.0) write (6,3010) 
     $                                 istep,iter,rbnorm,r0,tol
         goto 9999
      endif

C     Evaluate diagional pre-conidtioner for fluid solve
      call setprec (dpc,h1,h2,imesh,1)
      call setprec (wa ,h1,h2,imesh,2)
      call add2    (dpc,wa,n)
      if (ldim.eq.3) then
         call setprec (wa,h1,h2,imesh,3)
         call add2    (dpc,wa,n)
      endif
c     call rone (dpc,n)
c     call copy (dpc,binv,n)

      if (iffdm) then
         call set_fdm_prec_h1b(dpc,h1,h2,nel)
         call fdm_h1a (pp1,r1,dpc,nel,ktype(1,1,1),wa)
         call fdm_h1a (pp2,r2,dpc,nel,ktype(1,1,2),wa)
         call fdm_h1a (pp3,r3,dpc,nel,ktype(1,1,3),wa)
         call rmask   (pp1,pp2,pp3,nel)
         call opdssum (pp1,pp2,pp3)
      else
         call col3 (pp1,dpc,r1,n)
         call col3 (pp2,dpc,r2,n)
         if (if3d) call col3 (pp3,dpc,r3,n)
      endif
      if (ifcrsl) then
         call crs_strs(p1,p2,p3,r1,r2,r3)
         call rmask   (p1,p2,p3,nel)
      else
         call opzero(p1,p2,p3)
      endif
      call opadd2       (p1,p2,p3,pp1,pp2,pp3)
      rpp1 = op_glsc2_wt(p1,p2,p3,r1,r2,r3,rmult)

      maxit=200
      do 1000 iter=1,maxit
         call axhmsf  (ap1,ap2,ap3,p1,p2,p3,h1,h2,matmod)
         call rmask   (ap1,ap2,ap3,nel)
         call opdssum (ap1,ap2,ap3)
         pap   = op_glsc2_wt(p1,p2,p3,ap1,ap2,ap3,rmult)
         alpha = rpp1 / pap

         call opadds (u1,u2,u3,p1 ,p2 ,p3 , alpha,n,2)
         call opadds (r1,r2,r3,ap1,ap2,ap3,-alpha,n,2)

         call opdot  (wa,r1,r2,r3,r1,r2,r3,n)
         rbnorm = glsc3(wa,binv,rmult,n)
         rbnorm = sqrt (rbnorm/vol)

         if (iter.eq.1) r0 = rbnorm

         if (rbnorm.lt.tol) then
            ifin = iter
            if (nio.eq.0) then
               if (matmod.ge.0) write(6,3000) istep,ifin,rbnorm,r0,tol
               if (matmod.lt.0) write(6,3010) istep,ifin,rbnorm,r0,tol
            endif
            goto 9999
         endif

         if (iffdm) then
            call fdm_h1a (pp1,r1,dpc,nel,ktype(1,1,1),wa)
            call fdm_h1a (pp2,r2,dpc,nel,ktype(1,1,2),wa)
            call fdm_h1a (pp3,r3,dpc,nel,ktype(1,1,3),wa)
            call rmask   (pp1,pp2,pp3,nel)
            call opdssum (pp1,pp2,pp3)
         else
            call col3 (pp1,dpc,r1,n)
            call col3 (pp2,dpc,r2,n)
            if (if3d) call col3 (pp3,dpc,r3,n)
         endif

         if (ifcrsl) then
           call crs_strs(qq1,qq2,qq3,r1,r2,r3)
           call rmask   (qq1,qq2,qq3,nel)
           call opadd2  (pp1,pp2,pp3,qq1,qq2,qq3)
         endif

         call opdot (wa,r1,r2,r3,pp1,pp2,pp3,n)

         rpp2 = rpp1
         rpp1 = glsc2(wa,rmult,n)
         beta = rpp1/rpp2
         call opadds (p1,p2,p3,pp1,pp2,pp3,beta,n,1)

 1000 continue
      if (matmod.ge.0.and.nio.eq.0) write (6,3001) 
     $                              istep,iter,rbnorm,r0,tol
      if (matmod.lt.0.and.nio.eq.0) write (6,3011) 
     $                              istep,iter,rbnorm,r0,tol

 9999 continue
      ifsolv = .false.


 3000 format(i11,'  Helmh3 fluid  ',I6,1p3E13.4)
 3010 format(i11,'  Helmh3 mesh   ',I6,1p3E13.4)
 3001 format(i11,'  Helmh3 fluid unconverged! ',I6,1p3E13.4)
 3011 format(i11,'  Helmh3 mesh unconverged! ',I6,1p3E13.4)

      return
      end
c-----------------------------------------------------------------------
      subroutine setdt
c
c     Set the new time step. All cases covered.
c
      include 'SIZE'
      include 'SOLN'
      include 'MVGEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'PARALLEL'

      common /scruz/ cx(lx1*ly1*lz1*lelt)
     $ ,             cy(lx1,ly1,lz1,lelt)
     $ ,             cz(lx1,ly1,lz1,lelt)

      common /cprint/ ifprint
      logical         ifprint
      common /udxmax/ umax
      REAL     DTOLD
      SAVE     DTOLD
      DATA     DTOLD /0.0/
      REAL     DTOpf
      SAVE     DTOpf
      DATA     DTOpf /0.0/
      logical iffxdt
      save    iffxdt
      data    iffxdt /.false./
C

      if (param(12).lt.0.or.iffxdt) then
         iffxdt    = .true.
         param(12) = abs(param(12))
         dt        = param(12)
         dtopf     = dt
         if (ifmvbd) then
           call opsub3 (cx,cy,cz,vx,vy,vz,wx,wy,wz)
           call compute_cfl(umax,cx,cy,cz,1.0)
         else
           call compute_cfl(umax,vx,vy,vz,1.0)
         endif

         goto 200
      else IF (PARAM(84).NE.0.0) THEN
         if (dtold.eq.0.0) then
            dt   =param(84)
            dtold=param(84)
            dtopf=param(84)
            return
         else
            dtold=dt
            dtopf=dt
            dt=dtopf*param(85)
            dt=min(dt,param(12))
         endif
      endif

C
C     Find DT=DTCFL based on CFL-condition (if applicable)
C
      CALL SETDTC
      DTCFL = DT
C
C     Find DTFS based on surface tension (if applicable)
C
      CALL SETDTFS (DTFS)
C
C     Select appropriate DT
C
      IF ((DT.EQ.0.).AND.(DTFS.GT.0.)) THEN
          DT = DTFS
      ELSEIF ((DT.GT.0.).AND.(DTFS.GT.0.)) THEN
          DT = MIN(DT,DTFS)
      ELSEIF ((DT.EQ.0.).AND.(DTFS.EQ.0.)) THEN
          DT = 0.
          IF (IFFLOW.AND.NID.EQ.0.AND.IFPRINT) THEN
             WRITE (6,*) 'WARNING: CFL-condition & surface tension'
             WRITE (6,*) '         are not applicable'
          endif
      ELSEIF ((DT.GT.0.).AND.(DTFS.EQ.0.)) THEN
          DT = DT
      ELSE
          DT = 0.
          IF (NIO.EQ.0) WRITE (6,*) 'WARNING: DT<0 or DTFS<0'
          IF (NIO.EQ.0) WRITE (6,*) '         Reset DT      '
      endif
C
C     Check DT against user-specified input, DTINIT=PARAM(12).
C
      IF ((DT.GT.0.).AND.(DTINIT.GT.0.)) THEN
         DT = MIN(DT,DTINIT)
      ELSEIF ((DT.EQ.0.).AND.(DTINIT.GT.0.)) THEN
         DT = DTINIT
      ELSEIF ((DT.GT.0.).AND.(DTINIT.EQ.0.)) THEN
         DT = DT
      ELSEIF (.not.iffxdt) THEN
         DT = 0.001
         IF(NIO.EQ.0)WRITE (6,*) 'WARNING: Set DT=0.001 (arbitrarily)'
      endif
C
C     Check if final time (user specified) has been reached.
C
 200  IF (TIME+DT .GE. FINTIM .AND. FINTIM.NE.0.0) THEN
C        Last step
         LASTEP = 1
         DT = FINTIM-TIME
         IF (NIO.EQ.0) WRITE (6,*) 'Final time step = ',DT
      endif
C
      COURNO = DT*UMAX
      IF (NIO.EQ.0.AND.IFPRINT.AND.DT.NE.DTOLD)
     $   WRITE (6,100) DT,DTCFL,DTFS,DTINIT
 100     FORMAT(5X,'DT/DTCFL/DTFS/DTINIT',4E12.3)
C
C     Put limits on how much DT can change.
C
      IF (DTOLD.NE.0.0 .AND. LASTEP.NE.1) THEN
         DTMIN=0.8*DTOLD
         DTMAX=1.2*DTOLD
         DT = MIN(DTMAX,DT)
         DT = MAX(DTMIN,DT)
      endif
      DTOLD=DT

C      IF (PARAM(84).NE.0.0) THEN
C            dt=dtopf*param(85)
C            dt=min(dt,param(12))
C      endif

      if (iffxdt) dt=dtopf
      COURNO = DT*UMAX

! synchronize time step for multiple sessions
      if (ifneknek) dt = glmin_ms(dt,1)
c
      if (iffxdt.and.abs(courno).gt.10.*abs(ctarg)) then
         if (nid.eq.0) write(6,*) 'CFL, Ctarg!',courno,ctarg
         call emerxit
      endif


      return
      end
C
C--------------------------------------------------------
C
      subroutine cvgnlps (ifconv)
C----------------------------------------------------------------------
C
C     Check convergence for non-linear passisve scalar solver.
C     Relevant for solving heat transport problems with radiation b.c.
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      LOGICAL  IFCONV
C
      IF (IFNONL(IFIELD)) THEN
         IFCONV = .FALSE.
      ELSE
         IFCONV = .TRUE.
         return
      endif
C
      TNORM1 = TNRMH1(IFIELD-1)
      CALL UNORM
      TNORM2 = TNRMH1(IFIELD-1)
      EPS = ABS((TNORM2-TNORM1)/TNORM2)
      IF (EPS .LT. TOLNL) IFCONV = .TRUE.
C
      return
      end
C
      subroutine unorm
C---------------------------------------------------------------------
C
C     Norm calculation.
C
C--------------------------------------------------------------------- 
      include 'SIZE'
      include 'SOLN'
      include 'TSTEP'
C
      IF (IFIELD.EQ.1) THEN
C
C        Compute norms of the velocity.
C        Compute time mean (L2) of the inverse of the time step. 
C        Compute L2 in time, H1 in space of the velocity.
C
         CALL NORMVC (VNRMH1,VNRMSM,VNRML2,VNRML8,VX,VY,VZ)
         IF (ISTEP.EQ.0) return
         IF (ISTEP.EQ.1) THEN
            DTINVM = 1./DT
            VMEAN  = VNRML8
         ELSE
            tden   = time
            if (time.le.0) tden = abs(time)+1.e-9
            arg    = ((TIME-DT)*DTINVM**2+1./DT)/tden
            if (arg.gt.0) DTINVM = SQRT(arg)
            arg    = ((TIME-DT)*VMEAN**2+DT*VNRMH1**2)/tden
            if (arg.gt.0) VMEAN  = SQRT(arg)
         endif
      ELSE
C
C     Compute norms of a passive scalar
C
         CALL NORMSC (TNRMH1(IFIELD-1),TNRMSM(IFIELD-1),
     $                TNRML2(IFIELD-1),TNRML8(IFIELD-1),
     $                     T(1,1,1,1,IFIELD-1),IMESH)
         TMEAN(IFIELD-1) = 0.
      endif
C
      return
      end
C
      subroutine chktmg (tol,res,w1,w2,mult,mask,imesh)
C-------------------------------------------------------------------
C
C     Check that the tolerances are not too small for the MG-solver.
C     Important when calling the MG-solver (Gauss-Lobatto mesh).
C     Note: direct stiffness summation
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'EIGEN'
C
      REAL RES  (LX1,LY1,LZ1,1)
      REAL W1   (LX1,LY1,LZ1,1)
      REAL W2   (LX1,LY1,LZ1,1)
      REAL MULT (LX1,LY1,LZ1,1)
      REAL MASK (LX1,LY1,LZ1,1)
C
C     Single or double precision???
C
      DELTA = 1.E-9
      X     = 1.+DELTA
      Y     = 1.
      DIFF  = ABS(X-Y)
      IF (DIFF.EQ.0.) EPS = 1.E-6*EIGGA/EIGAA
      IF (DIFF.GT.0.) EPS = 1.E-13*EIGGA/EIGAA
C
      IF (IMESH.EQ.1) NL = NELV
      IF (IMESH.EQ.2) NL = NELT
      NTOT1 = lx1*ly1*lz1*NL
      CALL COPY (W1,RES,NTOT1)
C
      CALL DSSUM (W1,lx1,ly1,lz1)
C
      IF (IMESH.EQ.1) THEN
         CALL COL3 (W2,BINVM1,W1,NTOT1)
         RINIT  = SQRT(GLSC3 (W2,W1,MULT,NTOT1)/VOLVM1)
      ELSE
         CALL COL3 (W2,BINTM1,W1,NTOT1)
         RINIT  = SQRT(GLSC3 (W2,W1,MULT,NTOT1)/VOLTM1)
      endif
      RMIN   = EPS*RINIT
      IF (TOL.LT.RMIN) THEN
         TOLOLD=TOL
         TOL = RMIN
         IF (NIO.EQ.0)
     $   WRITE (6,*) 'New MG-tolerance (RINIT*epsm*cond) = ',TOL,TOLOLD
      endif
C
      CALL RONE (W1,NTOT1)
      BCNEU1 = GLSC3(W1,MASK,MULT,NTOT1)
      BCNEU2 = GLSC3(W1,W1  ,MULT,NTOT1)
      BCTEST = ABS(BCNEU1-BCNEU2)
      IF (BCTEST .LT. .1) THEN
         OTR = GLSUM (RES,NTOT1)
         TOLMIN = ABS(OTR)*EIGGA/EIGAA
         IF (TOL .LT. TOLMIN) THEN
             TOLOLD = TOL
             TOL = TOLMIN
             IF (NIO.EQ.0)
     $       WRITE (6,*) 'New MG-tolerance (OTR) = ',TOL,TOLOLD
         endif
      endif
C
      return
      end
C
C
      subroutine setdtc
C--------------------------------------------------------------
C
C     Compute new timestep based on CFL-condition
C
C--------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'MVGEOM'
      include 'MASS'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
C
      common /ctmp1/ u(lx1,ly1,lz1,lelv)
     $ ,             v(lx1,ly1,lz1,lelv)
     $ ,             w(lx1,ly1,lz1,lelv)
      common /ctmp0/ x(lx1,ly1,lz1,lelv)
     $ ,             r(lx1,ly1,lz1,lelv)
      common /udxmax/ umax

      common /scruz/ cx(lx1*ly1*lz1*lelv)
     $ ,             cy(lx1,ly1,lz1,lelv)
     $ ,             cz(lx1,ly1,lz1,lelv)
C
C
      REAL VCOUR
      SAVE VCOUR
C
      INTEGER IFIRST
      SAVE    IFIRST
      DATA    IFIRST/0/
C
C
C     Steady state => all done
C
C
      IF (.NOT.IFTRAN) THEN
         IFIRST=1
         LASTEP=1
         return
      endif
C
      irst = param(46)
      if (irst.gt.0) ifirst=1
C
C     First time around
C
      IF (IFIRST.EQ.0) THEN
         DT     = DTINIT
         IF (IFFLOW) THEN
            IFIELD = 1
            CALL BCDIRVC (VX,VY,VZ,v1mask,v2mask,v3mask)
         endif
      endif
      IFIRST=IFIRST+1
C
      DTOLD = DT
C
C     Convection ?
C
C     Don't enforce Courant condition if there is no convection.
C
C
      ICONV=0
      IF (IFFLOW .AND. IFNAV) ICONV=1
      IF (IFWCNO)             ICONV=1
      IF (IFHEAT) THEN
         DO 10 IPSCAL=0,NPSCAL
            IF (IFADVC(IPSCAL+2)) ICONV=1
   10    CONTINUE
      endif

      IF (ICONV.EQ.0) THEN
         DT=0.
         return
      endif
C
C
C     Find Courant and Umax
C
C
      NTOT   = lx1*ly1*lz1*NELV
      NTOTL  = LX1*LY1*LZ1*LELV
      NTOTD  = NTOTL*ldim
      COLD   = COURNO
      CMAX   = 1.2*CTARG
      CMIN   = 0.8*CTARG
 
      if (ifmvbd) then
        call opsub3 (cx,cy,cz,vx,vy,vz,wx,wy,wz)
           call compute_cfl(umax,cx,cy,cz,1.0)
      else
           call compute_cfl(umax,vx,vy,vz,1.0)
      endif

c      if (nio.eq.0) write(6,1) istep,time,umax,cmax
c   1  format(i9,1p3e12.4,' cmax')

C     Zero DT

      IF (DT .EQ. 0.0) THEN

         IF (UMAX .NE. 0.0) THEN
            DT = CTARG/UMAX
            VCOUR = UMAX
         ELSEIF (IFFLOW) THEN
C
C           We'll use the body force to predict max velocity
C
            CALL SETPROP
            IFIELD = 1
C
            CALL MAKEUF
            CALL OPDSSUM (BFX,BFY,BFZ)
            CALL OPCOLV  (BFX,BFY,BFZ,BINVM1)
            FMAX=0.0
            CALL RZERO (U,NTOTD)
            DO 600 I=1,NTOT
               U(I,1,1,1) = ABS(BFX(I,1,1,1))
               V(I,1,1,1) = ABS(BFY(I,1,1,1))
               W(I,1,1,1) = ABS(BFZ(I,1,1,1))
 600        CONTINUE
            FMAX    = GLMAX (U,NTOTD)
            DENSITY = AVTRAN(1)
            AMAX    = FMAX/DENSITY
            DXCHAR  = SQRT( (XM1(1,1,1,1)-XM1(2,1,1,1))**2 +
     $                      (YM1(1,1,1,1)-YM1(2,1,1,1))**2 + 
     $                      (ZM1(1,1,1,1)-ZM1(2,1,1,1))**2 ) 
            DXCHAR  = GLMIN (dxchar,1)
            IF (AMAX.NE.0.) THEN
               DT = SQRT(CTARG*DXCHAR/AMAX)
            ELSE
               IF (NID.EQ.0) 
     $         WRITE (6,*) 'CFL: Zero velocity and body force'
               DT = 0.0
               return
            endif
         ELSEIF (IFWCNO) THEN
            IF (NID.EQ.0) 
     $      WRITE (6,*) ' Stefan problem with no fluid flow'
            DT = 0.0
            return
         endif
C
      ELSEIF ((DT.GT.0.0).AND.(UMAX.NE.0.0)) THEN
C
C
C     Nonzero DT & nonzero velocity
C
C
      COURNO = DT*UMAX
      VOLD   = VCOUR
      VCOUR  = UMAX
      IF (IFIRST.EQ.1) THEN
         COLD = COURNO
         VOLD = VCOUR
      endif
      CPRED  = 2.*COURNO-COLD
C
C     Change DT if it is too big or if it is too small 
C
c     if (nid.eq.0) 
c    $write(6,917) dt,umax,vold,vcour,cpred,cmax,courno,cmin
c 917 format(' dt',4f9.5,4f10.6)
      IF(COURNO.GT.CMAX .OR. CPRED.GT.CMAX .OR. COURNO.LT.CMIN) THEN
C
            A=(VCOUR-VOLD)/DT
            B=VCOUR
C           -C IS Target Courant number
            C=-CTARG
            DISCR=B**2-4*A*C
            DTOLD=DT
            IF(DISCR.LE.0.0)THEN
c               if (nio.eq.0) 
c     $         PRINT*,'Problem calculating new DT Discriminant=',discr
               DT=DT*(CTARG/COURNO)
C               IF(DT.GT.DTOLD) DT=DTOLD
            ELSE IF(ABS((VCOUR-VOLD)/VCOUR).LT.0.001)THEN
C              Easy: same v as before (LINEARIZED)
               DT=DT*(CTARG/COURNO)
c     if (nid.eq.0) 
c    $write(6,918) dt,dthi,dtlow,discr,a,b,c
c 918 format(' d2',4f9.5,4f10.6)
            ELSE
               DTLOW=(-B+SQRT(DISCR) )/(2.0*A)
               DTHI =(-B-SQRT(DISCR) )/(2.0*A)
               IF(DTHI .GT. 0.0 .AND. DTLOW .GT. 0.0)THEN
                  DT = MIN (DTHI,DTLOW)
c     if (nid.eq.0) 
c    $write(6,919) dt,dthi,dtlow,discr,a,b,c
c 919 format(' d3',4f9.5,4f10.6)
               ELSE IF(DTHI .LE. 0.0 .AND. DTLOW .LE. 0.0)THEN
c                 PRINT*,'DTLOW,DTHI',DTLOW,DTHI
c                 PRINT*,'WARNING: Abnormal DT from CFL-condition'
c                 PRINT*,'         Keep going'
                  DT=DT*(CTARG/COURNO)
               ELSE
C                 Normal case; 1 positive root, one negative root
                  DT = MAX (DTHI,DTLOW)
c     if (nid.eq.0) 
c    $write(6,929) dt,dthi,dtlow,discr,a,b,c
c 929 format(' d4',4f9.5,4f10.6)
               endif
            endif
C           We'll increase gradually-- make it the geometric mean between
c     if (nid.eq.0) 
c    $write(6,939) dt,dtold
c 939 format(' d5',4f9.5,4f10.6)
            IF (DTOLD/DT .LT. 0.2) DT = DTOLD*5
      endif
C
      endif
C
      return
      end
C
      subroutine setdtfs (dtfs)
C
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'GEOM'
      include 'TSTEP'
      common /ctmp0/  stc(lx1,ly1,lz1),sigst(lx1,ly1),dtst(lx1,ly1)
      character cb*3,cb2*2
C
C     Applicable?
C
      IF (.NOT.IFSURT) THEN
         DTFS = 0.0
         return
      endif
C
      NFACE  = 2*ldim
      NXZ1   = lx1*lz1
      NTOT1  = lx1*ly1*lz1*NELV
      DTFS   = 1.1E+10
C
C     Kludge : for debugging purpose Fudge Factor comes in
C              as PARAM(45)
C
      FACTOR = PARAM(45)
      IF (FACTOR .LE. 0.0) FACTOR=1.0
      FACTOR = FACTOR / SQRT( PI**3 )
C
      IF (ISTEP.EQ.1) CALL SETPROP
C
      IFIELD = 1
      RHOMIN = GLMIN( VTRANS(1,1,1,1,IFIELD),NTOT1 )
C
      DO 100 IEL=1,NELV
      DO 100 IFC=1,NFACE
         CB =CBC(IFC,IEL,IFIELD)
         CB2=CBC(IFC,IEL,IFIELD)
         IF (CB2.NE.'MS' .AND. CB2.NE.'ms') GOTO 100
             IF (CB2(1:1).EQ.'M') THEN
                BC4 = BC(4,IFC,IEL,IFIELD)
                CALL CFILL  (SIGST,BC4,NXZ1)
             ELSE
                CALL FACEIS (CB,STC,IEL,IFC,lx1,ly1,lz1)
                CALL FACEXS (SIGST,STC,IFC,0)
             endif
             SIGMAX = VLMAX (SIGST,NXZ1)
             IF (SIGMAX.LE.0.0) GOTO 100
             RHOSIG = SQRT( RHOMIN/SIGMAX )
             IF (ldim.EQ.2) THEN
                CALL CDXMIN2 (DTST,RHOSIG,IEL,IFC,IFAXIS)
             ELSE
                CALL CDXMIN3 (DTST,RHOSIG,IEL,IFC)
             endif
             DTMIN = VLMIN( DTST,NXZ1 )
             DTFS  = MIN( DTFS,DTMIN )
  100 CONTINUE
      DTFS    = GLMIN( dtfs,1 )
C
      IF (DTFS .GT. 1.E+10) DTFS = 0.0
      DTFS = DTFS * FACTOR
C
      IF (DTFS.EQ.0.0) THEN
         IF (ISTEP.EQ.1.AND.NIO.EQ.0) THEN
            WRITE (6,*) ' Warning - zero surface-tension may results in'
            WRITE (6,*) ' instability of free-surface update'
         endif
      endif
C
      return
      end
      subroutine cdxmin2 (dtst,rhosig,iel,ifc,ifaxis)
C
      include 'SIZE'
      include 'GEOM'
      include 'DXYZ'
      common /delrst/ drst(lx1),drsti(lx1)
      common /ctmp0/  xfm1(lx1),yfm1(lx1),t1xf(lx1),t1yf(lx1)
      DIMENSION DTST(LX1,1)
      LOGICAL IFAXIS
C
      DELTA = 1.E-9
      X     = 1.+DELTA
      Y     = 1.
      DIFF  = ABS(X-Y)
      IF (DIFF.EQ.0.) EPS = 1.E-6
      IF (DIFF.GT.0.) EPS = 1.E-13
C
      CALL FACEC2 (XFM1,YFM1,XM1(1,1,1,IEL),YM1(1,1,1,IEL),IFC)
C
      IF (IFC.EQ.1 .OR. IFC.EQ.3) THEN
         CALL MXM (DXM1,lx1,XFM1,lx1,T1XF,1)
         CALL MXM (DXM1,lx1,YFM1,lx1,T1YF,1)
      ELSE
         IF (IFAXIS) CALL SETAXDY ( IFRZER(IEL) )
         CALL MXM (DYM1,ly1,XFM1,ly1,T1XF,1)
         CALL MXM (DYM1,ly1,YFM1,ly1,T1YF,1)
      endif
C
      IF (IFAXIS) THEN
         DO 100 IX=1,lx1
            IF (YFM1(IX) .LT. EPS) THEN
               DTST(IX,1) = 1.e+10
            ELSE
               XJ = SQRT( T1XF(IX)**2 + T1YF(IX)**2 )*DRST(IX)
               DTST(IX,1) = RHOSIG * SQRT( XJ**3 ) * YFM1(IX)
            endif
  100    CONTINUE
      ELSE
         DO 200 IX=1,lx1
            XJ = SQRT( T1XF(IX)**2 + T1YF(IX)**2 )*DRST(IX)
            DTST(IX,1) = RHOSIG * SQRT( XJ**3 )
  200    CONTINUE
      endif
C
      return
      end
      subroutine cdxmin3 (dtst,rhosig,iel,ifc)
C
      include 'SIZE'
      include 'GEOM'
      include 'DXYZ'
      common /delrst/ drst(lx1),drsti(lx1)
      common /ctmp0/  xfm1(lx1,ly1),yfm1(lx1,ly1),zfm1(lx1,ly1)
      common /ctmp1/  drm1(lx1,lx1),drtm1(lx1,ly1)
     $             ,  dsm1(lx1,lx1),dstm1(lx1,ly1)
      common /scrmg/  xrm1(lx1,ly1),yrm1(lx1,ly1),zrm1(lx1,ly1)
     $             ,  xsm1(lx1,ly1),ysm1(lx1,ly1),zsm1(lx1,ly1)
      dimension dtst(lx1,ly1)
C
      call facexv (xfm1,yfm1,zfm1,xm1(1,1,1,iel),ym1(1,1,1,iel),
     $             zm1(1,1,1,iel),ifc,0)
      call setdrs (drm1,drtm1,dsm1,dstm1,ifc)
C
      CALL MXM (DRM1,lx1, XFM1,lx1,XRM1,ly1)
      CALL MXM (DRM1,lx1, YFM1,lx1,YRM1,ly1)
      CALL MXM (DRM1,lx1, ZFM1,lx1,ZRM1,ly1)
      CALL MXM (XFM1,lx1,DSTM1,ly1,XSM1,ly1)
      CALL MXM (YFM1,lx1,DSTM1,ly1,YSM1,ly1)
      CALL MXM (ZFM1,lx1,DSTM1,ly1,ZSM1,ly1)
C
      DO 100 IX=1,lx1
      DO 100 IY=1,ly1
         DELR = XRM1(IX,IY)**2 + YRM1(IX,IY)**2 + ZRM1(IX,IY)**2
         DELS = XSM1(IX,IY)**2 + YSM1(IX,IY)**2 + ZSM1(IX,IY)**2
         DELR = SQRT( DELR )*DRST(IX)
         DELS = SQRT( DELS )*DRST(IY)
         XJ   = MIN( DELR,DELS )
         DTST(IX,IY) = RHOSIG * SQRT( XJ**3 )
  100 CONTINUE
C
      return
      end
C
      FUNCTION FACDOT(A,B,IFACE1)
C
C     Take the dot product of A and B on the surface IFACE1 of element IE.
C
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C         5 Jan 1989 15:12:22      PFF
C
      include 'SIZE'
      include 'TOPOL'
      DIMENSION A(LX1,LY1,LZ1),B(LX1,LY1)
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      SUM=0.0
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         SUM = SUM + A(J1,J2,1)*B(I,1)
  100 CONTINUE
C
      FACDOT = SUM
C
      return
      end
C
      subroutine fcaver(xaver,a,iel,iface1)
C------------------------------------------------------------------------
C
C     Compute the average of A over the face IFACE1 in element IEL.
C
C         A is a (NX,NY,NZ) data structure
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C------------------------------------------------------------------------
      include 'SIZE'
      include 'GEOM'
      include 'TOPOL'
      REAL A(LX1,LY1,LZ1,1)
C
      FCAREA = 0.
      XAVER  = 0.
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         FCAREA = FCAREA+AREA(I,1,IFACE1,IEL)
         XAVER  = XAVER +AREA(I,1,IFACE1,IEL)*A(J1,J2,1,IEL)
  100 CONTINUE
C
      XAVER = XAVER/FCAREA
      return
      end
      subroutine faccl2(a,b,iface1)
C
C     Collocate B with A on the surface IFACE1 of element IE.
C
C         A is a (NX,NY,NZ) data structure
C         B is a (NX,NY,IFACE) data structure
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C         5 Jan 1989 15:12:22      PFF
C
      include 'SIZE'
      include 'TOPOL'
      DIMENSION A(LX1,LY1,LZ1),B(LX1,LY1)
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         A(J1,J2,1) = A(J1,J2,1)*B(I,1)
  100 CONTINUE
C
      return
      end
C
      subroutine faccl3(a,b,c,iface1)
C
C     Collocate B with A on the surface IFACE1 of element IE.
C
C         A is a (NX,NY,NZ) data structure
C         B is a (NX,NY,IFACE) data structure
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C         5 Jan 1989 15:12:22      PFF
C
      include 'SIZE'
      include 'TOPOL'
      DIMENSION A(LX1,LY1,LZ1),B(LX1,LY1,LZ1),C(LX1,LY1)
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         A(J1,J2,1) = B(J1,J2,1)*C(I,1)
  100 CONTINUE
C
      return
      end
      subroutine faddcl3(a,b,c,iface1)
C
C     Collocate B with C and add to A on the surface IFACE1 of element IE.
C
C         A is a (NX,NY,NZ) data structure
C         B is a (NX,NY,NZ) data structure
C         C is a (NX,NY,IFACE) data structure
C         IFACE1 is in the preprocessor notation 
C         IFACE  is the dssum notation.
C         29 Jan 1990 18:00 PST   PFF
C
      include 'SIZE'
      include 'TOPOL'
      DIMENSION A(LX1,LY1,LZ1),B(LX1,LY1,LZ1),C(LX1,LY1)
C
C     Set up counters
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFACE1)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      I = 0
      DO 100 J2=JS2,JF2,JSKIP2
      DO 100 J1=JS1,JF1,JSKIP1
         I = I+1
         A(J1,J2,1) = A(J1,J2,1) + B(J1,J2,1)*C(I,1)
  100 CONTINUE
C
      return
      end
c-----------------------------------------------------------------------
      subroutine sethlm (h1,h2,intloc)
 
c     Set the variable property arrays H1 and H2
c     in the Helmholtz equation.
c     (associated with variable IFIELD)
c     INTLOC =      integration type

      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'

      real h1(1),h2(1)

      nel   = nelfld(ifield)
      ntot1 = lx1*ly1*lz1*nel

      if (iftran) then
         dtbd = bd(1)/dt
         call copy  (h1,vdiff (1,1,1,1,ifield),ntot1)
         if (intloc.eq.0) then
            call rzero (h2,ntot1)
         else
            call cmult2 (h2,vtrans(1,1,1,1,ifield),dtbd,ntot1)
         endif

c        if (ifield.eq.1 .and. ifanls) then   ! this should be replaced
c           const = 2.                        ! with a correct stress
c           call cmult (h1,const,ntot1)       ! formulation
c        endif

      ELSE
         CALL COPY  (H1,VDIFF (1,1,1,1,IFIELD),NTOT1)
         CALL RZERO (H2,NTOT1)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine nekuvp (iel)
C------------------------------------------------------------------
C
C     Generate user-specified material properties
C
C------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      include 'PARALLEL'
      include 'NEKUSE'
      ielg = lglel(iel)
c     IF (IFSTRS .AND. IFIELD.EQ.1) CALL STNRINV ! don't call! pff, 2007
      DO 10 K=1,lz1
      DO 10 J=1,ly1
      DO 10 I=1,lx1
         if (optlevel.le.2) CALL NEKASGN (I,J,K,IEL)
         CALL USERVP  (I,J,K,IELG)
         VDIFF (I,J,K,IEL,IFIELD) = UDIFF
         VTRANS(I,J,K,IEL,IFIELD) = UTRANS
 10   CONTINUE
      return
      end
C
      subroutine diagnos
      return
      end

c-----------------------------------------------------------------------
      subroutine setsolv
      include 'SIZE'

      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv

      ifsolv = .false.

      return
      end
c-----------------------------------------------------------------------
      subroutine hmhzsf (name,u1,u2,u3,r1,r2,r3,h1,h2,
     $                   rmask1,rmask2,rmask3,rmult,
     $                   tol,maxit,matmod)

c     Solve coupled Helmholtz equations (stress formulation)


      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'   ! For outpost diagnostic call
      include 'TSTEP'
      include 'ORTHOSTRS'
      include 'CTIMER'

      real u1(1),u2(1),u3(1),r1(1),r2(1),r3(1),h1(1),h2(1)
      real rmask1(1),rmask2(1),rmask3(1),rmult(1)
      character name*4

#ifdef TIMER
      nhmhz = nhmhz + 1
      etime1 = dnekclock()
#endif

      nel = nelfld(ifield)
      vol = volfld(ifield)
      n   = lx1*ly1*lz1*nel

      napproxstrs(1) = 0
      iproj = 0
      if (ifprojfld(ifield)) iproj = param(94)
      if (iproj.gt.0.and.istep.ge.iproj) napproxstrs(1)=param(93)
      napproxstrs(1)=min(napproxstrs(1),mxprev)

      call rmask   (r1,r2,r3,nel)
      call opdssum (r1,r2,r3)
      call rzero3  (u1,u2,u3,n)

      if (imesh.eq.1) then
         call chktcgs (r1,r2,r3,rmask1,rmask2,rmask3,rmult,binvm1
     $                ,vol,tol,nel)

         call strs_project_a(r1,r2,r3,h1,h2,rmult,ifield,ierr,matmod)

         call cggosf  (u1,u2,u3,r1,r2,r3,h1,h2,rmult,binvm1
     $                ,vol,tol,maxit,matmod)

         call strs_project_b(u1,u2,u3,h1,h2,rmult,ifield,ierr)

      else

         call chktcgs (r1,r2,r3,rmask1,rmask2,rmask3,rmult,bintm1
     $                ,vol,tol,nel)
         call cggosf  (u1,u2,u3,r1,r2,r3,h1,h2,rmult,bintm1
     $                ,vol,tol,maxit,matmod)

      endif

#ifdef TIMER
      thmhz=thmhz+(dnekclock()-etime1)
#endif

      return
      end

      subroutine chktcgs (r1,r2,r3,rmask1,rmask2,rmask3,rmult,binv,
     $                    vol,tol,nel)
C-------------------------------------------------------------------
C
C     Check that the tolerances are not too small for the CG-solver.
C     Important when calling the CG-solver (Gauss-Lobatto mesh) with
C     zero Neumann b.c.
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'EIGEN'
      common /cprint/ ifprint
      logical         ifprint
      common /ctmp0/ wa (lx1,ly1,lz1,lelt)
C
      dimension r1    (lx1,ly1,lz1,1)
     $        , r2    (lx1,ly1,lz1,1)
     $        , r3    (lx1,ly1,lz1,1)
     $        , rmask1(lx1,ly1,lz1,1)
     $        , rmask2(lx1,ly1,lz1,1)
     $        , rmask3(lx1,ly1,lz1,1)
     $        , rmult (lx1,ly1,lz1,1)
     $        , binv  (lx1,ly1,lz1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL
C
      IF (EIGAA .NE. 0.0) THEN
         ACONDNO = EIGGA/EIGAA
      ELSE
         ACONDNO = 10.0
      endif
C
C     Check Single or double precision
C
      DELTA = 1.0E-9
      X     = 1.0 + DELTA
      Y     = 1.0
      DIFF  = ABS(X - Y)
      IF (DIFF .EQ. 0.0) EPS = 1.0E-6
      IF (DIFF .GT. 0.0) EPS = 1.0E-13
C
      CALL OPDOT (WA,R1,R2,R3,R1,R2,R3,NTOT1)
      RINIT = GLSC3(WA,BINV,RMULT,NTOT1)
      RINIT = SQRT (RINIT/VOL)
      RMIN  = EPS*RINIT
C
      IF (TOL.LT.RMIN) THEN
       TOLOLD = TOL
       TOL = RMIN
c       IF (NIO.EQ.0 .AND. IFPRINT)
c     $ WRITE(6,*)'New CG1(stress)-tolerance (RINIT*epsm) = ',TOL,TOLOLD
      endif
C
      IF (ldim.EQ.2) THEN
         CALL ADD3 (WA,RMASK1,RMASK2,NTOT1)
      ELSE
         CALL ADD4 (WA,RMASK1,RMASK2,RMASK3,NTOT1)
      endif
      BCNEU1 = GLSC2 (WA,RMULT,NTOT1)
      BCNEU2 = (ldim) * GLSUM(RMULT,NTOT1)
      BCTEST = ABS(BCNEU1 - BCNEU2)
      IF (BCTEST .LT. 0.1) THEN
         IF (ldim.EQ.2) THEN
            CALL ADD3 (WA,R1,R2,NTOT1)
         ELSE
            CALL ADD4 (WA,R1,R2,R3,NTOT1)
         endif
         OTR    = GLSC2(WA,RMULT,NTOT1) / ( ldim )
         TOLMIN = ABS(OTR) * ACONDNO
         IF (TOL .LT. TOLMIN) THEN
            TOLOLD = TOL
            TOL = TOLMIN
c            IF (NIO.EQ.0)
c     $      WRITE (6,*) 'New CG1(stress)-tolerance (OTR) = ',TOL,TOLOLD
         endif
      endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine axhmsf (au1,au2,au3,u1,u2,u3,h1,h2,matmod)

C     Compute the coupled Helmholtz matrix-vector products

C     Fluid (MATMOD .GE. 0) :  Hij Uj = Aij*Uj + H2*B*Ui 
C     Solid (MATMOD .LT. 0) :  Hij Uj = Kij*Uj
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'MASS'
      include 'TSTEP'
      include 'CTIMER'

      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
C
      dimension au1(lx1,ly1,lz1,1)
     $        , au2(lx1,ly1,lz1,1)
     $        , au3(lx1,ly1,lz1,1)
     $        , u1 (lx1*ly1*lz1*1)
     $        , u2 (lx1*ly1*lz1*1)
     $        , u3 (lx1*ly1*lz1*1)
     $        , h1 (lx1,ly1,lz1,1)
     $        , h2 (lx1,ly1,lz1,1)

      naxhm = naxhm + 1
      etime1 = dnekclock()

      nel   = nelfld(ifield)
      ntot1 = lx1*ly1*lz1*nel

c      if (ifaxis.and.ifsplit) call exitti(
c     $'Axisymmetric stress w/PnPn not yet supported.$',istep)

c     icase = 1 --- axsf_fast (no axisymmetry)
c     icase = 2 --- stress formulation and supports axisymmetry
c     icase = 3 --- 3 separate axhelm calls

      icase = 1                ! Fast mode for stress
      if (ifaxis)      icase=2 ! Slow for stress, but supports axisymmetry
      if (matmod.lt.0) icase=2 ! Elasticity case
c     if (matmod.lt.0) icase=3 ! Block-diagonal Axhelm
      if (matmod.lt.0) icase=1 ! Elasticity case (faster, 7/28/17,pff)
      if (.not.ifstrs) icase=3 ! Block-diagonal Axhelm

      if (icase.eq.1) then

        call axsf_fast(au1,au2,au3,u1,u2,u3,h1,h2,ifield)

      elseif (icase.eq.3) then

        call axhelm(au1,u1,h1,h2,1,1)
        call axhelm(au2,u2,h1,h2,1,2)
        if (if3d) call axhelm(au3,u3,h1,h2,1,3)

      else  !  calculate coupled  Aij Uj  products

        if ( .not.ifsolv ) call setfast (h1,h2,imesh)

        call stnrate (u1,u2,u3,nel,matmod)
        call stress  (h1,h2,nel,matmod,ifaxis)
        call aijuj   (au1,au2,au3,nel,ifaxis)

        if (ifh2 .and. matmod.ge.0) then ! add Helmholtz contributions
           call addcol4 (au1,bm1,h2,u1,ntot1)
           call addcol4 (au2,bm1,h2,u2,ntot1)
           if (ldim.eq.3) call addcol4 (au3,bm1,h2,u3,ntot1)
        endif

      endif

      taxhm=taxhm+(dnekclock()-etime1)

      return
      end
c-----------------------------------------------------------------------
      subroutine stnrate (u1,u2,u3,nel,matmod)
C
C     Compute strainrates
C
C     CAUTION : Stresses and strainrates share the same scratch commons
C
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'TSTEP'

      common /ctmp0/ exz(lx1*ly1*lz1*lelt)
     $             , eyz(lx1*ly1*lz1*lelt)
      common /ctmp1/ exx(lx1*ly1*lz1*lelt)
     $             , exy(lx1*ly1*lz1*lelt)
     $             , eyy(lx1*ly1*lz1*lelt)
     $             , ezz(lx1*ly1*lz1*lelt)
c
      dimension u1(lx1,ly1,lz1,1)
     $        , u2(lx1,ly1,lz1,1)
     $        , u3(lx1,ly1,lz1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL

      CALL RZERO3 (EXX,EYY,EZZ,NTOT1)
      CALL RZERO3 (EXY,EXZ,EYZ,NTOT1)

      CALL UXYZ  (U1,EXX,EXY,EXZ,NEL)
      CALL UXYZ  (U2,EXY,EYY,EYZ,NEL)
      IF (ldim.EQ.3) CALL UXYZ   (U3,EXZ,EYZ,EZZ,NEL)

      CALL INVCOL2 (EXX,JACM1,NTOT1)
      CALL INVCOL2 (EXY,JACM1,NTOT1)
      CALL INVCOL2 (EYY,JACM1,NTOT1)
 
      IF (IFAXIS) CALL AXIEZZ (U2,EYY,EZZ,NEL)
C
      IF (ldim.EQ.3) THEN
         CALL INVCOL2 (EXZ,JACM1,NTOT1)
         CALL INVCOL2 (EYZ,JACM1,NTOT1)
         CALL INVCOL2 (EZZ,JACM1,NTOT1)
      endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine stress (h1,h2,nel,matmod,ifaxis)
C
C     MATMOD.GE.0        Fluid material models
C     MATMOD.LT.0        Solid material models
C
C     CAUTION : Stresses and strainrates share the same scratch commons
C
      include 'SIZE'
      common /ctmp1/ txx(lx1,ly1,lz1,lelt)
     $             , txy(lx1,ly1,lz1,lelt)
     $             , tyy(lx1,ly1,lz1,lelt)
     $             , tzz(lx1,ly1,lz1,lelt)
      common /ctmp0/ txz(lx1,ly1,lz1,lelt)
     $             , tyz(lx1,ly1,lz1,lelt)
      common /scrsf/ t11(lx1,ly1,lz1,lelt)
     $             , t22(lx1,ly1,lz1,lelt)
     $             , t33(lx1,ly1,lz1,lelt)
     $             , hii(lx1,ly1,lz1,lelt)
C
      DIMENSION H1(LX1,LY1,LZ1,1),H2(LX1,LY1,LZ1,1)
      LOGICAL IFAXIS

      NTOT1 = lx1*ly1*lz1*NEL

      IF (MATMOD.EQ.0) THEN

C        Newtonian fluids

         CONST = 2.0
         CALL CMULT2 (HII,H1,CONST,NTOT1)
         CALL COL2   (TXX,HII,NTOT1)
         CALL COL2   (TXY,H1 ,NTOT1)
         CALL COL2   (TYY,HII,NTOT1)
         IF (IFAXIS .OR. ldim.EQ.3) CALL COL2 (TZZ,HII,NTOT1)
         IF (ldim.EQ.3) THEN
            CALL COL2 (TXZ,H1 ,NTOT1)
            CALL COL2 (TYZ,H1 ,NTOT1)
         endif
C
      ELSEIF (MATMOD.EQ.-1) THEN
C
C        Elastic solids
C
         CONST = 2.0
         CALL ADD3S   (HII,H1,H2,CONST,NTOT1)
         CALL COPY    (T11,TXX,NTOT1)
         CALL COPY    (T22,TYY,NTOT1)
         CALL COL3    (TXX,HII,T11,NTOT1)
         CALL ADDCOL3 (TXX,H1 ,T22,NTOT1)
         CALL COL3    (TYY,H1 ,T11,NTOT1)
         CALL ADDCOL3 (TYY,HII,T22,NTOT1)
         CALL COL2    (TXY,H2     ,NTOT1)
         IF (IFAXIS .OR. ldim.EQ.3) THEN
            CALL COPY (T33,TZZ,NTOT1)
            CALL COL3    (TZZ,H1 ,T11,NTOT1)
            CALL ADDCOL3 (TZZ,H1 ,T22,NTOT1)
            CALL ADDCOL3 (TZZ,HII,T33,NTOT1)
            CALL ADDCOL3 (TXX,H1 ,T33,NTOT1)
            CALL ADDCOL3 (TYY,H1 ,T33,NTOT1)
         endif
         IF (ldim.EQ.3) THEN
            CALL COL2 (TXZ,H2     ,NTOT1)
            CALL COL2 (TYZ,H2     ,NTOT1)
         endif
C
      endif
C
      return
      end
c-----------------------------------------------------------------------
      subroutine aijuj (au1,au2,au3,nel,ifaxis)
C
      include 'SIZE'
      common /ctmp1/ txx(lx1,ly1,lz1,lelt)
     $             , txy(lx1,ly1,lz1,lelt)
     $             , tyy(lx1,ly1,lz1,lelt)
     $             , tzz(lx1,ly1,lz1,lelt)
      common /ctmp0/ txz(lx1,ly1,lz1,lelt)
     $             , tyz(lx1,ly1,lz1,lelt)
C
      DIMENSION AU1(LX1,LY1,LZ1,1)
     $        , AU2(LX1,LY1,LZ1,1)
     $        , AU3(LX1,LY1,LZ1,1)
      LOGICAL IFAXIS
C
      CALL TTXYZ (AU1,TXX,TXY,TXZ,NEL)
      CALL TTXYZ (AU2,TXY,TYY,TYZ,NEL)
      IF (IFAXIS)    CALL AXITZZ (AU2,TZZ,NEL)
      IF (ldim.EQ.3) CALL TTXYZ  (AU3,TXZ,TYZ,TZZ,NEL)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine uxyz (u,ex,ey,ez,nel)
C
      include 'SIZE'
      include 'GEOM'
      common /scrsf/ ur(lx1,ly1,lz1,lelt)
     $             , us(lx1,ly1,lz1,lelt)
     $             , ut(lx1,ly1,lz1,lelt)
c
      dimension u (lx1,ly1,lz1,1)
     $        , ex(lx1,ly1,lz1,1)
     $        , ey(lx1,ly1,lz1,1)
     $        , ez(lx1,ly1,lz1,1)
C
      NTOT1 = lx1*ly1*lz1*NEL
C
      CALL URST (U,UR,US,UT,NEL)
C
      CALL ADDCOL3 (EX,RXM1,UR,NTOT1)
      CALL ADDCOL3 (EX,SXM1,US,NTOT1)
      CALL ADDCOL3 (EY,RYM1,UR,NTOT1)
      CALL ADDCOL3 (EY,SYM1,US,NTOT1)
C
      IF (ldim.EQ.3) THEN
         CALL ADDCOL3 (EZ,RZM1,UR,NTOT1)
         CALL ADDCOL3 (EZ,SZM1,US,NTOT1)
         CALL ADDCOL3 (EZ,TZM1,UT,NTOT1)
         CALL ADDCOL3 (EX,TXM1,UT,NTOT1)
         CALL ADDCOL3 (EY,TYM1,UT,NTOT1)
      endif
C
      return
      end
      subroutine urst (u,ur,us,ut,nel)
C
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
C
      DIMENSION U (LX1,LY1,LZ1,1)
     $        , UR(LX1,LY1,LZ1,1)
     $        , US(LX1,LY1,LZ1,1)
     $        , UT(LX1,LY1,LZ1,1)
C
      DO 100 IEL=1,NEL
         IF (IFAXIS) CALL SETAXDY ( IFRZER(IEL) )
         CALL DDRST (U (1,1,1,IEL),UR(1,1,1,IEL),
     $               US(1,1,1,IEL),UT(1,1,1,IEL))
  100 CONTINUE
C
      return
      end
      subroutine ddrst (u,ur,us,ut)
C
      include 'SIZE'
      include 'DXYZ'
C
      DIMENSION U (LX1,LY1,LZ1)
     $        , UR(LX1,LY1,LZ1)
     $        , US(LX1,LY1,LZ1)
     $        , UT(LX1,LY1,LZ1)
C
      NXY1 = lx1*ly1
      NYZ1 = ly1*lz1
C
      CALL MXM (DXM1,lx1,U,lx1,UR,NYZ1)
      IF (ldim.EQ.2) THEN
         CALL MXM (U,lx1,DYTM1,ly1,US,ly1)
      ELSE
         DO 10 IZ=1,lz1
         CALL MXM (U(1,1,IZ),lx1,DYTM1,ly1,US(1,1,IZ),ly1)
   10    CONTINUE
         CALL MXM (U,NXY1,DZTM1,lz1,UT,lz1)
      endif
C
      return
      end
      subroutine axiezz (u2,eyy,ezz,nel)
C
      include 'SIZE'
      include 'GEOM'
C
      DIMENSION U2 (LX1,LY1,LZ1,1)
     $        , EYY(LX1,LY1,LZ1,1)
     $        , EZZ(LX1,LY1,LZ1,1)
C
      NXYZ1  = lx1*ly1*lz1
C
      DO 100 IEL=1,NEL
         IF ( IFRZER(IEL) ) THEN
            DO 200 IX=1,lx1
               EZZ(IX, 1,1,IEL) = EYY(IX,1,1,IEL)
            DO 200 IY=2,ly1
               EZZ(IX,IY,1,IEL) = U2(IX,IY,1,IEL) / YM1(IX,IY,1,IEL)
  200       CONTINUE
         ELSE
            CALL INVCOL3 (EZZ(1,1,1,IEL),U2(1,1,1,IEL),YM1(1,1,1,IEL),
     $                    NXYZ1)
         endif
  100 CONTINUE
C
      return
      end
c-----------------------------------------------------------------------
      subroutine flush_io
      return
      end
c-----------------------------------------------------------------------
      subroutine fcsum2(xsum,asum,x,e,f)
c
c     Compute the weighted sum of X over face f of element e
c
c     x is an (NX,NY,NZ) data structure
c     f  is in the preprocessor notation 
c
c     xsum is sum (X*area)
c     asum is sum (area)


      include 'SIZE'
      include 'GEOM'
      include 'TOPOL'
      real x(lx1,ly1,lz1,1)
      integer e,f,fd

      asum = 0.
      xsum = 0.

c     Set up counters ;  fd is the dssum notation.
      call dsset(lx1,ly1,lz1)
      fd     = eface1(f)
      js1    = skpdat(1,fd)
      jf1    = skpdat(2,fd)
      jskip1 = skpdat(3,fd)
      js2    = skpdat(4,fd)
      jf2    = skpdat(5,fd)
      jskip2 = skpdat(6,fd)

      i = 0
      do j2=js2,jf2,jskip2
      do j1=js1,jf1,jskip1
         i = i+1
         xsum = xsum+area(i,1,f,e)*x(j1,j2,1,e)
         asum = asum+area(i,1,f,e)
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      function surf_mean(u,ifld,bc_in,ierr)

      include 'SIZE'
      include 'TOTAL'

      real u(1)

      integer e,f
      character*3 bc_in

      usum = 0
      asum = 0

      nface = 2*ldim
      do e=1,nelfld(ifld)
      do f=1,nface
         if (cbc(f,e,ifld).eq.bc_in) then
            call fcsum2(usum_f,asum_f,u,e,f)
            usum = usum + usum_f
            asum = asum + asum_f
         endif
      enddo
      enddo

      usum = glsum(usum,1)  ! sum across processors
      asum = glsum(asum,1)

      surf_mean = usum
      ierr      = 1

      if (asum.gt.0) surf_mean = usum/asum
      if (asum.gt.0) ierr      = 0

      return
      end
c-----------------------------------------------------------------------
      subroutine fdm_h1a(z,r,d,nel,kt,rr)
      include 'SIZE'
      include 'TOTAL'
c
      common /ctmp0/ w(lx1,ly1,lz1)
c
      include 'FDMH1'
c
c     Overlapping Schwarz, FDM based
c
      real z(lx1,ly1,lz1,1)
      real r(lx1,ly1,lz1,1)
      real d(lx1,ly1,lz1,1)
      real rr(lx1,ly1,lz1,1)

      integer kt(lelt,3)

      integer icalld
      save    icalld
      data    icalld /0/

      n1 = lx1
      n2 = lx1*lx1
      n3 = lx1*lx1*lx1
      n  = lx1*ly1*lz1*nel

      if (ifbhalf) then
         call col3(rr,r,bhalf,n)
      else
         call copy(rr,r,n)
      endif
      icalld = icalld+1
c
c
      do ie=1,nel
         if (if3d) then
c           Transfer to wave space:  
            call mxm(fdst(1,kt(ie,1)),n1,rr(1,1,1,ie),n1,w,n2)
            do iz=1,n1
              call mxm(w(1,1,iz),n1,fds (1,kt(ie,2)),n1,z(1,1,iz,ie),n1)
            enddo
            call mxm(z(1,1,1,ie),n2,fds (1,kt(ie,3)),n1,w,n1)
c
c           fdsolve:
c
            call col2(w,d(1,1,1,ie),n3)
c
c           Transfer to physical space:  
c
            call mxm(w,n2,fdst(1,kt(ie,3)),n1,z(1,1,1,ie),n1)
            do iz=1,n1
              call mxm(z(1,1,iz,ie),n1,fdst(1,kt(ie,2)),n1,w(1,1,iz),n1)
            enddo
            call mxm(fds (1,kt(ie,1)),n1,w,n1,z(1,1,1,ie),n2)
c
         else
c           Transfer to wave space:  
            call mxm(fdst(1,kt(ie,1)),n1,rr(1,1,1,ie),n1,w,n1)
            call mxm(w,n1,fds (1,kt(ie,2)),n1,z(1,1,1,ie),n1)
c
c           fdsolve:
c
            call col2(z(1,1,1,ie),d(1,1,1,ie),n2)
c
c           Transfer to physical space:  
c
            call mxm(z(1,1,1,ie),n1,fdst(1,kt(ie,2)),n1,w,n1)
            call mxm(fds (1,kt(ie,1)),n1,w,n1,z(1,1,1,ie),n1)
c
         endif
      enddo

      if (ifbhalf) call col2(z,bhalf,n)

      call dssum(z,lx1,ly1,lz1)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_vert_strs(glo_num,ngv,nx,nel,vertex,ifcenter)

c     Given global array, vertex, pointing to hex vertices, set up
c     a new array of global pointers for an nx^ldim set of elements.

      include 'SIZE'
      include 'INPUT'

      integer*8 glo_num(1),ngv
      integer vertex(1),nx
      logical ifcenter

      common /ctmp0/ gnum(lx1*ly1*lz1*lelt)
      integer*8 gnum

      call set_vert(gnum,ngv,nx,nel,vertex,ifcenter)

      n=nel*nx*nx
      if (if3d) n=n*nx

c     Pack vertex ids component-wise (u,v,w_1; u,v,w_2; etc.)

      k=0
      do i=1,n

         do j=1,ldim
            glo_num(k+j) = ldim*(gnum(i)-1) + j
         enddo
         k=k+ldim

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine get_strs_mask(mask,nxc,nzc,nel)
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'

      real mask(ldim,nxc,nxc,nzc,nel)
      integer e


      if (if3d) then
         do e=1,nel
         do kc=1,nxc
         do jc=1,nxc
         do ic=1,nxc
            k = 1 + ((lx1-1)*(kc-1))/(nxc-1)
            j = 1 + ((lx1-1)*(jc-1))/(nxc-1)
            i = 1 + ((lx1-1)*(ic-1))/(nxc-1)
            mask(1,ic,jc,kc,e) = v1mask(i,j,k,e)
            mask(2,ic,jc,kc,e) = v2mask(i,j,k,e)
            mask(3,ic,jc,kc,e) = v3mask(i,j,k,e)
         enddo
         enddo
         enddo
         enddo
      else
         do e=1,nel
         do jc=1,nxc
         do ic=1,nxc
            j = 1 + ((lx1-1)*(jc-1))/(nxc-1)
            i = 1 + ((lx1-1)*(ic-1))/(nxc-1)
            mask(1,ic,jc,1,e) = v1mask(i,j,1,e)
            mask(2,ic,jc,1,e) = v2mask(i,j,1,e)
         enddo
         enddo
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine axstrs(a1,a2,a3,p1,p2,p3,h1,h2,matmod,nel)
      real a1(1),a2(1),a3(1),p1(1),p2(1),p3(1)

      call axhmsf  (a1,a2,a3,p1,p2,p3,h1,h2,matmod)
      call rmask   (a1,a2,a3,nel)
      call opdssum (a1,a2,a3)

      return
      end
c-----------------------------------------------------------------------
      subroutine axstrs_nds(a1,a2,a3,p1,p2,p3,h1,h2,matmod,nel)
      real a1(1),a2(1),a3(1),p1(1),p2(1),p3(1)

      call rmask   (p1,p2,p3,nel)
      call axhmsf  (a1,a2,a3,p1,p2,p3,h1,h2,matmod)
      call rmask   (a1,a2,a3,nel)
c     call opdssum (a1,a2,a3)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_local_crs_galerkin_strs(a,ncl,nxc,h1,h2,matmod)

c     This routine generates Nelv submatrices of order ncl using
c     Galerkin projection

      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'


      real    a(ldim,ncl,ldim,ncl,1),h1(1),h2(1)
c     real    a(ncl,ldim,ncl,ldim,1),h1(1),h2(1)

      common /scrcr2/ a1(lx1*ly1*lz1,lelt),w1(lx1*ly1*lz1,lelt)
     $              , a2(lx1*ly1*lz1,lelt),w2(lx1*ly1*lz1,lelt)
      common /scrcr3/ a3(lx1*ly1*lz1,lelt),w3(lx1*ly1*lz1,lelt)
     $              , b (lx1*ly1*lz1,8)

      integer e

      nel = nelfld(ifield)

      do j=1,ncl ! bi- or tri-linear interpolant, ONLY
         call gen_crs_basis(b(1,j),j)
      enddo

      nxyz = lx1*ly1*lz1
      do k = 1,ldim
       call opzero(w1,w2,w3)
       call opzero(a1,a2,a3)

       do j = 1,ncl

         do e = 1,nel
            if (k.eq.1) call copy(w1(1,e),b(1,j),nxyz)
            if (k.eq.2) call copy(w2(1,e),b(1,j),nxyz)
            if (k.eq.3) call copy(w3(1,e),b(1,j),nxyz)
         enddo

         call axstrs_nds(a1,a2,a3,w1,w2,w3,h1,h2,matmod,nel)

         do e = 1,nel
         do i = 1,ncl

            a(1,i,k,j,e)=vlsc2(b(1,i),a1(1,e),nxyz)  ! bi^T * A^e * bj
            a(2,i,k,j,e)=vlsc2(b(1,i),a2(1,e),nxyz)  ! bi^T * A^e * bj
            if (if3d)
     $      a(3,i,k,j,e)=vlsc2(b(1,i),a3(1,e),nxyz)  ! bi^T * A^e * bj

         enddo
         enddo

       enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine crs_strs(u1,u2,u3,v1,v2,v3)
c     Given an input vector v, this generates the H1 coarse-grid solution
      include 'SIZE'
      include 'DOMAIN'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'PARALLEL'
      include 'CTIMER'
      include 'TSTEP'

      real u1(1),u2(1),u3(1),v1(1),v2(1),v3(1)

      common /scrpr3/ uc1(lcr*lelt),uc2(lcr*lelt),uc3(lcr*lelt)
      common /scrpr2/ vc1(lcr*lelt),vc2(lcr*lelt),vc3(lcr*lelt)

      integer icalld1
      save    icalld1
      data    icalld1 /0/

      
      if (icalld1.eq.0) then ! timer info
         ncrsl=0
         tcrsl=0.0
         icalld1=1
      endif
      ncrsl  = ncrsl  + 1

      n = nelv*lx1*ly1*lz1
      m = nelv*lcr

      call map_f_to_c_h1_bilin(uc1,v1)   ! additive Schwarz
      call map_f_to_c_h1_bilin(uc2,v2)   ! additive Schwarz
      if (if3d) call map_f_to_c_h1_bilin(uc3,v3)   ! additive Schwarz

      k=0
      if (if3d) then
         do i=1,m
            vc1(k+1)=uc1(i)
            vc1(k+2)=uc2(i)
            vc1(k+3)=uc3(i)
            k=k+3
         enddo
      else
         do i=1,m
            vc1(k+1)=uc1(i)
            vc1(k+2)=uc2(i)
            k=k+2
         enddo
      endif

      etime1=dnekclock()
      call fgslib_crs_solve(xxth_strs,uc1,vc1)
      tcrsl=tcrsl+dnekclock()-etime1

      k=0
      if (if3d) then
         do i=1,m
            vc1(i)=uc1(k+1)
            vc2(i)=uc1(k+2)
            vc3(i)=uc1(k+3)
            k=k+3
         enddo
      else
         do i=1,m
            vc1(i)=uc1(k+1)
            vc2(i)=uc1(k+2)
            k=k+2
         enddo
      endif
      call map_c_to_f_h1_bilin(u1,vc1)
      call map_c_to_f_h1_bilin(u2,vc2)
      if (if3d) call map_c_to_f_h1_bilin(u3,vc3)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_up_h1_crs_strs(h1,h2,ifld,matmod)

      include 'SIZE'
      include 'GEOM'
      include 'DOMAIN'
      include 'INPUT'
      include 'PARALLEL'
      include 'TSTEP'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      common /ivrtx/ vertex ((2**ldim)*lelt)
      integer vertex

      integer null_space,e

      character*3 cb
      common /scrxxti/ ia(ldim*ldim*lcr*lcr*lelv)
     $               , ja(ldim*ldim*lcr*lcr*lelv)

      parameter (lcc=2**ldim)
      common /scrcr1/ a(ldim*ldim*lcc*lcc*lelt)
      real mask(ldim,lcr,lelv)
      equivalence (mask,a)

      integer*8 ngv

      t0 = dnekclock()

      nel = nelfld(ifld)

      nxc = 2
      nzc = 1
      if (if3d) nzc=nxc

      ncr = nxc**ldim
      mcr = ldim*ncr  ! Dimension of unassembled sub-block
      n   = mcr*nel

c     Set SEM_to_GLOB index
      call get_vertex
      call set_vert_strs(se_to_gcrs,ngv,nxc,nel,vertex,.true.)


c     Set mask for full array
      call get_strs_mask   (mask,nxc,nzc,nel) ! Set mask
      call set_jl_crs_mask (n,mask,se_to_gcrs)


c     Setup local SEM-based Neumann operators (for now, just full...)
      call get_local_crs_galerkin_strs(a,ncr,nxc,h1,h2,matmod)
      call set_mat_ij(ia,ja,mcr,nel)
      nnz=mcr*mcr*nel

      null_space=0

      t1 = dnekclock()-t0
      if (nio.eq.0) 
     $  write(6,*) 'start:: setup h1 coarse grid ',t1, ' sec'
     $             ,nnz,mcr,ncr,nel

      do i=1,nnz
         write(44,44) ia(i),ja(i),a(i)
      enddo
   44 format(2i9,1pe22.13)
c     stop

      imode = param(40)
      call fgslib_crs_setup(xxth_strs,imode,nekcomm,mp,n,se_to_gcrs,
     $                      nnz,ia,ja,a,null_space)

      t0 = dnekclock()-t0
      if (nio.eq.0) then
         write(6,*) 'done :: setup h1 coarse grid ',t0, ' sec',xxth_strs
         write(6,*) ' '
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine axsf_e_3d(au,av,aw,u,v,w,h1,h2,ur,e)
c
c                                         du_i
c     Compute the gradient tensor G_ij := ----  ,  for element e
c                                         du_j
c
      include 'SIZE'
      include 'TOTAL'

      real au(1),av(1),aw(1),u(1),v(1),w(1),h1(1),h2(1)

      parameter (l=lx1*ly1*lz1)
      real ur(l,ldim,ldim)

      integer e,p

      p = lx1-1      ! Polynomial degree
      n = lx1*ly1*lz1


      call local_grad3(ur(1,1,1),ur(1,2,1),ur(1,3,1),u,p,1,dxm1,dxtm1)
      call local_grad3(ur(1,1,2),ur(1,2,2),ur(1,3,2),v,p,1,dxm1,dxtm1)
      call local_grad3(ur(1,1,3),ur(1,2,3),ur(1,3,3),w,p,1,dxm1,dxtm1)

      do i=1,n

         u1 = ur(i,1,1)*rxm1(i,1,1,e) + ur(i,2,1)*sxm1(i,1,1,e)
     $                                + ur(i,3,1)*txm1(i,1,1,e)
         u2 = ur(i,1,1)*rym1(i,1,1,e) + ur(i,2,1)*sym1(i,1,1,e)
     $                                + ur(i,3,1)*tym1(i,1,1,e)
         u3 = ur(i,1,1)*rzm1(i,1,1,e) + ur(i,2,1)*szm1(i,1,1,e)
     $                                + ur(i,3,1)*tzm1(i,1,1,e)

         v1 = ur(i,1,2)*rxm1(i,1,1,e) + ur(i,2,2)*sxm1(i,1,1,e)
     $                                + ur(i,3,2)*txm1(i,1,1,e)
         v2 = ur(i,1,2)*rym1(i,1,1,e) + ur(i,2,2)*sym1(i,1,1,e)
     $                                + ur(i,3,2)*tym1(i,1,1,e)
         v3 = ur(i,1,2)*rzm1(i,1,1,e) + ur(i,2,2)*szm1(i,1,1,e)
     $                                + ur(i,3,2)*tzm1(i,1,1,e)

         w1 = ur(i,1,3)*rxm1(i,1,1,e) + ur(i,2,3)*sxm1(i,1,1,e)
     $                                + ur(i,3,3)*txm1(i,1,1,e)
         w2 = ur(i,1,3)*rym1(i,1,1,e) + ur(i,2,3)*sym1(i,1,1,e)
     $                                + ur(i,3,3)*tym1(i,1,1,e)
         w3 = ur(i,1,3)*rzm1(i,1,1,e) + ur(i,2,3)*szm1(i,1,1,e)
     $                                + ur(i,3,3)*tzm1(i,1,1,e)

         dj  = h1(i)*w3m1(i,1,1)*jacmi(i,e)
         s11 = dj*(u1 + u1)! S_ij
         s12 = dj*(u2 + v1)
         s13 = dj*(u3 + w1)
         s21 = dj*(v1 + u2)
         s22 = dj*(v2 + v2)
         s23 = dj*(v3 + w2)
         s31 = dj*(w1 + u3)
         s32 = dj*(w2 + v3)
         s33 = dj*(w3 + w3)

c        Sum_j : (r_p/x_j) h1 J S_ij

         ur(i,1,1)=rxm1(i,1,1,e)*s11+rym1(i,1,1,e)*s12+rzm1(i,1,1,e)*s13
         ur(i,2,1)=sxm1(i,1,1,e)*s11+sym1(i,1,1,e)*s12+szm1(i,1,1,e)*s13
         ur(i,3,1)=txm1(i,1,1,e)*s11+tym1(i,1,1,e)*s12+tzm1(i,1,1,e)*s13

         ur(i,1,2)=rxm1(i,1,1,e)*s21+rym1(i,1,1,e)*s22+rzm1(i,1,1,e)*s23
         ur(i,2,2)=sxm1(i,1,1,e)*s21+sym1(i,1,1,e)*s22+szm1(i,1,1,e)*s23
         ur(i,3,2)=txm1(i,1,1,e)*s21+tym1(i,1,1,e)*s22+tzm1(i,1,1,e)*s23

         ur(i,1,3)=rxm1(i,1,1,e)*s31+rym1(i,1,1,e)*s32+rzm1(i,1,1,e)*s33
         ur(i,2,3)=sxm1(i,1,1,e)*s31+sym1(i,1,1,e)*s32+szm1(i,1,1,e)*s33
         ur(i,3,3)=txm1(i,1,1,e)*s31+tym1(i,1,1,e)*s32+tzm1(i,1,1,e)*s33

      enddo
      call local_grad3_t
     $    (au,ur(1,1,1),ur(1,2,1),ur(1,3,1),p,1,dxm1,dxtm1,av)
      call local_grad3_t
     $    (av,ur(1,1,2),ur(1,2,2),ur(1,3,2),p,1,dxm1,dxtm1,ur)
      call local_grad3_t
     $    (aw,ur(1,1,3),ur(1,2,3),ur(1,3,3),p,1,dxm1,dxtm1,ur)

      do i=1,n
         au(i)=au(i) + h2(i)*bm1(i,1,1,e)*u(i)
         av(i)=av(i) + h2(i)*bm1(i,1,1,e)*v(i)
         aw(i)=aw(i) + h2(i)*bm1(i,1,1,e)*w(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine axsf_e_2d(au,av,u,v,h1,h2,ur,e)
c
c                                         du_i
c     Compute the gradient tensor G_ij := ----  ,  for element e
c                                         du_j
c
      include 'SIZE'
      include 'TOTAL'

      real au(1),av(1),u(1),v(1),h1(1),h2(1)

      parameter (l=lx1*ly1*lz1)
      real ur(l,ldim,ldim)

      integer e,p

      p = lx1-1      ! Polynomial degree
      n = lx1*ly1*lz1


      call local_grad2(ur(1,1,1),ur(1,2,1),u,p,1,dxm1,dxtm1)
      call local_grad2(ur(1,1,2),ur(1,2,2),v,p,1,dxm1,dxtm1)

      do i=1,n
         dj = h1(i)*w3m1(i,1,1)*jacmi(i,e)

         u1 = ur(i,1,1)*rxm1(i,1,1,e) + ur(i,2,1)*sxm1(i,1,1,e) !ux
         u2 = ur(i,1,1)*rym1(i,1,1,e) + ur(i,2,1)*sym1(i,1,1,e) !uy
         v1 = ur(i,1,2)*rxm1(i,1,1,e) + ur(i,2,2)*sxm1(i,1,1,e) !vx
         v2 = ur(i,1,2)*rym1(i,1,1,e) + ur(i,2,2)*sym1(i,1,1,e) !vy

         s11 = dj*( u1 + u1 ) ! h1*rho*S_ij
         s12 = dj*( u2 + v1 )
         s21 = dj*( v1 + u2 )
         s22 = dj*( v2 + v2 )

c        Sum_j : (r_k/x_j) h1 J S_ij

         ur(i,1,1)=rxm1(i,1,1,e)*s11+rym1(i,1,1,e)*s12 ! i=1,k=1
         ur(i,2,1)=sxm1(i,1,1,e)*s11+sym1(i,1,1,e)*s12 ! i=1,k=2

         ur(i,1,2)=rxm1(i,1,1,e)*s21+rym1(i,1,1,e)*s22 ! i=2,k=1
         ur(i,2,2)=sxm1(i,1,1,e)*s21+sym1(i,1,1,e)*s22 ! i=2,k=2

      enddo

      call local_grad2_t(au,ur(1,1,1),ur(1,2,1),p,1,dxm1,dxtm1,av)
      call local_grad2_t(av,ur(1,1,2),ur(1,2,2),p,1,dxm1,dxtm1,ur)

      do i=1,n
         au(i)=au(i) + h2(i)*bm1(i,1,1,e)*u(i)
         av(i)=av(i) + h2(i)*bm1(i,1,1,e)*v(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine axsf_fast(au,av,aw,u,v,w,h1,h2,ifld)
      include 'SIZE'
      include 'TOTAL'

      parameter (l=lx1*ly1*lz1)
      real au(l,1),av(l,1),aw(l,1),u(l,1),v(l,1),w(l,1),h1(l,1),h2(l,1)

      common /btmp0/ ur(l,ldim,ldim)

      integer e

      nel = nelfld(ifld)

      if (if3d) then
        do e=1,nel
          call axsf_e_3d(au(1,e),av(1,e),aw(1,e),u(1,e),v(1,e),w(1,e)
     $                                         ,h1(1,e),h2(1,e),ur,e)
        enddo
      else
        do e=1,nel
          call axsf_e_2d(au(1,e),av(1,e),u(1,e),v(1,e)
     $                          ,h1(1,e),h2(1,e),ur,e)
        enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine ttxyz (ff,tx,ty,tz,nel)
C
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'WZ'

      DIMENSION TX(LX1,LY1,LZ1,1)
     $        , TY(LX1,LY1,LZ1,1)
     $        , TZ(LX1,LY1,LZ1,1)
     $        , FF(LX1*LY1*LZ1,1)

      common /scrsf/ fr(lx1*ly1*lz1,lelt)
     $             , fs(lx1*ly1*lz1,lelt)
     $             , ft(lx1*ly1*lz1,lelt)
      real           wa(lx1,ly1,lz1,lelt)
      equivalence   (wa,ft)
      real ys(lx1)

      NXYZ1 = lx1*ly1*lz1
      NTOT1 = NXYZ1*NEL

      CALL COL3    (FR,RXM1,TX,NTOT1)
      CALL ADDCOL3 (FR,RYM1,TY,NTOT1)
      CALL COL3    (FS,SXM1,TX,NTOT1)
      CALL ADDCOL3 (FS,SYM1,TY,NTOT1)

      IF (ldim.EQ.3) THEN
         CALL ADDCOL3 (FR,RZM1,TZ,NTOT1)
         CALL ADDCOL3 (FS,SZM1,TZ,NTOT1)
         CALL COL3    (FT,TXM1,TX,NTOT1)
         CALL ADDCOL3 (FT,TYM1,TY,NTOT1)
         CALL ADDCOL3 (FT,TZM1,TZ,NTOT1)
      endif
C
      IF (IFAXIS) THEN
         DO 100 IEL=1,NEL
         IF ( IFRZER(IEL) ) THEN
            CALL MXM (YM1(1,1,1,IEL),lx1,DATM1,ly1,YS,1)
            DO 120 IX=1,lx1
               IY = 1
               WA(IX,IY,1,IEL)=YS(IX)*W2AM1(IX,IY)
            DO 120 IY=2,ly1
               DNR = 1.0 + ZAM1(IY)
               WA(IX,IY,1,IEL)=YM1(IX,IY,1,IEL)*W2AM1(IX,IY)/DNR
  120       CONTINUE
         ELSE
            CALL COL3 (WA(1,1,1,IEL),YM1(1,1,1,IEL),W2CM1,NXYZ1)
         endif
  100    CONTINUE
         CALL COL2 (FR,WA,NTOT1)      
         CALL COL2 (FS,WA,NTOT1)      
      else
         do 180 iel=1,nel
            call col2(fr(1,iel),w3m1,nxyz1)
            call col2(fs(1,iel),w3m1,nxyz1)
            call col2(ft(1,iel),w3m1,nxyz1)
  180    continue
      endif


      DO 200 IEL=1,NEL
         IF (IFAXIS) CALL SETAXDY ( IFRZER(IEL) )
         CALL TTRST (FF(1,IEL),FR(1,IEL),FS(1,IEL),
     $               FT(1,IEL),FR(1,IEL)) ! FR work array
  200 CONTINUE
C
      return
      end
c-----------------------------------------------------------------------
      subroutine ttrst (ff,fr,fs,ft,ta)

      include 'SIZE'
      include 'DXYZ'
      include 'TSTEP'

      DIMENSION FF(LX1,LY1,LZ1)
     $        , FR(LX1,LY1,LZ1)
     $        , FS(LX1,LY1,LZ1)
     $        , FT(LX1,LY1,LZ1)
     $        , TA(LX1,LY1,LZ1)

      NXY1  = lx1*ly1
      NYZ1  = ly1*lz1
      NXYZ1 = NXY1*lz1

      CALL MXM (DXTM1,lx1,FR,lx1,FF,NYZ1)
      IF (ldim.EQ.2) THEN
         CALL MXM  (FS,lx1,DYM1,ly1,TA,ly1)
         CALL ADD2 (FF,TA,NXYZ1)
      ELSE
         DO 10 IZ=1,lz1
         CALL MXM  (FS(1,1,IZ),lx1,DYM1,ly1,TA(1,1,IZ),ly1)
   10    CONTINUE
         CALL ADD2 (FF,TA,NXYZ1)
         CALL MXM  (FT,NXY1,DZM1,lz1,TA,lz1)
         CALL ADD2 (FF,TA,NXYZ1)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine axitzz (vfy,tzz,nel)
C
      include 'SIZE'
      include 'DXYZ'
      include 'GEOM'
      include 'WZ'
      common /ctmp0/ phi(lx1,ly1)
C
      DIMENSION VFY(LX1,LY1,LZ1,1)
     $        , TZZ(LX1,LY1,LZ1,1)
C
      NXYZ1 = lx1*ly1*lz1
C
      DO 100 IEL=1,NEL
         CALL SETAXW1 ( IFRZER(IEL) )
         CALL COL4 (PHI,TZZ(1,1,1,IEL),JACM1(1,1,1,IEL),W3M1,NXYZ1)
         IF ( IFRZER(IEL) ) THEN
            DO 120 IX=1,lx1
            DO 120 IY=2,ly1
               DNR = PHI(IX,IY)/( 1.0 + ZAM1(IY) )
               DDS = WXM1(IX) * WAM1(1) * DATM1(IY,1) *
     $               JACM1(IX,1,1,IEL) * TZZ(IX,1,1,IEL)
               VFY(IX,IY,1,IEL)=VFY(IX,IY,1,IEL) + DNR + DDS
  120       CONTINUE
         ELSE
            CALL ADD2 (VFY(1,1,1,IEL),PHI,NXYZ1)
         endif
  100 CONTINUE
C
      return
      end
c-----------------------------------------------------------------------
      subroutine setaxdy (ifaxdy)
C
      include 'SIZE'
      include 'DXYZ'
C
      LOGICAL IFAXDY
C
      IF (IFAXDY) THEN
         CALL COPY (DYM1 ,DAM1 ,ly1*ly1)
         CALL COPY (DYTM1,DATM1,ly1*ly1)
      ELSE
         CALL COPY (DYM1 ,DCM1 ,ly1*ly1)
         CALL COPY (DYTM1,DCTM1,ly1*ly1)
      endif
C
      return
      end
c-----------------------------------------------------------------------
      function opnorm2w(v1,v2,v3,w)
      include 'SIZE'
      include 'TOTAL'
c
      real v1(1) , v2(1), v3(1), w(1)

      n=lx1*ly1*lz1*nelv
      s=vlsc3(v1,w,v1,n)+vlsc3(v2,w,v2,n)
      if(if3d) s=s+vlsc3(v3,w,v3,n)
      s=glsum(s,1)
      
      if (s.gt.0) s=sqrt(s/volvm1)
      opnorm2w = s

      return
      end
c-----------------------------------------------------------------------
      subroutine strs_project_a(b1,b2,b3,h1,h2,wt,ifld,ierr,matmod)

c     Assumes if uservp is true and thus reorthogonalizes every step

      include 'SIZE'
      include 'TOTAL'
      include 'ORTHOSTRS'  ! Storage of approximation space
      include 'CTIMER'

      real b1(1),b2(1),b3(1),h1(1),h2(1),wt(1)
      common /ctmp1/ w(lx1*ly1*lz1*lelt,ldim)
      real l2a,l2b

      kmax = napproxstrs(1)
      k    = napproxstrs(2)
      n    = lx1*ly1*lz1*nelv
      m    = n*ldim

      if (k.eq.0.or.kmax.eq.0) return

      etime0 = dnekclock()

      l2b=opnorm2w(b1,b2,b3,binvm1)

c     Reorthogonalize basis

      dh1max=difmax(bstrs(1  ),h1,n)
      call col3(bstrs,h2,bm1,n)
      dh2max=difmax(bstrs(1+n),bstrs,n)

      if (dh1max.gt.0.or.dh2max.gt.0) then
        call strs_ortho_all(xstrs(1+m),bstrs(1+m),n,k,h1,h2,wt,ifld,w
     $                   ,ierr,matmod)
      else
        call strs_ortho_one(xstrs(1+m),bstrs(1+m),n,k,h1,h2,wt,ifld,w
     $                   ,ierr,matmod)
      endif

      napproxstrs(2) = k

      call opcopy(bstrs(1),bstrs(1+n),bstrs(1+2*n),b1,b2,b3)
      call opzero(xstrs(1),xstrs(1+n),xstrs(1+2*n))

      do i=1,k
         i1 = 1 + 0*n + (i-1)*m + m
         i2 = 1 + 1*n + (i-1)*m + m
         i3 = 1 + 2*n + (i-1)*m + m
         alpha=op_glsc2_wt(bstrs(1),bstrs(1+n),bstrs(1+2*n)
     $                    ,xstrs(i1),xstrs(i2),xstrs(i3),wt)

         alphm=-alpha
         call opadds(bstrs(1),bstrs(1+n),bstrs(1+2*n)
     $              ,bstrs(i1),bstrs(i2),bstrs(i3),alphm,n,2)

         call opadds(xstrs(1),xstrs(1+n),xstrs(1+2*n)
     $              ,xstrs(i1),xstrs(i2),xstrs(i3),alpha,n,2)


      enddo

      call opcopy(b1,b2,b3,bstrs(1),bstrs(1+n),bstrs(1+2*n))
      l2a=opnorm2w(b1,b2,b3,binvm1)

      call copy(bstrs(1  ),h1,n)  ! Save h1 and h2 for comparison
      call col3(bstrs(1+n),bm1,h2,n)

      ratio=l2b/l2a
      if (nio.eq.0) write(6,1) istep,'  Project ' // 'HELM3 ',
     $              l2a,l2b,ratio,k,kmax
    1 format(i11,a,6x,1p3e13.4,i4,i4)

c     if (ierr.ne.0) call exitti(' h3proj quit$',ierr)

      tproj = tproj + dnekclock() - etime0

      return
      end
c-----------------------------------------------------------------------
      subroutine strs_project_b(x1,x2,x3,h1,h2,wt,ifld,ierr)

c     Reconstruct solution; don't bother to orthonomalize bases

      include 'SIZE'
      include 'TOTAL'
      include 'ORTHOSTRS'  ! Storage of approximation space
      include 'CTIMER'

      real x1(1),x2(1),x3(1),h1(1),h2(1),wt(1)
      common /cptst/ xs(lx1*ly1*lz1*lelt*ldim)

      kmax = napproxstrs(1)
      k    = napproxstrs(2)
      n    = lx1*ly1*lz1*nelv
      m    = n*ldim

      if (kmax.eq.0) return

      etime0 = dnekclock()

      if (k.eq.0) then                                          !      _
         call opadd2(x1,x2,x3,xstrs(1),xstrs(1+n),xstrs(1+2*n)) ! x=dx+x
         k=1
         k1 = 1 + 0*n + (k-1)*m + m
         k2 = 1 + 1*n + (k-1)*m + m
         k3 = 1 + 2*n + (k-1)*m + m
         call opcopy(xstrs(k1),xstrs(k2),xstrs(k3),x1,x2,x3)    ! x1=x^n
      elseif (k.eq.kmax) then                                   !      _
         call opadd2(x1,x2,x3,xstrs(1),xstrs(1+n),xstrs(1+2*n)) ! x=dx+x
         k=1
         k1 = 1 + 0*n + (k-1)*m + m
         k2 = 1 + 1*n + (k-1)*m + m
         k3 = 1 + 2*n + (k-1)*m + m
         call opcopy(xstrs(k1),xstrs(k2),xstrs(k3),x1,x2,x3)    ! x1=x^n
c        k=2
c        k1 = 1 + 0*n + (k-1)*m + m
c        k2 = 1 + 1*n + (k-1)*m + m
c        k3 = 1 + 2*n + (k-1)*m + m
c        call opcopy(xstrs(k1),xstrs(k2),xstrs(k3),xs(1),xs(1+n),xs(1+2*n))
      else
         k=k+1
         k1 = 1 + 0*n + (k-1)*m + m
         k2 = 1 + 1*n + (k-1)*m + m
         k3 = 1 + 2*n + (k-1)*m + m
         call opcopy(xstrs(k1),xstrs(k2),xstrs(k3),x1,x2,x3)    ! xk=dx  _
         call opadd2(x1,x2,x3,xstrs(1),xstrs(1+n),xstrs(1+2*n)) ! x=dx + x
      endif

c     if (k.eq.kmax) call opcopy(xs(1),xs(1+n),xs(1+2*n),x1,x2,x3) ! presave

      napproxstrs(2)=k

      tproj = tproj + dnekclock() - etime0

      return
      end
c-----------------------------------------------------------------------
      subroutine strs_orthok(x,b,n,k,h1,h2,wt,ifld,w,ierr,matmod)

c     Orthonormalize the kth element of X against x_j, j < k.

      include 'SIZE'
      include 'TOTAL'

      real x(n,ldim,k),b(n,ldim,k),h1(n),h2(n),wt(n),w(n,ldim)
      real al(mxprev),bt(mxprev)

      m   = n*ldim ! vector length
      nel = nelfld(ifld)

      call axhmsf   (b(1,1,k),b(1,2,k),b(1,3,k)
     $              ,x(1,1,k),x(1,2,k),x(1,3,k),h1,h2,matmod)
      call rmask    (b(1,1,k),b(1,2,k),b(1,3,k),nel)
      call opdssum  (b(1,1,k),b(1,2,k),b(1,3,k))

      xax0 = op_glsc2_wt(b(1,1,k),b(1,2,k),b(1,3,k)
     $                  ,x(1,1,k),x(1,2,k),x(1,3,k),wt)
      ierr = 3

      if (xax0.le.0.and.nio.eq.0)write(6,*)istep,ierr,k,xax0,'Proj3 ERR'
      if (xax0.le.0) return

      s = 0.
      do j=1,k-1! Modifed Gram-Schmidt

        betaj = ( op_vlsc2_wt(b(1,1,j),b(1,2,j),b(1,3,j)
     $                       ,x(1,1,k),x(1,2,k),x(1,3,k),wt)
     $          + op_vlsc2_wt(b(1,1,k),b(1,2,k),b(1,3,k)
     $                       ,x(1,1,j),x(1,2,j),x(1,3,j),wt))/2.
        betam = -glsum(betaj,1)
        call add2s2 (x(1,1,k),x(1,1,j),betam,m) ! Full-vector-subtract
        call add2s2 (b(1,1,k),b(1,1,j),betam,m) !

        s = s + betam**2

      enddo

      xax1 = xax0-s
      xax2 = op_glsc2_wt(b(1,1,k),b(1,2,k),b(1,3,k)
     $                  ,x(1,1,k),x(1,2,k),x(1,3,k),wt)
      scale = xax2

      eps  = 1.e-8
      ierr = 0
      if (scale/xax0.lt.eps) ierr=1

c      if(nio.eq.0) write(6,*) istep,ierr,k,scale,xax0,' SCALE'
c      if(nio.eq.0.and.(istep.lt.10.or.mod(istep,100).eq.0.or.ierr.gt.0))
c     $write(6,3) istep,k,ierr,xax1/xax0,xax2/xax0
c    3 format(i9,2i3,1p2e12.4,' scale ortho')

      if (scale.gt.0) then
         scale = 1./sqrt(scale)     
         call cmult(x(1,1,k),scale,m)
         call cmult(b(1,1,k),scale,m)
      else
         ierr=2
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine strs_ortho_one(x,b,n,k,h1,h2,wt,ifld,w,ierr,matmod)

      include 'SIZE'
      include 'TOTAL'

      real x(n,ldim,k),b(n,ldim,k),h1(n),h2(n),wt(n),w(n,ldim)

      m = n*ldim

      js=k
      do j=k,k
         if (js.lt.j) call copy(x(1,1,js),x(1,1,j),m)
         call strs_orthok(x,b,n,js,h1,h2,wt,ifld,w,ierr,matmod)
         if (ierr.eq.0) js=js+1
      enddo
      k = js-1

      return
      end
c-----------------------------------------------------------------------
      subroutine strs_ortho_all(x,b,n,k,h1,h2,wt,ifld,w,ierr,matmod)

      include 'SIZE'
      include 'TOTAL'

      real x(n,ldim,k),b(n,ldim,k),h1(n),h2(n),wt(n),w(n,ldim)

      m = n*ldim

      js=1
      do j=1,k
         if (js.lt.j) call copy(x(1,1,js),x(1,1,j),m)
         call strs_orthok(x,b,n,js,h1,h2,wt,ifld,w,ierr,matmod)
         if (ierr.eq.0) js=js+1
      enddo
      k = js-1

      return
      end
c-----------------------------------------------------------------------
      subroutine setprop
C------------------------------------------------------------------------
C
C     Set variable property arrays
C
C------------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
C
C     Caution: 2nd and 3rd strainrate invariants residing in scratch
C              common /SCREV/ are used in STNRINV and NEKASGN
C
      common /screv/ sii (lx1,ly1,lz1,lelt),siii(lx1,ly1,lz1,lelt)

      if (nio.eq.0.and.loglevel.gt.2)
     $   write(6,*) 'setprop'

#ifdef TIMER
      if (icalld.eq.0) tspro=0.0
      icalld=icalld+1
      nspro=icalld
      etime1=dnekclock()
#endif

      NXYZ1 = lx1*ly1*lz1
      MFIELD=2
      IF (IFFLOW) MFIELD=1
      nfldt = nfield
      if (ifmhd) nfldt = nfield+1

      ifld = ifield

      do ifield=mfield,nfldt
         if (idpss(ifield-1).eq.-1) goto 100
         call vprops
 100     continue
      enddo

      ifield = ifld

#ifdef TIMER
      tspro=tspro+(dnekclock()-etime1)
#endif

      return
      end
c-----------------------------------------------------------------------
