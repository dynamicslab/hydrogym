c=======================================================================
      subroutine hmholtz(name,u,rhs,h1,h2,mask,mult,imsh,tli,maxit,isd)
      include 'SIZE'
      include 'TOTAL'
      include 'FDMH1'
      include 'CTIMER'

      CHARACTER      NAME*4
      REAL           U    (LX1,LY1,LZ1,1)
      REAL           RHS  (LX1,LY1,LZ1,1)
      REAL           H1   (LX1,LY1,LZ1,1)
      REAL           H2   (LX1,LY1,LZ1,1)
      REAL           MASK (LX1,LY1,LZ1,1)
      REAL           MULT (LX1,LY1,LZ1,1)

      logical iffdm
      character*3 nam3

      tol = abs(tli)

      iffdm = .false.
      if (ifsplit) iffdm = .true.
      if (icalld.eq.0.and.iffdm) call set_fdm_prec_h1A
      icalld = icalld+1

#ifdef TIMER
      if (name.ne.'PRES') then
        nhmhz = nhmhz + 1
        etime1 = dnekclock()
      endif
#endif

      ntot = lx1*ly1*lz1*nelfld(ifield)
      if (imsh.eq.1) ntot = lx1*ly1*lz1*nelv
      if (imsh.eq.2) ntot = lx1*ly1*lz1*nelt

C     Determine which field is being computed for FDM based preconditioner bc's
c
      call chcopy(nam3,name,3)
c
                          kfldfdm = -1
c     if (nam3.eq.'TEM' ) kfldfdm =  0
c     if (name.eq.'TEM1') kfldfdm =  0  ! hardcode for temp only, for mpaul
c     if (name.eq.'VELX') kfldfdm =  1
c     if (name.eq.'VELY') kfldfdm =  2
c     if (name.eq.'VELZ') kfldfdm =  3
      if (name.eq.'PRES') kfldfdm =  ldim+1
c     if (.not.iffdm) kfldfdm=-1
C
      call dssum   (rhs,lx1,ly1,lz1)
      call col2    (rhs,mask,ntot)
c      if (nio.eq.0.and.istep.le.10) 
c     $    write(6,*) param(22),' p22 ',istep,imsh
      if (param(22).eq.0.or.istep.le.10)
     $    call chktcg1 (tol,rhs,h1,h2,mask,mult,imsh,isd)

      if (tli.lt.0) tol=tli ! caller-specified relative tolerance

      if (imsh.eq.1) call cggo
     $   (u,rhs,h1,h2,mask,mult,imsh,tol,maxit,isd,binvm1,name)
      if (imsh.eq.2) call cggo
     $   (u,rhs,h1,h2,mask,mult,imsh,tol,maxit,isd,bintm1,name)

#ifdef TIMER
      if (name.ne.'PRES') thmhz=thmhz+(dnekclock()-etime1)
#endif

      return
      END
C
c=======================================================================
      subroutine axhelm (au,u,helm1,helm2,imesh,isd)
C------------------------------------------------------------------
C
C     Compute the (Helmholtz) matrix-vector product,
C     AU = helm1*[A]u + helm2*[B]u, for NEL elements.
C
C------------------------------------------------------------------
      include 'SIZE'
      include 'WZ'
      include 'DXYZ'
      include 'GEOM'
      include 'MASS'
      include 'INPUT'
      include 'PARALLEL'
      include 'CTIMER'
C
      COMMON /FASTAX/ WDDX(LX1,LX1),WDDYT(LY1,LY1),WDDZT(LZ1,LZ1)
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
C
      REAL           AU    (LX1,LY1,LZ1,1)
     $ ,             U     (LX1,LY1,LZ1,1)
     $ ,             HELM1 (LX1,LY1,LZ1,1)
     $ ,             HELM2 (LX1,LY1,LZ1,1)
      COMMON /CTMP1/ DUDR  (LX1,LY1,LZ1)
     $ ,             DUDS  (LX1,LY1,LZ1)
     $ ,             DUDT  (LX1,LY1,LZ1)
     $ ,             TMP1  (LX1,LY1,LZ1)
     $ ,             TMP2  (LX1,LY1,LZ1)
     $ ,             TMP3  (LX1,LY1,LZ1)

      REAL           TM1   (LX1,LY1,LZ1)
      REAL           TM2   (LX1,LY1,LZ1)
      REAL           TM3   (LX1,LY1,LZ1)
      REAL           DUAX  (LX1)
      REAL           YSM1  (LX1)
      EQUIVALENCE    (DUDR,TM1),(DUDS,TM2),(DUDT,TM3)

      integer e

      naxhm = naxhm + 1
      etime1 = dnekclock()

      nel=nelt
      if (imesh.eq.1) nel=nelv

      NXY=lx1*ly1
      NYZ=ly1*lz1
      NXZ=lx1*lz1
      NXYZ=lx1*ly1*lz1
      NTOT=NXYZ*NEL

      IF (.NOT.IFSOLV) CALL SETFAST(HELM1,HELM2,IMESH)
      CALL RZERO (AU,NTOT)

      do 100 e=1,nel
C
        if (ifaxis) call setaxdy ( ifrzer(e) )
C
        IF (ldim.EQ.2) THEN
C
C       2-d case ...............
C
           if (iffast(e)) then
C
C          Fast 2-d mode: constant properties and undeformed element
C
           h1 = helm1(1,1,1,e)
           call mxm   (wddx,lx1,u(1,1,1,e),lx1,tm1,nyz)
           call mxm   (u(1,1,1,e),lx1,wddyt,ly1,tm2,ly1)
           call col2  (tm1,g4m1(1,1,1,e),nxyz)
           call col2  (tm2,g5m1(1,1,1,e),nxyz)
           call add3  (au(1,1,1,e),tm1,tm2,nxyz)
           call cmult (au(1,1,1,e),h1,nxyz)
C
           else
C
C
           call mxm  (dxm1,lx1,u(1,1,1,e),lx1,dudr,nyz)
           call mxm  (u(1,1,1,e),lx1,dytm1,ly1,duds,ly1)
           call col3 (tmp1,dudr,g1m1(1,1,1,e),nxyz)
           call col3 (tmp2,duds,g2m1(1,1,1,e),nxyz)
           if (ifdfrm(e)) then
              call addcol3 (tmp1,duds,g4m1(1,1,1,e),nxyz)
              call addcol3 (tmp2,dudr,g4m1(1,1,1,e),nxyz)
           endif
           call col2 (tmp1,helm1(1,1,1,e),nxyz)
           call col2 (tmp2,helm1(1,1,1,e),nxyz)
           call mxm  (dxtm1,lx1,tmp1,lx1,tm1,nyz)
           call mxm  (tmp2,lx1,dym1,ly1,tm2,ly1)
           call add2 (au(1,1,1,e),tm1,nxyz)
           call add2 (au(1,1,1,e),tm2,nxyz)

        endif
C
        else
C
C       3-d case ...............
C
           if (iffast(e)) then
C
C          Fast 3-d mode: constant properties and undeformed element
C
           h1 = helm1(1,1,1,e)
           call mxm   (wddx,lx1,u(1,1,1,e),lx1,tm1,nyz)
           do 5 iz=1,lz1
           call mxm   (u(1,1,iz,e),lx1,wddyt,ly1,tm2(1,1,iz),ly1)
 5         continue
           call mxm   (u(1,1,1,e),nxy,wddzt,lz1,tm3,lz1)
           call col2  (tm1,g4m1(1,1,1,e),nxyz)
           call col2  (tm2,g5m1(1,1,1,e),nxyz)
           call col2  (tm3,g6m1(1,1,1,e),nxyz)
           call add3  (au(1,1,1,e),tm1,tm2,nxyz)
           call add2  (au(1,1,1,e),tm3,nxyz)
           call cmult (au(1,1,1,e),h1,nxyz)
C
           else
C
C
           call mxm(dxm1,lx1,u(1,1,1,e),lx1,dudr,nyz)
           do 10 iz=1,lz1
              call mxm(u(1,1,iz,e),lx1,dytm1,ly1,duds(1,1,iz),ly1)
   10      continue
           call mxm     (u(1,1,1,e),nxy,dztm1,lz1,dudt,lz1)
           call col3    (tmp1,dudr,g1m1(1,1,1,e),nxyz)
           call col3    (tmp2,duds,g2m1(1,1,1,e),nxyz)
           call col3    (tmp3,dudt,g3m1(1,1,1,e),nxyz)
           if (ifdfrm(e)) then
              call addcol3 (tmp1,duds,g4m1(1,1,1,e),nxyz)
              call addcol3 (tmp1,dudt,g5m1(1,1,1,e),nxyz)
              call addcol3 (tmp2,dudr,g4m1(1,1,1,e),nxyz)
              call addcol3 (tmp2,dudt,g6m1(1,1,1,e),nxyz)
              call addcol3 (tmp3,dudr,g5m1(1,1,1,e),nxyz)
              call addcol3 (tmp3,duds,g6m1(1,1,1,e),nxyz)
           endif
           call col2 (tmp1,helm1(1,1,1,e),nxyz)
           call col2 (tmp2,helm1(1,1,1,e),nxyz)
           call col2 (tmp3,helm1(1,1,1,e),nxyz)
           call mxm  (dxtm1,lx1,tmp1,lx1,tm1,nyz)
           do 20 iz=1,lz1
              call mxm(tmp2(1,1,iz),lx1,dym1,ly1,tm2(1,1,iz),ly1)
   20      continue
           call mxm  (tmp3,nxy,dzm1,lz1,tm3,lz1)
           call add2 (au(1,1,1,e),tm1,nxyz)
           call add2 (au(1,1,1,e),tm2,nxyz)
           call add2 (au(1,1,1,e),tm3,nxyz)
C
           endif
c
        endif
C
 100  continue
C
      if (ifh2) call addcol4 (au,helm2,bm1,u,ntot)
C
C     If axisymmetric, add a diagonal term in the radial direction (ISD=2)
C
      if (ifaxis.and.(isd.eq.2)) then
         do 200 e=1,nel
C
            if (ifrzer(e)) then
               call mxm(u  (1,1,1,e),lx1,datm1,ly1,duax,1)
               call mxm(ym1(1,1,1,e),lx1,datm1,ly1,ysm1,1)
            endif
c
            do 190 j=1,ly1
            do 190 i=1,lx1
C               if (ym1(i,j,1,e).ne.0.) then
                  if (ifrzer(e)) then
                     term1 = 0.0
                     if(j.ne.1) 
     $             term1 = bm1(i,j,1,e)*u(i,j,1,e)/ym1(i,j,1,e)**2
                     term2 =  wxm1(i)*wam1(1)*dam1(1,j)*duax(i)
     $                       *jacm1(i,1,1,e)/ysm1(i)
                  else
                   term1 = bm1(i,j,1,e)*u(i,j,1,e)/ym1(i,j,1,e)**2
                     term2 = 0.
                  endif
                  au(i,j,1,e) = au(i,j,1,e)
     $                          + helm1(i,j,1,e)*(term1+term2)
C               endif
  190       continue
  200    continue
      endif

      taxhm=taxhm+(dnekclock()-etime1)
      return
      end
C
c=======================================================================
      subroutine setfast (helm1,helm2,imesh)
C-------------------------------------------------------------------
C
C     Set logicals for fast evaluation of A*x
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
      REAL HELM1(lx1,ly1,lz1,1), HELM2(lx1,ly1,lz1,1)
C
      IF (IMESH.EQ.1) NEL=NELV
      IF (IMESH.EQ.2) NEL=NELT
      NXYZ = lx1*ly1*lz1
      NTOT = NXYZ*NEL
C
      DELTA = 1.E-9
      X    = 1.+DELTA
      Y    = 1.
      DIFF = ABS(X-Y)
      IF (DIFF.EQ.0.0) EPSM = 1.E-6
      IF (DIFF.GT.0.0) EPSM = 1.E-13
C
      DO 100 ie=1,NEL
         IFFAST(ie) = .FALSE.
         IF (IFDFRM(ie).OR.IFAXIS) THEN
            IFFAST(ie) = .FALSE.
         ELSE
           H1MIN  = VLMIN(HELM1(1,1,1,ie),NXYZ)
           H1MAX  = VLMAX(HELM1(1,1,1,ie),NXYZ)
           den    = abs(h1max)+abs(h1min)
           if (den.gt.0) then
              TESTH1 = ABS((H1MAX-H1MIN)/(H1MAX+H1MIN))
              IF (TESTH1.LT.EPSM) IFFAST(ie) = .TRUE.
           else
              iffast(ie) = .true.
           endif
         ENDIF
 100  CONTINUE
c
      IFH2   = .FALSE.
      TESTH2 =  VLAMAX(HELM2,NTOT)
      IF (TESTH2.GT.0.) IFH2 = .TRUE.
      return
      END
C
c=======================================================================
      subroutine sfastax
C----------------------------------------------------------------------
C
C     For undeformed elements, set up appropriate elemental matrices
C     and geometric factors for fast evaluation of Ax.
C
C----------------------------------------------------------------------
      include 'SIZE'
      include 'WZ'
      include 'DXYZ'
      include 'GEOM'
      COMMON /FASTAX/ WDDX(LX1,LY1),WDDYT(LY1,LY1),WDDZT(LZ1,LZ1)
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
      LOGICAL IFIRST
      SAVE    IFIRST
      DATA    IFIRST /.TRUE./
C
      NXX=lx1*lx1
      IF (IFIRST) THEN
         CALL RZERO(WDDX,NXX)
         DO 100 I=1,lx1
         DO 100 J=1,lx1
         DO 100 IP=1,lx1
            WDDX(I,J) = WDDX(I,J) + WXM1(IP)*DXM1(IP,I)*DXM1(IP,J)
  100    CONTINUE
         NYY=ly1*ly1
         CALL RZERO(WDDYT,NYY)
         DO 200 I=1,ly1
         DO 200 J=1,ly1
         DO 200 IP=1,ly1
            WDDYT(J,I) = WDDYT(J,I) + WYM1(IP)*DYM1(IP,I)*DYM1(IP,J)
  200    CONTINUE
         NZZ=lz1*lz1
         CALL RZERO(WDDZT,NZZ)
         DO 300 I=1,lz1
         DO 300 J=1,lz1
         DO 300 IP=1,lz1
            WDDZT(J,I) = WDDZT(J,I) + WZM1(IP)*DZM1(IP,I)*DZM1(IP,J)
  300    CONTINUE
         IFIRST=.FALSE.
      ENDIF
C
      IF (ldim.EQ.3) THEN
         DO 1001 IE=1,NELT
            IF (.NOT.IFDFRM(IE)) THEN
               DO 1000 IZ=1,lz1
               DO 1000 IY=1,ly1
               DO 1000 IX=1,lx1
                  G4M1(IX,IY,IZ,IE)=G1M1(IX,IY,IZ,IE)/WXM1(IX)
                  G5M1(IX,IY,IZ,IE)=G2M1(IX,IY,IZ,IE)/WYM1(IY)
                  G6M1(IX,IY,IZ,IE)=G3M1(IX,IY,IZ,IE)/WZM1(IZ)
 1000          CONTINUE
            ENDIF
 1001    CONTINUE
      ELSE
         DO 2001 IE=1,NELT
            IF (.NOT.IFDFRM(IE)) THEN
               DO 2000 IY=1,ly1
               DO 2000 IX=1,lx1
                  G4M1(IX,IY,1,IE)=G1M1(IX,IY,1,IE)/WXM1(IX)
                  G5M1(IX,IY,1,IE)=G2M1(IX,IY,1,IE)/WYM1(IY)
 2000          CONTINUE
            ENDIF
 2001    CONTINUE
      ENDIF
      return
      END
C
c=======================================================================
      subroutine setprec (dpcm1,helm1,helm2,imsh,isd)
C-------------------------------------------------------------------
C
C     Generate diagonal preconditioner for the Helmholtz operator.
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'WZ'
      include 'DXYZ'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
      include 'MASS'
      REAL            DPCM1 (LX1,LY1,LZ1,1)
      COMMON /FASTMD/ IFDFRM(LELT), IFFAST(LELT), IFH2, IFSOLV
      LOGICAL IFDFRM, IFFAST, IFH2, IFSOLV
      REAL            HELM1(lx1,ly1,lz1,1), HELM2(lx1,ly1,lz1,1)
      REAL YSM1(LY1)

      nel=nelt
      if (imsh.eq.1) nel=nelv

      ntot = nel*lx1*ly1*lz1

c     The following lines provide a convenient debugging option
c     call rone(dpcm1,ntot)
c     if (ifield.eq.1) call copy(dpcm1,binvm1,ntot)
c     if (ifield.eq.2) call copy(dpcm1,bintm1,ntot)
c     return

      CALL RZERO(DPCM1,NTOT)
      DO 1000 IE=1,NEL

        IF (IFAXIS) CALL SETAXDY ( IFRZER(IE) )

        DO 320 IQ=1,lx1
        DO 320 IZ=1,lz1
        DO 320 IY=1,ly1
        DO 320 IX=1,lx1
           DPCM1(IX,IY,IZ,IE) = DPCM1(IX,IY,IZ,IE) + 
     $                          G1M1(IQ,IY,IZ,IE) * DXTM1(IX,IQ)**2
  320   CONTINUE
        DO 340 IQ=1,ly1
        DO 340 IZ=1,lz1
        DO 340 IY=1,ly1
        DO 340 IX=1,lx1
           DPCM1(IX,IY,IZ,IE) = DPCM1(IX,IY,IZ,IE) + 
     $                          G2M1(IX,IQ,IZ,IE) * DYTM1(IY,IQ)**2
  340   CONTINUE
        IF (LDIM.EQ.3) THEN
           DO 360 IQ=1,lz1
           DO 360 IZ=1,lz1
           DO 360 IY=1,ly1
           DO 360 IX=1,lx1
              DPCM1(IX,IY,IZ,IE) = DPCM1(IX,IY,IZ,IE) + 
     $                             G3M1(IX,IY,IQ,IE) * DZTM1(IZ,IQ)**2
  360      CONTINUE
C
C          Add cross terms if element is deformed.
C
           IF (IFDFRM(IE)) THEN
              DO 600 IY=1,ly1,ly1-1
              DO 600 IZ=1,lz1,max(1,lz1-1)
              DPCM1(1,IY,IZ,IE) = DPCM1(1,IY,IZ,IE)
     $            + G4M1(1,IY,IZ,IE) * DXTM1(1,1)*DYTM1(IY,IY)
     $            + G5M1(1,IY,IZ,IE) * DXTM1(1,1)*DZTM1(IZ,IZ)
              DPCM1(lx1,IY,IZ,IE) = DPCM1(lx1,IY,IZ,IE)
     $            + G4M1(lx1,IY,IZ,IE) * DXTM1(lx1,lx1)*DYTM1(IY,IY)
     $            + G5M1(lx1,IY,IZ,IE) * DXTM1(lx1,lx1)*DZTM1(IZ,IZ)
  600         CONTINUE
              DO 700 IX=1,lx1,lx1-1
              DO 700 IZ=1,lz1,max(1,lz1-1)
                 DPCM1(IX,1,IZ,IE) = DPCM1(IX,1,IZ,IE)
     $            + G4M1(IX,1,IZ,IE) * DYTM1(1,1)*DXTM1(IX,IX)
     $            + G6M1(IX,1,IZ,IE) * DYTM1(1,1)*DZTM1(IZ,IZ)
                 DPCM1(IX,ly1,IZ,IE) = DPCM1(IX,ly1,IZ,IE)
     $            + G4M1(IX,ly1,IZ,IE) * DYTM1(ly1,ly1)*DXTM1(IX,IX)
     $            + G6M1(IX,ly1,IZ,IE) * DYTM1(ly1,ly1)*DZTM1(IZ,IZ)
  700         CONTINUE
              DO 800 IX=1,lx1,lx1-1
              DO 800 IY=1,ly1,ly1-1
                 DPCM1(IX,IY,1,IE) = DPCM1(IX,IY,1,IE)
     $                + G5M1(IX,IY,1,IE) * DZTM1(1,1)*DXTM1(IX,IX)
     $                + G6M1(IX,IY,1,IE) * DZTM1(1,1)*DYTM1(IY,IY)
                 DPCM1(IX,IY,lz1,IE) = DPCM1(IX,IY,lz1,IE)
     $                + G5M1(IX,IY,lz1,IE) * DZTM1(lz1,lz1)*DXTM1(IX,IX)
     $                + G6M1(IX,IY,lz1,IE) * DZTM1(lz1,lz1)*DYTM1(IY,IY)
  800         CONTINUE
           ENDIF

        ELSE  ! 2D

           IZ=1
           IF (IFDFRM(IE)) THEN
              DO 602 IY=1,ly1,ly1-1
                 DPCM1(1,IY,IZ,IE) = DPCM1(1,IY,IZ,IE)
     $                + G4M1(1,IY,IZ,IE) * DXTM1(1,1)*DYTM1(IY,IY)
                 DPCM1(lx1,IY,IZ,IE) = DPCM1(lx1,IY,IZ,IE)
     $                + G4M1(lx1,IY,IZ,IE) * DXTM1(lx1,lx1)*DYTM1(IY,IY)
  602         CONTINUE
              DO 702 IX=1,lx1,lx1-1
                 DPCM1(IX,1,IZ,IE) = DPCM1(IX,1,IZ,IE)
     $                + G4M1(IX,1,IZ,IE) * DYTM1(1,1)*DXTM1(IX,IX)
                 DPCM1(IX,ly1,IZ,IE) = DPCM1(IX,ly1,IZ,IE)
     $                + G4M1(IX,ly1,IZ,IE) * DYTM1(ly1,ly1)*DXTM1(IX,IX)
  702         CONTINUE
           ENDIF

        ENDIF
 1000 CONTINUE
C
      CALL COL2    (DPCM1,HELM1,NTOT)
      CALL ADDCOL3 (DPCM1,HELM2,BM1,NTOT)
C
C     If axisymmetric, add a diagonal term in the radial direction (ISD=2)
C
      IF (IFAXIS.AND.(ISD.EQ.2)) THEN
         DO 1200 IEL=1,NEL
C
            IF (IFRZER(IEL)) THEN
               CALL MXM(YM1(1,1,1,IEL),lx1,DATM1,ly1,YSM1,1)
            ENDIF
C
            DO 1190 J=1,ly1
            DO 1190 I=1,lx1
               IF (YM1(I,J,1,IEL).NE.0.) THEN
                  TERM1 = BM1(I,J,1,IEL)/YM1(I,J,1,IEL)**2
                  IF (IFRZER(IEL)) THEN
                     TERM2 =  WXM1(I)*WAM1(1)*DAM1(1,J)
     $                       *JACM1(I,1,1,IEL)/YSM1(I)
                  ELSE
                     TERM2 = 0.
                  ENDIF
                  DPCM1(I,J,1,IEL) = DPCM1(I,J,1,IEL)
     $                             + HELM1(I,J,1,IEL)*(TERM1+TERM2)
               ENDIF
 1190       CONTINUE
 1200    CONTINUE
      ENDIF
C
      CALL DSSUM (DPCM1,lx1,ly1,lz1)
      CALL INVCOL1 (DPCM1,NTOT)
C
      return
      END
C
c=======================================================================
      subroutine chktcg1 (tol,res,h1,h2,mask,mult,imesh,isd)
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
      COMMON  /CPRINT/ IFPRINT
      LOGICAL          IFPRINT
      COMMON /CTMP0/ W1   (LX1,LY1,LZ1,LELT)
     $ ,             W2   (LX1,LY1,LZ1,LELT)
      REAL RES  (LX1,LY1,LZ1,1)
      REAL H1   (LX1,LY1,LZ1,1)
      REAL H2   (LX1,LY1,LZ1,1)
      REAL MULT (LX1,LY1,LZ1,1)
      REAL MASK (LX1,LY1,LZ1,1)
C
      IF (EIGAA.NE.0.) THEN
         ACONDNO = EIGGA/EIGAA
      ELSE
         ACONDNO = 10.
      ENDIF
C
C     Single or double precision???
C
      DELTA = 1.E-9
      X     = 1.+DELTA
      Y     = 1.
      DIFF  = ABS(X-Y)
      IF (DIFF.EQ.0.) EPS = 1.E-6
      IF (DIFF.GT.0.) EPS = 1.E-13
C
      IF (IMESH.EQ.1) THEN
          NL  = NELV
          VOL = VOLVM1
      ELSEIF (IMESH.EQ.2) THEN
          NL  = NELT
          VOL = VOLTM1
      ENDIF
      NTOT1 = lx1*ly1*lz1*NL
      CALL COPY (W1,RES,NTOT1)
C
      IF (IMESH.EQ.1) THEN
         CALL COL3 (W2,BINVM1,W1,NTOT1)
         RINIT  = SQRT(GLSC3 (W2,W1,MULT,NTOT1)/VOLVM1)
      ELSE
         CALL COL3 (W2,BINTM1,W1,NTOT1)
         RINIT  = SQRT(GLSC3 (W2,W1,MULT,NTOT1)/VOLTM1)
      ENDIF
      RMIN   = EPS*RINIT
      IF (TOL.LT.RMIN) THEN
         IF (NIO.EQ.0.AND.IFPRINT)
     $   WRITE (6,*) 'New CG1-tolerance (RINIT*epsm) = ',RMIN,TOL
         TOL = RMIN
      ENDIF
C
      CALL RONE (W1,NTOT1)
      BCNEU1 = GLSC3(W1,MASK,MULT,NTOT1)
      BCNEU2 = GLSC3(W1,W1  ,MULT,NTOT1)
      BCTEST = ABS(BCNEU1-BCNEU2)
C
      CALL AXHELM (W2,W1,H1,H2,IMESH,ISD)
      CALL COL2   (W2,W2,NTOT1)
      CALL COL2   (W2,BM1,NTOT1)
      BCROB  = SQRT(GLSUM(W2,NTOT1)/VOL)
C
      IF ((BCTEST .LT. .1).AND.(BCROB.LT.(EPS*ACONDNO))) THEN
C         OTR = GLSC3 (W1,RES,MULT,NTOT1)
         TOLMIN = RINIT*EPS*10.
         IF (TOL .LT. TOLMIN) THEN
             TOL = TOLMIN
             IF (NIO.EQ.0.AND.IFPRINT)
     $       WRITE(6,*) 'New CG1-tolerance (Neumann) = ',TOLMIN
         ENDIF
      ENDIF
C
      return
      end
c=======================================================================
      subroutine cggo(x,f,h1,h2,mask,mult,imsh,tin,maxit,isd,binv,name)
C-------------------------------------------------------------------------
C
C     Solve the Helmholtz equation, H*U = RHS,
C     using preconditioned conjugate gradient iteration.
C     Preconditioner: diag(H).
C
C------------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      include 'FDMH1'
 
      COMMON  /CPRINT/ IFPRINT, IFHZPC
      LOGICAL          IFPRINT, IFHZPC
 
      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv

      logical ifmcor,ifprint_hmh
 
      real x(1),f(1),h1(1),h2(1),mask(1),mult(1),binv(1)
      parameter        (lg=lx1*ly1*lz1*lelt)
      COMMON /SCRCG/ d (lg) , scalar(2)
      common /SCRMG/ r (lg) , w (lg) , p (lg) , z (lg)
c
      parameter (maxcg=900)
      common /tdarray/ diagt(maxcg),upper(maxcg)
      common /iterhm/ niterhm
      character*4 name
c
      if (ifsplit.and.name.eq.'PRES') then
         if (param(42).eq.0) then
           n = lx1*ly1*lz1*nelv
           call copy      (x,f,n)
           iter = maxit
           call hmh_gmres (x,h1,h2,mult,iter)
           niterhm = iter
           return
         elseif(param(42).eq.2) then 
           n = lx1*ly1*lz1*nelv
           call copy       (x,f,n)
           iter = maxit
           call hmh_flex_cg(x,h1,h2,mult,iter)
           niterhm = iter
           return
         endif
      endif

c **  zero out stuff for Lanczos eigenvalue estimator
      call rzero(diagt,maxcg)
      call rzero(upper,maxcg)
      rho = 0.00
C
C     Initialization
C
      NXYZ   = lx1*ly1*lz1
      NEL    = NELV
      VOL    = VOLVM1
      IF (IMSH.EQ.2) NEL=NELT
      IF (IMSH.EQ.2) VOL=VOLTM1
      n      = NEL*NXYZ

      tol=abs(tin)

c     overrule input tolerance
      if (restol(ifield).ne.0) tol=restol(ifield)
      if (name.eq.'PRES'.and.param(21).ne.0) tol=abs(param(21))

      if (tin.lt.0) tol=abs(tin)
      niter = min(maxit,maxcg)

      if (.not.ifsolv) then
         call setfast(h1,h2,imsh)
         ifsolv = .true.
      endif
C
C     Set up diag preconditioner.
C
      if (kfldfdm.lt.0) then
         call setprec(D,h1,h2,imsh,isd)
      elseif(param(100).ne.2) then
         call set_fdm_prec_h1b(d,h1,h2,nel)
      endif

      call copy (r,f,n)
      call rzero(x,n)
      call rzero(p,n)

      fmax = glamax(f,n)
      if (fmax.eq.0.0) return

c     Check for non-trivial null-space

      ifmcor = .false.
      h2max = glmax(h2  ,n)
      skmin = glmin(mask,n)
      if (skmin.gt.0.and.h2max.eq.0) ifmcor = .true.
C
      if (name.eq.'PRES') then
c        call ortho (r)           ! Commented out March 15, 2011,pff
      elseif (ifmcor) then

         smean = -1./glsum(bm1,n) ! Modified 5/4/12 pff
         rmean = smean*glsc2(r,mult,n)
         call copy(x,bm1,n)
         call dssum(x,lx1,ly1,lz1)
         call add2s2(r,x,rmean,n)
         call rzero(x,n)
      endif
C
      krylov = 0
      rtz1=1.0
      niterhm = 0

      do iter=1,niter
C
         if (kfldfdm.lt.0) then  ! Jacobi Preconditioner
c           call copy(z,r,n)
            call col3(z,r,d,n)
         else                                       ! Schwarz Preconditioner
            if (name.eq.'PRES'.and.param(100).eq.2) then
               call h1_overlap_2(z,r,mask)
               call crs_solve_h1 (w,r)  ! Currently, crs grd only for P
               call add2         (z,w,n)
            else   
               call fdm_h1(z,r,d,mask,mult,nel,ktype(1,1,kfldfdm),w)
               if (name.eq.'PRES') then 
                 call crs_solve_h1 (w,r)  ! Currently, crs grd only for P
                 call add2         (z,w,n)
               endif
            endif
         endif
c
         if (name.eq.'PRES') then
            call ortho (z)
         elseif (ifmcor) then
            rmean = smean*glsc2(z,bm1,n)
            call cadd(z,rmean,n)
         endif
c        write(6,*) rmean,ifmcor,' ifmcor'
c
         rtz2=rtz1
         scalar(1)=vlsc3 (z,r,mult,n)
         if(param(18).eq.1) then
           scalar(2)=vlsc3(r,r,mult,n)
         else 
          scalar(2)=vlsc32(r,mult,binv,n)
         endif
         call gop(scalar,w,'+  ',2)
         rtz1=scalar(1)
         rbn2=sqrt(scalar(2)/vol)
         if (iter.eq.1) rbn0 = rbn2
         if (param(22).lt.0) tol=abs(param(22))*rbn0
         if (tin.lt.0)       tol=abs(tin)*rbn0

         ifprint_hmh = .false.
         if (nio.eq.0.and.ifprint.and.param(74).ne.0) ifprint_hmh=.true.
         if (nio.eq.0.and.istep.eq.1)                 ifprint_hmh=.true.

         if (ifprint_hmh)
     &      write(6,3002) istep,'  Hmholtz ' // name,
     &                    iter,rbn2,h1(1),tol,h2(1),ifmcor


c        Always take at least one iteration   (for projection) pff 11/23/98
#ifndef FIXITER
         IF (rbn2.LE.TOL.and.(iter.gt.1 .or. istep.le.5)) THEN
#else
         iter_max = param(150)
         if (name.eq.'PRES') iter_max = param(151)
         if (iter.gt.iter_max) then
#endif
c        IF (rbn2.LE.TOL) THEN
            NITER = ITER-1
c           IF(NID.EQ.0.AND.((.NOT.IFHZPC).OR.IFPRINT))
            if (nio.eq.0)
     &         write(6,3000) istep,'  Hmholtz ' // name,
     &                       niter,rbn2,rbn0,tol
            goto 9999
         ENDIF
c
         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0
         call add2s1 (p,z,beta,n)
         call axhelm (w,p,h1,h2,imsh,isd)
         call dssum  (w,lx1,ly1,lz1)
         call col2   (w,mask,n)
c
         rho0 = rho
         rho  = glsc3(w,p,mult,n)
         alpha=rtz1/rho
         alphm=-alpha
         call add2s2(x,p ,alpha,n)
         call add2s2(r,w ,alphm,n)
c
c        Generate tridiagonal matrix for Lanczos scheme
         if (iter.eq.1) then
            krylov = krylov+1
            diagt(iter) = rho/rtz1
         elseif (iter.le.maxcg) then
            krylov = krylov+1
            diagt(iter)    = (beta**2 * rho0 + rho ) / rtz1
            upper(iter-1)  = -beta * rho0 / sqrt(rtz2 * rtz1)
         endif
 1000 enddo
      niter = iter-1
c
      if (nio.eq.0) write (6,3001) istep, '  Error Hmholtz ' // name,
     &                             niter,rbn2,rbn0,tol


 3000 format(i11,a,1x,I7,1p4E13.4)
 3001 format(i11,a,1x,I7,1p4E13.4)
 3002 format(i11,a,1x,I7,1p4E13.4,l4)
 9999 continue
      niterhm = niter
      ifsolv = .false.
c
c
c     Call eigenvalue routine for Lanczos scheme:
c          two work arrays are req'd if you want to save "diag & upper"
c
c     if (iter.ge.3) then
c        niter = iter-1
c        call calc (diagt,upper,w,z,krylov,dmax,dmin)
c        cond = dmax/dmin
c        if (nid.eq.0) write(6,6) istep,cond,dmin,dmax,' lambda'
c     endif
c   6 format(i9,1p3e12.4,4x,a7)
c
c     if (n.gt.0) write(6,*) 'quit in cggo'
c     if (n.gt.0) call exitt
c     call exitt
      return
      end
c=======================================================================
      function vlsc32(r,b,m,n)
      real r(1),b(1),m(1)
      s = 0.
      do i=1,n
         s = s + b(i)*m(i)*r(i)*r(i)
      enddo
      vlsc32 = s
      return
      end
c=======================================================================
      subroutine calc (diag,upper,d,e,n,dmax,dmin)
c
      dimension diag(n),upper(n)
      dimension d(n),e(n)
c
      call copy (d,diag ,n)
      call copy (e,upper,n)
c
      do 15 l=1,n
         iter = 0
c
 1       do 12 m=l,n-1
            dd = abs( d(m) ) + abs( d(m+1) )
            if ( abs(e(m)) + dd .eq. dd ) goto 2
 12      continue
c
         m = n
 2       if ( m .ne. l ) then
c
            if ( iter .eq. 30 ) then
               write (6,*) 'too many iterations'
               return
            endif
c
         iter = iter + 1
         g = ( d(l+1) - d(l) ) / ( 2.0 * e(l) )
         r = sqrt( g**2 + 1.0 )
c
c sign is defined as a(2) * abs( a(1) )
c
         g = d(m) - d(l) + e(l)/(g+sign(r,g))
         s = 1.0
         c = 1.0
         p = 0.0
c
         do 14 i = m-1,l,-1
            f = s * e(i)
            b = c * e(i)
            if ( abs(f) .ge. abs(g) ) then
               c = g/f
               r = sqrt( c**2 + 1.0 )
               e(i+1) = f*r
               s = 1.0/r
               c = c*s
            else
               s = f/g
               r = sqrt( s**2 + 1.0 )
               e(i+1) = g*r
               c = 1.0 / r
               s = s * c
            endif
c
            g = d(i+1) - p
            r = ( d(i) - g ) * s + 2.0 * c * b
            p = s * r
            d(i+1) = g + p
            g = c*r - b
 14      continue
c
         d(l) = d(l) - p
         e(l) = g
         e(m) = 0.0
         goto 1
c
         endif
c
 15   continue
c
      dmax = 0.0
      dmin = d(1)
c
      do 40 i = 1 , n
        dmax = abs( max( d(i) , dmax ) )
        dmin = abs( min( d(i) , dmin ) )
 40   continue
c
      return
      end
c-----------------------------------------------------------------------
      subroutine fdm_h1(z,r,d,mask,mult,nel,kt,rr)
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
      real mask(lx1,ly1,lz1,1)
      real mult(lx1,ly1,lz1,1)
      real rr(lx1,ly1,lz1,1)
c
      integer kt(lelt,3)
c
      integer icalld
      save    icalld
      data    icalld /0/
c
      n1 = lx1
      n2 = lx1*lx1
      n3 = lx1*lx1*lx1
      ntot = lx1*ly1*lz1*nel
c
      if (ifbhalf) then
         call col3(rr,r,bhalf,ntot)
      else
         call copy(rr,r,ntot)
c        call col2(rr,mult,ntot)
      endif
c      if (nid.eq.0.and.icalld.eq.0) write(6,*) 'In fdm_h1',nel
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
c
c     call copy(vx,rr,ntot)
c     call copy(vy,z,ntot)
c     call prepost(.true.)
c     write(6,*) 'quit in fdm'
c     call exitt
c
      if (ifbhalf) call col2(z,bhalf,ntot)
c
c     call col2 (z,mult,ntot)
      call dssum(z,lx1,ly1,lz1)
      call col2 (z,mask,ntot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_fdm_prec_h1A_gen
c
      include 'SIZE'
      include 'DXYZ'
      include 'INPUT'
      include 'MASS'
      include 'WZ'
c
      include 'FDMH1'
c
      COMMON /CTMP0/ W(LX1,LX1),aa(lx1,lx1),bb(lx1,lx1)
c
      integer left,right
c
c     Set up generic operators for fdm applied to H1 operator (Helmholtz)
c
c     3 cases:   E (or P),   "D"  or "N"  for E-E bc, Dirichlet, or Neuamann.
c
c     Since there are 2 endpoints, there are a total of 9 types.
c
c
      n  = lx1
      n2 = lx1*lx1
c
      delta = abs( zgm1(2,1) - zgm1(1,1) )
      bbh   = 0.5*delta
      aah   = 1./delta
c
      l = 0
      do right = 1,3
      do left  = 1,3
         l = l+1
c
         call rzero(bb,n2)
         do i=1,lx1
            bb(i,i) = wxm1(i)
         enddo
c
c        A = D^T B D
c
         call mxm(BB,n,Dxm1 ,n,w,n)
         call mxm(Dxtm1,n,w,n,AA,n)
         if (left.eq.1) then
c           Internal
            bb(1,1) = bb(1,1) + bbh
            aa(1,1) = aa(1,1) + aah
         elseif (left.eq.2) then
c           Dirichlet
            bb(1,1) = 1.
            do i=1,n
               aa(i,1) = 0.
               aa(1,i) = 0.
            enddo
            aa(1,1) = 1.
         endif
c
         if (right.eq.1) then
c           Internal
            bb(n,n) = bb(n,n) + bbh
            aa(n,n) = aa(n,n) + aah
         elseif (right.eq.2) then
c           Dirichlet
            bb(n,n) = 1.
            do i=1,n
               aa(i,n) = 0.
               aa(n,i) = 0.
            enddo
            aa(n,n) = 1.
         endif
c
c        Scale out mass matrix, so we can precondition w/ binvhf.
c
c        ifbhalf = .true.

         ifbhalf = .false.
         if (ifbhalf) call rescale_abhalf (aa,bb,w,n)
c
c        Now, compute eigenvectors/eigenvalues
c
         call generalev(aa,bb,dd(1,l),n,w)
         call copy(fds(1,l),aa,n*n)
         call transpose(fdst(1,l),n,fds(1,l),n)
c
      enddo
      enddo
      ntot = lx1*ly1*lz1*nelv
      if (ifbhalf) call copy (bhalf,binvm1,ntot)
      if (ifbhalf) call vsqrt(bhalf,ntot)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_fdm_prec_h1A_els
c
      include 'SIZE'
      include 'DXYZ'
      include 'FDMH1'
      include 'GEOM'
      include 'INPUT'
      include 'SOLN'
      include 'TOPOL'
      include 'WZ'
c
      COMMON /CTMP0/ W(LX1,LX1),aa(lx1,lx1),bb(lx1,lx1)
     $             , mask(lx1,ly1,lz1,lelt)
      real mask
      character*3 cb
c
c
c     Set up element specific information
c
c     3 cases:   E (or P),   "D"  or "N"  for E-E bc, Dirichlet, or Neuamann.
c
c     Since there are 2 endpoints, there are a total of 9 types.
c
c
      ntot = lx1*ly1*lz1*nelt
      kf0 = 1
      kf1 = 0
      if (ifheat)  kf0 = 0
      if (ifflow)  kf1 = ldim
      if (ifsplit) kf1 = ldim+1
      do kfld=kf0,kf1
         ifld = 1
         if (kfld.eq.0) ifld = 2
c
         if (kfld.eq.0)      call copy(mask, tmask,ntot)
         if (kfld.eq.1)      call copy(mask,v1mask,ntot)
         if (kfld.eq.2)      call copy(mask,v2mask,ntot)
         if (kfld.eq.3)      call copy(mask,v3mask,ntot)
         if (kfld.eq.ldim+1) call copy(mask, pmask,ntot)
c
         do ie=1,nelv
            do ifacedim = 1,ldim
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c              Mask pointers
c
               ii = 2
               jj = 2
               kk = 2
c
               if (ifacedim.eq.1) ii = 1
               if (ifacedim.eq.2) jj = 1
               if (ifacedim.eq.3) kk = 1
               k1 = ii+lx1*(jj-1)
               if (if3d) k1 = ii+lx1*(jj-1) + lx1*lx1*(kk-1)
c
               if (ifacedim.eq.1) ii = lx1
               if (ifacedim.eq.2) jj = lx1
               if (ifacedim.eq.3) kk = lx1
               k2 = ii+lx1*(jj-1)
               if (if3d) k2 = ii+lx1*(jj-1) + lx1*lx1*(kk-1)
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
               iface = 2*ifacedim-1
               jface = iface+1
c
c              Convert to preproc   :(
               iface = eface(iface)
               jface = eface(jface)
c
c              "left" bc
c
               cb = cbc(iface,ie,ifld)
               if (cb.eq.'E  '.or.cb.eq.'P  '.or.cb.eq.'p  ') then
c                 Internal
                  ic1 = 1
               elseif (mask(k1,1,1,ie).eq.0) then
c                 Dirichlet
                  ic1 = 2
               else
c                 Neumann
                  ic1 = 3
               endif
c              write(6,*) ie,iface,'cbl: ',cb,ic1,k1,mask(k1,1,1,ie)
c
c              "right" bc
c
               cb = cbc(jface,ie,ifld)
               if (cb.eq.'E  '.or.cb.eq.'P  '.or.cb.eq.'p  ') then
c                 Internal
                  jc1 = 1
               elseif (mask(k2,1,1,ie).eq.0) then
c                 Dirichlet
                  jc1 = 2
               else
c                 Neumann
                  jc1 = 3
               endif
c              write(6,*) ie,jface,'cbr: ',cb,jc1,k2,mask(k2,1,1,ie)
c
               ijc = ic1 + 3*(jc1-1)
               ktype(ie,ifacedim,kfld) = ijc
c
            enddo
         enddo
      enddo
c
c     Boundary condition issues resolved... now resolve length scales
c
c
c
      do ie = 1,nelt
      do idim=1,ldim
         k1 = 1
         k2 = lz1
         if (idim.eq.3.or.ldim.eq.2) k2=1
         j1 = 1
         j2 = ly1
         if (idim.eq.2) j2=1
         i1 = 1
         i2 = lx1
         if (idim.eq.1) i2=1
c
c        l -- face 1,  l+jump = face 2
c
         jump = (lx1-1)*lx1**(idim-1)
         l    = 0
         dlm  = 0
         wgt  = 0
         do k=k1,k2
         do j=j1,j2
         do i=i1,i2
            l = l+1
            dl2 = (xm1(i+jump,j,k,ie)-xm1(i,j,k,ie))**2
     $          + (ym1(i+jump,j,k,ie)-ym1(i,j,k,ie))**2
     $          + (zm1(i+jump,j,k,ie)-zm1(i,j,k,ie))**2
            dlm = dlm + dl2*wxm1(i)*wxm1(j)*wxm1(k)
            wgt = wgt + wxm1(i)*wxm1(j)*wxm1(k)
c
         enddo
         enddo
         enddo
c
         dlm             = sqrt(dlm/wgt)
         elsize(idim,ie) = dlm/2.
c
      enddo
c        write(6,1) ie,' elsize:',(elsize(k,ie),k=1,ldim)
      enddo
    1 format(i8,a8,1p3e15.4)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_fdm_prec_h1b(d,h1,h2,nel)
      include 'SIZE'
      include 'FDMH1'
      include 'INPUT'
      include 'GEOM'
      real d (lx1,ly1,lz1,1)
      real h1(lx1,ly1,lz1,1)
      real h2(lx1,ly1,lz1,1)
c
c     Set up diagonal for FDM for each spectral element 
c
      nxyz = lx1*ly1*lz1
      if (if3d) then
         do ie=1,nel
            h1b = vlsum(h1(1,1,1,ie),nxyz)/nxyz
            h2b = vlsum(h2(1,1,1,ie),nxyz)/nxyz
            k1 = ktype(ie,1,kfldfdm)
            k2 = ktype(ie,2,kfldfdm)
            k3 = ktype(ie,3,kfldfdm)
            vol = elsize(1,ie)*elsize(2,ie)*elsize(3,ie)
            vl1 = elsize(2,ie)*elsize(3,ie)/elsize(1,ie)
            vl2 = elsize(1,ie)*elsize(3,ie)/elsize(2,ie)
            vl3 = elsize(1,ie)*elsize(2,ie)/elsize(3,ie)
            do i3=1,lz1
            do i2=1,ly1
            do i1=1,lx1
               den = h1b*(vl1*dd(i1,k1) + vl2*dd(i2,k2) + vl3*dd(i3,k3))
     $             + h2b*vol
               if (ifbhalf) den = den/vol
               if (den.ne.0) then
                  d(i1,i2,i3,ie) = 1./den
               else
                  d(i1,i2,i3,ie) = 0.
c
c                 write(6,3) 'd=0:'
c    $                 ,h1(i1,i2,i3,ie),dd(i1,k1),dd(i2,k2),dd(i3,k3)
c    $                 ,i1,i2,i3,ie,kfldfdm,k1,k2,k3
    3             format(a4,1p4e12.4,8i8)
c
               endif
            enddo
            enddo
            enddo
         enddo
      else
         do ie=1,nel
            if (ifaxis) then
               h1b = vlsc2(h1(1,1,1,ie),ym1(1,1,1,ie),nxyz)/nxyz
               h2b = vlsc2(h2(1,1,1,ie),ym1(1,1,1,ie),nxyz)/nxyz
            else
               h1b = vlsum(h1(1,1,1,ie),nxyz)/nxyz
               h2b = vlsum(h2(1,1,1,ie),nxyz)/nxyz
            endif
            k1 = ktype(ie,1,kfldfdm)
            k2 = ktype(ie,2,kfldfdm)
            vol = elsize(1,ie)*elsize(2,ie)
            vl1 = elsize(2,ie)/elsize(1,ie)
            vl2 = elsize(1,ie)/elsize(2,ie)
            i3=1
            do i2=1,ly1
            do i1=1,lx1
               den = h1b*( vl1*dd(i1,k1) + vl2*dd(i2,k2) )
     $             + h2b*vol
               if (ifbhalf) den = den/vol
               if (den.ne.0) then
                  d(i1,i2,i3,ie) = 1./den
c                 write(6,3) 'dn0:'
c    $                 ,d(i1,i2,i3,ie),dd(i1,k1),dd(i2,k2)
c    $                 ,i1,i2,i3,ie,kfldfdm,k1,k2
               else
                  d(i1,i2,i3,ie) = 0.
c                 write(6,3) 'd=0:'
c    $                 ,h1(i1,i2,i3,ie),dd(i1,k1),dd(i2,k2)
c    $                 ,i1,i2,i3,ie,kfldfdm,k1,k2
    2             format(a4,1p3e12.4,8i8)
               endif
c           write(6,1) ie,i1,i2,k1,k2,'d:',d(i1,i2,i3,ie),vol,vl1,vl2
c   1       format(5i3,2x,a2,1p4e12.4)
            enddo
            enddo
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_fdm_prec_h1A
      include 'SIZE'
c
      call set_fdm_prec_h1A_gen
      call set_fdm_prec_h1A_els
c
      return
      end
c-----------------------------------------------------------------------
      subroutine generalev(a,b,lam,n,w)
c
c     Solve the generalized eigenvalue problem  A x = lam B x
c
c     A -- symm.
c     B -- symm., pos. definite
c
c     "SIZE" is included here only to deduce WDSIZE, the working
c     precision, in bytes, so as to know whether dsygv or ssygv
c     should be called.
c
      include 'SIZE'
      include 'PARALLEL'
c
      real a(n,n),b(n,n),lam(n),w(n,n)
      real aa(100),bb(100)
c
      parameter (lbw=4*lx1*ly1*lz1*lelv)
      common /bigw/ bw(lbw)
c
      lw = n*n
c     write(6,*) 'in generalev, =',info,n,ninf
c
c     call outmat2(a,n,n,n,'aa  ')
c     call outmat2(b,n,n,n,'bb  ')
c
      call copy(aa,a,100)
      call copy(bb,b,100)
c
      call dsygv(1,'V','U',n,a,n,b,n,lam,bw,lbw,info)
c
c     call outmat2(a,n,n,n,'Aeig')
c     call outmat2(lam,1,n,n,'Deig')
c
      if (info.ne.0) then
c
         if (nid.eq.0) then
            call outmat2(aa ,n,n,n,'aa  ')
            call outmat2(bb ,n,n,n,'bb  ')
            call outmat2(a  ,n,n,n,'Aeig')
            call outmat2(lam,1,n,n,'Deig')
         endif
c
         ninf = n-info
         write(6,*) 'Error in generalev, info=',info,n,ninf
         call exitt
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat2(a,m,n,k,name)
      include 'SIZE'
      real a(m,n)
      character*4 name
c
      n2 = min(n,8)
      write(6,2) nid,name,m,n,k
      do i=1,m
         write(6,1) nid,name,(a(i,j),j=1,n2)
      enddo
c   1 format(i3,1x,a4,16f6.2)
    1 format(i3,1x,a4,1p8e14.5)
    2 format(/,'Matrix: ',i3,1x,a4,3i8)
      return
      end
c-----------------------------------------------------------------------
      subroutine rescale_abhalf (a,b,w,n)
      real a(n,n),b(n,n),w(n)
c
c             -1/2      -1/2
c     Set A = B    A  B
c
c
c     NOTE:   B is *diagonal*
c
c
      do i=1,n
         w(i) = 1./sqrt(b(i,i))
      enddo
c
      do j=1,n
      do i=1,n
         a(i,j) = a(i,j)*w(i)*w(j)
      enddo
      enddo
c
c     duh....  don't forget to change B ...  duh...
c
      call ident(b,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine hmholtz_dg(name,u,rhs,h1,h2,mask,tol,maxit)
      include 'SIZE'
      include 'CTIMER'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'
C
      character      name*4
      real           u    (lx1,ly1,lz1,1)
      real           rhs  (lx1,ly1,lz1,1)
      real           h1   (lx1,ly1,lz1,1)
      real           h2   (lx1,ly1,lz1,1)
      real           mask (lx1,ly1,lz1,1)

      icalld=icalld+1
      nhmhz=icalld
      etime1=dnekclock()

      if (ifield.eq.2) then
         call cggo_dg (u,rhs,h1,h2,bintm1,mask,name,tol,maxit)
      else
         call cggo_dg (u,rhs,h1,h2,binvm1,mask,name,tol,maxit)
      endif

      thmhz=thmhz+(dnekclock()-etime1)

      return
      end
C
c=======================================================================
      subroutine cggo_dg(x,f,h1,h2,binv,mask,name,tin,maxit)
C-------------------------------------------------------------------------
C
C     Solve the Helmholtz equation, H*U = RHS,
C     using preconditioned conjugate gradient iteration.
C     Preconditioner: diag(H).
C
C------------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'


      real x(1),f(1),h1(1),h2(1),binv(1),mask(1)
      parameter        (lg=lx1*ly1*lz1*lelt)
      common /scrcg/ d (lg) , scalar(2)
      common /scrmg/ r (lg) , w (lg) , p (lg) , z (lg)

      parameter (maxcg=900)
      common /tdarray/ diagt(maxcg),upper(maxcg)
      common /iterhm/ niterhm
      character*4 name

      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
 
      common  /cprint/ ifprint, ifhzpc
      logical          ifprint, ifhzpc

      logical ifmcor


c **  zero out stuff for Lanczos eigenvalue estimator
      call rzero(diagt,maxcg)
      call rzero(upper,maxcg)

c     Initialization

      nxyz   = lx1*ly1*lz1
      nel    = nelv
      vol    = volvm1
      if (ifield.eq.2) nel=nelt
      if (ifield.eq.2) vol=voltm1
      n  = nel*nxyz

      tol=tin
      if (param(22).ne.0) tol=abs(param(22))
      niter = min(maxit,maxcg)
 
      imsh = ifield
      call setprec_dg(d,h1,h2,imsh,1) !  diag preconditioner
c     call invers2   (d,bm1,n) !  diag preconditioner
 
      call copy (r,f,n)
      call rzero(x,n)
      call rzero(p,n)
 
c     Check for non-trivial null-space
 
      ifmcor = .false.
      h2max = glmax(h2  ,n)
      skmin = glmin(mask,n)
      if (skmin.gt.0.and.h2max.eq.0) ifmcor = .true.
 
      if (ifmcor) then
         rmean = glsum(r,n)
         call cadd(r,rmean,n)
      endif
 
      krylov = 0
      rtz1=1.0
      niterhm = 0
      do 1000 iter=1,niter
 
c        call copy(z,r,n)   ! No     preconditioner
         call col3(z,r,d,n) ! Jacobi Preconditioner
 
         rtz2=rtz1
         scalar(1)=vlsc2 (z,r,n)
         scalar(2)=vlsc3 (r,r,binv,n)
         call gop(scalar,w,'+  ',2)
         rtz1=scalar(1)
         rbn2=sqrt(scalar(2)/vol)
         if (iter.eq.1) rbn0 = rbn2
         if (param(22).lt.0) tol=abs(param(22))*rbn0
 
         if (ifprint.and.nid.eq.0.and.param(74).ne.0) then
            write(6,3002) istep,iter,name,ifmcor,rbn2,TOL,h1(1),h2(1)
         endif
 
         if (rbn2.le.tol) then
            niter = iter-1
            if(nid.eq.0.and.((.not.ifhzpc).or.ifprint))
     $      write(6,3000) ISTEP,NAME,niter,RBN2,RBN0,tol
            go to 9999
         endif
 
         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0
         call add2s1 (p,z,beta,n)
         call hxdg   (w,p,h1,h2)
 
         rho0 = rho
         rho  = glsc2(w,p,n)
         alpha=rtz1/rho
         alphm=-alpha
         call add2s2(x,p ,alpha,n)
         call add2s2(r,w ,alphm,n)

c        Generate tridiagonal matrix for Lanczos scheme
         if (iter.eq.1) then
            krylov = krylov+1
            diagt(iter) = rho/rtz1
         elseif (iter.le.maxcg) then
            krylov = krylov+1
            diagt(iter)    = (beta**2 * rho0 + rho ) / rtz1
            upper(iter-1)  = -beta * rho0 / sqrt(rtz2 * rtz1)
         endif
 1000 continue
      niter = iter-1
c
      if (nid.eq.0) write (6,3001) istep,niter,name,rbn2,rbn0,tol
 3000 format(i9,4x,'hmh dg ',a4,': ',I6,1p6E13.4)
 3001 format(2i6,' **ERROR**: Failed in hmh_dg: ',a4,1p6E13.4)
 3002 format(i3,i6,' HMH dg ',a4,1x,l4,':',1p6E13.4)
 9999 continue
      niterhm = niter
      ifsolv = .false.
c
c
c     Call eigenvalue routine for Lanczos scheme:
c          two work arrays are req'd if you want to save "diag & upper"
c
c     if (iter.ge.3) then
c        niter = iter-1
c        call calc (diagt,upper,w,z,krylov,dmax,dmin)
c        cond = dmax/dmin
c        if (nid.eq.0) write(6,6) istep,cond,dmin,dmax,' lambda'
c     endif
c   6 format(i9,1p3e12.4,4x,a7)
c
c     if (n.gt.0) write(6,*) 'quit in cggo'
c     if (n.gt.0) call exitt
c     call exitt
      return
      end
c-----------------------------------------------------------------------
      subroutine outmax(a,m,n,name6,ie)
      real a(m,n)
      character*6 name6
c
      n18 = min(n,18)
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      do i=1,m
         write(6,6) ie,name6,(a(i,j),j=1,n18)
      enddo
    6 format(i3,1x,a6,18f7.2)
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine outmat4(a,l,m,n,nel,name6,ie)
      real a(l,m,n,nel)
      character*6 name6
c
      n18 = min(n,18)
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      do ll=1,l
      do k=1,nel
         write(6,*) ie,' matrix: ',name6,ll,k
         do j=1,4
            write(6,6) ie,name6,(a(ll,i,j,k),i=1,m)
         enddo
      enddo
      enddo
    6 format(i3,1x,a6,18f7.2)
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine ioutmat4(a,l,m,n,nel,name6,ie)
      integer a(l,m,n,nel)
      character*6 name6
c
      n18 = min(n,18)
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      do ll=1,l
      do k=1,nel
         write(6,*) ie,' matrix: ',name6,ll,k
         do j=1,n
            write(6,6) ie,name6,(a(ll,i,j,k),i=1,m)
         enddo
      enddo
      enddo
    6 format(i3,1x,a6,18i7)
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine ioutfld(a,m,n,nel,name6,ie)
      integer a(m,n,nel)
      character*6 name6

      n18 = min(n,18)
      write(6,*) 
      write(6,*) ie,' matrix: ',name6,m,n
      do j=1,n
         if (m.eq.3) write(6,3) ie,name6,((a(i,j,k),i=1,m),k=1,2)
         if (m.eq.4) write(6,4) ie,name6,((a(i,j,k),i=1,m),k=1,2)
         if (m.eq.5) write(6,5) ie,name6,((a(i,j,k),i=1,m),k=1,2)
         if (m.eq.6) write(6,6) ie,name6,((a(i,j,k),i=1,m),k=1,2)
      enddo
    3 format(i3,1x,a6,2(3i7,2x))
    4 format(i3,1x,a6,2(4i7,2x))
    5 format(i3,1x,a6,2(5i7,2x))
    6 format(i3,1x,a6,2(6i7,2x))
      write(6,*) 
      return
      end
c-----------------------------------------------------------------------
      subroutine gradr(ur,us,ut,u,Dr,Dst,Dtt,nr,ns,nt,if3d)
c
c     Output: ur,us,ut         Input: u
c
      real ur(nr,ns,nt),us(nr,ns,nt),ut(nr,ns,nt)
      real u (nr,ns,nt)
      real Dr(nr,nr),Dst(ns,ns),Dtt(nt,nt)
c
      logical if3d
c
      nst = ns*nt
      nrs = nr*ns
c
      if (if3d) then
         call mxm(Dr,nr,u,nr,ur,nst)
         do k=1,nt
            call mxm(u(1,1,k),nr,Dst,ns,us(1,1,k),nt)
         enddo
         call mxm(u,nrs,Dtt,nt,ut,nt)
      else
         call mxm(Dr,nr,u  ,nr,ur,ns)
         call mxm(u ,nr,Dst,ns,us,ns)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gradrta(u,ur,us,ut,Drt,Ds,Dt,nr,ns,nt,if3d)
c
c                      T      T      T
c     Output: u = u + D ur + D us + D ut         Input: ur,us,ut
c                      r      s      t
c
      real u (nr,ns,nt)
      real ur(nr,ns,nt),us(nr,ns,nt),ut(nr,ns,nt)
      real Dr(nr,nr),Dst(ns,ns),Dtt(nt,nt)
c
      logical if3d
c
      nst = ns*nt
      nrs = nr*ns
c
      if (if3d) then
         call mxma(Drt,nr,ur,nr,u,nst)
         do k=1,nt
            call mxma(us(1,1,k),nr,Ds,ns,u(1,1,k),nt)
         enddo
         call mxma(ut,nrs,Dt,nt,u,nt)
      else
         call mxma(Drt,nr,ur,nr,u,ns)
         call mxma(us ,nr,Ds,ns,u,ns)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine face_diff (u,d,gsh_loc,w) ! difference: e_f - e'_f

      include 'SIZE'
      include 'TOPOL'
      include 'PARALLEL'

      integer d,gsh_loc
      real u(lx1*lz1*2*ldim*lelt,2) 
      real w(lx1*lz1*2*ldim*lelt,2) 

      n = 2*ldim*lx1*lz1*nelt

      do j=1,d
         do i=1,n
            w(i,j) = u(i,j)
         enddo
         call fgslib_gs_op (gsh_loc,w(1,j),1,1,0)  ! 1 ==> +

         do i=1,n
            u(i,j) = 2*u(i,j)-w(i,j)
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine setprec_dg (d,h1,h2,imsh,isd)
C-------------------------------------------------------------------
C
C     Generate diagonal preconditioner for the DG Helmholtz operator.
C
C-------------------------------------------------------------------
      include 'SIZE'
      include 'TOTAL'
      real            d(lx1,ly1,lz1,1)
      common /fastmd/ ifdfrm(lelt), iffast(lelt), ifh2, ifsolv
      logical ifdfrm, iffast, ifh2, ifsolv
      real            h1(lx1*ly1*lz1,1), h2(lx1*ly1*lz1,1)
      real ysm1(ly1)
      integer e,f,pf

      nel=nelt
      if (imsh.eq.1) nel=nelv

      call dsset(lx1,ly1,lz1)

      n     = nel*lx1*ly1*lz1
      nxyz  = lx1*ly1*lz1
      nface = 2*ldim

      do 1000 e=1,nel

        call rzero(d(1,1,1,e),nxyz)

        if (ldim.eq.3) then

         do 320 iz=1,lz1
         do 320 iy=1,ly1
         do 320 ix=1,lx1
         do 320 iq=1,lx1
           d(ix,iy,iz,e) = d(ix,iy,iz,e)
     $                   + g1m1(iq,iy,iz,e) * dxm1(iq,ix)**2
     $                   + g2m1(ix,iq,iz,e) * dxm1(iq,iy)**2
     $                   + g3m1(ix,iy,iq,e) * dxm1(iq,iz)**2
  320    continue
c
c        Add cross terms if element is deformed.
c
         if (ifdfrm(e)) then

           do i2=1,ly1,ly1-1  
           do i1=1,lx1,lx1-1
              d(1,i1,i2,e) = d(1,i1,i2,e)
     $            + g4m1(1,i1,i2,e) * dxtm1(1,1)*dytm1(i1,i1)
     $            + g5m1(1,i1,i2,e) * dxtm1(1,1)*dztm1(i2,i2)
              d(lx1,i1,i2,e) = d(lx1,i1,i2,e)
     $            + g4m1(lx1,i1,i2,e) * dxtm1(lx1,lx1)*dytm1(i1,i1)
     $            + g5m1(lx1,i1,i2,e) * dxtm1(lx1,lx1)*dztm1(i2,i2)
              d(i1,1,i2,e) = d(i1,1,i2,e)
     $            + g4m1(i1,1,i2,e) * dytm1(1,1)*dxtm1(i1,i1)
     $            + g6m1(i1,1,i2,e) * dytm1(1,1)*dztm1(i2,i2)
              d(i1,ly1,i2,e) = d(i1,ly1,i2,e)
     $            + g4m1(i1,ly1,i2,e) * dytm1(ly1,ly1)*dxtm1(i1,i1)
     $            + g6m1(i1,ly1,i2,e) * dytm1(ly1,ly1)*dztm1(i2,i2)
              d(i1,i2,1,e) = d(i1,i2,1,e)
     $            + g5m1(i1,i2,1,e) * dztm1(1,1)*dxtm1(i1,i1)
     $            + g6m1(i1,i2,1,e) * dztm1(1,1)*dytm1(i2,i2)
              d(i1,i2,lz1,e) = d(i1,i2,lz1,e)
     $            + g5m1(i1,i2,lz1,e) * dztm1(lz1,lz1)*dxtm1(i1,i1)
     $            + g6m1(i1,i2,lz1,e) * dztm1(lz1,lz1)*dytm1(i2,i2)

           enddo
           enddo
         endif

        else  ! 2d

         iz=1
         if (ifaxis) call setaxdy ( ifrzer(e) )

         do 220 iy=1,ly1
         do 220 ix=1,lx1
         do 220 iq=1,lx1
           d(ix,iy,iz,e) = d(ix,iy,iz,e)
     $                   + g1m1(iq,iy,iz,e) * dxm1(iq,ix)**2
     $                   + g2m1(ix,iq,iz,e) * dxm1(iq,iy)**2
  220    continue
c

         if (ifdfrm(e)) then

           do i1=1,ly1,ly1-1
              d(1,i1,iz,e) = d(1,i1,iz,e)
     $            + g4m1(1,i1,iz,e) * dxm1(1,1)*dym1(i1,i1)
              d(lx1,i1,iz,e) = d(lx1,i1,iz,e)
     $            + g4m1(lx1,i1,iz,e) * dxm1(lx1,lx1)*dym1(i1,i1)
              d(i1,1,iz,e) = d(i1,1,iz,e)
     $            + g4m1(i1,1,iz,e) * dym1(1,1)*dxm1(i1,i1)
              d(i1,ly1,iz,e) = d(i1,ly1,iz,e)
     $            + g4m1(i1,ly1,iz,e) * dym1(ly1,ly1)*dxm1(i1,i1)
           enddo
         endif

        endif

c       Here, we add DG surface terms (11/06/16)

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
              d(j1,j2,1,e) = d(j1,j2,1,e) + etalph(i,f,e)
           enddo
           enddo
        enddo

        i=0
        nx=lx1
        if (ldim.eq.3) then
         do i2=1,ly1
         do i1=1,lx1
           i=i+1
           d( 1,i1,i2,e)=d( 1,i1,i2,e)-2*fw(4,e)*unr(i,4,e)*dxm1( 1, 1)
           d(nx,i1,i2,e)=d(nx,i1,i2,e)-2*fw(2,e)*unr(i,2,e)*dxm1(nx,nx)
           d(i1, 1,i2,e)=d(i1, 1,i2,e)-2*fw(1,e)*uns(i,1,e)*dym1( 1, 1)
           d(i1,nx,i2,e)=d(i1,nx,i2,e)-2*fw(3,e)*uns(i,3,e)*dym1(nx,nx)
           d(i1,i2, 1,e)=d(i1,i2, 1,e)-2*fw(5,e)*unt(i,5,e)*dzm1( 1, 1)
           d(i1,i2,nx,e)=d(i1,i2,nx,e)-2*fw(6,e)*unt(i,6,e)*dzm1(nx,nx)
         enddo
         enddo
        else  ! 2D
         do i1=1,lx1
           i=i+1
           d( 1,i1,1,e)=d( 1,i1,1,e)-2*fw(4,e)*unr(i,4,e)*dxm1( 1, 1)
           d(nx,i1,1,e)=d(nx,i1,1,e)-2*fw(2,e)*unr(i,2,e)*dxm1(nx,nx)
           d(i1, 1,1,e)=d(i1, 1,1,e)-2*fw(1,e)*uns(i,1,e)*dym1( 1, 1)
           d(i1,nx,1,e)=d(i1,nx,1,e)-2*fw(3,e)*uns(i,3,e)*dym1(nx,nx)
         enddo
        endif

        do i=1,nxyz
           d(i,1,1,e)=1./(d(i,1,1,e)*h1(i,e)+h2(i,e)*bm1(i,1,1,e))
        enddo

 1000 continue ! element loop

c     If axisymmetric, add a diagonal term in the radial direction (ISD=2)

      if (ifaxis.and.(isd.eq.2)) then
         call invcol1 (d,n)
         do 1200 e=1,nel

            if (ifrzer(e)) call mxm(ym1(1,1,1,e),lx1,datm1,ly1,ysm1,1)

            k=0
            do 1190 j=1,ly1
            do 1190 i=1,lx1
               k=k+1
               if (ym1(i,j,1,e).ne.0.) then
                  term1 = bm1(i,j,1,e)/ym1(i,j,1,e)**2
                  if (ifrzer(e)) then
                     term2 =  wxm1(i)*wam1(1)*dam1(1,j)
     $                       *jacm1(i,1,1,e)/ysm1(i)
                  else
                     term2 = 0.
                  endif
                  d(i,j,1,e) = d(i,j,1,e)
     $                             + h1(k,e)*(term1+term2)
               endif
 1190       continue
 1200    continue

         call invcol1 (d,n)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine hxdg_surfa(au,u,h1,h2)

c     Helmholtz matrix-vector product: Au = Au + surface term 

      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1)

      real au(lx1,ly1,lz1,lelt),u(lx1,ly1,lz1,lelt)
      real h1(lx1,ly1,lz1,lelt),h2(1)

      common /ytmp9/ qr(lx1,ly1,lz1),qs(lx1,ly1,lz1),qt(lx1,ly1,lz1)

      integer e,f,pf


      call dsset(lx1,ly1,lz1)
      nface = 2*ldim
      n     = lx1*ly1*lz1*nelfld(ifield)

      do e=1,nelfld(ifield)
         iflag=0
         do f=1,nface
           if (fw(f,e).gt.0.6) iflag=1
         enddo
         if (iflag.gt.0) then

          if (ifaxis) call setaxdy(ifrzer(e))

          do i=1,lxyz
            qr(i,1,1)=0
            qs(i,1,1)=0
            qt(i,1,1)=0
          enddo

          do f=1,nface
           if (fw(f,e).gt.0.6) then
             pf     = eface1(f)
             js1    = skpdat(1,pf)
             jf1    = skpdat(2,pf)
             jskip1 = skpdat(3,pf)
             js2    = skpdat(4,pf)
             jf2    = skpdat(5,pf)
             jskip2 = skpdat(6,pf)

             fwtbc=1.
             if (cbc(f,e,ifield).eq.'O  '.or.cbc(f,e,ifield).eq.'I  ') 
     $          fwtbc=0

             i = 0
             do j2=js2,jf2,jskip2
             do j1=js1,jf1,jskip1
                i = i+1
                fwt = fwtbc *       h1(j1,j2,1,e)*u(j1,j2,1,e)
                et1 = etalph(i,f,e)*h1(j1,j2,1,e)*u(j1,j2,1,e)
                qr(j1,j2,1) = qr(j1,j2,1)-fwt*unr(i,f,e)
                qs(j1,j2,1) = qs(j1,j2,1)-fwt*uns(i,f,e)
                qt(j1,j2,1) = qt(j1,j2,1)-fwt*unt(i,f,e)
                au(j1,j2,1,e) = au(j1,j2,1,e)+et1*fwtbc
             enddo
             enddo
           endif
          enddo

          call gradrta(au(1,1,1,e),qr,qs,qt        ! NOTE FIX in gradr()! 3D!
     $        ,dxtm1,dym1,dzm1,lx1,ly1,lz1,if3d)
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine hxdg (au,u,h1,h2)

c     Helmholtz matrix-vector product: Au = h1*[A]u + h2*[B]u

      include 'SIZE'
      include 'TOTAL'

      parameter(lxyz=lx1*ly1*lz1)
      real au(lx1,ly1,lz1,1),u(lx1,ly1,lz1,1),h1(lx1,ly1,lz1,1),h2(1)

      common /ctmp0/ w(2*lx1*lz1*2*ldim*lelt)
      common /ctmp1/ ur(lx1,ly1,lz1,lelt),us(lx1,ly1,lz1,lelt)
     $                                   ,ut(lx1,ly1,lz1,lelt)
      common /ytmp9/ qr(lx1,ly1,lz1),qs(lx1,ly1,lz1),qt(lx1,ly1,lz1)
      common /ytmp0/ uf(lx1*lz1,2*ldim,lelt,2)

      integer e,f,pf

      n     = lx1*ly1*lz1*nelfld(ifield)
      nface = 2*ldim

      call dsset(lx1,ly1,lz1)

      call col4(au,h2,bm1,u,n)                     ! au = h2 B u

      do e=1,nelfld(ifield)

         if (ifaxis) call setaxdy(ifrzer(e))

         call gradr(ur(1,1,1,e),us(1,1,1,e),ut(1,1,1,e) ! NOTE FIX in gradr()! 3D!
     $             ,u (1,1,1,e),dxm1,dytm1,dztm1,lx1,ly1,lz1,if3d)

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
c             Normally, we'd store this as a 2-vector: uf(2,...)
              uf(i,f,e,1) = u(j1,j2,1,e)*h1(j1,j2,1,e)
              uf(i,f,e,2) = (unr(i,f,e)*ur(j1,j2,1,e)
     $                    +  uns(i,f,e)*us(j1,j2,1,e)
     $                    +  unt(i,f,e)*ut(j1,j2,1,e))*h1(j1,j2,1,e)
           enddo
           enddo
         enddo
      enddo

      call face_diff (uf,2,dg_hndlx,w) ! difference: e_f - e'_f

      do e=1,nelfld(ifield)
         if (ifaxis) call setaxdy(ifrzer(e))
         do i=1,lxyz
           qr(i,1,1) = (g1m1(i,1,1,e)*ur(i,1,1,e)
     $                 +g4m1(i,1,1,e)*us(i,1,1,e)
     $                 +g5m1(i,1,1,e)*ut(i,1,1,e))*h1(i,1,1,e)
           qs(i,1,1) = (g4m1(i,1,1,e)*ur(i,1,1,e)
     $                 +g2m1(i,1,1,e)*us(i,1,1,e)
     $                 +g6m1(i,1,1,e)*ut(i,1,1,e))*h1(i,1,1,e)
           qt(i,1,1) = (g5m1(i,1,1,e)*ur(i,1,1,e)
     $                 +g6m1(i,1,1,e)*us(i,1,1,e)
     $                 +g3m1(i,1,1,e)*ut(i,1,1,e))*h1(i,1,1,e)
         enddo

         do f=1,nface

           fwtbc=1.
           if (cbc(f,e,ifield).eq.'O  '.or.cbc(f,e,ifield).eq.'I  ') 
     $        fwtbc=0
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
              fwt = fw(f,e)*fwtbc
              qr(j1,j2,1)=qr(j1,j2,1)-fwt*unr(i,f,e)*uf(i,f,e,1)
              qs(j1,j2,1)=qs(j1,j2,1)-fwt*uns(i,f,e)*uf(i,f,e,1)
              qt(j1,j2,1)=qt(j1,j2,1)-fwt*unt(i,f,e)*uf(i,f,e,1)
              au(j1,j2,1,e) = au(j1,j2,1,e)-fwt*uf(i,f,e,2)
     $                          + etalph(i,f,e)*uf(i,f,e,1)*fwtbc
           enddo
           enddo

         enddo

         call gradrta(au(1,1,1,e),qr,qs,qt        ! NOTE FIX in gradr()! 3D!
     $      ,dxtm1,dym1,dzm1,lx1,ly1,lz1,if3d)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine hmh_flex_cg(res,h1,h2,wt,iter)

c     Solve the Helmholtz equation by right-preconditioned 
c     GMRES iteration.

     
      include 'SIZE'
      include 'TOTAL'
      include 'FDMH1'
      include 'GMRES'
      common  /ctolpr/ divex
      common  /cprint/ ifprint
      logical          ifprint
      real             res  (lx1*ly1*lz1*lelv)
      real             h1   (lx1,ly1,lz1,lelv)
      real             h2   (lx1,ly1,lz1,lelv)
      real             wt   (lx1,ly1,lz1,lelv)

      parameter (lt=lx1*ly1*lz1*lelt)
      common /scrcg/ r(lt),z(lt),p(lt),w(lt)
      common /scrmg/ r1(lt)

      common /cgmres1/ y(lgmres)
      common /ctmp0/   wk1(lgmres),wk2(lgmres)
      real alpha, l, temp
      integer outer

      logical iflag,if_hyb
      save    iflag,if_hyb
c     data    iflag,if_hyb  /.false. , .true. /
      data    iflag,if_hyb  /.false. , .false. /
      real    norm_fac
      save    norm_fac

      real*8 etime1,dnekclock

      n = lx1*ly1*lz1*nelv

            div0 = gamma_gmres(1)*norm_fac

      etime1 = dnekclock()
      etime_p = 0.
      divex = 0.
      maxit = iter
      iter  = 0


      call chktcg1(tolps,res,h1,h2,pmask,vmult,1,1)
      if (param(21).gt.0.and.tolps.gt.abs(param(21))) 
     $   tolps = abs(param(21))
      if (istep.eq.0) tolps = 1.e-4
      tolpss = tolps

      iconv = 0
      call copy (r,res,n)    ! Residual
      call rzero(r1 ,n)      ! Lagged residual for flexible CG
      call rzero(p,n)        ! Search direction
      call rzero(res,n)      ! Solution vector
      rho1 = 1
                                               !            ______
      div0  = sqrt(glsc3(r,wt,r,n)/volvm1) ! gamma  = \/ (r,r) 

      if (param(21).lt.0) tolpss=abs(param(21))*div0


      do k=1,maxit

         if (param(40).ge.0 .and. param(40).le.2) then
            call h1mg_solve(z,r,if_hyb) ! z  = M   w
         else if (param(40).eq.3) then
            call fem_amg_solve(z,r)
         endif

         call sub2(r1,r,n)
         rho0 = rho1
         rho1 =  glsc3(z,wt,r,n)   ! Inner product weighted by multiplicity
         rho2 = -glsc3(z,wt,r1,n)  ! Inner product weighted by multiplicity
         beta = rho2/rho0          ! Flexible GMRES

         call copy(r1,r,n)         ! Save prior residual
         call add2s1(p,z,beta,n)

         call ax(w,p,h1,h2,n)      ! w = A p
         den = glsc3(w,wt,p,n)
         alpha = rho1/den
         rnorm = 0.0
         do i = 1,n
            res(i) = res(i) + alpha*p(i)
            r(i)   = r(i)   - alpha*w(i)
            rnorm  = rnorm  + r(i)*r(i)*wt(i,1,1,1)
         enddo
         call gop(rnorm,temp,'+  ',1)

c         rnorm = sqrt(glsc3(r,wt,r,n)/volvm1) ! gamma  = \/ (r,r) 
         rnorm = sqrt(rnorm/volvm1) ! gamma  = \/ (r,r) 
         ratio = rnorm/div0

         iter=iter+1
         if (ifprint.and.nio.eq.0) 
     $         write (6,66) iter,tolpss,rnorm,div0,ratio,istep
   66    format(i5,1p4e12.5,i8,' Divergence')

         if (rnorm .lt. tolpss) goto 900  !converged

      enddo

  900 iconv = 1
      divex = rnorm
      call ortho   (res) ! Orthogonalize wrt null space, if present
      etime1 = dnekclock()-etime1
      if (nio.eq.0) write(6,9999) istep,iter,divex,div0,tolpss,etime_p,
     &                            etime1,if_hyb
 9999 format(4x,i7,'  PRES cgflex',4x,i5,1p5e13.4,1x,l4)

      if (outer.le.2) if_hyb = .false.

      return
      end
c-----------------------------------------------------------------------
