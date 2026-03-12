c-----------------------------------------------------------------------
      SUBROUTINE SETLOG(ifecho)
C                                                                     
C     Subroutine to initialize logical flags
C                                                                     
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'CTIMER'
      INCLUDE 'ADJOINT'

      logical ifecho

      COMMON  /CPRINT/ IFPRINT
C
      common  /nekcb/ cb
      CHARACTER CB*3
      LOGICAL IFALGN,IFNORX,IFNORY,IFNORZ,IFPRINT
C
      NFACE  = 2*ldim
C
      IFPRINT = .TRUE.
      IFVCOR  = .TRUE.
      IFGEOM  = .FALSE.
      IFINTQ  = .FALSE.
      IFSURT  = .FALSE.
      IFWCNO  = .FALSE.
      DO 10 IFIELD=1,NFIELD
         IFNONL(IFIELD) = .FALSE.
 10   CONTINUE
C
      CALL LFALSE (IFEPPM,NFACE*NELV)
      CALL LFALSE (IFQINP,NFACE*NELV)
C
      IF (IFMVBD) THEN
         IFGEOM = .TRUE.
         IF ( IFFLOW .AND. .NOT.IFNAV )       IFWCNO          = .TRUE.
         IF ( IFMELT .AND. .NOT.IFFLOW )      IFWCNO          = .TRUE.
      ENDIF
C
      IF (IFFLOW) THEN
         IERR = 0
         IFIELD = 1
         DO 100 IEL=1,NELV
         DO 100 IFC=1,NFACE
            CB = CBC(IFC,IEL,IFIELD)
            CALL CHKNORD (IFALGN,IFNORX,IFNORY,IFNORZ,IFC,IEL)
            CALL CHKCBC  (CB,IEL,IFC,IFALGN,IERR)
            IF  (CB.EQ.'O  ' .OR. CB.EQ.'o  ' .OR.
     $           CB.EQ.'ON ' .OR. CB.EQ.'on ' .OR.
     $           CB.EQ.'S  ' .OR. CB.EQ.'s  ' .OR.
     $           CB.EQ.'SL ' .OR. CB.EQ.'sl ' .OR.
     $           CB.EQ.'MM ' .OR. CB.EQ.'mm ' .OR.
     $           CB.EQ.'MS ' .OR. CB.EQ.'ms ')  THEN
                                              IFVCOR          = .FALSE.
                                              IFEPPM(IFC,IEL) = .TRUE.
            ENDIF
            IF  (CB.EQ.'VL ' .OR. CB.EQ.'vl ' .OR.
     $           CB.EQ.'WSL' .OR. CB.EQ.'wsl' .OR.
     $           CB.EQ.'SL ' .OR. CB.EQ.'sl ' .OR.
     $           CB.EQ.'SHL' .OR. CB.EQ.'shl' .OR.
     $           CB.EQ.'MM ' .OR. CB.EQ.'mm ' .OR.
     $           CB.EQ.'MS ' .OR. CB.EQ.'ms ' .OR.
     $           CB.EQ.'O  ' .OR. CB.EQ.'o  ' .OR.
     $           CB.EQ.'ON ' .OR. CB.EQ.'on ')  THEN
                                              IFQINP(IFC,IEL) = .TRUE.
            ENDIF
            IF  (CB.EQ.'MS ' .OR. CB.EQ.'ms ' .OR.
     $           CB.EQ.'MM ' .OR. CB.EQ.'mm ' .OR.
     $           CB.EQ.'MSI' .OR. CB.EQ.'msi' ) THEN
                                              IFSURT          = .TRUE.
            ENDIF
  100    CONTINUE

         ierr = iglsum(ierr,1)
         if (ierr.gt.0) call exitt 
      ENDIF
C
      IF (IFHEAT) THEN
C
         DO 250 IFIELD=2,NFIELD
         DO 250 IEL=1,NELFLD(IFIELD)
         DO 250 IFC=1,NFACE
            CB=CBC(IFC,IEL,IFIELD)
            IF  (CB.EQ.'r  ' .OR. CB.EQ.'R  ') THEN
                                              IFNONL(IFIELD)  = .TRUE.
            ENDIF
  250    CONTINUE
C
      ENDIF

      if (ifmhd) call set_ifbcor
C
C     Establish global consistency of LOGICALS amongst all processors.
C
      CALL GLLOG(IFVCOR , .FALSE.)
      CALL GLLOG(IFSURT , .TRUE. )
      CALL GLLOG(IFWCNO , .TRUE. )
      DO 400 IFIELD=2,NFIELD
         CALL GLLOG(IFNONL(IFIELD),.TRUE.)
  400 CONTINUE
C
      IF (NIO.EQ.0 .AND. ifecho) THEN
         WRITE (6,*) 'IFTRAN    =',IFTRAN
         WRITE (6,*) 'IFFLOW    =',IFFLOW
         WRITE (6,*) 'IFHEAT    =',IFHEAT
         WRITE (6,*) 'IFSPLIT   =',IFSPLIT
         WRITE (6,*) 'IFLOMACH  =',IFLOMACH
         WRITE (6,*) 'IFUSERVP  =',IFUSERVP
         WRITE (6,*) 'IFUSERMV  =',IFUSERMV
         WRITE (6,*) 'IFPERT    =',IFPERT
         WRITE (6,*) 'IFADJ     =',IFADJ
         WRITE (6,*) 'IFSTRS    =',IFSTRS
         WRITE (6,*) 'IFCHAR    =',IFCHAR
         WRITE (6,*) 'IFCYCLIC  =',IFCYCLIC
         WRITE (6,*) 'IFAXIS    =',IFAXIS
         WRITE (6,*) 'IFMVBD    =',IFMVBD
         WRITE (6,*) 'IFMELT    =',IFMELT
         WRITE (6,*) 'IFNEKNEK  =',IFNEKNEK
         WRITE (6,*) 'IFNEKNEKC =',IFNEKNEKC
         WRITE (6,*) 'IFSYNC    =',IFSYNC
         WRITE (6,*) '  '
         WRITE (6,*) 'IFVCOR    =',IFVCOR
         WRITE (6,*) 'IFINTQ    =',IFINTQ
         WRITE (6,*) 'IFGEOM    =',IFGEOM
         WRITE (6,*) 'IFSURT    =',IFSURT
         WRITE (6,*) 'IFWCNO    =',IFWCNO

         DO 500 IFIELD=1,NFIELD
            WRITE (6,*) '  '
            WRITE (6,*) 'IFTMSH for field',IFIELD,'   = ',IFTMSH(IFIELD)
            WRITE (6,*) 'IFADVC for field',IFIELD,'   = ',IFADVC(IFIELD)
            WRITE (6,*) 'IFNONL for field',IFIELD,'   = ',IFNONL(IFIELD)
 500     CONTINUE
         WRITE (6,*) '  '
         if (param(99).gt.0) write(6,*) 'Dealiasing enabled, nxd=', nxd
      ENDIF
C
      RETURN
      END
C
c-----------------------------------------------------------------------
      SUBROUTINE SETRZER
C-------------------------------------------------------------------
C
C     Check for axisymmetric case.
C     Are some of the elements close to the axis?
C
C-------------------------------------------------------------------
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
C
C     Single or double precision???
C
      DELTA = 1.E-9
      X     = 1.+DELTA
      Y     = 1.
      DIFF  = ABS(X-Y)
      IF (DIFF.EQ.0.) EPS = 1.E-7
      IF (DIFF.GT.0.) EPS = 1.E-14
      eps1 = 1.e-6 ! for prenek mesh in real*4
C
      DO 100 IEL=1,NELT
         IFRZER(IEL) = .FALSE.
         IF (IFAXIS) THEN
            NVERT = 0
            DO 10 IC=1,4
               IF(ABS(YC(IC,IEL)).LT.EPS1) THEN
                  NVERT = NVERT+1
                  YC(IC,IEL) = 0.0  ! exactly on the axis
               ENDIF
 10         CONTINUE
         ENDIF
         IEDGE = 1
         IF ((NVERT.EQ.2).AND.(CCURVE(IEDGE,IEL).EQ.' '))
     $       IFRZER(IEL) = .TRUE.
 100  CONTINUE
      RETURN
      END
C
c-----------------------------------------------------------------------
      SUBROUTINE CHKNORD (IFALGN,IFNORX,IFNORY,IFNORZ,IFC,IEL)
C
C      Check direction of normal of an element face for
C      alignment with the X, Y, or Z axis.
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
C
      LOGICAL IFALGN,IFNORX,IFNORY,IFNORZ
C
      SUMX    = 0.0
      SUMY    = 0.0
      SUMZ    = 0.0
      TOLNOR  = 1.0e-3
      IFALGN  = .FALSE.
      IFNORX  = .FALSE.
      IFNORY  = .FALSE.
      IFNORZ  = .FALSE.
C
      IF (ldim.EQ.2) THEN
C
         NCPF = lx1
         DO 100 IX=1,lx1
            SUMX = SUMX + ABS( ABS(UNX(IX,1,IFC,IEL)) - 1.0 )
            SUMY = SUMY + ABS( ABS(UNY(IX,1,IFC,IEL)) - 1.0 )
  100    CONTINUE
         SUMX = SUMX / NCPF
         SUMY = SUMY / NCPF
         IF ( SUMX.LT.TOLNOR ) THEN
            IFNORX  = .TRUE.
            IFALGN = .TRUE.
         ENDIF
         IF ( SUMY.LT.TOLNOR ) THEN
            IFNORY  = .TRUE.
            IFALGN = .TRUE.
         ENDIF
C
      ELSE
C
         NCPF = lx1*lx1
         DO 200 IX=1,lx1
         DO 200 IY=1,ly1
            SUMX = SUMX + ABS( ABS(UNX(IX,IY,IFC,IEL)) - 1.0 )
            SUMY = SUMY + ABS( ABS(UNY(IX,IY,IFC,IEL)) - 1.0 )
            SUMZ = SUMZ + ABS( ABS(UNZ(IX,IY,IFC,IEL)) - 1.0 )
  200    CONTINUE
         SUMX = SUMX / NCPF
         SUMY = SUMY / NCPF
         SUMZ = SUMZ / NCPF
         IF ( SUMX.LT.TOLNOR ) THEN
            IFNORX  = .TRUE.
            IFALGN = .TRUE.
         ENDIF
         IF ( SUMY.LT.TOLNOR ) THEN
            IFNORY  = .TRUE.
            IFALGN = .TRUE.
         ENDIF
         IF ( SUMZ.LT.TOLNOR ) THEN
            IFNORZ  = .TRUE.
            IFALGN = .TRUE.
         ENDIF
C
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHKAXCB
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      CHARACTER CB*3
C
      IFLD  = 1
      NFACE = 2*ldim
C
      DO 100 IEL=1,NELV
      DO 100 IFC=1,NFACE
         CB = CBC(IFC,IEL,IFLD)
         IF  (CB.EQ.'A  ' .AND. IFC.NE.1)  GOTO 9000
  100 CONTINUE
C
      RETURN
C
 9000 WRITE (6,*) ' Element face on the axis of symmetry must be FACE 1'
      WRITE (6,*) ' Element',IEL,'   face',IFC,'  is on the axis.'
      call exitt
C
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHKCBC (CB,IEL,IFC,IFALGN,IERR)
      include 'SIZE' 
      include 'PARALLEL' 
      include 'INPUT' 
C
C     Check for illegal boundary conditions
C
      CHARACTER CB*3
      LOGICAL IFALGN

      if (ifstrs) return 

      ieg  = lglel(iel)

C     Laplacian formulation only
      IF  (CB.EQ.'SH ' .OR.  CB.EQ.'sh ' .OR.
     $     CB.EQ.'SHL' .OR.  CB.EQ.'shl' .OR.
     $     CB.EQ.'S  ' .OR.  CB.EQ.'s  ' .OR.
     $     CB.EQ.'SL ' .OR.  CB.EQ.'sl ' .OR.
     $     CB.EQ.'MM ' .OR.  CB.EQ.'mm ' .OR.
     $     CB.EQ.'MS ' .OR.  CB.EQ.'ms ' .OR.
     $     CB.EQ.'MSI' .OR.  CB.EQ.'msi'    )                GOTO 9001

       IF ( .NOT.IFALGN .AND.
     $    (CB.EQ.'ON ' .OR.  CB.EQ.'on ' .OR. CB.EQ.'SYM') ) GOTO 9010

      RETURN

 9001 WRITE (6,*) ' Illegal traction boundary conditions detected for'
      GOTO 9999

 9010 WRITE (6,*) ' Mixed B.C. on a side nonaligned with either the X,Y,
     $ or Z axis detected for'

 9999 WRITE (6,*) ' Element',ieg,'   side',IFC
      WRITE (6,*) ' Requires PN/PN-2 STRESS FORMULATION'

      ierr = err + 1

      END
c-----------------------------------------------------------------------
      SUBROUTINE BCMASK
C
C     Zero out masks corresponding to Dirichlet boundary points.
C
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'MVGEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TOPOL'

      common  /nekcb/ cb
      character*3 cb
      character*1 cb1(3)
      equivalence (cb1,cb)

      logical ifalgn,ifnorx,ifnory,ifnorz
      integer e,f

      NFACES=2*ldim
      NXYZ  =lx1*ly1*lz1

C
C     Masks for moving mesh
C
      IF (IFMVBD) THEN
         IFIELD = 0
         CALL STSMASK (W1MASK,W2MASK,W3MASK)
         do e=1,nelv
         do f=1,nfaces
            if (cbc(f,e,1).eq.'msi'.or.cbc(f,e,1).eq.'msi') then
               call facev(w1mask,e,f,0.0,lx1,ly1,lz1)
               call facev(w2mask,e,f,0.0,lx1,ly1,lz1)
               call facev(w3mask,e,f,0.0,lx1,ly1,lz1)
            endif
         enddo
         enddo
      ENDIF
C
C     Masks for flow variables
C
      IF (IFFLOW) THEN
         IFIELD = 1
         NEL    = NELFLD(IFIELD)
         NTOT   = NXYZ*NEL
C
C        Pressure mask
C
         call rone(pmask,ntot)
         do 50 iel=1,nelt
         do 50 iface=1,nfaces
            cb=cbc(iface,iel,ifield)
            if (cb.eq.'O  ' .or. cb.eq.'ON ' .or.
     $          cb.eq.'o  ' .or. cb.eq.'on ')
     $         call facev(pmask,iel,iface,0.0,lx1,ly1,lz1)
   50    continue
         if (nelt.gt.nelv) then
            nn=lx1*ly1*lz1*(nelt-nelv)
            call rzero(pmask(1,1,1,nelv+1),nn)
         endif
C
C        Zero out mask at Neumann-Dirichlet interfaces
C
         CALL DSOP(PMASK,'MUL',lx1,ly1,lz1)
C
C        Velocity masks
C
         IF (IFSTRS) THEN
           CALL STSMASK (V1MASK,V2MASK,V3MASK)
         ELSE
C
           CALL RONE(V1MASK,NTOT)
           CALL RONE(V2MASK,NTOT)
           CALL RONE(V3MASK,NTOT)
C
           DO 100 IEL=1,NELV
           DO 100 IFACE=1,NFACES
              CB =CBC(IFACE,IEL,IFIELD)
              CALL CHKNORD (IFALGN,IFNORX,IFNORY,IFNORZ,IFACE,IEL)
C
C            All-Dirichlet boundary conditions
C
           IF (CB.EQ.'v  ' .OR. CB.EQ.'V  ' .OR. CB.EQ.'vl ' .OR.
     $         cb.eq.'MV ' .or. cb.eq.'mv '                  .or.
     $         CB.EQ.'VL ' .OR. CB.EQ.'W  ') THEN
             CALL FACEV (V1MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             CALL FACEV (V2MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             CALL FACEV (V3MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             GOTO 100
           ENDIF
C
C        Mixed-Dirichlet-Neumann boundary conditions
C
         IF (CB.EQ.'SYM') THEN
             IF ( .NOT.IFALGN .OR. IFNORX )
     $            CALL FACEV (V1MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             IF ( IFNORY )
     $            CALL FACEV (V2MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             IF ( IFNORZ )
     $            CALL FACEV (V3MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             GOTO 100
         ENDIF

         IF (CB.EQ.'ON ' .OR. CB.EQ.'on ') THEN
             IF ( IFNORY .OR. IFNORZ )
     $            CALL FACEV (V1MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             IF ( .NOT.IFALGN .OR. IFNORX .OR. IFNORZ )
     $            CALL FACEV (V2MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             IF ( .NOT.IFALGN .OR. IFNORX .OR. IFNORY )
     $            CALL FACEV (V3MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
             GOTO 100
         ENDIF
         IF (CB.EQ.'A  ') THEN
             CALL FACEV (V2MASK,IEL,IFACE,0.0,lx1,ly1,lz1)
         ENDIF
  100    CONTINUE

         call opdsop(v1mask,v2mask,v3mask,'MUL') ! no rotation for mul

       ENDIF

       CALL RONE(OMASK,NTOT)
       DO 200 IEL=1,NELV
       DO 200 IFACE=1,NFACES
          CB =CBC(IFACE,IEL,IFIELD)
          IF (CB.EQ.'A  ') THEN
              CALL FACEV (OMASK,IEL,IFACE,0.0,lx1,ly1,lz1)
          ENDIF
 200   CONTINUE
       CALL DSOP(OMASK,'MUL',lx1,ly1,lz1)
C
      ENDIF
C
C     Masks for passive scalars
C
      IF (IFHEAT) THEN
C
         DO 1200 IFIELD=2,NFIELD
            IPSCAL = IFIELD-1
            NEL    = NELFLD(IFIELD)
            NTOT   = NXYZ*NEL
            CALL RONE (TMASK(1,1,1,1,IPSCAL),NTOT)
         DO 1100 IEL=1,NEL
         DO 1100 IFACE=1,NFACES
            CB =CBC(IFACE,IEL,IFIELD)
C
C           Assign mask values.
C
            IF  (CB.EQ.'T  ' .OR. CB.EQ.'t  ' .OR. 
     $          (CB.EQ.'A  ' .AND. IFAZIV)    .OR.
     $           CB.EQ.'MCI' .OR. CB.EQ.'MLI' .OR.
     $           CB.EQ.'KD ' .OR. CB.EQ.'kd ' .OR.
     $           CB.EQ.'ED ' .OR. CB.EQ.'ed ' .OR.
     $           CB.EQ.'KW ' .OR. CB.EQ.'KWS' .OR. CB.EQ.'EWS')
     $           CALL FACEV (TMASK(1,1,1,1,IPSCAL),
     $                       IEL,IFACE,0.0,lx1,ly1,lz1)
 1100       CONTINUE
         CALL DSOP (TMASK(1,1,1,1,IPSCAL),'MUL',lx1,ly1,lz1)
 1200    CONTINUE
C
      ENDIF
C
C     Masks for B-field
C
      if (ifmhd) then
         ifield = ifldmhd
         nel    = nelfld(ifield)
         ntot   = nxyz*nel
C
C        B-field pressure mask
C
         call rone(bpmask,ntot)
         do iel=1,nelv
         do iface=1,nfaces
            cb=cbc(iface,iel,ifield)
            if (cb.eq.'O  ' .or. cb.eq.'ON ')
     $         call facev(bpmask,iel,iface,0.0,lx1,ly1,lz1)
         enddo
         enddo
C
C        Zero out mask at Neumann-Dirichlet interfaces
C
         call dsop(bpmask,'MUL',lx1,ly1,lz1)
C
C        B-field masks
C
         if (ifstrs) then
           call stsmask (b1mask,b2mask,b3mask)
         else
C
           call rone(b1mask,ntot)
           call rone(b2mask,ntot)
           call rone(b3mask,ntot)
C
           do iel=1,nelv
           do iface=1,nfaces
              cb =cbc(iface,iel,ifield)
              call chknord (ifalgn,ifnorx,ifnory,ifnorz,iface,iel)
c
              if (cb.eq.'v  ' .or. cb.eq.'V  ' .or. cb.eq.'vl ' .or.
     $           cb.eq.'VL ' .or. cb.eq.'W  ') then
c
c               All-Dirichlet boundary conditions
c
                call facev (b1mask,iel,iface,0.0,lx1,ly1,lz1)
                call facev (b2mask,iel,iface,0.0,lx1,ly1,lz1)
                call facev (b3mask,iel,iface,0.0,lx1,ly1,lz1)
c
              elseif (cb.eq.'SYM') then
c
c               Mixed-Dirichlet-Neumann boundary conditions
c
                if ( .not.ifalgn .or. ifnorx )
     $            call facev (b1mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( ifnory )
     $            call facev (b2mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( ifnorz )
     $            call facev (b3mask,iel,iface,0.0,lx1,ly1,lz1)
c
              elseif (cb.eq.'ON ') then
c
c               Mixed-Dirichlet-Neumann boundary conditions
c
                if ( ifnory .or. ifnorz )
     $            call facev (b1mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( .not.ifalgn .or. ifnorx .or. ifnorz )
     $            call facev (b2mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( .not.ifalgn .or. ifnorx .or. ifnory )
     $            call facev (b3mask,iel,iface,0.0,lx1,ly1,lz1)
c
              elseif (cb.eq.'A  ') then
c
c               axisymmetric centerline
c
                call facev (b2mask,iel,iface,0.0,lx1,ly1,lz1)
c
              else
c
                if ( cb1(1).eq.'d' ) 
     $            call facev (b1mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( cb1(2).eq.'d' ) 
     $            call facev (b2mask,iel,iface,0.0,lx1,ly1,lz1)
                if ( cb1(3).eq.'d' .and. if3d ) 
     $            call facev (b3mask,iel,iface,0.0,lx1,ly1,lz1)
c
              endif
           enddo
           enddo
c
           call dsop(b1mask,'MUL',lx1,ly1,lz1)
           call dsop(b2mask,'MUL',lx1,ly1,lz1)
           if (ldim.eq.3) call dsop(b3mask,'MUL',lx1,ly1,lz1)
         endif
      endif
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BCDIRVC(V1,V2,V3,mask1,mask2,mask3)
C
C     Apply Dirichlet boundary conditions to surface of vector (V1,V2,V3).
C     Use IFIELD as a guide to which boundary conditions are to be applied.
C
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'TOPOL'
      INCLUDE 'CTIMER'
      COMMON /SCRUZ/ TMP1(LX1,LY1,LZ1,LELV)
     $             , TMP2(LX1,LY1,LZ1,LELV)
     $             , TMP3(LX1,LY1,LZ1,LELV)
      COMMON /SCRMG/ TMQ1(LX1,LY1,LZ1,LELV)
     $             , TMQ2(LX1,LY1,LZ1,LELV)
     $             , TMQ3(LX1,LY1,LZ1,LELV)
C
      REAL V1(lx1,ly1,lz1,LELV),V2(lx1,ly1,lz1,LELV)
     $    ,V3(lx1,ly1,lz1,LELV)
      real mask1(lx1,ly1,lz1,lelv),mask2(lx1,ly1,lz1,lelv)
     $    ,mask3(lx1,ly1,lz1,lelv)
c
      common  /nekcb/ cb
      character cb*3
      character*1 cb1(3)
      equivalence (cb1,cb)
c
      logical ifonbc
c
      ifonbc = .false.
c
      if (icalld.eq.0) then
         tusbc=0.0
         nusbc=0
         icalld=icalld+1
      endif
      nusbc=nusbc+1
      etime1=dnekclock()
C
C
      NFACES=2*ldim
      NXYZ  =lx1*ly1*lz1
      NEL   =NELFLD(IFIELD)
      NTOT  =NXYZ*NEL
C
      CALL RZERO(TMP1,NTOT)
      CALL RZERO(TMP2,NTOT)
      IF (IF3D) CALL RZERO(TMP3,NTOT)
C
C     Velocity boundary conditions
C
c     write(6,*) 'BCDIRV: ifield',ifield
      DO 2100 ISWEEP=1,2
         DO 2000 IE=1,NEL
         DO 2000 IFACE=1,NFACES
            CB  = CBC(IFACE,IE,IFIELD)
            BC1 = BC(1,IFACE,IE,IFIELD)
            BC2 = BC(2,IFACE,IE,IFIELD)
            BC3 = BC(3,IFACE,IE,IFIELD)

            IF (CB.EQ.'V  ' .OR. CB.EQ.'VL '  .OR.
     $          CB.EQ.'WS ' .OR. CB.EQ.'WSL') THEN
               CALL FACEV (TMP1,IE,IFACE,BC1,lx1,ly1,lz1)
               CALL FACEV (TMP2,IE,IFACE,BC2,lx1,ly1,lz1)
               IF (IF3D) CALL FACEV (TMP3,IE,IFACE,BC3,lx1,ly1,lz1)
               IF ( IFQINP(IFACE,IE) )
     $         CALL GLOBROT (TMP1(1,1,1,IE),TMP2(1,1,1,IE),
     $                       TMP3(1,1,1,IE),IE,IFACE)
            ENDIF

            IF (CB.EQ.'v  ' .OR. CB.EQ.'vl ' .OR. 
     $          CB.EQ.'ws ' .OR. CB.EQ.'wsl' .OR.
     $          CB.EQ.'mv ' .OR. CB.EQ.'mvn' .OR.
     $          cb1(1).eq.'d'.or.cb1(2).eq.'d'.or.cb1(3).eq.'d') then

                call faceiv (cb,tmp1(1,1,1,ie),tmp2(1,1,1,ie),
     $                       tmp3(1,1,1,ie),ie,iface,lx1,ly1,lz1)

                IF ( IFQINP(IFACE,IE) )
     $          CALL GLOBROT (TMP1(1,1,1,IE),TMP2(1,1,1,IE),
     $                        TMP3(1,1,1,IE),IE,IFACE)
            ENDIF

            IF (CB.EQ.'ON ' .OR. CB.EQ.'on ') then   ! 5/21/01 pff
                ifonbc =.true.
                CALL FACEIV ('v  ',TMP1(1,1,1,IE),TMP2(1,1,1,IE),
     $                       TMP3(1,1,1,IE),IE,IFACE,lx1,ly1,lz1)
            ENDIF

 2000    CONTINUE
         DO 2010 IE=1,NEL
         DO 2010 IFACE=1,NFACES
            IF (CBC(IFACE,IE,IFIELD).EQ.'W  ') THEN
               CALL FACEV (TMP1,IE,IFACE,0.0,lx1,ly1,lz1)
               CALL FACEV (TMP2,IE,IFACE,0.0,lx1,ly1,lz1)
               IF (IF3D) CALL FACEV (TMP3,IE,IFACE,0.0,lx1,ly1,lz1)
            ENDIF
 2010    CONTINUE
C
C        Take care of Neumann-Dirichlet shared edges...
C
         if (isweep.eq.1) then
            call opdsop(tmp1,tmp2,tmp3,'MXA')
         else
            call opdsop(tmp1,tmp2,tmp3,'MNA')
         endif
 2100 CONTINUE
C
C     Copy temporary array to velocity arrays.
C
      IF ( .NOT.IFSTRS ) THEN
         CALL COL2(V1,mask1,NTOT)
         CALL COL2(V2,mask2,NTOT)
         IF (IF3D) CALL COL2(V3,mask3,NTOT)
         if (ifonbc) then
            call antimsk1(tmp1,mask1,ntot)
            call antimsk1(tmp2,mask2,ntot)
            if (if3d) call antimsk1(tmp3,mask3,ntot)
         endif
      ELSE
         CALL RMASK (V1,V2,V3,NELV)
      ENDIF

      CALL ADD2(V1,TMP1,NTOT)
      CALL ADD2(V2,TMP2,NTOT)
      IF (IF3D) CALL ADD2(V3,TMP3,NTOT)

      if (ifneknekc) call fix_surface_flux

      tusbc=tusbc+(dnekclock()-etime1)

      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BCDIRSC(S)
C
C     Apply Dirichlet boundary conditions to surface of scalar, S.
C     Use IFIELD as a guide to which boundary conditions are to be applied.
C
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TOPOL'
      INCLUDE 'CTIMER'
C
      DIMENSION S(LX1,LY1,LZ1,LELT)
      COMMON /SCRSF/ TMP(LX1,LY1,LZ1,LELT)
     $             , TMA(LX1,LY1,LZ1,LELT)
     $             , SMU(LX1,LY1,LZ1,LELT)
      common  /nekcb/ cb
      CHARACTER CB*3

      if (icalld.eq.0) then
         tusbc=0.0
         nusbc=0
         icalld=icalld+1
      endif
      nusbc=nusbc+1
      etime1=dnekclock()
C
      IFLD   = 1
      NFACES = 2*ldim
      NXYZ   = lx1*ly1*lz1
      NEL    = NELFLD(IFIELD)
      NTOT   = NXYZ*NEL
      NFLDT  = NFIELD - 1
C
      CALL RZERO(TMP,NTOT)
C
C     Temperature boundary condition
C
      DO 2100 ISWEEP=1,2
C
         DO 2010 IE=1,NEL
         DO 2010 IFACE=1,NFACES
            CB=CBC(IFACE,IE,IFIELD)
            BC1=BC(1,IFACE,IE,IFIELD)
            BC2=BC(2,IFACE,IE,IFIELD)
            BC3=BC(3,IFACE,IE,IFIELD)
            BC4=BC(4,IFACE,IE,IFIELD)
            BCK=BC(4,IFACE,IE,IFLD)
            BCE=BC(5,IFACE,IE,IFLD)
            IF (CB.EQ.'T  ') CALL FACEV (TMP,IE,IFACE,BC1,lx1,ly1,lz1)
            IF (CB.EQ.'MCI') CALL FACEV (TMP,IE,IFACE,BC4,lx1,ly1,lz1)
            IF (CB.EQ.'MLI') CALL FACEV (TMP,IE,IFACE,BC4,lx1,ly1,lz1)
            IF (CB.EQ.'KD ') CALL FACEV (TMP,IE,IFACE,BCK,lx1,ly1,lz1)
            IF (CB.EQ.'ED ') CALL FACEV (TMP,IE,IFACE,BCE,lx1,ly1,lz1)
            IF (CB.EQ.'t  ' .OR. CB.EQ.'kd ' .or.
     $          CB.EQ.'ed ' .or. cb.eq.'o  ' .or. cb.eq.'on ') 
     $          CALL FACEIS (CB,TMP(1,1,1,IE),IE,IFACE,lx1,ly1,lz1)
 2010    CONTINUE
C
C        Take care of Neumann-Dirichlet shared edges...
C
         IF (ISWEEP.EQ.1) CALL DSOP(TMP,'MXA',lx1,ly1,lz1)
         IF (ISWEEP.EQ.2) CALL DSOP(TMP,'MNA',lx1,ly1,lz1)
 2100 CONTINUE
C
C     Copy temporary array to temperature array.
C
      CALL COL2(S,TMASK(1,1,1,1,IFIELD-1),NTOT)
      CALL ADD2(S,TMP,NTOT)

      tusbc=tusbc+(dnekclock()-etime1)

      RETURN
      END
C
c-----------------------------------------------------------------------
      SUBROUTINE BCNEUSC(S,ITYPE)
C
C     Apply Neumann boundary conditions to surface of scalar, S.
C     Use IFIELD as a guide to which boundary conditions are to be applied.
C
C     If ITYPE = 1, then S is returned as the rhs contribution to the 
C                   volumetric flux.
C
C     If ITYPE =-1, then S is returned as the lhs contribution to the 
C                   diagonal of A.
C
C
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'
      INCLUDE 'CTIMER'
      INCLUDE 'NEKUSE'
C
      DIMENSION S(LX1,LY1,LZ1,LELT)
      common  /nekcb/ cb
      CHARACTER CB*3
C
      if (icalld.eq.0) then
         tusbc=0.0
         nusbc=0
         icalld=icalld+1
      endif
      nusbc=nusbc+1
      etime1=dnekclock()
C
      NFACES=2*ldim
      NXYZ  =lx1*ly1*lz1
      NEL   =NELFLD(IFIELD)
      NTOT  =NXYZ*NEL
      CALL RZERO(S,NTOT)
C
      IF (ITYPE.EQ.-1) THEN
C
C        Compute diagonal contributions to accomodate Robin boundary conditions
C
         DO 1000 IE=1,NEL
         DO 1000 IFACE=1,NFACES
            ieg=lglel(ie)
            CB =CBC(IFACE,IE,IFIELD)
            IF (CB.EQ.'C  ' .OR. CB.EQ.'c  ' .OR.
     $          CB.EQ.'R  ' .OR. CB.EQ.'r  ') THEN
C
               IF (CB.EQ.'C  ') HC   = BC(2,IFACE,IE,IFIELD)
               IF (CB.EQ.'R  ') THEN
                                TINF = BC(1,IFACE,IE,IFIELD)
                                HRAD = BC(2,IFACE,IE,IFIELD)
               ENDIF
               IA=0
C
C IA is areal counter, assumes advancing fastest index first. (IX...IY...IZ)
C
               CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,lx1,ly1,lz1,IFACE)
               DO 100 IZ=KZ1,KZ2
               DO 100 IY=KY1,KY2
               DO 100 IX=KX1,KX2
                  IA = IA + 1
                  TS = T(IX,IY,IZ,IE,IFIELD-1)
                  IF (CB.EQ.'c  ' .OR. CB.EQ.'r  ') THEN
                     if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IE)
                     CALL USERBC  (IX,IY,IZ,IFACE,IEG)
                  ENDIF
                  IF (CB.EQ.'r  ' .OR. CB.EQ.'R  ') 
     $               HC = HRAD * (TINF**2 + TS**2) * (TINF + TS)
                  S(IX,IY,IZ,IE) = S(IX,IY,IZ,IE) +
     $               HC*AREA(IA,1,IFACE,IE)/BM1(IX,IY,IZ,IE)
  100          CONTINUE
            ENDIF
 1000    CONTINUE
      ENDIF
      IF (ITYPE.EQ.1) THEN
C
C        Add passive scalar fluxes to rhs
C
         DO 2000 IE=1,NEL
         DO 2000 IFACE=1,NFACES
            ieg=lglel(ie)
            CB =CBC(IFACE,IE,IFIELD)
            IF (CB.EQ.'F  ' .OR. CB.EQ.'f  ' .OR.
     $          CB.EQ.'C  ' .OR. CB.EQ.'c  ' .OR. 
     $          CB.EQ.'R  ' .OR. CB.EQ.'r  ' ) THEN
C
                IF (CB.EQ.'F  ') FLUX=BC(1,IFACE,IE,IFIELD)
                IF (CB.EQ.'C  ') FLUX=BC(1,IFACE,IE,IFIELD)
     $                               *BC(2,IFACE,IE,IFIELD)
                IF (CB.EQ.'R  ') THEN
                                 TINF=BC(1,IFACE,IE,IFIELD)
                                 HRAD=BC(2,IFACE,IE,IFIELD)
                ENDIF
C
C              Add local weighted flux values to rhs, S.
C
C IA is areal counter, assumes advancing fastest index first. (IX...IY...IZ)
               IA=0
               CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,lx1,ly1,lz1,IFACE)
               DO 200 IZ=KZ1,KZ2
               DO 200 IY=KY1,KY2
               DO 200 IX=KX1,KX2
                  IA = IA + 1
                  TS = T(IX,IY,IZ,IE,IFIELD-1)
                  IF (CB.EQ.'f  ') THEN
                     if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IE)
                     CALL USERBC  (IX,IY,IZ,IFACE,IEG)
                  ENDIF
                  IF (CB.EQ.'c  ') THEN
                     if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IE)
                     CALL USERBC  (IX,IY,IZ,IFACE,IEG)
                     FLUX = TINF*HC
                  ENDIF
                  IF (CB.EQ.'r  ') THEN
                     if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IE)
                     CALL USERBC  (IX,IY,IZ,IFACE,IEG)
                  ENDIF
                  IF (CB.EQ.'R  ' .OR. CB.EQ.'r  ') 
     $               FLUX = HRAD*(TINF**2 + TS**2)*(TINF + TS) * TINF
C
C                 Add computed fluxes to boundary surfaces:
C
                  S(IX,IY,IZ,IE) = S(IX,IY,IZ,IE)
     $                           + FLUX*AREA(IA,1,IFACE,IE)
  200          CONTINUE
            ENDIF
 2000    CONTINUE
      ENDIF
C
      tusbc=tusbc+(dnekclock()-etime1)
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE FACEIS (CB,S,IEL,IFACE,NX,NY,NZ)
C
C     Assign inflow boundary conditions to face(IE,IFACE)
C     for scalar S.
C
      INCLUDE 'SIZE'
      INCLUDE 'PARALLEL'
      INCLUDE 'NEKUSE'
      INCLUDE 'TSTEP'     ! ifield    11/19/2010
      INCLUDE 'SOLN'      ! tmask()   11/19/2010
C
      DIMENSION S(LX1,LY1,LZ1)
      CHARACTER CB*3
c
      common  /nekcb/ cb3
      character*3 cb3
      cb3 = cb

      ifld1 = ifield-1


C     Passive scalar term

      ieg = lglel(iel)
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX,NY,NZ,IFACE)

      if (cb.eq.'t  ') then
         DO 100 IZ=KZ1,KZ2                           !  11/19/2010: The tmask() screen
         DO 100 IY=KY1,KY2                           !  added here so users can leave
         DO 100 IX=KX1,KX2                           !  certain points to be Neumann,
            if (tmask(ix,iy,iz,iel,ifld1).eq.0) then !  if desired.
               if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
               CALL USERBC  (IX,IY,IZ,IFACE,IEG)
               S(IX,IY,IZ) = TEMP
            endif
  100    CONTINUE
         RETURN

      elseif (cb.eq.'o  ' .or. cb.eq.'on ') then
         DO 101 IZ=KZ1,KZ2                           !  11/19/2010: The tmask() screen
         DO 101 IY=KY1,KY2                           !  added here so users can leave
         DO 101 IX=KX1,KX2                           !  certain points to be Neumann,
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            S(IX,IY,IZ) = PA
  101    CONTINUE
         RETURN

      ELSEIF (CB.EQ.'ms ' .OR. CB.EQ.'msi') THEN

         DO 200 IZ=KZ1,KZ2
         DO 200 IY=KY1,KY2
         DO 200 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            S(IX,IY,IZ) = SIGMA
  200    CONTINUE
C
      ELSEIF (CB.EQ.'kd ') THEN
C
         DO 300 IZ=KZ1,KZ2
         DO 300 IY=KY1,KY2
         DO 300 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            S(IX,IY,IZ) = TURBK
  300    CONTINUE
C
      ELSEIF (CB.EQ.'ed ') THEN
C
         DO 400 IZ=KZ1,KZ2
         DO 400 IY=KY1,KY2
         DO 400 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            S(IX,IY,IZ) = TURBE
  400    CONTINUE
C
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE FACEIV (CB,V1,V2,V3,IEL,IFACE,NX,NY,NZ)
C
C     Assign fortran function boundary conditions to 
C     face IFACE of element IEL for vector (V1,V2,V3).
C
      INCLUDE 'SIZE'
      INCLUDE 'NEKUSE'
      INCLUDE 'PARALLEL'
C
      dimension v1(nx,ny,nz),v2(nx,ny,nz),v3(nx,ny,nz)
      character cb*3
c
      character*1 cb1(3)
c
      common  /nekcb/ cb3
      character*3 cb3
      cb3 = cb
c
      call chcopy(cb1,cb,3)
c
      ieg = lglel(iel)
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX,NY,NZ,IFACE)
C
      IF (CB.EQ.'v  ' .OR. CB.EQ.'ws ' .OR. CB.EQ.'mv '.OR. 
     $    CB.EQ.'mvn') THEN
C
         DO 100 IZ=KZ1,KZ2
         DO 100 IY=KY1,KY2
         DO 100 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = UX
            V2(IX,IY,IZ) = UY
            V3(IX,IY,IZ) = UZ
  100    CONTINUE
         RETURN
C
      elseif (cb1(1).eq.'d'.or.cb1(2).eq.'d'.or.cb1(3).eq.'d') then
C
         do iz=kz1,kz2
         do iy=ky1,ky2
         do ix=kx1,kx2
            if (optlevel.le.2) call nekasgn (ix,iy,iz,iel)
            call userbc  (ix,iy,iz,iface,ieg)
            if (cb1(1).eq.'d') v1(ix,iy,iz) = ux
            if (cb1(2).eq.'d') v2(ix,iy,iz) = uy
            if (cb1(3).eq.'d') v3(ix,iy,iz) = uz
         enddo
         enddo
         enddo
         return
C
      ELSEIF (CB.EQ.'vl ' .OR. CB.EQ.'wsl') THEN
C
         DO 120 IZ=KZ1,KZ2
         DO 120 IY=KY1,KY2
         DO 120 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = UN
            V2(IX,IY,IZ) = U1
            V3(IX,IY,IZ) = U2
  120    CONTINUE
         RETURN
C
      ELSEIF (CB.EQ.'s  ' .OR. CB.EQ.'sh ') THEN
C
         DO 200 IZ=KZ1,KZ2
         DO 200 IY=KY1,KY2
         DO 200 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = TRX
            V2(IX,IY,IZ) = TRY
            V3(IX,IY,IZ) = TRZ
  200    CONTINUE
         RETURN
C
      ELSEIF (CB.EQ.'sl ' .OR. CB.EQ.'shl') THEN
C
         DO 220 IZ=KZ1,KZ2
         DO 220 IY=KY1,KY2
         DO 220 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = TRN
            V2(IX,IY,IZ) = TR1
            V3(IX,IY,IZ) = TR2
  220    CONTINUE
C
      ELSEIF (CB.EQ.'ms ') THEN
C
         DO 240 IZ=KZ1,KZ2
         DO 240 IY=KY1,KY2
         DO 240 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = -PA
            V2(IX,IY,IZ) = TR1
            V3(IX,IY,IZ) = TR2
  240    CONTINUE
C
      ELSEIF (CB.EQ.'on ' .OR. CB.EQ.'o  ') THEN
C
         DO 270 IZ=KZ1,KZ2
         DO 270 IY=KY1,KY2
         DO 270 IX=KX1,KX2
            if (optlevel.le.2) CALL NEKASGN (IX,IY,IZ,IEL)
            CALL USERBC  (IX,IY,IZ,IFACE,IEG)
            V1(IX,IY,IZ) = -PA
            V2(IX,IY,IZ) = 0.0
            V3(IX,IY,IZ) = 0.0
  270    CONTINUE
C
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine nekasgn (ix,iy,iz,e)
C
C     Assign NEKTON variables for definition (by user) of
C     boundary conditions at collocation point (IX,IY,IZ)
C     of element IEL.
C
C       X             X-coordinate
C       Y             Y-coordinate
C       Z             Z-coordinate
C       UX            X-velocity
C       UY            Y-velocity
C       UZ            Z-velocity
C       TEMP          Temperature
C       PS1           Passive scalar No. 1
C       PS2           Passive scalar No. 2
C        .             .
C        .             .
C       PS9           Passive scalar No. 9
C       SI2           Strainrate invariant II
C       SI3           Strainrate invariant III
C
C     Variables to be defined by user for imposition of
C     boundary conditions :
C
C       SH1           Shear component No. 1
C       SH2           Shear component No. 2
C       TRX           X-traction
C       TRY           Y-traction
C       TRZ           Z-traction
C       SIGMA         Surface-tension coefficient
C       FLUX          Flux
C       HC            Convection heat transfer coefficient
C       HRAD          Radiation  heat transfer coefficient
C       TINF          Temperature at infinity
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'NEKUSE'

      integer e

      common  /nekcb/ cb
      character cb*3

      COMMON /SCREV / SII (LX1,LY1,LZ1,LELT)
     $              , SIII(LX1,LY1,LZ1,LELT)

      x     = xm1(ix,iy,iz,e)
      y     = ym1(ix,iy,iz,e)
      z     = zm1(ix,iy,iz,e)

      r     = x**2+y**2
      if (r.gt.0.0) r=sqrt(r)
      if (x.ne.0.0 .or. y.ne.0.0) theta = atan2(y,x)

      ux    = vx(ix,iy,iz,e)
      uy    = vy(ix,iy,iz,e)
      uz    = vz(ix,iy,iz,e)
      temp  = t(ix,iy,iz,e,1)
      do ips=1,npscal
         ps(ips) = t(ix,iy,iz,e,ips+1)
      enddo

      pa = pr(ix,iy,iz,e)
      p0 = p0th

      si2   = sii (ix,iy,iz,e)
      si3   = siii(ix,iy,iz,e)
      udiff = vdiff (ix,iy,iz,e,ifield)
      utrans= vtrans(ix,iy,iz,e,ifield)

      cbu   = cb

      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE BCNEUTR
C
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      COMMON /SCRSF/ TRX(LX1,LY1,LZ1)
     $             , TRY(LX1,LY1,LZ1)
     $             , TRZ(LX1,LY1,LZ1)
      COMMON /CTMP0/ STC(LX1,LY1,LZ1)
      REAL SIGST(LX1,LY1)
C
      LOGICAL IFALGN,IFNORX,IFNORY,IFNORZ
      common  /nekcb/ cb
      CHARACTER CB*3
C
      IFLD  = 1
      NFACE = 2*ldim
      NXY1  = lx1*ly1
      NXYZ1 = lx1*ly1*lz1
C
      DO 100 IEL=1,NELV
      DO 100 IFC=1,NFACE
C
         CB  = CBC (IFC,IEL,IFLD)
         BC1 = BC(1,IFC,IEL,IFLD)
         BC2 = BC(2,IFC,IEL,IFLD)
         BC3 = BC(3,IFC,IEL,IFLD)
         BC4 = BC(4,IFC,IEL,IFLD)
         CALL RZERO3 (TRX,TRY,TRZ,NXYZ1)
C
C        Prescribed tractions and shear tractions
C
         IF (CB.EQ.'S  ' .OR. CB.EQ.'SL ' .OR.
     $       CB.EQ.'SH ' .OR. CB.EQ.'SHL' ) THEN
             CALL TRCON (TRX,TRY,TRZ,BC1,BC2,BC3,IEL,IFC)
             IF (IFQINP(IFC,IEL)) CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             GOTO 120
         ENDIF
         IF (CB.EQ.'s  ' .OR. CB.EQ.'sl ' .OR.
     $       CB.EQ.'sh ' .OR. CB.EQ.'shl' ) THEN
             CALL FACEIV (CB,TRX,TRY,TRZ,IEL,IFC,lx1,ly1,lz1)
             CALL FACCVS (TRX,TRY,TRZ,AREA(1,1,IFC,IEL),IFC)
             IF (IFQINP(IFC,IEL)) CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             GOTO 120
         ENDIF
C
C        Prescribed outflow ambient pressure
C
         IF (CB.EQ.'ON ' .OR. CB.EQ.'O  ') THEN
             BCN = -BC1
             BC2 =  0.0
             BC3 =  0.0
             CALL TRCON   (TRX,TRY,TRZ,BCN,BC2,BC3,IEL,IFC)
             CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             GOTO 120
         ENDIF
         IF (CB.EQ.'on ' .OR. CB.EQ.'o  ') THEN
             CALL FACEIV  (CB,TRX,TRY,TRZ,IEL,IFC,lx1,ly1,lz1)
             CALL FACCVS  (TRX,TRY,TRZ,AREA(1,1,IFC,IEL),IFC)
             CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             GOTO 120
         ENDIF
C
C     Surface-tension
C
         IF (CB.EQ.'MS ' .OR. CB.EQ.'MSI' .OR.
     $       CB.EQ.'MM ' .OR. CB.EQ.'mm ' .OR.
     $       CB.EQ.'ms ' .OR. CB.EQ.'msi') THEN
             IF (CB.EQ.'MS '.or.cb.eq.'MM ') THEN
                BCN = -BC1
                CALL TRCON   (TRX,TRY,TRZ,BCN,BC2,BC3,IEL,IFC)
                CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             ENDIF
c            IF (CB.EQ.'ms '.or.cb.eq.'mm ') THEN
             IF (CB.EQ.'ms '.or.cb.eq.'msi') THEN
                CALL FACEIV  (CB,TRX,TRY,TRZ,IEL,IFC,lx1,ly1,lz1)
                CALL FACCVS  (TRX,TRY,TRZ,AREA(1,1,IFC,IEL),IFC)
                CALL GLOBROT (TRX,TRY,TRZ,IEL,IFC)
             ENDIF
             IF (CB(1:1).EQ.'M') THEN
                CALL CFILL  (SIGST,BC4,NXY1)
             ELSE
                CALL FACEIS (CB,STC,IEL,IFC,lx1,ly1,lz1)
                CALL FACEXS (SIGST,STC,IFC,0)
             ENDIF
             IF (IFAXIS) THEN
                CALL TRSTAX (TRX,TRY,SIGST,IEL,IFC)
             ELSEIF (ldim.EQ.2) THEN
                CALL TRST2D (TRX,TRY,SIGST,IEL,IFC)
             ELSE
                CALL TRST3D (TRX,TRY,TRZ,SIGST,IEL,IFC)
             ENDIF
         ENDIF
C
  120    CALL ADD2 (BFX(1,1,1,IEL),TRX,NXYZ1)
         CALL ADD2 (BFY(1,1,1,IEL),TRY,NXYZ1)
         IF (ldim.EQ.3) CALL ADD2 (BFZ(1,1,1,IEL),TRZ,NXYZ1)
C
  100 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE TRCON (TRX,TRY,TRZ,TR1,TR2,TR3,IEL,IFC)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TOPOL'
C
      DIMENSION TRX(LX1,LY1,LZ1)
     $        , TRY(LX1,LY1,LZ1)
     $        , TRZ(LX1,LY1,LZ1)
C
      CALL DSSET(lx1,ly1,lz1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      IF (ldim.EQ.2) THEN
         DO 100 J2=JS2,JF2,JSKIP2
         DO 100 J1=JS1,JF1,JSKIP1
            I = I + 1
            TRX(J1,J2,1) = TR1*AREA(I,1,IFC,IEL)
            TRY(J1,J2,1) = TR2*AREA(I,1,IFC,IEL)
  100    CONTINUE
      ELSE
         DO 200 J2=JS2,JF2,JSKIP2
         DO 200 J1=JS1,JF1,JSKIP1
            I = I + 1
            TRX(J1,J2,1) = TR1*AREA(I,1,IFC,IEL)
            TRY(J1,J2,1) = TR2*AREA(I,1,IFC,IEL)
            TRZ(J1,J2,1) = TR3*AREA(I,1,IFC,IEL)
  200    CONTINUE
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE TRST2D (TRX,TRY,SIGST,IEL,IFC)
C
C     Compute taction due to surface tension (2D)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'DXYZ'
      INCLUDE 'TOPOL'
      INCLUDE 'WZ'
      COMMON /CTMP1/ A1X(LX1),A1Y(LX1),STX(LX1),STY(LX1)
C
      DIMENSION TRX(LX1,LY1,LZ1),TRY(LX1,LY1,LZ1),SIGST(LX1,1)
      DIMENSION CANG(2),SANG(2)
      DIMENSION IXN(2),IYN(2),IAN(2)
C
      DO 100 IX=1,lx1
         AA = SIGST(IX,1) * WXM1(IX)
         STX(IX) = T1X(IX,1,IFC,IEL) * AA
         STY(IX) = T1Y(IX,1,IFC,IEL) * AA
  100 CONTINUE
C 
      IF (IFC.EQ.3 .OR. IFC.EQ.4) THEN
         CALL CHSIGN (STX,lx1)
         CALL CHSIGN (STY,lx1)
      ENDIF
C
      IF (IFC.EQ.1 .OR. IFC.EQ.3) THEN
         CALL MXM (DXTM1,lx1,STX,lx1,A1X,1)
         CALL MXM (DXTM1,lx1,STY,lx1,A1Y,1)
      ELSE
         CALL MXM (DYTM1,ly1,STX,ly1,A1X,1)
         CALL MXM (DYTM1,ly1,STY,ly1,A1Y,1)
      ENDIF
C
      CALL DSSET (lx1,ly1,lz1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      DO 200 J2=JS2,JF2,JSKIP2
      DO 200 J1=JS1,JF1,JSKIP1
         I = I + 1
         TRX(J1,J2,1) = TRX(J1,J2,1) - A1X(I)
         TRY(J1,J2,1) = TRY(J1,J2,1) - A1Y(I)
  200 CONTINUE
C
C     Contact angle corrections
C
      CALL CTANG2D (CANG,SANG,IXN,IYN,IAN,IFC,IEL)
      DO 500 I=1,2
         IX = IXN(I)
         IY = IYN(I)
         IA = IAN(I)
         TRX(IX,IY,1)=TRX(IX,IY,1) + SIGST(IA,1)*CANG(I)
         TRY(IX,IY,1)=TRY(IX,IY,1) + SIGST(IA,1)*SANG(I)
  500 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE TRSTAX (TRX,TRY,SIGST,IEL,IFC)
C
C     Compute taction due to surface tension (axisymmetric)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'DXYZ'
      INCLUDE 'TOPOL'
      INCLUDE 'WZ'
      COMMON /CTMP1/ A1X(LX1),A1Y(LX1),A2X(LX1),A2Y(LX1)
     $             , STX(LX1),STY(LX1),XJM1(LX1)
      COMMON /CTMP0/ XFM1(LX1),YFM1(LX1),T1XF(LX1),T1YF(LX1)
C
      DIMENSION TRX(LX1,LY1,LZ1),TRY(LX1,LY1,LZ1),SIGST(LX1,LY1)
      DIMENSION CANG(2),SANG(2)
      DIMENSION IXN(2),IYN(2),IAN(2)
      LOGICAL IFGLJ
C
      IFGLJ = .FALSE.
      IF ( IFRZER(IEL) .AND. (IFC.EQ.2 .OR. IFC.EQ.4) ) IFGLJ = .TRUE.
      CALL FACEC2 (XFM1,YFM1,XM1(1,1,1,IEL),YM1(1,1,1,IEL),IFC)
C
      IF (IFGLJ) THEN
         CALL MXM (DAM1,ly1,XFM1,ly1,T1XF,1)
         CALL MXM (DAM1,ly1,YFM1,ly1,T1YF,1)
         YS0 = T1YF(1)
      ELSE
         CALL MXM (DXM1,lx1,XFM1,lx1,T1XF,1)
         CALL MXM (DXM1,lx1,YFM1,lx1,T1YF,1)
      ENDIF
C
      DO 10 IX=1,lx1
         XJM1(IX)=SQRT( T1XF(IX)**2 + T1YF(IX)**2 )
         T1XF(IX)=T1XF(IX) / XJM1(IX)
         T1YF(IX)=T1YF(IX) / XJM1(IX)
   10 CONTINUE
C
      IF ( IFGLJ ) THEN
         CALL MXM (DAM1,1,T1XF,ly1,T1XS0,1)
         CALL MXM (DAM1,1,UNY(1,1,IFC,IEL),ly1,UNYS0,1)
         DDX    = WAM1(1)*SIGST(1,1)*T1XS0*YS0
         DDY    = WAM1(1)*SIGST(1,1)*T1YF(1)*YS0*2.0
         A2X(1) = WAM1(1)*SIGST(1,1)*XJM1(1)*UNX(1,1,IFC,IEL)*UNYS0
         A2Y(1) = 0.0
         STX(1) = 0.0
         STY(1) = 0.0
         DO 100 IY=2,ly1
            AA = WAM1(IY) * SIGST(IY,1) / (1.0 + ZAM1(IY))
            STX(IY) = T1XF(IY) * AA
            STY(IY) = T1YF(IY) * AA
            AA = AA * XJM1(IY) * UNY(IY,1,IFC,IEL)
            A2X(IY) = UNX(IY,1,IFC,IEL) * AA
            A2Y(IY) = UNY(IY,1,IFC,IEL) * AA
  100    CONTINUE
      ELSE
         DO 200 IX=1,lx1
            AA = SIGST(IX,1) * WXM1(IX)
            STX(IX) = T1XF(IX) * AA
            STY(IX) = T1YF(IX) * AA
            AA = AA * XJM1(IX) * UNY(IX,1,IFC,IEL)
            A2X(IX) = UNX(IX,1,IFC,IEL) * AA
            A2Y(IX) = UNY(IX,1,IFC,IEL) * AA
  200    CONTINUE
      ENDIF
C
      IF (IFGLJ) THEN
         DO 220 IY=1,ly1
            YSIY = T1YF(IY)*XJM1(IY)
            DTX1 = 0.0
            DTY1 = DATM1(IY,1)*DDY
            DTX2 = YSIY*STX(IY)
            DTY2 = YSIY*STY(IY)
            DTY3 = 0.0
            DO 240 J=2,ly1
               DTYS = DATM1(IY,J)*YFM1(J)
               DTX1 = DTX1 + DTYS*STX(J)
               DTY3 = DTY3 + DTYS*STY(J)
  240       CONTINUE
            A1X(IY) = DTX1 + DTX2
            A1Y(IY) = DTY1 + DTY2 + DTY3
  220    CONTINUE
            A1X(1)  = A1X(1) + DDX
      ELSE
         CALL MXM  (DXTM1,lx1,STX,lx1,A1X,1)
         CALL MXM  (DXTM1,lx1,STY,lx1,A1Y,1)
         CALL COL2 (A1X,YFM1,lx1)
         CALL COL2 (A1Y,YFM1,lx1)
      ENDIF
C
      CALL DSSET (lx1,ly1,lz1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      DO 300 J2=JS2,JF2,JSKIP2
      DO 300 J1=JS1,JF1,JSKIP1
         I  = I + 1
         TRX(J1,J2,1) = TRX(J1,J2,1) - A2X(I) - A1X(I)
         TRY(J1,J2,1) = TRY(J1,J2,1) - A2Y(I) - A1Y(I)
  300 CONTINUE
C
C     Contact angle corrections
C
      CALL CTANG2D (CANG,SANG,IXN,IYN,IAN,IFC,IEL)
      DO 500 I=1,2
         IX = IXN(I)
         IY = IYN(I)
         IA = IAN(I)
         AA = SIGST(IA,1)*YM1(IX,IY,1,IEL)
         TRX(IX,IY,1)=TRX(IX,IY,1) + AA*CANG(I)
         TRY(IX,IY,1)=TRY(IX,IY,1) + AA*SANG(I)
  500 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CTANG2D (CANG,SANG,IXN,IYN,IAN,IFC,IEL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      INCLUDE 'INPUT'
C
      DIMENSION CANG(2),SANG(2)
      DIMENSION IXN(2),IYN(2),IAN(2),ISN(2),NEBPT(4,2)
      CHARACTER CBN*3
C
      DATA NEBPT /4,1,2,3, 2,3,4,1/
      IFLD = 1
      EPS  = 1.e-6
C
      DO 100 I=1,2
         IFCN    = NEBPT(IFC,I)
         CBN     = CBC(IFCN,IEL,IFLD)
         IXN(I)  = 1
         IYN(I)  = 1
         IAN(I)  = 1
         ISN(I)  = 1
         CANG(I) = 0.0
         SANG(I) = 0.0
         IF (CBN.EQ.'E  '.OR.CBN.EQ.'P  '.OR.cbn.eq.'p  '.or.
     $       CBN(1:1).EQ.'M' .OR. CBN(1:1).EQ.'m') GOTO 100
         NC = IFC
         IF (I.EQ.2) NC=IFCN
         IF (NC  .EQ.2 .OR. NC  .EQ.3) IXN(I) = lx1
         IF (NC  .EQ.3 .OR. NC  .EQ.4) IYN(I) = ly1
         IF (IFC .EQ.2 .OR. IFC .EQ.3) ISN(I) = lx1
         IF (IFCN.EQ.2 .OR. IFCN.EQ.3) IAN(I) = lx1
         IX = IXN(I)
         IY = IYN(I)
         IA = IAN(I)
         IS = ISN(I)
         IF (CBN(1:1).EQ.'V'   .OR. CBN(1:1).EQ.'v'   .OR. 
     $       CBN     .EQ.'S  ' .OR. CBN     .EQ.'s  ' .OR. 
     $       CBN     .EQ.'SL ' .OR. CBN     .EQ.'sl ' .OR. 
     $       CBN(1:1).EQ.'O'   .OR. CBN(1:1).EQ.'o' ) THEN
             UX=VX(IX,IY,1,IEL)
             UY=VY(IX,IY,1,IEL)
             UM=UX**2 + UY**2
             IF (UM.GT.EPS) THEN
                 UNLX=UNX(IS,1,IFCN,IEL)
                 UNLY=UNY(IS,1,IFCN,IEL)
                 UM=SQRT(UM)
                 DOT =UX*UNLX + UY*UNLY
                 IF (DOT.LT.0.0) UM=-UM
                 CANG(I)=UX/UM
                 SANG(I)=UY/UM
                 GOTO 100
             ENDIF
         ENDIF
         CANG(I)=UNX(IS,1,IFCN,IEL)
         SANG(I)=UNY(IS,1,IFCN,IEL)
  100 CONTINUE
C
      RETURN      
      END
c-----------------------------------------------------------------------
      SUBROUTINE TRST3D (TRX,TRY,TRZ,SIGST,IEL,IFC)
C
C     Compute taction due to surface tension (3D)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'WZ'
      COMMON /CTMP0/  XFM1(LX1,LY1),YFM1(LX1,LY1),ZFM1(LX1,LY1)
      COMMON /CTMP1/  DRM1(LX1,LX1),DRTM1(LX1,LY1)
     $             ,  DSM1(LX1,LX1),DSTM1(LX1,LY1)
     $             ,  WGS(LX1,LY1)
      COMMON /SCRMG/  XRM1(LX1,LY1),YRM1(LX1,LY1),ZRM1(LX1,LY1)
     $             ,  XSM1(LX1,LY1),YSM1(LX1,LY1),ZSM1(LX1,LY1)
      COMMON /SCRUZ/  S1X(LX1,LY1),S1Y(LX1,LY1),S1Z(LX1,LY1)
     $             ,  S2X(LX1,LY1),S2Y(LX1,LY1),S2Z(LX1,LY1)
      COMMON /SCRNS/  G1X(LX1,LY1),G1Y(LX1,LY1),G1Z(LX1,LY1)
     $             ,  G2X(LX1,LY1),G2Y(LX1,LY1),G2Z(LX1,LY1)
     $             ,  GBS(LX1,LY1),GB1L(LX1,LY1),GB2L(LX1,LY1)
C
      DIMENSION TRX(LX1,LY1,LZ1),TRY(LX1,LY1,LZ1),TRZ(LX1,LY1,LZ1)
      DIMENSION SIGST(LX1,LY1)
C
      NXY1 = lx1*ly1
C
      CALL RZERO3 (S1X,S1Y,S1Z,NXY1)
      CALL RZERO3 (S2X,S2Y,S2Z,NXY1)
      CALL FACEXV (XFM1,YFM1,ZFM1,XM1(1,1,1,IEL),YM1(1,1,1,IEL),
     $             ZM1(1,1,1,IEL),IFC,0)
      CALL SETDRS (DRM1,DRTM1,DSM1,DSTM1,IFC)
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
         GB1X=XRM1(IX,IY)
         GB1Y=YRM1(IX,IY)
         GB1Z=ZRM1(IX,IY)
         GB2X=XSM1(IX,IY)
         GB2Y=YSM1(IX,IY)
         GB2Z=ZSM1(IX,IY)
         GB11=GB1X*GB1X + GB1Y*GB1Y + GB1Z*GB1Z
         GB12=GB1X*GB2X + GB1Y*GB2Y + GB1Z*GB2Z
         GB22=GB2X*GB2X + GB2Y*GB2Y + GB2Z*GB2Z
         GDET=GB11*GB22 - GB12*GB12
         IF (GDET .LT. 1.E-20) GO TO 9001
         GT11= GB22/GDET
         GT12=-GB12/GDET
         GT22= GB11/GDET
         GB1L(IX,IY)=SQRT(GB11)
         GB2L(IX,IY)=SQRT(GB22)
         GBS (IX,IY)=SQRT(GDET)
         WGS (IX,IY)=WXM1(IX)*WYM1(IY)*SIGST(IX,IY)
         BB = GBS(IX,IY) * WGS(IX,IY)
         G1X(IX,IY) = BB * ( GT11*GB1X + GT12*GB2X )
         G1Y(IX,IY) = BB * ( GT11*GB1Y + GT12*GB2Y )
         G1Z(IX,IY) = BB * ( GT11*GB1Z + GT12*GB2Z )
         G2X(IX,IY) = BB * ( GT12*GB1X + GT22*GB2X )
         G2Y(IX,IY) = BB * ( GT12*GB1Y + GT22*GB2Y )
         G2Z(IX,IY) = BB * ( GT12*GB1Z + GT22*GB2Z )
  100    CONTINUE
C
      CALL MXM (DRTM1,lx1,G1X,lx1,S1X,ly1)
      CALL MXM (DRTM1,lx1,G1Y,lx1,S1Y,ly1)
      CALL MXM (DRTM1,lx1,G1Z,lx1,S1Z,ly1)
C
      CALL MXM (G2X,lx1,DSM1,ly1,S2X,ly1)
      CALL MXM (G2Y,lx1,DSM1,ly1,S2Y,ly1)
      CALL MXM (G2Z,lx1,DSM1,ly1,S2Z,ly1)
C
      CALL ADD2 (S1X,S2X,NXY1)
      CALL ADD2 (S1Y,S2Y,NXY1)
      CALL ADD2 (S1Z,S2Z,NXY1)
C
C     Contact angle option on hold
C
C      ICONTAC=INT(BC2)
C      IF (ICONTAC.NE.0) THEN
C         IX=1
C         IY=1
C         IF (ICONTAC.GE.3) IY=ly1
C         IF (ICONTAC.EQ.2 .OR. ICONTAC.EQ.3) IX=lx1
C         ANG = BC3 * PI / 180.00
C         RR  = YM1(IX,IY,IZ,IEL)
C         TRX(IX,IY,IZ)=TRX(IX,IY,IZ) + RR*SIGST*COS( ANG )
C         TRY(IX,IY,IZ)=TRY(IX,IY,IZ) + RR*SIGST*SIN( ANG )
C      ENDIF
C
      CALL FACSUB2 (TRX,TRY,TRZ,S1X,S1Y,S1Z,IFC)
C
      RETURN
C
 9001 WRITE ( 6,*) 'Zero area for Element=',IEL,'    Face=',IFC
      call exitt
C
      END
c-----------------------------------------------------------------------
      SUBROUTINE SETDRS (DRM1,DRTM1,DSM1,DSTM1,IFC)
C
      INCLUDE 'SIZE'
      INCLUDE 'DXYZ'
C
      DIMENSION DRM1(LX1,LX1),DRTM1(LX1,LX1)
     $        , DSM1(LY1,LY1),DSTM1(LY1,LY1)
C
      NXY1=lx1*ly1
C
      IF (IFC.EQ.5 .OR. IFC.EQ.6) THEN
         CALL COPY (DRM1 ,DXM1 ,NXY1)
         CALL COPY (DSM1 ,DYM1 ,NXY1)
         CALL COPY (DRTM1,DXTM1,NXY1)
         CALL COPY (DSTM1,DYTM1,NXY1)
      ELSEIF (IFC.EQ.2 .OR. IFC.EQ.4) THEN
         CALL COPY (DRM1 ,DYM1 ,NXY1)
         CALL COPY (DSM1 ,DZM1 ,NXY1)
         CALL COPY (DRTM1,DYTM1,NXY1)
         CALL COPY (DSTM1,DZTM1 ,NXY1)
      ELSE     
         CALL COPY (DRM1 ,DZM1 ,NXY1)
         CALL COPY (DSM1 ,DXM1 ,NXY1)
         CALL COPY (DRTM1,DZTM1,NXY1)
         CALL COPY (DSTM1,DXTM1,NXY1)
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE GLOBROT (R1,R2,R3,IEL,IFC)
C
C     Rotate vector components R1,R2,R3 at face IFC 
C     of element IEL from local to global system.
C
C     R1, R2, R3 have the (NX,NY,NZ) data structure
C     IFACE1 is in the preprocessor notation 
C     IFACE  is the dssum notation.
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'TOPOL'
C
      DIMENSION R1(LX1,LY1,LZ1)
     $        , R2(LX1,LY1,LZ1)
     $        , R3(LX1,LY1,LZ1)
C
      CALL DSSET (lx1,ly1,lz1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
      I = 0
C
      IF (ldim.EQ.2) THEN
         DO 200 J2=JS2,JF2,JSKIP2
         DO 200 J1=JS1,JF1,JSKIP1
            I = I+1
            RNORL = R1(J1,J2,1)
            RTAN1 = R2(J1,J2,1)
            R1(J1,J2,1) = RNORL*UNX(I,1,IFC,IEL) +
     $                    RTAN1*T1X(I,1,IFC,IEL)
            R2(J1,J2,1) = RNORL*UNY(I,1,IFC,IEL) +
     $                    RTAN1*T1Y(I,1,IFC,IEL)
  200    CONTINUE
      ELSE
         DO 300 J2=JS2,JF2,JSKIP2
         DO 300 J1=JS1,JF1,JSKIP1
            I = I+1
            RNORL = R1(J1,J2,1)    
            RTAN1 = R2(J1,J2,1)    
            RTAN2 = R3(J1,J2,1)    
            R1(J1,J2,1) = RNORL*UNX(I,1,IFC,IEL) +
     $                    RTAN1*T1X(I,1,IFC,IEL) +
     $                    RTAN2*T2X(I,1,IFC,IEL)
            R2(J1,J2,1) = RNORL*UNY(I,1,IFC,IEL) +
     $                    RTAN1*T1Y(I,1,IFC,IEL) +
     $                    RTAN2*T2Y(I,1,IFC,IEL)
            R3(J1,J2,1) = RNORL*UNZ(I,1,IFC,IEL) +
     $                    RTAN1*T1Z(I,1,IFC,IEL) +
     $                    RTAN2*T2Z(I,1,IFC,IEL)
  300       CONTINUE
         ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE FACEC2 (A1,A2,B1,B2,IFC)
C
C     2-D Geometry only
C     Extract A1,A2 from B1,B2 on surface IFC.
C
C     A1, A2 have the (lx1,  1,NFACE) data structure
C     B1, B2 have the (lx1,ly1,    1) data structure
C
      INCLUDE 'SIZE'
C
      DIMENSION A1(LX1),A2(LX1),B1(LX1,LY1),B2(LX1,LY1)
C
      IX=1
      IY=1
      IF (IFC.EQ.1 .OR. IFC.EQ.3) THEN
         IF (IFC.EQ.3) IY = ly1
         DO 10 IX=1,lx1
            A1(IX)=B1(IX,IY)
            A2(IX)=B2(IX,IY)
   10    CONTINUE
      ELSE
         IF (IFC.EQ.2) IX = lx1
         DO 20 IY=1,ly1
            A1(IY)=B1(IX,IY)
            A2(IY)=B2(IX,IY)
   20   CONTINUE
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE LFALSE (IFA,N)
      LOGICAL IFA(1)
      DO 100 I=1,N
      IFA(I)=.FALSE.
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE RZERO3 (A,B,C,N)
      DIMENSION A(1),B(1),C(1)
      DO 100 I=1,N
         A(I)=0.0
         B(I)=0.0
         C(I)=0.0
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE UNITVEC (X,Y,Z,N)
      DIMENSION X(1),Y(1),Z(1)
      DO 100 I=1,N
      XLNGTH = SQRT( X(I)**2 + Y(I)**2 + Z(I)**2 )
      IF (XLNGTH.NE.0.0) THEN
         X(I) = X(I)/XLNGTH
         Y(I) = Y(I)/XLNGTH
         Z(I) = Z(I)/XLNGTH
      ENDIF
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE CHKZVN (VMAX,IEL,IFC,IVNORL)
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'SOLN'
      COMMON /SCRMG/ V1(LX1,LY1,LZ1,LELV)
     $             , V2(LX1,LY1,LZ1,LELV)
     $             , V3(LX1,LY1,LZ1,LELV)
     $             , VV(LX1,LY1,LZ1,LELV)
C
      NXZ1  = lx1*lz1
      TOLV  = 0.01*VMAX
C
      VNOR1 = FACDOT(V1(1,1,1,IEL),UNX(1,1,IFC,IEL),IFC)
      VNOR2 = FACDOT(V2(1,1,1,IEL),UNY(1,1,IFC,IEL),IFC)
      VNOR  = VNOR1 + VNOR2
      IF (ldim.EQ.3) THEN
          VNOR3 = FACDOT(V3(1,1,1,IEL),UNZ(1,1,IFC,IEL),IFC)
          VNOR  = VNOR + VNOR3
      ENDIF
      VNOR = ABS(VNOR) / NXZ1
C
      IVNORL = 1
      IF (VNOR .LT. TOLV) IVNORL = 0
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE BCTWALL (TMP1,TMP2,TMP3)
C
C     Apply Dirichlet boundary conditions to surface of vector (V1,V2,V3)
C     (No antimask operation is applied).
C
      INCLUDE 'SIZE'
      INCLUDE 'GEOM'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
C
      DIMENSION TMP1(lx1,ly1,lz1,1)
     $        , TMP2(lx1,ly1,lz1,1)
     $        , TMP3(lx1,ly1,lz1,1)
      common  /nekcb/ cb
      CHARACTER CB*3
C
      NFACE = 2*ldim
      NTOT1 = lx1*ly1*lz1*NELV
C
      CALL RZERO (TMP1,NTOT1)
      CALL RZERO (TMP2,NTOT1)
      IF (IF3D) CALL RZERO (TMP3,NTOT1)
C
      DO 2000 IEL=1,NELV
      DO 2000 IFC=1,NFACE
         CB  = CBC (IFC,IEL,IFIELD)
         BC1 = BC(1,IFC,IEL,IFIELD)
         BC2 = BC(2,IFC,IEL,IFIELD)
         BC3 = BC(3,IFC,IEL,IFIELD)
         IF (CB.EQ.'V  ' .OR. CB.EQ.'VL '  .OR.
     $       CB.EQ.'WS ' .OR. CB.EQ.'WSL') THEN
             CALL FACEV (TMP1,IEL,IFC,BC1,lx1,ly1,lz1)
             CALL FACEV (TMP2,IEL,IFC,BC2,lx1,ly1,lz1)
             IF (ldim.EQ.3) CALL FACEV (TMP3,IEL,IFC,BC3,lx1,ly1,lz1)
             IF (CB.EQ.'VL ' .OR. CB.EQ.'WSL')
     $       CALL GLOBROT (TMP1(1,1,1,IEL),TMP2(1,1,1,IEL),
     $                     TMP3(1,1,1,IEL),IEL,IFC)
         ENDIF
         IF (CB.EQ.'v  ' .OR. CB.EQ.'vl ' .OR. 
     $       CB.EQ.'ws ' .OR. CB.EQ.'wsl' .OR.
     $       CB.EQ.'mv ' .OR. CB.EQ.'mvn') THEN
             CALL FACEIV (CB,TMP1(1,1,1,IEL),TMP2(1,1,1,IEL),
     $                    TMP3(1,1,1,IEL),IEL,IFC,lx1,ly1,lz1)
             IF (CB.EQ.'vl ' .OR. CB.EQ.'wsl')
     $       CALL GLOBROT (TMP1(1,1,1,IEL),TMP2(1,1,1,IEL),
     $                     TMP3(1,1,1,IEL),IEL,IFC)
         ENDIF
 2000 CONTINUE
C
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE ANTIMSK1(X,XMASK,N)
C------------------------------------------------------------------
C
C     Return only Dirichlet boundary values of X
C
C-------------------------------------------------------------------
      REAL  X(1),XMASK(1)
      include 'OPCTR'
C
      DO 100 I=1,N
         X(I) = X(I)*(1.-XMASK(I))
 100  CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine check_cyclic  ! check for cyclic bcs
      include 'SIZE'
      include 'TOTAL'

      common /scrmg/ v1(lx1,ly1,lz1,lelt)
     $             , v2(lx1,ly1,lz1,lelt)
     $             , v3(lx1,ly1,lz1,lelt)

      integer e,f

      nface = 2*ldim

      n = lx1*ly1*lz1*nelt
      call rzero(v1,n)
      call rzero(v2,n)
      call rzero(v3,n)

      ifield = 1
      do e=1,nelt   ! possibly U or B field
      do f=1,nface

         if (cbc(f,e,ifield).eq.'P  '.or.cbc(f,e,ifield).eq.'p  ') then
            call facind2 (js1,jf1,jskip1,js2,jf2,jskip2,f)
            k = 0
            do j2=js2,jf2,jskip2
            do j1=js1,jf1,jskip1
               k = k+1
               v1(j1,j2,1,e) = unx(j1,j2,1,e)
               v2(j1,j2,1,e) = uny(j1,j2,1,e)
               v3(j1,j2,1,e) = unz(j1,j2,1,e)
            enddo
            enddo
         endif

      enddo
      enddo

      ifcyclic = .false.
      call opdssum(v1,v2,v3)

      eps = 1.e-4
      if (ldim.eq.2) call rzero(v3,n)

      do e=1,nelt   ! Check for turning angle
      do f=1,nface

         if (cbc(f,e,ifield).eq.'P  '.or.cbc(f,e,ifield).eq.'p  ') then

            call facindr(i0,i1,j0,j1,k0,k1,lx1,ly1,lz1,f) ! restricted indx
            snorm = 0.
            dnorm = 0.
            do k=k0,k1
            do j=j0,j1
            do i=i0,i1
               snorm = abs(v1(i,j,k,e))
     $               + abs(v2(i,j,k,e))
     $               + abs(v3(i,j,k,e))
            enddo
            enddo
            enddo
            if (snorm.gt.eps) ifcyclic = .true.

         endif

      enddo
      enddo

      itest = 0
      if (ifcyclic) itest = 1
      itest = iglmax(itest,1)

      if (itest.gt.0) ifcyclic = .true.

      return
      end
c-----------------------------------------------------------------------
      real function glcflux(tx,ty,tz)
c
      include 'SIZE'
      include 'TOTAL'

      real tx(lx1,ly1,lz1,lelv)
      real ty(lx1,ly1,lz1,lelv)
      real tz(lx1,ly1,lz1,lelv)

      character cb*3

      nxyz1= lx1*ly1*lz1
      ntot1= nxyz1*nelv
      nfaces = 2*ldim

      termA = 0.0
      termVL= 0.0

      do 100 iel=1,nelv
      do 100 iface=1,nfaces
         cb = cbc(iface,iel,1)
         if (cb.eq.'v  ' .or. cb.eq.'V  ' .or. cb.eq.'mv ') then
            call facind(kx1,kx2,ky1,ky2,kz1,kz2,lx1,ly1,lz1,iface)
            ia = 0
            do 10 iz=kz1,kz2
            do 10 iy=ky1,ky2
            do 10 ix=kx1,kx2
               ia =ia + 1
               termxyz = tx(ix,iy,iz,iel)*unx(ia,1,iface,iel)
     $                 + ty(ix,iy,iz,iel)*uny(ia,1,iface,iel)
     $                 + tz(ix,iy,iz,iel)*unz(ia,1,iface,iel)
               termA  = termA + area(ia,1,iface,iel)
               termVL = termVL+ termxyz * area(ia,1,iface,iel)
 10         continue
         endif
 100  continue

      glcflux = glsum(termVL,1)  ! sum across processors

      return
      end
c-----------------------------------------------------------------------
      subroutine local_bflux(flux,tx,ty,tz,ifld)
c
      include 'SIZE'
      include 'TOTAL'

      real   tx(lx1,ly1,lz1,1),
     $       ty(lx1,ly1,lz1,1),
     $       tz(lx1,ly1,lz1,1),
     $     flux(lx1,ly1,lz1,1)

      character cb*3

      nel    = nelfld(ifld)
      nxyz   = lx1*ly1*lz1
      ntot   = nxyz*nel
      nfaces = 2*ldim

      call rzero(flux,ntot)

      do 100 iel=1,nel
      do 100 iface=1,nfaces
         cb = cbc(iface,iel,ifld)
         if (cb.ne.'E  ') then
            call facind(kx1,kx2,ky1,ky2,kz1,kz2,lx1,ly1,lz1,iface)
            ia = 0
            do 10 iz=kz1,kz2
            do 10 iy=ky1,ky2
            do 10 ix=kx1,kx2
               ia =ia + 1
               dtmp =   tx(ix,iy,iz,iel)*unx(ia,1,iface,iel)
     $                + ty(ix,iy,iz,iel)*uny(ia,1,iface,iel)
     $                + tz(ix,iy,iz,iel)*unz(ia,1,iface,iel)
               flux(ix,iy,iz,iel) = flux(ix,iy,iz,iel)
     $                              + dtmp*area(ia,1,iface,iel)
 10         continue
         endif
 100  continue

      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE FACIND2 (JS1,JF1,JSKIP1,JS2,JF2,JSKIP2,IFC)
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
      CALL DSSET (lx1,ly1,lz1)
      IFACE  = EFACE1(IFC)
      JS1    = SKPDAT(1,IFACE)
      JF1    = SKPDAT(2,IFACE)
      JSKIP1 = SKPDAT(3,IFACE)
      JS2    = SKPDAT(4,IFACE)
      JF2    = SKPDAT(5,IFACE)
      JSKIP2 = SKPDAT(6,IFACE)
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine create_obj(iobjo,sid_list,n)
c
c     defines an object for a given list of surface ids
c
      include 'SIZE'
      include 'TOTAL'

      integer sid_list(n)

      integer e,f

      nobj = nobj + 1
      iobj = nobj

      if (maxobj.lt.nobj)
     $   call exitti('maxobj too small, increate in SIZE.$',ierr)

      do e=1,nelv
      do f=1,2*ndim
      do i=1,n
         if (boundaryID(f,e) .eq. sid_list(i)) then
            nmember(iobj) = nmember(iobj) + 1
            mem = nmember(iobj)
            ieg = lglel(e)
            object(iobj,mem,1) = ieg
            object(iobj,mem,2) = f
c            write(6,1) iobj,mem,f,ieg,e,nid,' OBJ'
    1       format(6i9,a4)
         endif
      enddo
      enddo
      enddo

      iobjo  = iobj

      return
      end
c-----------------------------------------------------------------------
      subroutine setbc(bid,ifld,cbci)
c
c     sets boundary condition for a given surface id and field
c
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'

      character*3 cbci
      integer bid

      if (bid.lt.1 .or. bid.gt.lbid)
     $  call exitti('invalid boundary id!$',bid)

      cbc_bmap(bid,ifld) = cbci
 
      if (iftmsh(ifld)) then
        do iel = 1,nelt
        do ifc = 1,2*ndim
           if (boundaryIDt(ifc,iel).eq.bid)
     $       cbc(ifc,iel,ifld) = cbc_bmap(bid,ifld)
        enddo
        enddo
      else
        do iel = 1,nelv
        do ifc = 1,2*ndim
           if (boundaryID(ifc,iel).eq.bid)
     $       cbc(ifc,iel,ifld) = cbc_bmap(bid,ifld)
        enddo
        enddo
      endif

      return
      end
