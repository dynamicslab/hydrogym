C------------------------------------------------------------------------------
C
C                          NEKTON 2.6  2/8/90
C
C			Copyright (C) 1990, by the 
C
C		Massachusetts Institute of Technology  and Nektonics, Inc.
C
C All Rights Reserved
C
C This program is a licenced product of MIT and Nektonics, Inc.,  and it is 
C not to be disclosed to others, copied, distributed, or displayed 
C without prior authorization.
C
C------------------------------------------------------------------------------
C
      subroutine FPREP
c     PROGRAM POSTNK
C23456789012345678901234567890123456789012345678901234567890123456789012
C     PostProcessor for Spectral element code.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      common /quantc/ quanto
      character*20 quanto
c
      LOGICAL IFTMP
      COMMON /FILLFG/ IFILLM
      character*12 menu_hdr
      common /regularize/ regdir
      character*1  regdir

      integer buf(30)
      logical ifbswap
      
c
C
      CALL INIT0
c
      CALL DATA
      CALL INIT
      CALL GNABLE
      CALL HARD
c
c     8/11/95 pff
c
      call initstate
c
c     Force initialization of interp
      call interp(v1,xp,yp,zp,idum,wkv1,ierr)
c
      call setwrk(.TRUE.)
c
      call drmesh
      CALL DRFRAM
      CALL NOHARD
c
1     CONTINUE
c
C     Don't need to draw menu first time around; menu does it automatically
                                  nchoic=1
c     this will be commented out later
c     ITEM(nchoic)='CUTTING PLANE'
c                                 nchoic=nchoic+1
c     ITEM(nchoic)='set_surf_layer'
c                                 nchoic=nchoic+1
      ITEM(nchoic)='EXIT'
                                  nchoic=nchoic+1
      ITEM(nchoic)='CLEAR'
                                  nchoic=nchoic+1
      ITEM(nchoic)='PLOT'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET QUANTITY'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET LOCATION'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET PLOT FORMAT'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET TIME'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET ATTRIBUTE'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SET SCREEN'
                                  nchoic=nchoic+1
      ITEM(nchoic)='SAVE STATE'
                                  nchoic=nchoic+1
      ITEM(nchoic)='MAKE MOVIE'
                                  nchoic=nchoic+1
      ITEM(nchoic)='OUTPUT FILE'
c                                 nchoic=nchoic+1
c     ITEM(nchoic)='REDRAW'
c
      write(menu_hdr,66) nel,nx
   66 format('K=',i5,' N=',i2)
      CALL MENU(XMOUSE,YMOUSE,BUTTON,menu_hdr)
C
C     Check if it needs to choose a dump
      IF( CHOICE.EQ.'SET QUANTITY')THEN
C        Here we allow the user to set the quantity to any the can be
C        Dumped with this eqtype.  If the data was not dumped (in output
C        field, or in history) he will get an error message when he tries
C        the actual plo
263      CONTINUE
         NCHOIC=0
c
         NCHOIC=NCHOIC+1
         ITEM(NCHOIC)='UP MENU'
c
         IF(IFXYH)THEN
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='COORDINATE'
         ENDIF
         IF(IFFLOW)THEN
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='VELOCITY'
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='VORTICITY'
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='PRESSURE'
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='TOTAL PRESSURE'
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='DIVERGENCE'
            if (if3d) then
               NCHOIC=NCHOIC+1
               ITEM(NCHOIC)='STRESS VEC'
               NCHOIC=NCHOIC+1
               ITEM(NCHOIC)='NORMAL VEC'
            endif
            IF(NSOBJS.GT.0)THEN
               NCHOIC=NCHOIC+1
               ITEM(NCHOIC)='FORCE'
            ENDIF
            IF(NDIM.EQ.2)THEN
               NCHOIC=NCHOIC+1
               ITEM(NCHOIC)='STREAMFUNCTION'
            ENDIF
         ENDIF
         IF(IFTO)THEN
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='TEMPERATURE'
c           NCHOIC=NCHOIC+1
c           ITEM(NCHOIC)='FLUX' ??Should Flux be temperature in format vector?
         ENDIF
c
         NCHOIC=NCHOIC+1
         ITEM(NCHOIC)='USER FUNC'
c
c        NCHOIC=NCHOIC+1
c        ITEM(NCHOIC)='Proc. Map'
c
c        if (if3d.and.ifflow) NCHOIC=NCHOIC+1
c        if (if3d.and.ifflow) ITEM(NCHOIC)='VORTEX'
c
         NCHOIC=NCHOIC+1
         ITEM(NCHOIC)='FILTER'
c
         if (ifflow) NCHOIC=NCHOIC+1
         if (ifflow) ITEM(NCHOIC)='VORTEX'
c
c        if (if3d.and.ifflow) NCHOIC=NCHOIC+1
c        if (if3d.and.ifflow) ITEM(NCHOIC)='V.W'
c
         NCHOIC=NCHOIC+1
         ITEM(NCHOIC)='Linear Combination'
c        NCHOIC=NCHOIC+1
c        ITEM(NCHOIC)='rms-avg'
c        NCHOIC=NCHOIC+1
c        ITEM(NCHOIC)='rms-uvw'

         nnops=nchoic

         write(6,*)
     $     'npscal',npscal,nchoic,param(23),(ifpsco(k),k=1,npscal)

         if(npscal.gt.0)then
            do 1221 i=1,npscal
               write(6,*)
     $         'npsca2',npscal,nchoic,param(23),ifpsco(i),psname(i)
               if (ifpsco(i)) then ! we dumped this passive scalar
                  nchoic=nchoic+1
                  item(nchoic)=psname(i)
               endif
1221        continue
         endif

c        NCHOIC=NCHOIC+1
c        IF(.NOT.IFERROR)ITEM(NCHOIC)='ERROR NO'
c        IF(     IFERROR)ITEM(NCHOIC)='ERROR YES'
         NCHOIC=NCHOIC+1
         LINE='DERIVATIVE '
         LINE(12:16)=DERIV
         ITEM(NCHOIC)=LINE
         quanto = quanty
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET QUANTITY')
         IF(CHOICE(1:10).EQ.'DERIVATIVE')THEN
            ITEM(1)='NO'
            ITEM(2)='D/DX'
            ITEM(3)='D/DY'
            ITEM(4)='D/DZ'
            ITEM(5)='GRAD'
            NCHOIC=5
            CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET DERIVATIVE')
            DERIV=CHOICE
            GO TO 263
         ENDIF
         QUANTY=CHOICE
         IF(QUANTY.EQ.'ERROR YES')THEN
            ITEM(NCHOIC)='ERROR NO'
            IFERROR=.FALSE.
            GO TO 263
         ELSE IF(QUANTY.EQ.'ERROR NO')THEN
            ITEM(NCHOIC)='ERROR YES'
            IFERROR=.TRUE.
            GO TO 263
         ENDIF
         IF(ICHOIC.GT.NNOPS)QUANTY='PASSIVE SCALAR'
         LPSDMP=ICHOIC-NNOPS
         DO 119 I=1,NPSCAL
            IF(CHOICE.EQ.PSNAME(I))NPSDMP=I
119      CONTINUE
         IF(QUANTY.EQ.'COORDINATE'.OR.QUANTY.EQ.'FORCE')THEN
            CALL PRS(' ENTER COMPONENT (For scalar plots)$')
            ITEM(1)='X'
            ITEM(2)='Y'
            NCHOIC=2
            IF(IF3D)THEN
               ITEM(3)='Z'
               NCHOIC=NCHOIC+1
            ENDIF
            CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET COMPONENT')
            COMPON=CHOICE(1:1)
         ENDIF
         IF(QUANTY.EQ.'VELOCITY'              .OR.
     $     (QUANTY.EQ.'VORTICITY'.AND.IF3D)   .OR.
     $      DERIV .EQ.'GRAD'                ) THEN
            IF(.NOT.IF3D)CALL PRS(' ENTER COMPONENT (FOR X-Y PLOTS)$')
            IF(IF3D)CALL PRS(' ENTER COMPONENT (For scalar plots)$')
            ITEM(1)='X'
            ITEM(2)='Y'
            NCHOIC=2
            IF(IF3D)THEN
               ITEM(3)='Z'
               ITEM(4)='Normal to Plane'
               NCHOIC=NCHOIC+2
            ELSE
               ITEM(3)='Normal to line'
               NCHOIC=NCHOIC+1
            ENDIF
            NCHOIC=NCHOIC+1
            ITEM(NCHOIC)='Magnitude'
            if (quanty.eq.'VORTICITY') then
               NCHOIC=NCHOIC+1
               ITEM(NCHOIC)='Helicity'
            endif
            CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET COMPONENT')
            COMPON=CHOICE(1:1)
         ENDIF
         IF(QUANTY.EQ.'STRESS VEC') SUATTR='VECTOR PLOT'
         IF(QUANTY.EQ.'NORMAL VEC') SUATTR='VECTOR PLOT'
C
C        End of QUANTITY Selection operation - set the quantity.
C                                                    7-1-90 pff
c
         if (QUANTY.EQ.'Linear Combination') then
            call lin_combo
            quanty=quanto
            call setwrk(.true.)
         elseif (QUANTY.EQ.'rms-avg') then
            call rms_avg
            quanty=quanto
            call setwrk(.true.)
         elseif (QUANTY.EQ.'rms-uvw') then
            call rms_uvw
            quanty=quanto
            call setwrk(.true.)
         else
            call setwrk(.false.)
         endif
C
      ELSE IF( CHOICE.EQ.'SET TIME')THEN
         CALL PRS('Input dump number:$')
         CALL REI(NEWDUMP)
         IF(IFLEARN)WRITE(3,*) NEWDUMP,' Dump Number.'
         IF (PLTMOD.EQ.'VOLUME') THEN
            call getfld(newdump,ierr,.true.,.true.)
            if (ierr.ne.0) then
               CALL PRSI('Unable to open dump number:$',newdump)
c           else
c              this is already embedded in getfld  (pff 6/26/99)
c              call setwrk(.FALSE.)
            endif
         ELSE
            NOPEN=NEWDUMP-1
         ENDIF
      ELSE IF( CHOICE.EQ.'PLOT')THEN
         IF (PLFORM.NE.'MULTIPLOT') THEN
            IFMLTI=.FALSE.
            CALL HARD
            if (ifportrait) then
               CALL CLEAR
               CALL DRMESH
               CALL PLOTIT(1)
            else
               CALL PLOTIT(1)
            endif
            CALL NOHARD
         ELSE
            IFMLTI=.TRUE.
            CALL HARD
            CALL MLTIPLT
            CALL NOHARD
         ENDIF
      ELSE IF(CHOICE.EQ.'EXIT')THEN
         CALL PRS('CONFIRM by clicking RIGHT button.$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF (BUTTON.EQ.'RIGHT') THEN
            CALL dmpstate
            CALL CLEAR
            CALL GINDIS
            sesion(11:14)='   '
            CALL PRS('Deleting tmp.* files.$')
            CALL DELTMP
            CALL PRS('Exiting '//sesion//'Postnek Session$')
            CALL DEVEX
            CALL EXITT
         ENDIF
      ELSE IF (CHOICE.EQ.'CLEAR') THEN
         call clear
         CALL HARD
         CALL HEJECT
         CALL DRMESH
         CALL DRFRAM
         CALL NOHARD
      ELSE IF (CHOICE.EQ.'SHOW') THEN
         CALL SHOW(ARG,IA)
      ELSE IF(CHOICE.EQ.'CUTTING PLANE') THEN
         call cutting_plane
      ELSE IF(CHOICE.EQ.'set_surf_layer') THEN
         call set_surf_layer(-99.,99.,-99.,99.,-99.,99.)
      ELSE IF (CHOICE.EQ.'SET LOCATION') THEN
         CALL SETLOC
      ELSE IF (CHOICE.EQ.'SET ATTRIBUTE') THEN
         CALL SETATT
      ELSE IF (CHOICE.EQ.'SET PLOT FORMAT') THEN
c
C        PLFORM specifies 0,1, or 2 Dimensional plot
c
                                                     nchoic = 1
         ITEM(nchoic)='UP MENU'
c                                                    nchoic = nchoic+1
c        ITEM(nchoic)='VALUE AT POINT'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='X-Y PLOT'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='SURFACE PLOT'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='VALUES'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='VOLUME'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='MULTIPLOT'
c                                                    nchoic = nchoic+1
c        ITEM(nchoic)='ANIMATE'
c                                                    nchoic = nchoic+1
c        ITEM(nchoic)='SURFACE ONLY'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='DRAW MESH NODES'
                                                     nchoic = nchoic+1
         ITEM(nchoic)='LABEL MESH NODES'
         if (ndim.eq.2)                              nchoic = nchoic+1
         if (ndim.eq.2) ITEM(nchoic)='DRAW MESH'
c
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET PLOT FORMAT')
         PLFORM=CHOICE
         IF(PLFORM.EQ.'VALUE AT POINT')THEN
         ELSE IF(PLFORM.EQ.'VOLUME')THEN
            CALL SETCVOL
         ELSE IF(PLFORM.EQ.'X-Y PLOT')THEN
c           ITEM(1)='HISTORY'
c           ITEM(2)='INTEGRAL QUANTITY'
c
            ITEM(1)='UP MENU'
            ITEM(2)='STREAMLINE'
            ITEM(3)='PROFILE'
            ITEM(4)='SET OF PROFILES'
            ITEM(5)='LINE OF VECTORS'
c           ITEM(6)='PLANAR MAX'
            NCHOIC=5
            CALL MENU(XMOUSE,YMOUSE,BUTTON,'X-Y PLOT')
            if (choice.ne.'UP MENU') XYATTR=CHOICE
         ELSE IF(PLFORM.EQ.'SURFACE PLOT'.or.PLFORM.EQ.'SURFACE ONLY')
     $    THEN
            ITEM(1)='SCALAR PLOT'
            ITEM(2)='VECTOR PLOT'
            ITEM(3)='ELEMENT PLOT'
            ITEM(4)='POINT PLOT'
            ITEM(5)='COMPUTE STRESSES'
            ITEM(6)='SURFACE CHECK'
            ITEM(7)='MESH SMOOTH'
c           ITEM(8)='Groove channel'
            ITEM(8)='Periodic Check'
            NCHOIC=8
            CALL MENU(XMOUSE,YMOUSE,BUTTON,'SURFACE PLOT')
            ifptplt=.false.
            SUATTR=CHOICE
            IF(CHOICE.EQ.'SCALAR PLOT') THEN
               ITEM(1)='MAIN MENU'
               ITEM(2)='FISHNET GRID'
               ITEM(3)='COLOR FILL'
               ITEM(4)='CONTOUR LINES'
               NCHOIC=4
               CALL MENU(XMOUSE,YMOUSE,BUTTON,'SCALAR PLOT')
               IF(CHOICE.NE.'MAIN MENU')SCALPT=CHOICE
            ELSE IF(CHOICE.EQ.'VECTOR PLOT') THEN
C              ONLY ARROW PLOT CURRENTLY AVAILABLE
            ELSE IF(CHOICE.EQ.'POINT PLOT') THEN
               ifptplt=.true.
            ELSE IF(CHOICE.EQ.'COMPUTE STRESSES') THEN
               call compute_stresses
            ELSE IF(CHOICE.EQ.'SURFACE CHECK') THEN
               call surfchk
            ELSE IF(CHOICE.EQ.'MESH SMOOTH') THEN
               call mesh_smooth
            ELSE IF(CHOICE.EQ.'Periodic Check') THEN
               call periodic_check
            ELSE IF(CHOICE.EQ.'Groove channel') THEN
               call compute_miles
            ELSE IF(CHOICE.EQ.'ELEMENT PLOT') THEN
  917          CALL PRS('Input element number to highlight: (0=quit)$')
               CALL REI(IE)
               IF (IE.GT.0) THEN
                  if (nhi.gt.lhi-1) nhi=0
                  nhi=nhi+1
                  ieh(nhi)=ie
                  CALL HILIGHT(IE,9)
                  GOTO 917
               ENDIF
               IF (IE.LT.0) nhi=0
               write(6,*) 'this is nhi:',nhi,ie
            ENDIF
         ENDIF
         IF (PLFORM.EQ.'SURFACE ONLY') THEN
            PLTMOD='SURFACE'
         ELSE
            PLTMOD='VOLUME'
         ENDIF
      ELSE IF(CHOICE.EQ.'MAKE MOVIE') THEN
         CALL MLTIPLT2
      ELSE IF(CHOICE.EQ.'REDRAW') THEN
         CALL REDRAW
      ELSE IF(CHOICE.EQ.'SAVE STATE') THEN
         call prs(
     $'Enter "0" if you wish to refresh the state buffer, -1 return.$')
         call prsi('Current number of states:$',nstat)
         call rei(is)
         if (is.eq.0) then
            nstat=1
            istat=1
            call savstate(istat,nstat,ir,ic,il,ii,'save')
         elseif (is.gt.0) then
            nstat=nstat+1
            istat=nstat
            call savstate(istat,nstat,ir,ic,il,ii,'save')
         endif
         ans='n'
c        call prs('Tile?$')
c        call res (ans,1)
         if (ans.eq.'y'.or.ans.eq.'Y') then

            rad = 1.154               ! P/D for Cheng & Todreas
            xzero_0 = xzero
            yzero_0 = yzero

            one = 1.
            pi  = 4.*atan(one)
            pi3 = pi/3.

            do i=0,5
               ang = i*pi3
               ca  = cos(ang)
               sa  = sin(ang)
            
               xzero=xzero_0+rad*ca
               yzero=yzero_0+rad*sa
C              reset clipping window
               wfr=1.3
               wt = yphy(.995)
               wb = yphy(.005)
               wl = xphy(.005)
               wr = xphy(.995)
               if (iffull) then
                  wr = xphy(wfr)
               endif
               nstat=nstat+1
               istat=nstat
               call savstate(istat,nstat,ir,ic,il,ii,'save')
            enddo

         endif

      ELSE IF(CHOICE.EQ.'OUTPUT FILE') THEN
         call fileout
      ELSE IF(CHOICE.EQ.'SET SCREEN') THEN
         CALL SETSCR
      ELSE IF(BUTTON.EQ.'RIGHT') THEN
         CALL hmt_refresh
c        call scrn_out
      ENDIF
      GO TO 1
      END
c-----------------------------------------------------------------------
      subroutine PLOTIT(iprompt)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER CVEL*20
C
C     Everything has been set up; check for reasonableness and then plot.
C     Set up screen

c     k=1
c     do ie=1,nel
c        call quickfill(work(k))
c        k=k+nx*ny*nz
c     enddo



      IF(IFERROR)THEN
C        Special overriding case-  you get color fill plot
         CALL PLERR
         return
      ENDIF
      IF(PLFORM.EQ.'X-Y PLOT'.OR.PLFORM.EQ.'VALUE AT POINT') THEN
C        ??!!FIX: 2-D RUNS SHOULD BE ABLE TO HANDLE NORMAL X-Y PLOTS
         IF(QUANTY.EQ.'VELOCITY' .AND. COMPON.EQ.'N'.AND.
     $     (XYATTR.NE.'PROFILE'.AND.XYATTR.NE.'SET OF PROFILES')) THEN
           CALL PRS
     $      ('COMPONENT OF VELOCITY SET TO ''NORMAL TO PLANE.''$')
           CALL PRS('COMPONENT MUST BE SET TO X,Y, OR Z FOR X-Y PLOT$')
           CALL PRS('OR POINT PLOT.$')
           return
         ENDIF
      ENDIF
C
      if (ifptplt) then
         call strplt(.true.)
         return
      endif
      IF(PLFORM.EQ.'X-Y PLOT') THEN
         IF(XYATTR.EQ.'HISTORY')THEN
            CALL HEJECT
            CALL HISTRY
         ELSE IF(XYATTR.EQ.'INTEGRAL QUANTITY')THEN
            CALL HEJECT
            CALL HISTRY
         ELSE IF(XYATTR.EQ.'STREAMLINE')THEN
            CALL STREAM
         ELSE IF(XYATTR.EQ.'PARTICLE PATH')THEN
            call particle_paths
         ELSE IF(XYATTR.EQ.'PLANAR MAX')THEN
            CALL HEJECT
            CALL nekplanar
         ELSE IF(XYATTR.EQ.'PROFILE')THEN
            CALL HEJECT
            CALL nekprofil
         ELSE IF(XYATTR.EQ.'LINE OF VECTORS')THEN
               CALL VFILES
         ELSE IF(XYATTR.EQ.'SET OF PROFILES')THEN
               call pfiles
         ENDIF
      ELSE IF(PLFORM.EQ.'VALUE AT POINT') THEN
         CALL VALUE
      ELSE IF(PLFORM.EQ.'ANIMATE')THEN
         CALL ANIMATE
      ELSE IF(PLFORM.EQ.'SURFACE ONLY')THEN
         write(6,*) 'calling srfplot'
         CALL SRFPLOT
      ELSE IF(PLFORM.EQ.'DRAW MESH') THEN
         CALL drwmsh
      ELSE IF(PLFORM.EQ.'DRAW MESH NODES') THEN
         CALL drwmsh2
      ELSE IF(PLFORM.EQ.'LABEL MESH NODES') THEN
         call prs('Place int(vx) next to each node.$')
         CALL drwmsh3
      ELSE IF(PLFORM.EQ.'VALUES') THEN
         CALL VALUES
C...     CALL VALUES  .... to be installed later   5 Feb 1989 15:11:39
      ELSE IF(PLFORM.EQ.'VOLUME') THEN
           CALL CVOLUME
      ELSE IF(PLFORM.EQ.'SURFACE PLOT') THEN
         IF(SUATTR.EQ.'SCALAR PLOT')THEN
            IF(QUANTY.EQ.'FLUX'       )             CALL TEM
            IF(QUANTY.EQ.'USER FUNC'  )             CALL TEM
            IF(QUANTY.EQ.'Proc. Map'  )             CALL TEM
            IF(QUANTY.EQ.'VELOCITY'   )             CALL TEM
            IF(QUANTY.EQ.'DIVERGENCE' )             CALL TEM
            IF(QUANTY.EQ.'T CONTOUR'  )             CALL TEM
            IF(QUANTY.EQ.'VORTICITY'  )             CALL TEM
            IF(QUANTY.EQ.'STREAMFUNCTION'  )        CALL TEM
            IF(QUANTY.EQ.'TEMPERATURE')             CALL TEM
            IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')    CALL TEM
            IF(QUANTY.EQ.'PRESSURE'   )             CALL TEM
            IF(QUANTY.EQ.'TOTAL PRESSURE')          CALL TEM
            IF(QUANTY.EQ.'VORTEX'        )          CALL TEM
            IF(QUANTY.EQ.'V.W'        )             CALL TEM
C
         ELSE IF(SUATTR.EQ.'VECTOR PLOT'.AND.DERIV .EQ.'GRAD') THEN
                                                    CALL VEL 
         ELSE IF(SUATTR.EQ.'VECTOR PLOT')THEN
            IF(DERIV .EQ.'GRAD'       )             CALL VEL 
C           Arrow plot !!?? Fix for other vectors (flux..)
            IF(QUANTY.EQ.'VELOCITY'   )             CALL VEL
            IF(QUANTY.EQ.'NORMAL VEC' )             CALL VEL
            IF(QUANTY.EQ.'USER FUNC'  )             CALL VEL 
            IF(QUANTY.EQ.'Proc. Map'  )             CALL VEL 
            IF(QUANTY.EQ.'T CONTOUR'  )             CALL TEM
            IF(QUANTY.EQ.'VORTICITY'.AND..NOT.IF3D) CALL TEM
            IF(QUANTY.EQ.'VORTICITY'.AND.     IF3D) CALL VEL
            IF(QUANTY.EQ.'DIVERGENCE' )             CALL TEM
            IF(QUANTY.EQ.'STREAMFUNCTION'  )        CALL TEM
            IF(QUANTY.EQ.'TEMPERATURE')             CALL TEM
            IF(QUANTY(1:14).EQ.'PASSIVE SCALAR')    CALL TEM
            IF(QUANTY.EQ.'PRESSURE'   )             CALL TEM
            IF(QUANTY.EQ.'TOTAL PRESSURE')          CALL TEM
            IF(QUANTY.EQ.'VORTEX'        )          CALL TEM
            IF(QUANTY.EQ.'V.W'        )             CALL TEM
            IF(QUANTY.EQ.'STRESS VEC' )             CALL TEMVEL ! new, 12/8/03 pff
         ENDIF
      ENDIF
      IF (IFMLTI) return
 
      if (.not.ifportrait) then
c
c        Put headers
c
c        next line moved 10/15/97 for true border free
         if (jhard.eq.3) ifhard=.false.
         IF(QUANTY.EQ.'VELOCITY') THEN
            WRITE(CVEL,'(G11.4,''$'')')UVMAX
            CALL GWRITE(XPHY(1.12),YPHY(0.79),1.0,CVEL)
            CALL GWRITE(XPHY(1.03),YPHY(0.82),1.0,'Max Velocity$')
         ENDIF
C        Draw box to overwrite any old time labels
         if (jhard.eq.3) ifhard=.false.
         YTOP =0.93
         YBOT =0.85
c        CALL FILLP(-4)
         CALL COLOR(13)
         CALL FILLP(-0)
         CALL BEGINB(XPHY(1.01),YPHY(YTOP))
         CALL MOVESC(     1.29 ,     YTOP)
         CALL MOVESC(     1.29 ,     YBOT)
         CALL MOVESC(     1.01 ,     YBOT)
         CALL MOVESC(     1.01 ,     YTOP)
         CALL ENDP
c
         IF(IFTRAN)THEN
            WRITE(CVEL,'(I11,''$'')')ISTEP
            CALL GWRITE(XPHY(1.10),YPHY(0.86),1.0,CVEL)
            CALL GWRITE(XPHY(1.03),YPHY(0.86),1.0,'Step$')
            WRITE(CVEL,'(G11.5,''$'')')TIME
         ELSE
            CVEL=' = Infinity$'
         ENDIF
C
         CALL GWRITE(XPHY(1.10),YPHY(0.90),1.0,CVEL)
         CALL GWRITE(XPHY(1.03),YPHY(0.90),1.0,'Time$')
         sesion(11:14)='$'
         if (jhard.eq.3) ifhard=.true.
      endif
C
      IF (iprompt.eq.1.and.
     $   PLFORM.EQ.'SURFACE PLOT'.AND..NOT.IFDEMO) THEN
         CALL PRS(
     $   'Hit left button for menu, right button for screen dump$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         index = 1
c
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c        IF(BUTTON.EQ.'RIGHT')  CALL grab_window('x12_movie\0',index)
c        IF(BUTTON.EQ.'RIGHT')  CALL SCRDMP
c        IF(BUTTON.EQ.'MIDDLE') CALL SCRUDMP
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine READLN(LOCATN)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
C       Reads a line of text.  Jumps to appropriate place on help.
C
C       Writes out line to read file; don't worry, if you don't like it
C       you can backspace the read file and write a cleaner line
C
C
      CHARACTER HFILE*20,HLINE*80
C
C1     READ(5,'(A70)',END=50,ERR=50) LINE
1     CALL RES(LINE,70)
      IF(IFLEARN)WRITE(3,'(A70)') LINE
      CALL CAPIT(LINE,2)
      IF(ANS.EQ.'H'.and.LINES(2).ne.'I')then
              ARG(2)=' '
              HFILE='nekton:helpnek.'
              WRITE(HFILE(16:18),'(I3)') LOCATN
              CALL OPENF(30,HFILE,'OLD',2,IERR)
              IF(IERR.EQ.0)THEN
                  DO 10 ILINE=1,1000
                     READ(30,'(A70)',END=11) HLINE
                     CALL PRS(HLINE//'$')
10                CONTINUE
11                CLOSE(UNIT=30)
              ENDIF
C
              CALL PRS(' <cr> to continue$')
              GO TO 1
      ENDIF
2     CONTINUE
C
      WRITE (X13,'(A70)') LINE
      return
50    IN=5
      GO TO 1
      END
c-----------------------------------------------------------------------
      subroutine DRMESH
C     Draws Mesh.  Draws walls white.
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION ICRVS(4),IWBC(4,NELM)
      INTEGER FCORNS (4,6)
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
c
      character*3 c3
      logical ifdhis

      real xx(3,9)


C     Draw Element Sides
C     DRISO Good for 2- or 3-D SIMULATION
C     Pseudo-hidden line removal:Draw in order of increasing zscxreen
C     and fill with color zero
C     Let's fill up a zscreen array (there's probably a better place for it
c
      if (.not. ifmesh) RETURN
c
      IF (IFXYO.OR.ifgngm) THEN
         CALL DRISOG
      ELSE
         CALL DRISO
      ENDIF

      do iel=1,4
        call rzero(xx,9)
        do j=5,8
         xx(1,1)=xx(1,1)+.25*x(iel,j)
         xx(2,1)=xx(2,1)+.25*y(iel,j)
         xx(3,1)=xx(3,1)+.25*z(iel,j)
        enddo
        do j=5,8
         xx(1,2)=max(xx(1,2),abs(xx(1,1)-x(iel,j)))
         xx(2,2)=max(xx(2,2),abs(xx(2,1)-y(iel,j)))
         xx(3,2)=max(xx(3,2),abs(xx(3,1)-z(iel,j)))
        enddo
        do j=5,8
         xxm=x(iel,j)!+ .12*(xx(1,1)-x(iel,j))**2 / xx(1,2)
         yym=y(iel,j)!+ .12*(xx(2,1)-x(iel,j))**2 / xx(2,2)
         zzm=z(iel,j)!+ .12*(xx(3,1)-x(iel,j))**2 / xx(3,2)
         if (j.eq.5) call g3writ(xxm,yym,zzm,1.0,'5$')
         if (j.eq.6) call g3writ(xxm,yym,zzm,1.0,'6$')
         if (j.eq.7) call g3writ(xxm,yym,zzm,1.0,'7$')
         if (j.eq.8) call g3writ(xxm,yym,zzm,1.0,'8$')
        enddo
      enddo

C     History points
c     IF (IFDMSH.AND.NNHIS.GT.0.AND.NDIM.EQ.2) THEN
c        CALL COLOR(15)
c        DO 62 I=1,NNHIS
c           IF(HCODE(10,I).EQ.'H'.OR.HCODE(10,I).EQ.'P')
c    $      CALL XDOT(XPTS(LOCHIS(1,I),LOCHIS(2,I),LOCHIS(4,I)),
c    $                YPTS(LOCHIS(1,I),LOCHIS(2,I),LOCHIS(4,I)))
62       CONTINUE
c        CALL COLOR(1)
c     ENDIF
c
c     Draw history points
c
      ifdhis = .true.
      if (ifdhis.and.nnhis.gt.0) then
         do i=1,nnhis
            write(c3,133) i
  133       format(i2,'$')
            call hard
c
            lochis(1,i) = min(nx,lochis(1,i))
            lochis(2,i) = min(ny,lochis(2,i))
            lochis(3,i) = min(nz,lochis(3,i))
c
            ix = lochis(1,i)
            iy = lochis(2,i)
            iz = lochis(3,i)
            ie = lochis(4,i)
c
            ii = ix + nx*(iy-1+ny*(iz-1+nz*(ie-1)))
c           write(6,144) i,ix,iy,iz,ie,ii,xp(ii),yp(ii),zp(ii)
c 144       format(4i4,2i8,1p3e12.4,' HIS')
            call g3writ   (xp(ii),yp(ii),zp(ii),1.0,c3)
            call diamd3   (xp(ii),yp(ii),zp(ii),9)
         enddo
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine READHS
      return
      END
c-----------------------------------------------------------------------
      subroutine COMAND(ARG,IA)
C       Main Subroutine which interprets commands and sends control of
C       the program flow to the proper branch.
C
      CHARACTER ANSWER,TEMP(40),FILE*10,
     $CVEL*20
      LOGICAL GAP
      INTEGER IAS(50),IES(50)
      CHARACTER*5 STRING
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
200   FORMAT(G11.4)
101   FORMAT(a1,a5)
201   FORMAT(G11.2)
202   FORMAT(I11)
C
C
      DO 300 I=1,NCMD
              IF(ARG(1).EQ.CMD(I)) return
              IF(ARG(1).EQ.'SHOW') return
              IF(ARG(1).EQ.'E         ') return
300   CONTINUE
      CALL PRS('No such command as  '//ARG(1)//' Exists$')
      CALL PRS('Available Commands:$')
      WRITE(S,'(5A13)')(CMD(I),I=1,NCMD)
      CALL PRS(S//'$')
      return
110   FORMAT(7I6)
120   FORMAT(2G15.7,2I6)
130   FORMAT(2I6,2G15.7,2X,A1)
112   FORMAT(G16.8,5X,A10)
      END
c-----------------------------------------------------------------------
      subroutine SETATT
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
c
      common /sfldwrk/ odumpw,odump
      integer odumpw,odump
c
c
C     ADD STUFF IN BASICSP  WIRE OR FILL; FRAME; SPECIFY INDUVIDUAL ELEMENTS;
C     ZOOM AND PAN; HIDE MENU? SET FILL COARSENESS;
1     CALL PRS('CURRENT SETUP:$')
      nchoic=1
      ITEM(nchoic) ='MAIN MENU'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET RESOLUTION'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET ANNOTATION'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET RANGE'
c                                              nchoic=nchoic+1
c     ITEM(nchoic) ='THRESHOLD LEVEL'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='Regularize in X,Y, or Z'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='Average UVWPT'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='DIRECT STIFFNESS SUM'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET MIRROR'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET PORTRAIT'
c                                              nchoic=nchoic+1
c     ITEM(nchoic) ='SET BDRY CONDITIONS'
c                                              nchoic=nchoic+1
c     ITEM(nchoic) ='SHIFT CG'
c                                              nchoic=nchoic+1
c     ITEM(nchoic) ='SHIFT MOMENT OF INERT.'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='if_output_ijke'
                                               nchoic=nchoic+1
      ITEM(nchoic) ='SET HARD COPY'
c
c
c     items previously in set attribute
c
c
c     ITEM(nchoic) ='SET DRAW AXIS'       ! moved to SET SCREEN menu. pff 3-3-90
c     ITEM(nchoic) ='SET COLORS'
c     ITEM(nchoic) ='SET ARROW SIZE'      ! moved to SETATT, pff 9/21/98
c     ITEM(nchoic) ='SET FISHNET HEIGHT'  ! moved to SETATT, pff 9/21/98
c     ITEM(nchoic)='GENERATE FEM MESH'
c
c
c
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET ATTRIBUTE')
      IF(CHOICE.EQ.'MAIN MENU') THEN
         return
c     ELSE IF(CHOICE.EQ.'SET BDRY CONDITIONS') THEN
c        call resetbcs
      ELSE IF(CHOICE.EQ.'SHIFT CG') THEN
         call shiftcg
      ELSE IF(CHOICE.EQ.'SHIFT MOMENT OF INERT.') THEN
         call shiftmom
      ELSE IF(CHOICE.EQ.'SET RESOLUTION') THEN
         CALL SETRES
      ELSE IF(CHOICE.EQ.'SET MIRROR') THEN
         if (nmirror.eq.2) then
            call prs('Setting mirror to .false.$')
            nmirror = 1
         else
            call prs('Setting mirror to .true.$')
            nmirror = 2
         endif
      ELSE IF(CHOICE.EQ.'DIRECT STIFFNESS SUM') THEN
         if (ifdssum) then
            call prs('Setting dssum to .false.$')
            ifdssum = .false.
         else
            call prs('Setting dssum to .true.$')
            ifdssum = .true.
         endif
      ELSE IF(CHOICE.EQ.'Regularize in X,Y, or Z') then
         call prs('Note: logic for reg.xyz not robust!$')
         write(6,*) 'ifregz:',ifregz
         if (.not.ifregz) then
            call prs('Regularize in X, Y, or Z?$')
            call res(regdir,1)
            ifregz=.true.
            write(6,*) 'call mapz ',regdir
            call mapz(u,regdir)
            call mapz(v,regdir)
            call mapz(w,regdir)
            call mapz(p,regdir)
            call mapz(t,regdir)
            call mapz(xp,regdir)
            call mapz(yp,regdir)
            call mapz(zp,regdir)
         endif
      ELSE IF(CHOICE.EQ.'Average UVWPT') then
         if (ifavgupt) then
            call prs('Turning off Z-averaging$')
            ifavgupt = .false.
c
c           .reset "odump" ... so that nekton will re-read data
c           .call getfld to re-load u,v,w,p, and t
            newdump = odump
            odump   = -99
            call getfld(newdump,ierr,.true.,.true.)
         else
            call prs('Turning on Z-averaging$')
            ifavgupt = .true.
            call avg_uvwpt_regular
         endif
      ELSE IF(CHOICE.EQ.'GENERATE FEM MESH') THEN
         CALL GENFEM
      ELSE IF(CHOICE.EQ.'SET ANNOTATION') THEN
         CALL SETANN
      ELSE IF (CHOICE.EQ.'SET COLORS') THEN
         CALL PRS
     $   ('CURRENT BREAK POINTS (Normalized 0.0= MIN; 1.0 = MAX)$')
         WRITE(S,'(8F7.4)')(TX(III),III=1,15)
         CALL PRS(S//'$')
59       CALL PRS('ENTER DIVISION 1-15   OR NEGATIVE TO END:$')
         CALL REI(IDIV)
         IF(IFLEARN)WRITE(3,*) IDIV
         IF(IDIV .GE. 1  .AND. IDIV .LE.15) THEN
            WRITE(S,'(1X,A8,I4,A1,G11.3,I2)')
     $      'Division',IDIV,'[',TX(IDIV),']:'
            CALL PRS(S//'$')
            CALL RES(LINE,70)
            IF(IFLEARN)WRITE(3,'(A70)') LINE
            IF(LINE.NE.' ')CALL READER(TX(IDIV),IERR)
            IF(IERR.NE.0)GO TO 13
            WRITE(S,'(1X,A8,I4,A1,G11.3,I2)')
     $      'Division',IDIV,'[',TX(IDIV),']:'
            CALL PRS(S//'$')
            GO TO 59
         ELSE
            CALL PRS('No more modifications$')
            GO TO 60
         ENDIF
60       CONTINUE
         CALL DRBAR(TX,15)
      ELSE IF (CHOICE.EQ.'SET RANGE') THEN
         IFUSCL=.FALSE.
         CALL PRS('Autoscale for color fills? (y/n).$')
         CALL RES(LINE,70)
         IF(IFLEARN)WRITE(3,'(A70)') LINE
         IF(LINE.EQ.'n'.OR.LINE.EQ.'N')THEN
            IFUSCL=.TRUE.
            CALL PRS('Enter lower and upper values of scalar range:$')
            CALL RERR(USRMIN,USRMAX)
            IF(IFLEARN)WRITE(3,*) USRMIN,USRMAX,' user scalar range.'
            ttmin = usrmin
            ttmax = usrmax
         ENDIF
      ELSE IF (CHOICE.EQ.'THRESHOLD LEVEL') THEN
         CALL PRS('Input THRESHOLD LEVELS for color fills.$')
         CALL PRS('ENTER LEVMIN (0-15)$')
         CALL REI(LEVMIN)
         CALL PRS('ENTER LEVMAX (0-15)$')
         CALL REI(LEVMAX)
         IF(IFLEARN)WRITE(3,*) LEVMIN,' Threshold levels.'
         IF(IFLEARN)WRITE(3,*) LEVMAX,' Threshold levels.'
      ELSE IF(CHOICE.EQ.'if_output_ijke') then
         if (if_output_ijke) then
            call prs('Setting if_output_ijke to false.$')
            if_output_ijke = .false.
         else
            call prs('Setting if_output_ijke to true.$')
            if_output_ijke = .true.
         endif
      ELSE IF(CHOICE.EQ.'SET PORTRAIT') THEN
         if (ifportrait) then
            call prs('Turning PORTRAIT mode off.$')
            ifportrait = .false.
         else
            call prs('Turning PORTRAIT mode on.$')
            call prs('Screen will clear before plots.$')
            ifportrait = .true.
         endif
      ELSE IF(CHOICE.EQ.'SET HARD COPY') THEN
         CALL PRS
     $   ('Hard copy Menu.  Determines whether subsequent plots'//
     $    ' will be stored in file$')
         jhard = mod(jhard,3) + 1
         if (jhard.eq.1) then
            ifhard = .false.
            CALL PRS('Setting HARDCOPY to OFF$')
         elseif (jhard.eq.2) then
            ifhard = .true.
            CALL PRS('Setting HARDCOPY to ON (std)$')
         elseif (jhard.eq.3) then
            ifhard = .true.
            CALL PRS('Setting HARDCOPY to ON (border free)$')
         endif
      ENDIF
c     return
      GO TO 1
13    CALL PRS('Error reading input; plot parameter not changed$')
      GO TO 1
      END
c-----------------------------------------------------------------------
      subroutine SETRES
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
13    ITEM(1)='MAIN MENU'
      ITEM(2)='FISHNET GRID'
      ITEM(3)='COLOR FILL'
      ITEM(4)='CONTOUR LINES'
      ITEM(5)='PROFILE POINTS'
      ITEM(6)='ARROW DENSITY'
      ITEM(7)='CURVE SEGMENTS'
      ITEM(8) ='SET ARROW SIZE'
      ITEM(9) ='SET FISHNET HEIGHT'
      NCHOIC=9
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'RESOLUTION')
c
      IF(CHOICE.EQ.'FISHNET GRID')THEN
       CALL PRSIS(
     $ 'Enter surface grid density (lines/element): [$',NXGRID,']$')
       CALL RES(LINE,70)
       IF(IFLEARN)WRITE(3,'(A70)') LINE
       IF(LINE.NE.' ')CALL READER(VALUE,IERR)
       IF(IERR.NE.0)GO TO 13
       NXGRID=VALUE
c
      ELSE IF(CHOICE.EQ.'COLOR FILL')THEN
       CALL PRSIS(
     $  'Enter color fill   density (bands/element): [$',NXBAND,']$')
       CALL RES(LINE,70)
       IF(IFLEARN)WRITE(3,'(A70)') LINE
       IF(LINE.NE.' ')CALL READER(VALUE,IERR)
       IF(IERR.NE.0)GO TO 13
       NXBAND=VALUE
c
      ELSE IF (CHOICE.EQ.'SET ARROW SIZE') THEN
         CALL PRSRS('ENTER ARROW SIZE [$',SARROW,']$')
         CALL RES(LINE,70)
         IF(IFLEARN)WRITE(3,'(A70)') LINE
         IF(LINE.NE.' ')CALL READER(SARROW,IERR)
         IF(IERR.NE.0)GO TO 13
c
      ELSE IF (CHOICE.EQ.'SET FISHNET HEIGHT') THEN
         CALL PRS('FISHNET HEIGHT MAXIMUM DISTANCE (IN CHARACTERISTIC$')
         CALL PRS('DOMAIN LENGTHS) OF FISHNET PERTURBATION$')
         CALL PRSRS('ENTER FISHNET HEIGHT [$',HFISH,']$')
         CALL RES(LINE,70)
         IF(IFLEARN)WRITE(3,'(A70)') LINE
         IF(LINE.NE.' ')CALL READER(HFISH,IERR)
         IF(IERR.NE.0)GO TO 13
c
      ELSE IF(CHOICE.EQ.'CONTOUR LINES')THEN
       acont = abs(cont_lev)
       CALL PRSIS(
     $ 'Enter number of contour lines  [$',NXCONT,'],$')
       CALL PRSRS(
     $ 'a negative real number to specify contour level  [$'
     $ ,acont,' ], or$')
       CALL PRS(
     $ '0 to prompt for a file containing specific contour levels$')
c
       CALL RES(LINE,70)
       IF(IFLEARN)WRITE(3,'(A70)') LINE
       IF(LINE.NE.' ')CALL READER(VALUE,IERR)
       IF(IERR.NE.0)GO TO 13
c
       if_spec_cont = .false.
       if (value.eq.0.0) then
c
c         Specifying contours...
c
          CALL PRS
     $    ('Input name of file containing: nlev, lev1, ...lev_nlev.$')
          CALL blank(line,70)
          CALL RES(LINE,70)
c
          if_spec_cont = .true.
          open(unit=67,file=line,status='old',err=103)
          read(67,*,err=103,end=103) spec_cont_nlev
            spec_cont_nlev = spec_cont_nlev - 1
            if (spec_cont_nlev.gt.max_cont_nlev) then
               spec_cont_nlev = max_cont_nlev
               CALL PRSI('Resetting number of contours to max allowed:$'
     $         ,spec_cont_nlev)
            endif
            read(67,*,err=103,end=103) (spec_cont(k),k=0,spec_cont_nlev)
            call sortit(spec_cont,spec_ind,spec_cont_nlev+1)
            goto 104
  103     continue
            CALL PRS('Trouble finding/reading from file.$')
            if_spec_cont = .false.
  104     continue
          close(unit=67)
c
       else
c
c         Standard contour level specification...
c
          if (value.lt.1.0) value = -abs(value)
          cont_lev = value
          NXCONT   = VALUE
       endif
c
      ELSE IF(CHOICE.EQ.'PROFILE POINTS')THEN
       CALL PRSIS(
     $ 'Enter number of points in profile: [$',NPTPRO,']$')
       CALL RES(LINE,70)
       IF(IFLEARN)WRITE(3,'(A70)') LINE
       IF(LINE.NE.' ')CALL READER(VALUE,IERR)
       IF(IERR.NE.0)GO TO 13
       IF(LINE.NE.' ') NPTPRO=VALUE
       if (nptpro.gt.mxpro) then
          call pris(mxpro,
     $   ' is upper bound for # profile pts. Resetting.$')
          nptpro = mxpro
       endif
c
      ELSE IF(CHOICE.EQ.'CURVE SEGMENTS')THEN
       CALL PRSIS(
     $ 'Enter number of line segments in curved side: [$',
     $ NCSEGS,']$')
       CALL RES(LINE,70)
       IF(IFLEARN)WRITE(3,'(A70)') LINE
       IF(LINE.NE.' ')CALL READER(VALUE,IERR)
       IF(IERR.NE.0)GO TO 13
       IF(VALUE.GT.50 .OR. VALUE .LT.3)THEN
          CALL PRS('ENTER NUMBER BETWEEN 3 AND 50$')
       ELSE
          NCSEGS=VALUE
       ENDIF
c
      ELSE IF(CHOICE.EQ.'ARROW DENSITY')THEN
         ITEM(1)='HIGH'
         ITEM(2)='MEDIUM'
         ITEM(3)='LOW'
         NCHOIC=3
         WRITE(S,22) DARROW
   22    format('Current density: ',a20,'$')
         CALL PRS(S)
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'ARROW DENSITY')
         DARROW=CHOICE
      ELSE
         return
      ENDIF
      goto 13
      END
c-----------------------------------------------------------------------
      subroutine setloc
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,inewt
      INTEGER FCORNS (4,6)
      LOGICAL IFTMP,if_add_loc
C     surface qualifiers
      CHARACTER*1 SRFQAL(20)
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
      IOLD=ILEVEL
C
      write(6,*) 'pltmod in setloc:',pltmod
c
    1 continue
      ITEM(1)='MAIN MENU'
      nchoic=1
      IF (PLTMOD.EQ.'SURFACE') THEN
          ITEM(2)='ALL SURFACES'
                                         nchoic=nchoic+1
          CALL FINDNAM(ITEM(3),NCHOIC)
C
      ELSEIF(IF3D)THEN
                                         nchoic=nchoic+1
         ITEM(nchoic)='ADD LOCATION'
                                         nchoic=nchoic+1
         ITEM(nchoic)='X-PLANE'
                                         nchoic=nchoic+1
         ITEM(nchoic)='Y-PLANE'
                                         nchoic=nchoic+1
         ITEM(nchoic)='Z-PLANE'
                                         nchoic=nchoic+1
         ITEM(nchoic)='CLIP PLANES'
                                         nchoic=nchoic+1
c        ITEM(nchoic)='ALL WALLS'
c                                        nchoic=nchoic+1
c        ITEM(nchoic)='ALL BOUNDARIES'
c                                        nchoic=nchoic+1
         ITEM(nchoic)='BOUNDARIES'
c                                        nchoic=nchoic+1
c        ITEM(nchoic)='POINT'
c                                        nchoic=nchoic+1
c        ITEM(nchoic)='LINE'
                                         nchoic=nchoic+1
         ITEM(nchoic)='SET OF LINES'
                                         nchoic=nchoic+1
         ITEM(nchoic)='CUTTING PLANE'
      ELSE
         ITEM(2)='POINT'
         ITEM(3)='LINE'
         ITEM(4)='SET OF LINES'
         ITEM(5)='ELEMENT'
         NCHOIC=5
      ENDIF
      IF(PLTMOD.NE.'SURFACE'.AND.NHIS.GE.1)THEN
         nchoic=nchoic+1
         ITEM(NCHOIC)='HISTORY POINT'
      ENDIF
      IF(NSOBJS.GE.1)THEN
         nchoic=nchoic+1
         ITEM(NCHOIC)='OBJECT'
      ENDIF
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET LOCATION')
c
      if_add_loc = .false.
      IF (CHOICE.EQ.'ADD LOCATION') THEN
         if_add_loc = .true.
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'ADD LOCATION')
      ENDIF
c
c
      IF(CHOICE.EQ.'MAIN MENU')return
C
      IF (PLTMOD.EQ.'SURFACE') THEN
         IF (CHOICE.EQ.'ALL SURFACES') THEN
            CALL BLANK(NAMSEL,16)
         ELSE
            NAMSEL=CHOICE
         ENDIF
         return
      ENDIF
C
      IF (CHOICE.EQ.'LINE')THEN
         GRID=PARAM(18)
         IF(GRID .GT. .0099 .AND. .NOT. IF3D) THEN
C           PUT GRAPH PAPER ON SCREEN
            CALL COLOR(4)
            DO 10 I=1,IFIX(1/GRID - 1)
                XX=I*GRID * .9963
C               Don't write grid lines in bottom 1/10 of screen.
                IF(XX.GT.  0.11) THEN
                    CALL MOVESC(0.01,XX)
                    CALL DRAWSC(0.99,XX)
                ENDIF
                CALL MOVESC(XX,0.11)
                CALL DRAWSC(XX,0.99)
 10         CONTINUE
            CALL DRMESH
            CALL DRAXIS
         ENDIF
         CALL PRS
     $('Enter 1st Endpoint:'//
     $ ' (Rightmost button latches to nearest vertex$')
         CALL GETPT(XLINE(1),YLINE(1),ZLINE(1),' ')
         CALL PRS('Enter 2nd Endpoint:$')
         CALL GETPT(XLINE(2),YLINE(2),ZLINE(2),' ')
         CALL COLOR(1)
         IF(IF3D)THEN
            CALL MOVE3(XLINE(1),YLINE(1),ZLINE(1))
            CALL DRAW3(XLINE(2),YLINE(2),ZLINE(2))
            WRITE(S,'(1X,A5,3G11.3)')'From ',XLINE(1),YLINE(1),ZLINE(1)
            CALL PRS(S//'$')
            WRITE(S,'(1X,A5,3G11.3)')'To   ',XLINE(2),YLINE(2),ZLINE(2)
            CALL PRS(S//'$')
         ELSE
            CALL MOVE(XLINE(1),YLINE(1))
            CALL DRAW(XLINE(2),YLINE(2))
         ENDIF
      ELSE IF(CHOICE.EQ.'CUTTING PLANE') THEN
         call cutting_plane
      ELSE IF (CHOICE.EQ.'HISTORY POINT')THEN
         DO 30 I=1,NHIS
            DO 20 II=1,7
               LINE(II:II)=HCODE(II,I)
 20         CONTINUE
            WRITE(LINE( 8:16),'(3I2,I3)')(LOCHIS(IIIND,I),IIIND=1,4)
            ITEM(I)=LINE(1:26)
 30      CONTINUE
         NCHOIC=NHIS
         CALL PRS('Data saved from each point $')
         CALL PRS('I,J,K, Element # $')
         CALL MENU(XMOUSE,YMOUSE,BUTTON,'HISTORY POINT')
         IHISPT=ICHOIC
         CALL PRSI('History Point$',IHISPT)
C
      ELSE IF (CHOICE.EQ.'OBJECT')THEN
         if (nsobjs.le.4) then
            do i=1,nsobjs
               item(i)=sobj(i)
            enddo
            nchoic=nsobjs
            call menu(xmouse,ymouse,button,'OBJECT')
            isobj=ichoic
            call prsi('Object # $',ISOBJ)
         else
            call prs('Enter object #:$')
            call rei(isobj)
         endif
C
C        Set up the planes in the surface LIST
         if (.not.if_add_loc) nlstp = 0
         do isrf=1,nface(isobj)
            nlstp=nlstp+1
            iel  =ilsurf(1,isobj,isrf)
            iface=ilsurf(2,isobj,isrf)
            ijkpln=eface1(iface)
            ix=1
            if (mod(ijkpln,2).eq.0) ix=nx
            list(nlstp)=6*nx*(iel-1)+nx*(ijkpln-1)+ix
            call signpl(list(nlstp))
         enddo
C
      ELSE IF (CHOICE.EQ.'SET OF LINES')THEN
          CALL PRS('Multiple profiles not yet implemented$')
C         CALL PRS('Enter number of profiles:$')
      ELSE IF (CHOICE.EQ.'POINT')THEN
C        Make default 1st history point
         CALL PRS('Enter Single Point:$')
         CALL GETPT(XPOINT,YPOINT,ZPOINT,' ')
         RMIN=1.0E6
         l=0
         DO 200 IEL=1,NEL
           DO 200 K=1,NZ
           DO 200 J=1,NY
           DO 200 I=1,NX
              l=l+1
                  R=( XPOINT-XP(l) )**2 +
     $              ( YPOINT-YP(l) )**2 +
     $              ( ZPOINT-ZP(l) )**2 
                  IF(R.LT.RMIN)THEN
C                    Why do we need this?  XPOINT etc are sufficent for VALUE.
                     IPOINT=I
                     JPOINT=J
                     KPOINT=K
                     IELPNT=IEL
                  ENDIF
200      CONTINUE
C        Now find nearest HISTORICAL point.
C        Find historical point nearest to current point.
         IF(NHIS.GT.0)THEN
            RMIN=1.0E6
            DO 300 II=1,NHIS
               I    =LOCHIS(1,II)
               J    =LOCHIS(2,II)
               K    =LOCHIS(3,II)
               IEL  =LOCHIS(4,II)
               R=(XPOINT-XP(IND(I,J,K,IEL)) )**2 +
     $           (YPOINT-YP(IND(I,J,K,IEL)) )**2 +
     $           (ZPOINT-ZP(IND(I,J,K,IEL)) )**2 
                  IF (R.LT.RMIN)THEN
                     RMIN=R
                     IHISPT=II
                  ENDIF
300         CONTINUE
            CALL PRSI('History Point$',IHISPT)
         ENDIF
      ELSE IF (CHOICE.EQ.'ELEMENT')THEN
         CALL PRS
     $   ('Click mouse to select element, out of domain to quit$')
C
         IFSELE=.FALSE.
         IS=1
         DTZERO = 0.0
         INTDIR = 1
         DO 301 IEL=1,NEL
            IFPLOT(IEL)=.FALSE.
  301    CONTINUE
C
         IFTMP=IFGRID
         IFGRID=.FALSE.
         ZPOINT=0.0
         DO 302 IEL=1,NEL
            CALL MOUSE(XPOINT,YPOINT,BUTTON)
            CALL INTERP(VAL,XPOINT,YPOINT,ZPOINT,IDUM,WORK,ierr)
C
C           Check to see if button clicked within domain.
C
C
C                 If IEL for this particle is returned as zero then
C                 the particle belongs to no element, i.e., is lost.
C
            IF( IENTRP.NE.0 ) THEN
               CALL PRSI('Element$',IENTRP)
               IFSELE=.TRUE.
               IFPLOT(IENTRP)=.TRUE.
               CALL COLOR(1)
               CALL DIAMDA(XPOINT,YPOINT)
            ELSE
               GOTO 303
            ENDIF
  302 CONTINUE
  303 CONTINUE
      IFGRID=IFTMP
C
      ELSE IF(CHOICE.EQ.'X-PLANE' .OR.
     $        CHOICE.EQ.'Y-PLANE' .OR.
     $        CHOICE.EQ.'Z-PLANE' ) THEN

C        They specified a planar surface
         if (.not.if_add_loc) nlstp = 0
         PLANE=CHOICE
         PLANE2=CHOICE
         IF (CHOICE.EQ.'X-PLANE') CALL SETRSTP(1)
         IF (CHOICE.EQ.'Y-PLANE') CALL SETRSTP(2)
         IF (CHOICE.EQ.'Z-PLANE') CALL SETRSTP(3)
         call pris(nlstp,' surfaces found.$')
      ELSE IF(CHOICE.EQ.'CLIP PLANES') THEN
         call clip_plane
      ELSE IF(CHOICE.EQ.'BOUNDARIES') THEN
         call setloc_bdry(if_add_loc)
         IF (CHOICE.EQ.'UP MENU') goto 1
      ELSE IF(CHOICE.EQ.'ALL WALLS') THEN
         if (.not.if_add_loc) nlstp = 0
         NQUAL=1
         SRFQAL(1)='W'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
         call pris(nlstp,' surfaces found.$')
      ELSE IF(CHOICE.EQ.'ALL BOUNDARIES') THEN
         if (.not.if_add_loc) nlstp = 0
         NQUAL=-2
         SRFQAL(1)='E'
         SRFQAL(2)='P'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
         call pris(nlstp,' surfaces found.$')
      ENDIF
      goto 1
C
      END
c-----------------------------------------------------------------------
      subroutine GETPT(POINT1,POINT2,POINT3,FLAG)
      INCLUDE 'basics.inc'
      CHARACTER*26 FLAG
      INTEGER FCORNS (4,6)
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
c
      POINT3 = 0.0
c
C     Gets the x,y,z, coordinates of a point
      IF(IF3D)THEN
         IF(FLAG.EQ.'X-PLANE')THEN
            IDIR1=1
            IDIR2=1
         ELSE IF(FLAG.EQ.'Y-PLANE')THEN
            IDIR1=2
            IDIR2=2
         ELSE IF(FLAG.EQ.'Z-PLANE')THEN
            IDIR1=3
            IDIR2=3
         ELSE
C           Normal specification of all 3 coordinates
            IDIR1=1
            IDIR2=3
         ENDIF
         CALL PRS('Type in X,Y, and Z Coordinates (drop mouse!)$')
         CALL RERRR(POINT1,POINT2,POINT3)
         IF(IFLEARN)WRITE(3,*) POINT1,POINT2,POINT3
c
      ELSE
C        Two-Dimensional
         CALL MOUSE(POINT1,POINT2,BUTTON)
C
         IF(BUTTON.EQ.'RIGHT')THEN
C           Latch to closest point
            XC=POINT1
            YC=POINT2
C           Find Closest Corner (NOT IN SAME ELEMENT)
            RMIN = 1.0E8
            DO 22 IIEL=1,NEL
               DO 20 IICORN=1,4
                  XT=X(IIEL,IICORN)
                  YT=Y(IIEL,IICORN)
                  R=SQRT((XT-XC)**2+(YT-YC)**2)
                  IF(R.LT.RMIN) THEN
                     RMIN  = R
                     IELMIN= IIEL
                     ICMIN = IICORN
                  ENDIF
20             CONTINUE
21             CONTINUE
22          CONTINUE
            POINT1=X(IELMIN,ICMIN)
            POINT2=Y(IELMIN,ICMIN)
C
         ENDIF
         IF(XSCR(POINT1).GT.1)THEN
C           Menu area; use keypad
            CALL PRS('Entering x, y coordinates using keypad. $')
            CALL PRS('Enter x coordinate:$')
            CALL RER(POINT1)
            IF(IFLEARN)WRITE(3,*) POINT1
            CALL PRS('Enter y coordinate:$')
            CALL RER(POINT2)
            IF(IFLEARN)WRITE(3,*) POINT2
         ENDIF
      ENDIF
      IF(.NOT.IF3D)WRITE(S,'(1X,A6,2G11.3)')'x,y:  ',POINT1,POINT2
      IF(     IF3D)WRITE(S,'(1X,A6,3G11.3)')'x,y,z:',
     $POINT1,POINT2,POINT3
      CALL PRS(S//'$')
      IF (IF3D) THEN
         PNTX=XISO(POINT1,POINT2,POINT3)
         PNTY=ZISO(POINT1,POINT2,POINT3)
         PNTZ=ZISO(POINT1,POINT2,POINT3)
         WRITE(S,'(1X,A16,3G11.3)')'xiso,yiso,zhobs:',PNTX,PNTY,PNTZ
         CALL PRS(S//'$')
      ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine DRAXIS
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      IF(.NOT.IFDRAX)return
      CALL COLOR(8)
C
      CALL G3WRIT(XFAC/10,0.0    ,0.0    ,1.0,' X$')
      CALL G3WRIT(0.0    ,YFAC/10,0.0    ,1.0,' Y$')
      IF(IF3D)
     $CALL G3WRIT(0.0    ,0.0    ,XFAC/10,1.0,' Z$')
C
      CALL ARROW3(0.0,0.0,0.0,XFAC/10/SARROW,0.0,    0.0  ,PLANE,SARROW)
      CALL ARROW3(0.0,0.0,0.0,0.0,    YFAC/10/SARROW,0.0  ,PLANE,SARROW)
      IF(IF3D)
     $CALL ARROW3(0.0,0.0,0.0,0.0,    0.0,  XFAC/10/SARROW,PLANE,SARROW)
C
      return
      END
C
c-----------------------------------------------------------------------
      subroutine SHOW(ARG,IA)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      return
      END
c-----------------------------------------------------------------------
      subroutine INIT
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,inewt
c
      REAL PN(NXM)
      CHARACTER*40 s40
      CHARACTER*12 FSESSION
      CHARACTER*1 SRFQAL(20)
      CHARACTER*30 tfile
      CHARACTER*1  tfile1(30)
      CHARACTER*20 s20
      CHARACTER*3 chtmp3
      EQUIVALENCE (tfile1,tfile)
      LOGICAL IFFMAT

      integer fnamei(10)
      character*1 file2(40)
      EQUIVALENCE (file2,fnamei)
      
      integer buf(50)
      character*80 hdr
      character*5 ver
      integer wdsizi
      logical ifbswap,if_byte_swap_test
      real*4 test
      real*8 rtemp


      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
 
      XPHY0=0.0
      YPHY0=0.0
      ILGRNG=0
      ISCND =0
      IENTRP=0
C
      PLTMOD='VOLUME'
      CALL BLANK(NAMSEL,16)
      IFSELE=.FALSE.
      IFDRAX=.FALSE.
      call setgraph(ifgraf)  ! initialize in iolib.f
      ifdssum=.FALSE.
      nmirror=1
      ifregz=.false.
      IFHARD=.FALSE.
      jhard= 1
      IFPOST=.TRUE.
      IFPLAN=.FALSE.
      IFPROJ=.FALSE.
      IFDMSH=.TRUE.
      IFMESH=.TRUE.
      IFCFBD=.FALSE.
      IFCLRB=.TRUE.
      IFZOOM=.FALSE.
      IFUSCL=.FALSE.
      ifavgupt=.FALSE.
C
C     Hilighting element info
C
      nhi = 0
C
C     Assign EB's numbering scheme to PF's scheme.
C
      EFACE(1)=4
      EFACE(2)=2
      EFACE(3)=1
      EFACE(4)=3
      EFACE(5)=5
      EFACE(6)=6
C
C     Assign inverse of EB's numbering scheme to PF's scheme.
C
      EFACE1(1)=3
      EFACE1(2)=2
      EFACE1(3)=4
      EFACE1(4)=1
      EFACE1(5)=5
      EFACE1(6)=6
C
C     Assign group designation to each face to determine ordering of indices.
C
      GROUP(1)=0
      GROUP(2)=1
      GROUP(3)=1
      GROUP(4)=0
      GROUP(5)=0
      GROUP(6)=1
C
C
C     Default threshold levels for color fill plots
C
      LEVMIN = 0
      LEVMAX = 99
C
      DO 2 IEL=1,NELM
         IFPLOT(IEL)=.FALSE.
         DO 2 IFIELD=1,MAXFLD
              MASKEL(IEL,IFIELD) = 1
2     CONTINUE
C
C     Initialize some arrays
C
      CALL RZERO(X,NELM*8)
      CALL RZERO(Y,NELM*8)
      CALL RZERO(Z,NELM*8)
      CALL RZERO(U,MAXPTS)
      CALL RZERO(V,MAXPTS)
      CALL RZERO(W,MAXPTS)
      CALL RZERO(P,MAXPTS)
      CALL RZERO(T,MAXPTS)
      CALL RZERO(XP,MAXPTS)
      CALL RZERO(YP,MAXPTS)
      CALL RZERO(ZP,MAXPTS)
      CALL RZERO(RXM1,MAXPTS)
      CALL RZERO(RYM1,MAXPTS)
      CALL RZERO(RZM1,MAXPTS)
      CALL RZERO(SXM1,MAXPTS)
      CALL RZERO(SYM1,MAXPTS)
      CALL RZERO(SZM1,MAXPTS)
      CALL RZERO(TXM1,MAXPTS)
      CALL RZERO(TYM1,MAXPTS)
      CALL RZERO(TZM1,MAXPTS)
      CALL RZERO(wkv1,MAXPTS)
      CALL RZERO(wkv2,MAXPTS)
      CALL RZERO(wkv3,MAXPTS)
      CALL RZERO(wrk2,MAXPTS)
      CALL RZERO(vortex,MAXPTS)
      CALL RZERO(vrt1,lvrt)
      CALL RZERO(vrt2,lvrt)
      CALL RZERO(vrt3,lvrt)
      CALL RZERO(JACM1,MAXPTS)
      CALL RZERO(WORK,MAXPTS)
c     CALL RZERO(PSA,MAXPTS)
c     CALL RZERO(PSB,MAXPTS)
c     CALL RZERO(PSC,MAXPTS)
      CALL RZERO(XBMIN,NELM)
      CALL RZERO(YBMIN,NELM)
      CALL RZERO(ZBMIN,NELM)
      CALL RZERO(XBMAX,NELM)
      CALL RZERO(YBMAX,NELM)
      CALL RZERO(ZBMAX,NELM)
      CALL RZERO(RR    ,3*NPRT)
      CALL RZERO(XLG   ,4*NPRT)
      CALL RZERO(SCALLG,5*NPRT)
C
      CALL IZERO(ICRV,NELM)
C
C     GET THE DATA FROM THE .REA and .FLD files
C
5     CONTINUE
      open(unit=4,file='session.name',status='old',err=11)
      fsession='session.name'
   11 open(unit=4,file='SESSION.NAME',status='old',err=13)
      fsession='SESSION.NAME'
C     Default exists
      READ(4,'(A80)',err=13,end=13) sesion
      CLOSE(UNIT=4)
      CALL PRS('Enter Session Name --Default= '//sesion//'$')
C     READ(5,819) line
C 819 FORMAT(A70)
      CALL RES(line,70)
      IF (LINE.NE.' ') sesion=LINE
      GO TO 14
13    CONTINUE
C     No Default exists
      CALL PRS('Enter Session Name$')
      CALL BLANK(sesion,80)
      CALL BLANK(FILENM,40)
      CALL RES(sesion,80)
c     READ(5,818) sesion
c 818 format(a10)
14    CONTINUE
      IF(IFLEARN)WRITE(3,'(A80)')sesion
      IF(sesion.eq.'!')THEN
         CALL PRS('In DEMO Mode$')
         IFDEMO=.true.
         GO TO 5
      ENDIF
      IF(sesion.eq.'$')THEN
         CALL PRS('In Learning Mode$')
         IFLEARN=.true.
         OPEN(UNIT=3,FILE='DEMOP.COM',STATUS='NEW')
         WRITE(3,*)'postx'
         WRITE(3,*)'!'
         GO TO 5
      ENDIF
      IF(sesion.eq.'#')THEN
         CALL PRS('In non graphics mode$')
         IFGRAF=.FALSE.
         GO TO 5
      ENDIF
C
C     All set with session name
C
      CALL PRS(' Beginning Session '//sesion//'$')
      CALL OPENF(4,'SESSION.NAME','NEW',1,IERR)
      WRITE(4,'(A80)')sesion
C
C     Open .rea .fld .his .str files
C
C     Check How many letters in sesion name
      lastch=LTRUNC(sesion,80)
C     Make small letters in filename
      CALL BLANK(FILENM,40)
      CALL BLANK(session_name,80)
      call chcopy(session_name,sesion,lastch)
      write(6,*) 'Session name: ',session_name
c
      do 9 i=1,lastch
          int=ichar(sesion(i:i))
c         if(int.ge.65 .and. int.le.90) int=int+32
          filenm(i:i)=char(int)
   9  continue
      m=lastch+1
      n=m+3
c
C     Read file
      filenm(m:n)='.rea'
      CALL OPENF( 9,FILENM,'OLD',1,IERR)
      IF(IERR.NE.0) THEN
         CALL PRS(' ** ERROR **  Can''t open file '//filenm//'$')
         GOTO 13
      ENDIF
c     if(IERR.NE.0)STOP
C     Field Data  (we move this to AFTER reading in read file)
C     Streamline Data
      filenm(m:n)='.str'
      call blank(strfle,40)
      call chcopy(strfle,filenm,n)
C     CALL OPENF(18,FILENM,'OLD',1,IERR)  <--- moved to ROUTINE STREAM
C     Plot File - save filename stuff for successive plot files
c
      fplotnam        = filenm
      iplotnum        = 1
      iplotnam        = m
      fplotnam(m:n+2) = '.plt01'
c
      IF (IFGRAF) THEN
         CALL DEVINI('GENERIC')
         CALL SETENV
         CALL OPENF(45,fplotnam,'NEW',3,IERR)
         IF(IFLASE ) CALL INITQMS
         IF(IFPOSTS) CALL INITPS
      ENDIF
C
      filenm(m:n)='.re2'
C     Read in stuff
      READ(9,'(a1)',ERR=59)ans
      READ(9,*,ERR=59) VNEKOLD
      NKTONV=VNEKTON
      READ(9,*,ERR=59) NDIM
      IF(NDIM.EQ.3)IF3D=.TRUE.
      IF(NDIM.EQ.2)IF3D=.FALSE.

C     Read in Parameters
      READ(9,*,ERR=59) NPARAM
      DO 1051 IP=1,NPARAM
         READ(9,*,ERR=59) PARAM(IP)
1051  CONTINUE
      param(20) = 5    ! default from .rea file
      NPSCAL=PARAM(23)

      READ(9,*,ERR=59)NSKIP
      if(NSKIP.ne.0) then
        READ(9,*,ERR=59) (PCOND (I),I=3,11)
        READ(9,*,ERR=59) (PRHOCP(I),I=3,11)
      endif

C     Read in Logicals
      ifflow    = .false.
      ifheat    = .false.
      iftran    = .false.
      ifnav     = .false.
      ifaxis    = .false.
      READ(9,*,ERR=59) NLOGIC
      DO I = 1,NLOGIC
         call blank(s40,40)
         call capit(s40,40)
         READ(9,'(a40)',ERR=59) s40
         if    (indx1(s40,'IFFLOW',6).ne.0) then
                  read(s40,*) ifflow
         elseif(indx1(s40,'IFHEAT',6).ne.0) then
                  read(s40,*) ifheat
         elseif(indx1(s40,'IFTRAN',6).ne.0) then
                  read(s40,*) iftran
         elseif(indx1(s40,'IFADVC',6).ne.0) then
                  read(s40,*) ifnav 
         elseif(indx1(s40,'IFAXIS',6).ne.0) then
                  read(s40,*) ifaxis
         endif
      ENDDO
c     READ(9,*,ERR=59)IFFLOW
c     READ(9,*,ERR=59)IFHEAT
c     READ(9,*,ERR=59)IFTRAN
c     READ(9,*,ERR=59)IFNAV
c     READ(9,*,ERR=59)
c     READ(9,*,ERR=59)IFAXIS
C     Put if statements in case preprocessed version is out of date
c     IF(NLOGIC.GT.6)THEN
c        DO 1053 IL=1,NLOGIC-6
c          READ(9,*,ERR=59)
c053     CONTINUE
c     ENDIF
C
      NFLDS=1
      IF(IFHEAT)NFLDS=2+NPSCAL
c     NX=PARAM(20)
      NX=4                  ! this is a better default,pff 8/28/08
      PARAM(20) = NX
      NY=NX
      IF(NDIM.EQ.3)NZ=NX
      IF(NDIM.EQ.2)NZ=1
C     Premature?????!!!!
      IF(NKTONV.EQ.2)THEN
C        Set up Zpoints and derivative arrays
         CALL LEGEND(ZPTS,WGHT,NX)
         CALL GENPN1(PN,ZPTS,NX)
         CALL DERMAT(DFDR,DFDRT,PN,ZPTS,NX,NXM)
         CALL DERMAT(DGDR,DGDRT,PN,ZPTS,NX,NX )
      ENDIF
      IF(IFTRAN)THEN
         IOSTEP=PARAM(15)
         NSTEPS=PARAM(11)
         DT    =PARAM(12)
         IF(NKTONV.GT.1)THEN
C           Default if it can't read from end of history file
            NDUMPS=2
         ELSE IF(IOSTEP.NE.0)THEN
            NDUMPS=(NSTEPS-1)/IOSTEP +1
         ELSE
            NDUMPS=1
         ENDIF
         FINTIM=PARAM(10)
         IF(FINTIM.EQ.0.0)FINTIM=NSTEPS*DT
      ELSE
         NDUMPS=1
      ENDIF
c
      READ(9,*,ERR=40,end=34)XFAC,YFAC,XZERO,YZERO
C     Preserve original coordinates for 2D case
      XFACO=XFAC
      YFACO=YFAC
      XZEROO=XZERO
      YZEROO=YZERO
C     Read Elemental Mesh data
      READ(9,*,ERR=41,end=41)

      read(9,*,err=41,end=41)nelr,ndim

      ifsubset = .false.
      IFFMTIN  = .TRUE.
      IF (NELR.LT.0) IFFMTIN=.FALSE.
c     OPEN .re2
      if (.not.iffmtin) then
         ilen= ltrunc(filenm,40)
         call izero(fnamei,10)
         call chcopy(file2,filenm,ilen)

         call byte_open(file2)
         call byte_read(hdr,20)
         read(hdr,*) ver,nelr,ndim
         wdsizi = 4
         if(ver.eq.'#v002') wdsizi = 8
         call byte_read(test,1)
         ifbswap = if_byte_swap_test(test)
      endif
      nel = nelr
      if (nelr.gt.nelm) then
         call prsii('Num. elems > nelm:$',nelr,nelm)
         ifsubset = .true.
         nel = nelm-1
         call get_subset_list
      endif

      nelr = abs(nelr)
      IF(NX*NY*NZ*NELr.GE.MAXPTS)THEN
         CALL PRSIS(
     $   'Your simulation has more than the current maximum of$'
     $   ,MAXPTS,' allowed.$')
         CALL PRS('Please notify your NEKTONICS representative to get '
     $   //'this maximum increased.$')
         WRITE(6,*)
     $   'Your simulation has more than the current maximum of'
     $   ,MAXPTS,' allowed.'
         WRITE(6,*) 
         WRITE(6,*) 
     $  'Please notify your NEKTONICS representative to get '
         WRITE(6,*) 
     $   'this maximum increased.'
c        STOP
      ENDIF
c     IF(NEL.GE.NELM)THEN
c        CALL PRS
c    $   ('ERROR: This postprocessor has arrays dimensioned for$')
c        CALL PRSIS('a maximum of $',NELM-1,' Elements.$')
c        CALL PRS('Please notify your NEKTONICS representative to get '
c    $   //'this maximum increased.$')
c        WRITE(6,*)
c    $   'ERROR: This postprocessor has arrays dimensioned for'
c        NELM1=NELM-1
c        WRITE(6,*)
c    $   'a maximum of ',NELM1,' Elements.'
c        WRITE(6,*)
c    $   'Please notify your NEKTONICS representative to get '
c        WRITE(6,*)
c    $   'this maximum increased.'
c        STOP
c     ENDIF
      NSIDES=NDIM*2
      NEDGES=4
      IF(IF3D)NEDGES=8
      IF(IF3D.AND.NDIM.EQ.2)
     $CALL PRS('ERROR: PARAMETER SET TO 2-D BUT MESH DATA IS FOR 3-D$')
      IF(.NOT.IF3D.AND.NDIM.EQ.3)
     $CALL PRS('ERROR: PARAMETER SET TO 3-D BUT MESH DATA IS FOR 2-D$')
      CALL PRS(' Reading geometry from '//sesion//' .rea$')
      DO 98 IER=1,NELR
         IEL = min(ier,nelm)
C        READ(9,*,ERR=42,END=42) IDUM,NUMAPT(IEL)
         if (iffmtin) then
            read(9,*)        ! dummy read
            numapt(iel) = 1
            letapt(iel) = 'A'
c           READ(9,'(27X,I4,A1)',ERR=422,END=422)NUMAPT(IEL),LETAPT(IEL)
c           goto 423
c 422       continue
c           numapt(iel) = 1
c           letapt(iel) = 'A'
c 423       continue
         endif
         IF(NDIM.EQ.2)THEN
            if (iffmtin) then
               READ(9,*,ERR=43,END=43)(X(IEL,IC),IC=1,4)
               READ(9,*,ERR=43,END=43)(Y(IEL,IC),IC=1,4)
            else
               nwds =  (1 + ndim*(2**ndim))*(wdsizi/4)
               call byte_read(buf,nwds)
               if(wdsizi.eq.8) then
                 if (ifbswap) call byte_reverse8(buf,nwds)
                 call copyi(ig,buf(1 ),1)          !1-2
                 call copy8b(x,buf( 3),4,iel,nelm) !3-10
                 call copy8b(y,buf( 11),4,iel,nelm)!11-18
               else
                 if (ifbswap) call byte_reverse(buf,nwds)
                 ig=buf(1)
                 call copy4b(x(iel,1),buf( 2),4,iel,nelm)
                 call copy4b(y(iel,1),buf( 6),4,iel,nelm)
               endif
            endif
            DO 96 IC=5,8
               X(IEL,IC)=X(IEL,IC-4)
               Y(IEL,IC)=Y(IEL,IC-4)
96          CONTINUE
            if (iel.eq.1) then
               xxmin = x(iel,ic)
               yymin = y(iel,ic)
               xxmax = x(iel,ic)
               yymax = y(iel,ic)
            else
               do 997 ic=1,8
                  xxmin = min(xxmin,x(iel,ic))
                  yymin = min(yymin,y(iel,ic))
                  xxmax = max(xxmax,x(iel,ic))
                  yymax = max(yymax,y(iel,ic))
  997          continue
            endif
         ELSE IF(NDIM.EQ.3)THEN
            if (iffmtin) then
               READ(9,*,ERR=43,END=43)(X(IEL,IC),IC=1,4)
               READ(9,*,ERR=43,END=43)(Y(IEL,IC),IC=1,4)
               READ(9,*,ERR=43,END=43)(Z(IEL,IC),IC=1,4)
               READ(9,*,ERR=43,END=43)(X(IEL,IC),IC=5,8)
               READ(9,*,ERR=43,END=43)(Y(IEL,IC),IC=5,8)
               READ(9,*,ERR=43,END=43)(Z(IEL,IC),IC=5,8)
            else
               nwds =  (1 + ndim*(2**ndim))*(wdsizi/4)
 
               call byte_read(buf,nwds)
               if(wdsizi.eq.8) then
                 if(ifbswap) call byte_reverse8(buf,nwds)
                 call copyi(ig,buf(1),1)                  !1-2
                 call copy8b(x,buf( 3),8,iel,nelm) !3-18
                 call copy8b(y,buf(19),8,iel,nelm) !19-34
                 call copy8b(z,buf(35),8,iel,nelm) !35-50
               else
                 if(ifbswap) call byte_reverse(buf,nwds)
                 ig=buf(1)
                 call copy4b(x,buf( 2),8,iel,nelm)
                 call copy4b(y,buf(10),8,iel,nelm)
                 call copy4b(z,buf(18),8,iel,nelm)
               endif
            endif
            if (iel.eq.1) then
               xxmin = x(iel,1)
               yymin = y(iel,1)
               zzmin = z(iel,1)
               xxmax = x(iel,1)
               yymax = y(iel,1)
               zzmax = z(iel,1)
            endif
            do 97 ic=1,8
               xxmin = min(xxmin,x(iel,ic))
               yymin = min(yymin,y(iel,ic))
               zzmin = min(zzmin,z(iel,ic))
               xxmax = max(xxmax,x(iel,ic))
               yymax = max(yymax,y(iel,ic))
               zzmax = max(zzmax,z(iel,ic))
   97       continue
         ENDIF
98    CONTINUE
      call prs('XYZ Min,Max:$')
      call prrr(xxmin,xxmax)
      call prrr(yymin,yymax)
      if(ndim.eq.3) call prrr(ZZmin,ZZmax)
C     Read curved side data
      if (iffmtin) then
         READ(9,*,err=7201,end=7201)
         READ(9,*,ERR=7202,END=7202)NCURVE
      else
         if(wdsizi.eq.4) then
            call byte_read(ncurve,1)
            if(ifbswap) call byte_reverse(ncurve,1)
         else
            call byte_read(rtemp,2)
            if(ifbswap) call byte_reverse8(rtemp,2)
            ncurve = rtemp
         endif
      endif
      IF(NCURVE.GT.0)THEN
         DO 19 I=1,NCURVE
            IF (NEL.LT.1000.and.iffmtin) THEN
               READ(9,'(2I3,5G14.6,1X,A1)',ERR=7203,END=7203)
     $                       IEDGE,IEL,R1,R2,R3,R4,R5,ANS
            elseif (iffmtin.and.nel.lt.100 000) then
               read(9,*,err=7204,end=7204)
     $                       iedge,iel,r1,r2,r3,r4,r5,ans
            elseif (iffmtin.and.nel.lt.1 000 000) then
               read(9,'(i2,i6,5g14.6,1x,a1)',err=7224,end=7224)
     $                       iedge,iel,r1,r2,r3,r4,r5,ans
            elseif (iffmtin) then
               read(9,'(i2,i12,5g14.6,1x,a1)',err=7234,end=7234)
     $                       iedge,iel,r1,r2,r3,r4,r5,ans
            else
             if(wdsizi.eq.8) then
               call byte_read(buf,16)
               if(ifbswap) call byte_reverse8(buf,14)
               call copyi(iedge,buf(1),1) !1-2
               call copyi(iel  ,buf(3),1) !3-4
               call copy8r (r1,buf( 5),1) !5-6
               call copy8r (r2,buf( 7),1) !7-8
               call copy8r (r3,buf( 9),1) !9-10
               call copy8r (r4,buf(11),1) !11-12
               call copy8r (r5,buf(13),1) !13-14
               call chcopy(ans,buf(15),1)
             else
               call byte_read(buf,8)
               if(ifbswap) call byte_reverse(buf,7)
               iedge = buf(1)
               iel   = buf(2)
               call copy4r(r1,buf(3),1)
               call copy4r(r2,buf(4),1)
               call copy4r(r3,buf(5),1)
               call copy4r(r4,buf(6),1)
               call copy4r(r5,buf(7),1)
               call chcopy(ans,buf(8),1)
             endif
            ENDIF
            IEL=min(iel,nelm)
            CALL DDUMMY(IEDGE,IEL)
            CURVE(1,IEDGE,IEL)=R1
            CURVE(2,IEDGE,IEL)=R2
            CURVE(3,IEDGE,IEL)=R3
            CURVE(4,IEDGE,IEL)=R4
            CURVE(5,IEDGE,IEL)=R5
            CCURVE( IEDGE,IEL)=ANS
19       CONTINUE
      ENDIF
C     Now that we have the corner and curves side data, make the mesh!
      CALL MESPOS
C     Read Boundary Conditions (and connectivity data)
      if (iffmtin) READ(9,*,ERR=44,END=44)
      DO 90 IFLD=1,NFLDS
C        Fluid and/or thermal
C        !!?? NELF DIFFERENT FROM NEL??
         if (iffmtin) then 
             READ(9,*,ERR=44,END=44)
         else
           if(wdsizi.eq.8) then 
             call byte_read(rtemp,2)
             if (ifbswap) call byte_reverse8(rtemp,2)
             nbc_max=rtemp
           else
             call byte_read(nbc_max,1)
             if (ifbswap) call byte_reverse(nbc_max,1)
           endif
         endif
         IF( (IFLD.EQ.1.AND.IFFLOW) .OR. IFLD.GT.1)THEN
C         FIX UP FOR WHICH OF FIELDS TO BE USED

          write(6,*) nelr,iel,ifld,' TRYING TO READ BC'
          if(iffmtin) then
            DO 88 IER=1,NELR
            iel = min(ier,nelm)
            DO 88 ISIDE=1,NSIDES
C              !Fix to a4,i2 when you make cbc character*4
               NBCREA = 5
c              IF(VNEKOLD .LE. 2.5) NBCREA = 3
               IF (NEL.LT.1000) THEN
                  READ(9,'(1X,A3,1x,I2,I3,5G14.6)',ERR=44,END=44)
     $            CBC(ISIDE,IEL,IFLD),ID,ID,
     $            (BC(II,ISIDE,IEL,IFLD),II=1,NBCREA)
                  jse=bc(1,iside,iel,ifld)
                  jsi=bc(2,iside,iel,ifld)
c               write(6,*) cbc(iside,iel,ifld),iside,iel,jse,jsi,' cbc'
               ELSEIF (NEL.LT.100 000) then
                  READ(9,'(1X,A3,I5,I1,5G14.6)',ERR=44,END=44)
     $            CBC(ISIDE,IEL,IFLD),ID,ID,
     $            (BC(II,ISIDE,IEL,IFLD),II=1,NBCREA)
               ELSEIF (NEL.LT.1 000 000) then
                  READ(9,'(1X,A3,I6,5G14.6)',ERR=44,END=44)
     $            CBC(ISIDE,IEL,IFLD),ID,
     $            (BC(II,ISIDE,IEL,IFLD),II=1,NBCREA)
               ELSE
                  READ(9,'(1X,A3,I12,5G18.11)',ERR=44,END=44)
     $            CBC(ISIDE,IEL,IFLD),ID,
     $            (BC(II,ISIDE,IEL,IFLD),II=1,NBCREA)
               ENDIF
               CBC1=CBC(ISIDE,IEL,IFLD)
               IF(CBC1.EQ.'M'.OR.CBC1.EQ.'m')THEN
                  IFMOVB=.TRUE.
                  ifgngm=.TRUE.
               ENDIF
               ICBC1=ICHAR(CBC1)
               IF(ICBC1.GE.97 .AND. ICBC1.LE.122)THEN
                  CBC3 = CBC(ISIDE,IEL,IFLD)
                  IF(CBC3(3:3) .EQ. 'i')THEN
C                    Special storage for internal boundaries
                     NLINES=BC(4,ISIDE,IEL,IFLD)
                     BC(5,ISIDE,IEL,IFLD)=LOCLIN
                  ELSE
                     NLINES=BC(1,ISIDE,IEL,IFLD)
                     BC(2,ISIDE,IEL,IFLD)=LOCLIN
                  ENDIF
                  DO 86 I=1,NLINES
                    READ(9,'(A70)',ERR=44,END=44)INBC(LOCLIN)
                    LOCLIN=LOCLIN+1
86                CONTINUE
               ENDIF

c     write(6,*) iside,iel,ifld,' ',cbc(iside,iel,ifld),' cbc2'

88        CONTINUE
          else  !re2
             do  k=1,nbc_max
               if(wdsizi.eq.8) then
                 call byte_read(buf,16)
                 if (ifbswap) call byte_reverse(buf,14)
                 call copyi(iel,buf(1),1)  !1-2
                 call copyi(iside,buf(3),1)!3-4
                 call copy8r ( bc(1,iside,iel,ifld),buf(5) ,5)
                 call chcopy (cbc(  iside,iel,ifld),buf(15),3)
               else
                 call byte_read(buf,8)
                 if (ifbswap) call byte_reverse(buf,8-1)
                 iel   = buf(1)
                 iside = buf(2)
                 call copy4r ( bc(1,iside,iel,ifld),buf(3),5)
                 call chcopy (cbc(  iside,iel,ifld),buf(8),3)
               endif
             enddo
             cbc1=cbc(iside,iel,ifld)
             if(cbc1.eq.'m'.or.cbc1.eq.'m')then
               ifmovb=.true.
               ifgngm=.true.
             endif
          endif
          do 89 ier=1,nelr
             iel = min(ier,nelm)
             IF(IFMOVB)ICRV(IEL)=1+4+9+16
             DO 89 IEDGE=1,8
                IF(IFMOVB)CCURVE(IEDGE,IEL)='M'
89        CONTINUE
         ENDIF
90    CONTINUE
      IF(NFLDS.EQ.1.and.iffmtin)READ(9,*,err=7205,end=7205)
      if(.not.iffmtin) call byte_close() 
C     Read initial conditions
      IF(VNEKOLD.GE.2.5)THEN
         READ(9,'(A70)',ERR=7206,END=7206) LINE
         IF (INDX1(LINE,'RESTART',7).NE.0) THEN
            REWIND(13)
            WRITE(13,'(A70)')LINE
            REWIND(13)
            READ(13,*,ERR=7207,END=7207)NSKIP
            REWIND(13)
            DO 56 I=1,NSKIP
               READ(9,*,err=7208,end=7208)
56          CONTINUE
            READ(9,'(A70)',ERR=7209,END=7209) LINE
         ENDIF
         REWIND(13)
         WRITE(13,'(A70)')LINE
         REWIND(13)
         READ(13,*,ERR=7210,END=7210)NSKIP
         REWIND(13)
      ENDIF
      IF(VNEKOLD.LT.2.5)NSKIP=8
      DO 57 I=1,NSKIP
         READ(9,*,err=7211,end=7211)
57    CONTINUE
C     Read Drive force data
      READ(9,*,err=7212,end=7212)
      READ(9,*,err=7213,end=7213)NSKIP
      DO 58 I=1,NSKIP
         READ(9,*,err=7214,end=7214)
58    CONTINUE
C     Read conduction element data
      READ(9,*,err=7215,end=7215)
      READ(9,*,err=7216,end=7216)nskip
      DO 60 I=1,NSKIP
         READ(9,*,err=7217,end=7217)
60    CONTINUE
C     Read History data
C     HCODE(10) IS WHETHER IT IS HISTORY, STREAKLINE, PARTICLE, ETC.
      READ(9,*,err=7218,end=7218)
C
      READ(9,*,err=51)NNHIS
      filenm(m:n)='.his'
c     CALL OPENF(26,FILENM,'OLD',1,IERR)
c     IF(IERR.NE.0)CALL PRS(' No history file '//filenm//'$')
      NHIS=0
      IHISPT=1
      IF(NNHIS.GT.0)THEN
         IHSOUT=0
         DO 50 I=1,NNHIS
            READ(9,'(1X,11A1,1X,4I5)')
     $      (HCODE(II,I),II=1,11),(LOCHIS(I2,I),I2=1,4)
            IF(HCODE(10,I).EQ.'H'.OR.HCODE(10,I).EQ.'P') then
              NHIS=NHIS+1
c             lochis(1,i) = min(lochis(1,i),nx)   ! Here, nx not yet known!
c             lochis(2,i) = min(lochis(2,i),ny)   ! Here, nx not yet known!
c             lochis(3,i) = min(lochis(3,i),nz)   ! Here, nx not yet known!
            endif
            IF(NHIS.EQ.1) IHISPT=I
            IF(HCODE(10,I).EQ.'I')THEN
               NINTEG = NINTEG + 1
C              Iobj=lochis(1,I)
            ENDIF
            IF(HCODE(10,I).EQ.'H')THEN
               IF (IHSOUT.EQ.0) THEN
                  WRITE(S,914)
                  CALL PRS(S)
                  IHSOUT=1
               ENDIF
               IX=LOCHIS(1,I)
               IY=LOCHIS(2,I)
               IZ=LOCHIS(3,I)
               IE=LOCHIS(4,I)
               IF (IF3D) THEN
                  WRITE(S,916) I,XP(IND(IX,IY,IZ,IE))
     $           ,YP(IND(IX,IY,IZ,IE)),ZP(IND(IX,IY,IZ,IE)),I
               ELSE
                  WRITE(S,917) I,XP(IND(IX,IY,IZ,IE))
     $           ,YP(IND(IX,IY,IZ,IE)),I
               ENDIF
               CALL PRS(S)
  914          FORMAT('History points:$')
  916          FORMAT(I5,3G14.5,'$',I5)
  917          FORMAT(I5,2G14.5,'$',I5)
            ENDIF
50       CONTINUE
      ENDIF
C
C     Read output specs
      READ(9,*,ERR=51,END=51)
      READ(9,*,ERR=51,END=51)NOUTS
      READ(9,*,ERR=51,END=51)IFXYO
      READ(9,*,ERR=51,END=51)IFVO
      READ(9,*,ERR=51,END=51)IFPO
      READ(9,*,ERR=51,END=51)IFTO
      IF(NOUTS.GT.4)READ(9,*,ERR=51,END=51)IFTGO
      IF(NOUTS.GT.5)READ(9,*,ERR=51,END=51)IPSCO
      IF(IPSCO .GT.0)THEN
         DO 1221 I=1,IPSCO
c           READ(9,'(1X,L1,1X,A5)',ERR=51,END=51) IFPSCO(I),PSNAME(I)
            READ(9,*) IFPSCO(I)
1221     CONTINUE
      ENDIF
C     OBJECT SPECIFICATION DATA
      READ(9,*,ERR=64,END=64)
      READ(9,*,ERR=64,END=64)NSOBJS
      IF(NSOBJS   .GT. 0)THEN
          DO 62 IOBJ=1,NSOBJS  
             if (iobj.le.msobjs) then
                READ(9,'(4X,I4,35X,A20)',ERR=64,END=64)
     $          NFACE(IOBJ),SOBJ(IOBJ)
                DO 63 IFACE=1,NFACE(IOBJ)
                   if (iface.le.mfaces) then
                      READ(9,*,ERR=64,END=64)
     $                ILSURF(1,IOBJ,IFACE),ILSURF(2,IOBJ,IFACE)
                   else
                      READ(9,*,ERR=64,END=64) idum1
                   endif
63              CONTINUE
             else
                READ(9,*,ERR=64,END=64) NDUM
                do iface=1,ndum
                   READ(9,*,ERR=64,END=64) idum1
                enddo
             endif
62        CONTINUE
      ENDIF
      READ(9,*,ERR=64,END=64)NVOBJS
      READ(9,*,ERR=64,END=64)NEOBJS
      READ(9,*,ERR=64,END=64)NPOBJS
  64  continue
      close(unit=9)
C     Set up defaults for various attributes
      SCALPT='COLOR FILL'
      VECTPT='VECTOR ARROWS'
      IF(IFTRAN)THEN
         XYATTR='HISTORY'
      ELSE
         XYATTR='PROFILE'
      ENDIF
      PLFORM='SURFACE PLOT'
      IF(IFTO)THEN
         QUANTY='TEMPERATURE'
         SUATTR='SCALAR PLOT'
      ENDIF
      IF(IFFLOW)THEN
         QUANTY='VELOCITY'
         SUATTR='VECTOR PLOT'
      ENDIF
      COMPON='X'
C     Default First Line
      XLINE(1)=X(1,1)
      XLINE(2)=X(1,2)
      YLINE(1)=Y(1,1)
      YLINE(2)=Y(1,2)
      ZLINE(1)=Z(1,1)
      ZLINE(2)=Z(1,2)
C
C     MESS TO JUMP TO
      GO TO 1111
59    CALL PRS('PARAMETER DATA CORRUPTED OR MISSING FROM READ FILE$')
      WRITE(6,*) 'PARAMETER DATA CORRUPTED OR MISSING FROM READ FILE'
      CALL DEVEX
      CALL EXITT
33    CALL PRS
     $('ERROR: MESH DATA MISSING OR CORRUPTED.  CHECK READ FILE.$')
      WRITE(6,*)
     $'ERROR: MESH DATA MISSING OR CORRUPTED.  CHECK READ FILE.'
      CALL DEVEX
      CALL EXITT
34    CALL PRS(
     $'ERROR: FILE CONTAINS ONLY PARAMETER DATA. MESH DATA MISSING$')
      WRITE(6,*)
     $'ERROR: FILE CONTAINS ONLY PARAMETER DATA. MESH DATA MISSING'
      CALL DEVEX
      CALL EXITT
35    CALL PRS('FILE DOES NOT CONTAIN B.C. DATA.'//
     $'  CONTINUING WITH MESH ONLY.$')
      GO TO 1111
40    CALL PRS('Error reading Scale factors$')
      WRITE(6,*) 'Error reading Scale factors'
      CALL EXITT
41    CALL PRS('Error reading Number of elements.$')
      GO TO 1111
42    CALL PRSI(
     $'Error reading element number and letter near element$',IEL)
      GO TO 1111
43    CALL PRSI('Error reading mesh coordinates for element$',IEL)
      GO TO 1111
45    CALL PRS('Error reading options in .REA file$')
      GO TO 1111

7201  CALL PRS('Error reading options 7201 in .REA file$')
7202  CALL PRS('Error reading options 7202 in .REA file$')
7203  CALL PRS('Error reading options 7203 in .REA file$')
7204  CALL PRS('Error reading options 7204 in .REA file$')
7224  CALL PRS('Error reading options 7224 in .REA file$')
7234  CALL PRS('Error reading options 7234 in .REA file$')
7205  CALL PRS('Error reading options 7205 in .REA file$')
7206  CALL PRS('Error reading options 7206 in .REA file$')
7207  CALL PRS('Error reading options 7207 in .REA file$')
7208  CALL PRS('Error reading options 7208 in .REA file$')
7209  CALL PRS('Error reading options 7209 in .REA file$')
7210  CALL PRS('Error reading options 7210 in .REA file$')
7211  CALL PRS('Error reading options 7211 in .REA file$')
7212  CALL PRS('Error reading options 7212 in .REA file$')
7213  CALL PRS('Error reading options 7213 in .REA file$')
7214  CALL PRS('Error reading options 7214 in .REA file$')
7215  CALL PRS('Error reading options 7215 in .REA file$')
7216  CALL PRS('Error reading options 7216 in .REA file$')
7217  CALL PRS('Error reading options 7217 in .REA file$')
7218  CALL PRS('Error reading options 7218 in .REA file$')
      write(6,*) 'this is line:'
      write(6,*) LINE
      GO TO 1111

51    CALL PRS('No output specs$')
      GO TO 1111
52    CALL PRS('No object specifications$')
      GO TO 1111
512   CALL PRS('Trouble opening .re2 file')
      GO TO 1111
44    WRITE(S,'(1X,A36,I4,A6,I3)')
     $'Error reading B.C. data for element ',IEL,' side ',ISIDE
      CALL PRS(S//'$')
C     END OF MESS BLOCK (FOR ERRORS)
C
C
1111  CONTINUE
C
C     READ IN FIRST DUMP
      do i=1,1 
         call getfld(i,ierr,.true.,.true.)
         if (ierr.eq.0) goto 1112
      enddo
C     We didn't find a file.
      CALL PRS(' ** ERROR **  Can''t open file '//filenm//'$')
c
c     Reset nx to be a small number... just to ease viewing...
      
      CALL PRS('Resetting NX to 4.$')
      nx=4
      ny=4
      nz=4
      if (ndim.eq.2) nz=1
      param(20) = nx
 1112 continue
C
c
      ntot = nx*ny*nz*nel
      if (ntot.gt.maxpts) then
         CALL PRS(' ** ERROR **  Too many points!$')
         WRITE(6,*) 'Increase maxpts to',ntot,' in basicsp.inc'
         WRITE(6,*) 'EXITING'
         CALL EXITT
      endif
      if (nel.gt.nelm) then
         CALL PRS(' ** ERROR **  Too many elements!$')
         WRITE(6,*) 'Increase NELM to',nel,' in basics.inc'
         WRITE(6,*) 'EXITING'
         CALL EXITT
      endif
c
C     Now that we have the corner and curves side data, make the mesh!
C     Set up Zpoints and derivative arrays
      CALL LEGEND(ZPTS,WGHT,NX)
      CALL GENPN1(PN,ZPTS,NX)
      CALL DERMAT(DFDR,DFDRT,PN,ZPTS,NX,NXM)
      CALL DERMAT(DGDR,DGDRT,PN,ZPTS,NX,NX )
      CALL MESPOS
C
      CALL CLEAR
C     CLEAR DIALOG AREA
      CALL SCROLL(4)
      CALL READHS
      CALL CLRMAP('HEAT')
      IF (IFFLOW) THEN
C        Velocity, select all wall boundaries
         SUATTR='SCALAR PLOT'
         QUANTY='PRESSURE'
      ELSE
C        Temperature, select all boundaries except E and P.
         SUATTR='SCALAR PLOT'
         QUANTY='TEMPERATURE'
      ENDIF
      call setwrk(.FALSE.)
C     Initialize transformation matrix
      IF (IF3D) THEN
         THETA=20.0
         PHI=25.0
         IROT=0
      ELSE
C        For 2-d, use overhead perspective
         THETA =  90.
         PHI   = -90.
         IROT  = 0
      ENDIF
      THETAr=THETA*PI/180.0
      PHIr  =PHI  *PI/180.0
C
C     Direction toward eye (normal vector)
      VHOBS(1)=COS(THETAR)*COS(PHIR)
      VHOBS(2)=COS(THETAR)*SIN(PHIR)
      VHOBS(3)=SIN(THETAR)
C       Perpendicular on x-y plane
      XHOBS(1)=-1.0*SIN(PHIR)
      XHOBS(2)=COS(PHIR)
      XHOBS(3)=0.0
C       Perpendicular to above two
      YHOBS(1)=-1.0*SIN(THETAR)*COS(PHIR)
      YHOBS(2)=-1.0*SIN(THETAR)*SIN(PHIR)
      YHOBS(3)=COS(THETAR)
C     Rescale screen-world factors so it fits on screen
C     Rescale to accomodate transformation
      CALL RESCAL
C
      DO 1 IEL=1,NEL
         NLEVEL=MAX0(NUMAPT(IEL),NLEVEL)
1     CONTINUE
C
      IF (IF3D) THEN
C
C        Default 3D stuff
C
         IFDMSH=.FALSE.
         IFMESH=.TRUE.
         IFCFBD=.TRUE.
C
C        Now set viewing transformation
         THETA =  12.0
         PHI   = -55.0
         THETAr=THETA*PI/180.0
         PHIr  =PHI  *PI/180.0
C
C       Direction toward eye (normal vector)
         VHOBS(1)=COS(THETAR)*COS(PHIR)
         VHOBS(2)=COS(THETAR)*SIN(PHIR)
         VHOBS(3)=SIN(THETAR)
C      Perpendicular on x-y plane
         XHOBS(1)=-1.0*SIN(PHIR)
         XHOBS(2)=COS(PHIR)
         XHOBS(3)=0.0
C      Perpendicular to above two
         YHOBS(1)=-1.0*SIN(THETAR)*COS(PHIR)
         YHOBS(2)=-1.0*SIN(THETAR)*SIN(PHIR)
         YHOBS(3)=COS(THETAR)
C
C        Set Zoom factors, pff
C
         IFZOOM=.TRUE.
         xfac  =  5.27
         yfac  =  5.27
         xzero = -1.08
c        yzero = -2.66
         yzero = -1.88
         wt    =  3.38
         wb    = -1.84
         wl    = -1.85
         wr    =  3.36
C
C        Select initial surface location.
C
         IF (PLTMOD.EQ.'SURFACE') THEN
            PLFORM='SURFACE ONLY'
            QUANTY='VORTICITY'
            COMPON='Z'
         ELSEIF (IFFLOW) THEN
C           Velocity, select all wall boundaries
            write(6,*) 'calling setquad'
            CALL SETQUAD
            NQUAL=1
            SRFQAL(1)='W'
            IFLD=1
            write(6,*) 'calling setsrfp'
            CALL SETSRFP(SRFQAL,NQUAL,IFLD)
            write(6,*) 'done setsrfp'
            SUATTR='SCALAR PLOT'
            QUANTY='PRESSURE'
         ELSE
            write(6,*) 'calling setquad'
            CALL SETQUAD
C           Temperature, select all boundaries except E and P.
            NQUAL=-2
            SRFQAL(1)='E'
            SRFQAL(2)='P'
            IFLD=2
            write(6,*) 'calling setsrfp'
            CALL SETSRFP(SRFQAL,NQUAL,IFLD)
            write(6,*) 'done setsrfp'
            SUATTR='SCALAR PLOT'
            QUANTY='TEMPERATURE'
         ENDIF
      ENDIF
C
c
c     These calls replaced by interp   pff 9/21/98
c     CALL locglob
c     CALL COEF
      CALL INTERP(VAL,xp(1),yp(1),zp(1),idum,wkv1,ierr)
c
      return
      END
c-----------------------------------------------------------------------
      subroutine tem
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER LABEL1*10,LABEL2*10,LABEL3*6
      LOGICAL IFCRV(4)
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
      PARAMETER (NXM2=NXM*NYM)
      COMMON /SCRTCH/ TPLOT(NXM2),XPLOT(NXM2),YPLOT(NXM2)
     $               ,ZPLOT(NXM2)
     $               ,XVEC(NXM2) ,YVEC(NXM2) ,ZVEC(NXM2)
      COMMON /ccdiag/ iee
      common /ccntr/ iel
      common /surfavg/ xsavg,ysavg,zsavg,wsavg
      CHARACTER C*20
      SAVE IFIRST
      DATA IFIRST /0/
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
C     Draw colorbar
C
      if (jhard.eq.3)        ifhard=.false.
      if (.not.ifportrait)   CALL COLRBAR
      if (jhard.eq.3)        ifhard=.true.
C
      CALL PENW(1)
C
C     Set range and thresholding levels:
C
c     CALL PRS(' Enter range (-1,2):$')
c     CALL RERR(USRMIN,USRMAX)
c     CALL PRS(' Enter threshold (-1,2):$')
c     CALL RERR(THRMIN,THRMAX)
C
C     Plot element by element
      IF (.NOT.IF3D) THEN
         DO 1000 IEL=1,NEL
C
C           Check for selection by element in 2D - pff 9 May 1988 17:47:22
C
            IF (IFSELE.AND..NOT.IFPLOT(IEL)) GOTO 1000
C
C           New plotting algorithms for color fill and fishnet - pff 7-1-90
C           (3D will come later - 10-10-90)
C
            NXYZ=NX*NY*NZ
            IEOFF=NX*NY*NZ*(IEL-1)
            WSCALE=1.0
            IF (WKMAX.NE.WKMIN)            WSCALE=1.0/(WKMAX-WKMIN)
            IF(IFUSCL.AND.USRMIN.NE.USRMAX)WSCALE=1.0/(USRMAX-USRMIN)
            CURMIN=WKMIN
            IF(IFUSCL) CURMIN=USRMIN
C           Normalize temperatures
            DO 300 I=1,NXYZ
               I1=IEOFF+I
               TPLOT(I)=WSCALE*(WORK(I1)-CURMIN)
c              IF (IFUSCL) THEN
c                 TPLOT(I)=MIN(TPLOT(I),1.0)
c                 TPLOT(I)=MAX(TPLOT(I),0.0)
c              ENDIF
  300       CONTINUE
C
C           New routine
C
            IEOFF=IEOFF+1
            CALL LFALSE(IFCRV,4)
            IF (CCURVE(1,IEL).NE.' ') IFCRV(1)=.TRUE.
            IF (CCURVE(2,IEL).NE.' ') IFCRV(2)=.TRUE.
            IF (CCURVE(3,IEL).NE.' ') IFCRV(3)=.TRUE.
            IF (CCURVE(4,IEL).NE.' ') IFCRV(4)=.TRUE.
            IF (SCALPT.EQ.'FISHNET GRID') THEN
               INRM=1
               CALL HEATF
     $         (TPLOT,XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,INRM)
            ELSE
               KTR=1
               IF (SCALPT.EQ.'CONTOUR LINES') KTR=0
                iee=iel
               CALL HEATC
     $         (TPLOT,XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,KTR)
            ENDIF
 1000    CONTINUE
C
      ELSE
c
C        3-D
C        Sort the selected list of planes according to visibility
C
         CALL SORTL
C
         WORKMIN= 9.999E20
         WORKMAX=-9.999E20
         i = 0
c        write(48,46) nmirror,nlstp
c  46    format('New surf:',2i9)
         xsavg=0.
         ysavg=0.
         zsavg=0.
         wsavg=0.
         nsavg=0
c
         if (if_output_ijke .and. .not. ifauto) 
     $                          open(unit=34,file='ijke.dat')
         if (if_output_ijke .and. .not. ifauto) 
     $                          open(unit=33,file='ijke.out')
c
         kk = 0
         zero = 0.
         if (if_output_ijke) write(33,1011) time,zero,zero
         DO Imir=1,nmirror
         DO Ipla=1,NLSTP
            i = i+1
C
C           New plotting algorithms for color fill and fishnet - pff 7-1-90
C           (3D will come later - 10-10-90)
C
            LISTA=ABS(LISW(I))
            im   =    lmir(i)
            CALL DECOD(IPLANE,IPLN,IE,IDUM,LISTA,NX,6,NELM)
            I1=1
            I2=NX
            J1=1
            J2=NY
            K1=1
            K2=NZ
            IF (IPLN.LE.2) THEN
               I1=IPLANE
               I2=IPLANE
            ELSEIF (IPLN.LE.4) THEN
               J1=IPLANE
               J2=IPLANE
            ELSE
               K1=IPLANE
               K2=IPLANE
               ENDIF
            KTR=1
c           write(48,48) ie,i1,i2,j1,j2,k1,k2
c  48       format(i7,6i3)
            IF (SCALPT.EQ.'CONTOUR LINES') KTR=0
            IFCRV(1)=.TRUE.
            IFCRV(2)=.TRUE.
            IFCRV(3)=.TRUE.
            IFCRV(4)=.TRUE.
            WSCALE=1.0
            IF (WKMAX.NE.WKMIN)            WSCALE=1.0/(WKMAX-WKMIN)
            IF(IFUSCL.AND.USRMIN.NE.USRMAX)WSCALE=1.0/(USRMAX-USRMIN)
            CURMIN=WKMIN
            IF(IFUSCL) CURMIN=USRMIN
            II=0
            DO 5001 IZ=K1,K2
            DO 5001 IY=J1,J2
            DO 5001 IX=I1,I2
c           DO 5001 IY=J1,J2,NY-1
c           DO 5001 IX=I1,I2,NX-1
               kk = kk + 1
               IPOINT=IND(IX,IY,IZ,IE)
               xc1 = xp  (ipoint)
               yc1 = yp  (ipoint)
               zc1 = zp  (ipoint)
               ux1 = u   (ipoint)
               vy1 = v   (ipoint)
               wz1 = w   (ipoint)
               wk1 = work(ipoint)

c              if (if_output_ijke) write(33,1010) ix,iy,iz,ie
c    &                             ,xc1,yc1,zc1,ux1,vy1,wz1,time
c
c              if (if_output_ijke) write(33,1010) ix,iy,iz,kk
c    &                             ,xc1,yc1,zc1
c
               if (if_output_ijke.and..not.ifauto)
     $            write(34,1010) ix,iy,iz,ie,xc1,yc1,zc1
               if (if_output_ijke) write(33,1011) ux1,vy1,wz1,wk1
 1010          format(3i3,i8,1p7e13.4)
 1011          format(1p7e13.5)
               II=II+1
               XPLOT(II)=XP(IPOINT)
               YPLOT(II)=YP(IPOINT)
               ZPLOT(II)=ZP(IPOINT)
               WORKIP =WORK(IPOINT)
               TPLOT(II)=WSCALE*(WORKIP-CURMIN)
               xsavg=xsavg+xplot(ii)
               ysavg=ysavg+yplot(ii)
               zsavg=zsavg+zplot(ii)
               wsavg=wsavg+workip
               nsavg=nsavg+1
C
C              Find observed range:
C
               IF (WORKIP.LT.WORKMIN) THEN
                  WORKMIN=WORKIP
                  XWMIN=XP(IPOINT)
                  YWMIN=YP(IPOINT)
                  ZWMIN=ZP(IPOINT)
                  IEMIN=IE
               ENDIF
               IF (WORKIP.GT.WORKMAX) THEN
                  WORKMAX=WORKIP
                  XWMAX=XP(IPOINT)
                  YWMAX=YP(IPOINT)
                  ZWMAX=ZP(IPOINT)
                  IEMAX=IE
               ENDIF
C
               IF (IFUSCL) THEN
                  TPLOT(II)=MIN(TPLOT(II),1.0)
                  TPLOT(II)=MAX(TPLOT(II),0.0)
               ENDIF
 5001       CONTINUE
c
c           Check for mirroring
c
            if (im.gt.1) call vmirror(tplot,xplot,yplot,zplot,nx,im)
c
            IF (SCALPT.EQ.'FISHNET GRID') THEN
               INRM=1
c
c              lplane = lista
c              call signpl(lplane)
c              INRM=1
c              IF (lplane.LT.0) INRM=-1
c
               IF (lisw(i).LT.0) INRM=-1
               CALL HEATF(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,INRM)
            ELSE
               CALL HEATC(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,KTR)
            ENDIF
         enddo
         enddo
c
         if (if_output_ijke .and. .not. ifauto) close(unit=33)
         if (if_output_ijke .and. .not. ifauto) close(unit=34)
c
         if (nsavg.gt.0) then
            xsavg=xsavg/nsavg
            ysavg=ysavg/nsavg
            zsavg=zsavg/nsavg
            wsavg=wsavg/nsavg
         endif
C
C        Range information:
C
            WRITE(S,5004) wsavg,xsavg,ysavg,zsavg,nsavg
            CALL PRS(S//'$')
         IF(PLFORM.ne.'VOLUME')THEN
            WRITE(S,5002) WORKMIN,XWMIN,YWMIN,ZWMIN,IEMIN
            CALL PRS(S//'$')
            WRITE(S,5003) WORKMAX,XWMAX,YWMAX,ZWMAX,IEMAX
            CALL PRS(S//'$')
 5002       FORMAT(' Local Min:',E13.5,5x,'X:',3E12.3,I7,'$')
 5003       FORMAT(' Local Max:',E13.5,5x,'X:',3E12.3,I7,'$')
 5004       FORMAT(' Local Avg:',E13.5,5x,'X:',3E12.3,I7,'$')
         ENDIF
C
      ENDIF
C
      return
      end
c-----------------------------------------------------------------------
      subroutine fill_tx
 
c     fill color bar
 
      include 'basics.inc'
      include 'basicsp.inc'
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX

      wkmint = wkmin
      wkmaxt = wkmax
      if (ifuscl) then
         wkmint = usrmin
         wkmaxt = usrmax
      endif

      delta = (wkmaxt-wkmint)/14

      do i=1,15
         tx(i) = wkmint + (i-1)*delta
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine COLRBAR
C
C     Draw and label color bar
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER LABEL1*10,LABEL2*10,LABEL3*6
      LOGICAL IFCRV(4)
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
      CHARACTER C*20
      SAVE IFIRST
      DATA IFIRST /0/
C
C     Return if IFCLRB has been turned off
C
      IF (.NOT.IFCLRB) return
C
      IF(IFIRST.NE.1.OR.IFNOSEG)THEN
         IF(IFNOSEG)CALL DRCOVR(13)
         IFIRST=1
C        DRBAR Draws the color bar onto segment 12
         call fill_tx
         call drbar(tx,15)
      ELSE
C        Disappear menu and cover; appear Color bar.
         CALL SGVIS(13,0)
         CALL SGVIS(14,0)
         CALL SGVIS(15,0)
         CALL SGVIS(15,1)
      ENDIF
C
      IF (WKMAX.EQ.WKMIN) THEN
         IF(QUANTY.EQ.'PRESSURE') THEN
             CALL PRSRS
     $       ('*****Constant Field; P=$',WKMAX,'********$')
         ELSE IF(QUANTY(1:14).EQ.'PASSIVE SCALAR') THEN
             CALL PRSRS
     $       ('*****Constant Field; PS=$',WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'TEMPERATURE') THEN
             CALL PRSRS
     $       ('*****Constant Field; T=$',WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'STREAMFUNCTION') THEN
             CALL PRSRS
     $       ('*****Constant Field; PSI=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VORTEX')THEN
             CALL PRSRS
     $       ('*****Constant Field; VORTEX=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'V.W')THEN
             CALL PRSRS
     $       ('*****Constant Field; V.W=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VORTICITY')THEN
             CALL PRSRS
     $       ('*****Constant Field; VORTICITY=$',
     $       WKMAX,'********$')
         ELSE IF(QUANTY.EQ.'VELOCITY') THEN
             CALL PRSRS(
     $       '*****Constant Field; '//COMPON//' VELOCITY=$'
     $       ,WKMAX,'********$')
         ELSE 
             CALL PRSRS
     $       ('*****Constant Field; '//quanty//'=$'
     $       ,WKMAX,'********$')
         ENDIF
         return
      ENDIF
C     Draw box to overwrite any old labels

      YTOP2=0.75
      YBOT2=-.00
      CALL FILLP(0)
      CALL BEGINB(XPHY(1.00),YPHY(YTOP2))
      CALL MOVESC(     1.30 ,     YTOP2)
      CALL MOVESC(     1.30 ,     YBOT2)
      CALL MOVESC(     1.00 ,     YBOT2)
      CALL MOVESC(     1.00 ,     YTOP2)
      CALL ENDP
C
      IF(QUANTY.EQ.'VELOCITY') THEN
         WRITE(LABEL1,'(''V'',A1,''min$'')')COMPON
         WRITE(LABEL2,'(''V'',A1,''max$'')')COMPON
         IF(COMPON.EQ.'S')WRITE(LABEL1(1:2),'(''Sp'')')
         IF(COMPON.EQ.'S')WRITE(LABEL2(1:2),'(''Sp'')')
      ELSE IF(QUANTY.EQ.'PRESSURE') THEN
         LABEL1 = 'Pmin=$'
         LABEL2 = 'Pmax=$'
      ELSE IF(QUANTY.EQ.'VORTEX') THEN
         LABEL1 = 'VORTEXmn=$'
         LABEL2 = 'VORTEXmx=$'
      ELSE IF(QUANTY.EQ.'V.W') THEN
         LABEL1 = 'V.Wmn=$'
         LABEL2 = 'V.Wmx=$'
      ELSE IF(QUANTY.EQ.'VORTICITY') THEN
         LABEL1 = 'VRmin=$'
         LABEL2 = 'VRmax=$'
      ELSE IF(QUANTY.EQ.'TOTAL PRESSURE') THEN
         LABEL1 = 'T Pmin=$'
         LABEL2 = 'T Pmax=$'
      ELSE IF(QUANTY.EQ.'STREAMFUNCTION') THEN
         LABEL1 = 'PSImin=$'
         LABEL2 = 'PSImax=$'
      ELSE IF(QUANTY.EQ.'DIVERGENCE') THEN
         LABEL1 = 'DIVmin=$'
         LABEL2 = 'DIVmax=$'
      ELSE  IF(QUANTY.EQ.'TEMPERATURE') THEN
         LABEL1 = 'Tmin=$'
         LABEL2 = 'Tmax=$'
      ELSE  IF(QUANTY(1:14).EQ.'PASSIVE SCALAR') THEN
         LABEL1 = 'PSmin=$'
         LABEL2 = 'PSmax=$'
         LABEL3 = PSNAME(LPSDMP)
         LABEL3(6:6)='$'
         CALL GWRITE(XPHY(0.7),YPHY(0.95),1.0,LABEL3)
      ELSE  
         WRITE(LABEL1,'(a6,''min$'')')quanty(1:6)
         WRITE(LABEL2,'(a6,''max$'')')quanty(1:6)
      ENDIF

      call drbar(tx,15)

      wkmint = wkmin
      wkmaxt = wkmax
      if (ifuscl) then
         wkmint = usrmin
         wkmaxt = usrmax
      endif

      WRITE(C,'(G11.4,''$'')')wkmint
      CALL GWRITE(XPHY(1.10+.034),YPHY(0.07),1.0,C)
      CALL GWRITE(XPHY(1.02),YPHY(0.07),1.0,LABEL1)
      WRITE(C,'(G11.4,''$'')')wkmaxt
      CALL GWRITE(XPHY(1.10+.034),YPHY(0.72),1.0,C)
      CALL GWRITE(XPHY(1.02),YPHY(0.72),1.0,LABEL2)
      IF(DERIV.NE.'NO')THEN
         LABEL2=DERIV//'$'
         CALL GWRITE(XPHY(1.02),YPHY(0.76),1.0,LABEL2)
         CALL GWRITE(XPHY(1.02),YPHY(0.11),1.0,LABEL2)
      ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine DRFRAM
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CALL COLOR(1)
C     Draw Title
C
      if (jhard.eq.3) ifhard=.false.
      sesion(11:11)='$'
      CALL GWRITE(XPHY(1.10),YPHY(0.95),1.3,sesion)
c     CALL GWRITE(XPHY(0.50),YPHY(0.95),1.5,'NEKTON 2.6$')
c     CALL GWRITE(XPHY(0.81),YPHY(0.95),1.5,DATE//'$')
      CALL GWRITE(XPHY(0.50),YPHY(0.95),1.5,DATE//'$')
C     Draw Box around menu area
C
      CALL MOVESC(1.0,0.1)
      CALL DRAWSC(1.3,0.1)
      CALL DRAWSC(1.3,1.0)
      CALL DRAWSC(1.0,1.0)
      CALL DRAWSC(1.0,0.1)
C
C     DON'TDraw Elevator
C      IF(NLEVEL.GT.0) THEN
      IF(.FALSE.)THEN
         XELE  = 0.97
         YELEB = 0.1
         YELET = 0.4
         DXELE = 0.02
         CALL COLOR(1)
         CALL MOVESC(XELE,YELEB)
         CALL DRAWSC(XELE,YELET)
C        Draw ticks
         CALL COLOR(15)
         DO 10 I=0,NLEVEL
            YTICK=FLOAT(I)/NLEVEL *(YELET-YELEB) + YELEB
            CALL MOVESC(XELE-DXELE,YTICK)
            CALL DRAWSC(XELE+DXELE,YTICK)
10       CONTINUE
         CALL COLOR(1)
         IF(IELLEV.EQ.0)THEN
            ZFRAC=0
         ELSE
            ZFRAC=0
         ENDIF
         YTICK=((ILEVEL-1.) + ZFRAC )/NLEVEL*(YELET-YELEB) + YELEB
         CALL MOVESC(XELE-DXELE/2,YTICK)
         CALL DRAWSC(XELE+DXELE/2,YTICK)
      ENDIF
C
c
      if (jhard.eq.3) ifhard=.true.
      CALL DRAXIS
C
      return
      END
c-----------------------------------------------------------------------
      subroutine DRISO
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      logical ifcbdr,ifcbdn
c
      COMMON /FILLFG/ IFILLM
      INTEGER FCORNS (4,6),FEDGES(4,6),FFLIP(4,6)
      DIMENSION XISOM(8),YISOM(8),CSPACE(100),XCRVED(100),YCRVED(100)
      CHARACTER CCUR
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
      DATA FEDGES / 9  ,1, 10 ,5,
     $              10 ,2, 11 ,6,
     $              11 ,3, 12 ,7,
     $              12 ,4, 9  ,8,
     $              4  ,1, 2  ,3,
     $              8  ,7, 6  ,5 /
      DATA FFLIP  / -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $              -1, 1, 1, -1,
     $               1, 1, 1,  1,
     $              -1,-1,-1, -1 /
      DATA IFIRST /0/
      SAVE IFIRST
C
      IF(IFIRST.EQ.0)THEN
         IFIRST=1
         CALL RESCAL
      ENDIF
      NPOINT=NCSEGS
      DO 18 I=1,NPOINT
         CSPACE(I)=(I-1.0)/(NPOINT-1.0)
18    CONTINUE
      IF (IF3D) CALL SORTZ
      CALL COLOR(1)
      CALL PENW(1)
C !!??
      NLOOPS=NEL
      IF(IF3D)NLOOPS=NEL*6
      DO 400 NUMBER=1,NLOOPS
         IF(NDIM.EQ.2)THEN
           IEL=NUMBER
           IFACE=5
         ELSE
           IEL  =IZDPTH(NUMBER,1)
           IFACE=IZDPTH(NUMBER,2)
         ENDIF
         IFLD=2
         IF(IFFLOW)IFLD=1
         ifcbdr = .false.
         IF (  CBC(IFACE,IEL,IFLD).EQ.'v'  .OR.
     $         CBC(IFACE,IEL,IFLD).EQ.'V'  .OR.
     $         CBC(IFACE,IEL,IFLD).EQ.'W'  ) ifcbdr = .true.
C        Draw and fill this face
         ICORN=4
	 if (if3d) then
	    zavg=0.0
	    do 917 iii=1,4
               zavg=zavg+z(iel,iii)
  917       continue
	 ELSE
	    zavg=1.0
         endif
         IF(IFILLM.EQ.1.and.
     $    ((zavg.gt.0.001.and.iface.eq.5).or.
     $     (zavg.gt.0.001.and.iface.eq.6).or.
     $      iface.lt.5) ) THEN
C           FILL Mesh with colored panels
C           Take outward normal of face, dot it with view angle.  If we are
C           Looking at the inside of an element, make color darker.
C           Check Here for points not counter-clockwise
            ic1=fcorns(1,iface)
            ic2=fcorns(2,iface)
            ic3=fcorns(3,iface)
            ic4=fcorns(4,iface)
            x1=xiso(x(iel,ic1),y(iel,ic1),z(iel,ic1))
            x2=xiso(x(iel,ic2),y(iel,ic2),z(iel,ic2))
            x3=xiso(x(iel,ic3),y(iel,ic3),z(iel,ic3))
            x4=xiso(x(iel,ic4),y(iel,ic4),z(iel,ic4))
            y1=yiso(x(iel,ic1),y(iel,ic1),z(iel,ic1))
            y2=yiso(x(iel,ic2),y(iel,ic2),z(iel,ic2))
            y3=yiso(x(iel,ic3),y(iel,ic3),z(iel,ic3))
            y4=yiso(x(iel,ic4),y(iel,ic4),z(iel,ic4))
            elarea = x1*y2-x2*y1 +
     $               x2*y3-x3*y2 +
     $               x3*y4-x4*y3 +
     $               x4*y1-x1*y4
            IF(ELAREA .GT. 0.0) THEN
C              Regular colors
c              IF(IFACE.EQ.6) CALL FILLP(-8)
c              IF(IFACE.EQ.5) CALL FILLP(-8)
               IF(IFACE.EQ.6) CALL FILLP(-14)
               IF(IFACE.EQ.5) CALL FILLP(-14)
               IF(IFACE.EQ.4) CALL FILLP(-14)
               IF(IFACE.EQ.2) CALL FILLP(-14)
               IF(IFACE.EQ.3) CALL FILLP(-12)
               IF(IFACE.EQ.1) CALL FILLP(-12)
            ELSE
C              Darker Colors
c              IF(IFACE.EQ.6) CALL FILLP(-6)
c              IF(IFACE.EQ.5) CALL FILLP(-6)
               IF(IFACE.EQ.6) CALL FILLP(-15)
               IF(IFACE.EQ.5) CALL FILLP(-15)
               IF(IFACE.EQ.4) CALL FILLP(-15)
               IF(IFACE.EQ.2) CALL FILLP(-15)
               IF(IFACE.EQ.3) CALL FILLP(-13)
               IF(IFACE.EQ.1) CALL FILLP(-13)
            ENDIF
            IFILLC=1
         ELSE
C           Fill with black panels
            IFILLC=0
            CALL FILLP(0)
         ENDIF
C        Don't draw or fill elemental boundaries; only physical boundaries
         IFLD=2
         IF(IFFLOW)IFLD=1
         IF( (CBC(IFACE,IEL,IFLD).NE.'E'.AND.IFDMSH) .OR.
     $       (.NOT.IF3D)                             .OR.
     $       ifcbdr                                 ) then
            IF( (IFDMSH.AND..NOT.IF3D.AND.CBC(IFACE,IEL,IFLD).NE.'P  ')
     $         .OR.(IF3D.AND.CBC(IFACE,IEL,IFLD).NE.'P  ')
     $         .AND.(IFILLC.EQ.1)                              ) THEN
               CALL BEGINB(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))),
     $                     YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))))
            ELSE
               CALL MOVEC (XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))),
     $                     YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                          Y(IEL,FCORNS(ICORN,IFACE)),
     $                          Z(IEL,FCORNS(ICORN,IFACE))))
            ENDIF
C
C           Don't draw mesh on periodic boundaries if IFILLM=1
            IF(CBC(IFACE,IEL,IFLD).NE.'P  '.OR.IFILLM.NE.1)THEN
               DO 370 ICORN=1,4
                  IICORN=ICORN-1
                  IF(IICORN.EQ.0)IICORN=IICORN+4
                  ifcbdn = .false.
                  IF (  CBC(iicorn,IEL,IFLD).EQ.'v'  .OR.
     $                  CBC(iicorn,IEL,IFLD).EQ.'V'  .OR.
     $                  CBC(iicorn,IEL,IFLD).EQ.'W'  ) ifcbdn = .true.
                  IF (.NOT.IF3D.AND.ifcbdn) then
                     CALL COLOR(1)
                     CALL PENW(5)
                  ENDIF
C                 Find out which edge we are drawing
                  IEDGE=FEDGES(ICORN,IFACE)
                  IF(IEDGE.LE.8)CCUR=CCURVE(IEDGE,IEL)
                  IF(IEDGE.GT.8 .OR. CCUR.EQ.' ')THEN
C                    Straight side
                     IF(IFDMSH.OR.ifcbdn.or.if3d) then
                        CALL DRAWC(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                                 Y(IEL,FCORNS(ICORN,IFACE)),
     $                                 Z(IEL,FCORNS(ICORN,IFACE))),
     $                            YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                                 Y(IEL,FCORNS(ICORN,IFACE)),
     $                                 Z(IEL,FCORNS(ICORN,IFACE))))
                     ELSE
                        CALL MOVEC(XISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                                 Y(IEL,FCORNS(ICORN,IFACE)),
     $                                 Z(IEL,FCORNS(ICORN,IFACE))),
     $                            YISO(X(IEL,FCORNS(ICORN,IFACE)),
     $                                 Y(IEL,FCORNS(ICORN,IFACE)),
     $                                 Z(IEL,FCORNS(ICORN,IFACE))))
                     ENDIF
                  ELSE
C                    Draw curved side
                     CALL GETPTS(NPOINT,CSPACE,IEL,IEDGE,XCRVED,YCRVED)
                     IFLIP=FFLIP(ICORN,IFACE)
                     IF(IFLIP.EQ.1)THEN
                        IBEGIN=1
                        IEND  =NPOINT
                     ELSE
                        IBEGIN=NPOINT
                        IEND  =1
                     ENDIF
                     ZCRVED=Z(IEL,FCORNS(ICORN,IFACE))
C
C      Draw only if draw mesh is .TRUE.
                     IF(IFDMSH.OR.ifcbdn.OR.IF3D)THEN
                        DO 118 I=IBEGIN,IEND,IFLIP
                        CALL DRAWC( XISO(XCRVED(I),YCRVED(I),ZCRVED) ,
     $                             YISO(XCRVED(I),YCRVED(I),ZCRVED) )
  118                   CONTINUE
                     ELSE
                        I=IEND
                        CALL MOVEC( XISO(XCRVED(I),YCRVED(I),ZCRVED) ,
     $                             YISO(XCRVED(I),YCRVED(I),ZCRVED) )
                     ENDIF
                  ENDIF
                  IF(.NOT.IF3D.AND.ifcbdn)THEN
                     CALL COLOR(1)
                     CALL PENW(1)
                  ENDIF
370            CONTINUE
            ENDIF
            IF(IFILLC.EQ.1.AND.CBC(IFACE,IEL,IFLD).NE.'P  ') CALL ENDP
         ENDIF
400   CONTINUE
      CALL COLOR(1)
      CALL PENW(3)
      return
      END
c-----------------------------------------------------------------------
      subroutine plerr
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
      CHARACTER LABEL1*10,LABEL2*10,C*20
      DATA IFIRST /0/
      SAVE IFIRST
C     Turn on bar

      return  ! don't plot error (broken, 12/8/08, pff)

      IF(IFIRST.NE.1)THEN
         IFIRST=1
C        DRBAR Draws the color bar onto segment 12
         CALL DRBAR(TX,15)
      ELSE
C        Disappear menu and cover; appear Color bar.
         CALL SGVIS(13,0)
         CALL SGVIS(14,0)
         CALL SGVIS(15,0)
         CALL SGVIS(15,1)
      ENDIF
      CALL PRSRS('LOG ERROR IN $',quanty,compon//'$')
      CALL PRS(S//'$')
      IF(QUANTY.EQ.'VELOCITY') THEN
         LABEL1='Verr$'
         LABEL2='Verr$'
         IFERR=1
                  IF(COMPON.EQ.'X') CALL CALCERR(U,IFERR)
                  IF(COMPON.EQ.'Y') CALL CALCERR(V,IFERR)
                  IF(COMPON.EQ.'Z') CALL CALCERR(W,IFERR)
c                  IF(COMPON.EQ.'N')THEN
c                     IF(PLANE.EQ.'X-PLANE') CALL CALCERR(U,IFERR)
c                     IF(PLANE.EQ.'Y-PLANE') CALL CALCERR(V,IFERR)
c                     IF(PLANE.EQ.'Z-PLANE') CALL CALCERR(W,IFERR)
C                  ENDIF
      ELSE IF(QUANTY.EQ.'PRESSURE') THEN
         LABEL1 = 'Perr=$'
         LABEL2 = 'Perr=$'
         CALL PRS( 'ERROR IN P NOT AVAILABLE YET$')
          IFERR=3
C         CALL CALCERR(P,IFERR)
        return
      ELSE  IF(QUANTY.EQ.'TEMPERATURE') THEN
         LABEL1 = 'Terr=$'
         LABEL2 = 'Terr=$'
         IFERR=2
         CALL CALCERR(T,IFERR)
      ELSE  IF(QUANTY(1:14).EQ.'PASSIVE SCALAR') THEN
         LABEL1 = 'PSerr=$'
         LABEL2 = 'PSerr=$'
         IFERR=4
         CALL PRS( 'ERROR IN PS NOT AVAILABLE YET$')
C         CALL CALCERR(PSA,IFERR)  !---CHECK PSA
         return
      ENDIF
C     Plot log error
      ERRMIN=1.0E10
      ERRMAX=0.0
      DO 10 IEL=1,NEL
C        Find max and min errors
         ERRMIN=AMIN1(ERRMIN,ALOG10(CERROR(IFERR,IEL)))
         ERRMAX=AMAX1(ERRMAX,ALOG10(CERROR(IFERR,IEL)))
10    CONTINUE
      IF(ERRMIN.EQ.ERRMAX) THEN
                  CALL PRSRS('*****Constant Field; ERR=$',ERRmax,
     $            '********$')
                  return
                  ENDIF
C     Fill in boxes
C     Draw box to overwrite any old labels
      YTOP =0.10
      YBOT =0.06
      YTOP2=0.75
      YBOT2=0.71
      CALL FILLP(-2)
      CALL BEGINB(XPHY(1.01),YPHY(YTOP))
      CALL MOVESC(     1.29 ,     YTOP)
      CALL MOVESC(     1.29 ,     YBOT)
      CALL MOVESC(     1.01 ,     YBOT)
      CALL MOVESC(     1.01 ,     YTOP)
      CALL ENDP
      CALL FILLP(-14)
      CALL BEGINB(XPHY(1.01),YPHY(YTOP2))
      CALL MOVESC(     1.29 ,     YTOP2)
      CALL MOVESC(     1.29 ,     YBOT2)
      CALL MOVESC(     1.01 ,     YBOT2)
      CALL MOVESC(     1.01 ,     YTOP2)
      CALL ENDP
C
C
      WRITE(C,'(G11.4,''$'')')ERRMIN
      CALL GWRITE(XPHY(1.10),YPHY(0.07),1.0,C)
      CALL GWRITE(XPHY(1.02),YPHY(0.07),1.0,LABEL1)
      WRITE(C,'(G11.4,''$'')')ERRMAX
      CALL GWRITE(XPHY(1.10),YPHY(0.72),1.0,C)
      CALL GWRITE(XPHY(1.02),YPHY(0.72),1.0,LABEL2)
      CALL PENW(1)
C
      WRITE(41,*)
     #  'ERRMIN     ERRMAX     ERROR     NORMERR    LOG10(ERR)'
      DO 20 IEL=1,NEL
         ERRREL=1.-(ERRMAX-ALOG10(CERROR(IFERR,IEL)))/(ERRMAX-ERRMIN)
         write(41,*) errmin,errmax,cerror(iferr,iel),
     #              errrel,ALOG10(CERROR(IFERR,IEL))
         ICOLOR=ERRREL*14+2
         CALL DRERR(IEL,ICOLOR)
20    CONTINUE
      CALL PRS('Hit mouse button to continue$')
      CALL MOUSE(POINT1,POINT2,BUTTON)
      return
      END
c-----------------------------------------------------------------------
      subroutine DRERR(IEL,ICOLOR)
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
      CALL COLOR(1)
      CALL FILLP(-ICOLOR)
      K=1
      DO 20 IC=1,5
         IF(IC.EQ.1.OR.IC.EQ.5)THEN
            I=1
            J=1
         ELSE IF(IC.EQ.2)THEN
            I=NX
            J=1
         ELSE IF(IC.EQ.3)THEN
            I=NX
            J=NY
         ELSE IF(IC.EQ.4)THEN
            I=1
            J=NY
         ENDIF
         IF(IC.EQ.1)THEN
            CALL BEGINP(XISO(
     $      XP(IND(I,J,K,IEL)),YP(IND(I,J,K,IEL)),ZP(IND(I,J,K,IEL))),
     $      YISO(
     $      XP(IND(I,J,K,IEL)),YP(IND(I,J,K,IEL)),ZP(IND(I,J,K,IEL))))
         ELSE
c            CALL DRAWED(IEL,IC,1)
            CALL DRAW3
     $      (XP(IND(I,J,K,IEL)),YP(IND(I,J,K,IEL)),ZP(IND(I,J,K,IEL)))
         ENDIF
20    CONTINUE
      CALL ENDP
       IF(.NOT.IF3D) THEN
      DO 30 IC=1,4
30    CALL DRAWED(IEL,IC,1)
        CALL HARD
        call color(1)
C         Draw only those on this floor
          IF(NUMAPT(IEL).EQ.ILEVEL)THEN
            IF(ICRV(IEL).EQ.0)THEN
               DO 209 I = 1,NX
                 CALL MOVEC(XP(ind(I,1 ,1,IEL)),YP(ind(I,1 ,1,IEL)))
                 CALL DRAWC(XP(ind(I,NY,1,IEL)),YP(ind(I,NY,1,IEL)))
209            CONTINUE
               DO 211 J = 1,NY
                 CALL MOVEC(XP(ind(1 ,J,1,IEL)),YP(ind(1 ,J,1,IEL)))
                 CALL DRAWC(XP(ind(NX,J,1,IEL)),YP(ind(NX,J,1,IEL)))
211            CONTINUE
            ELSE
C              Draw it the hard, slow, curvy way
               DO 1109 J=1,NY-1
                   DO 1109 I = 1,NX
                    IF(CCURVE(2,IEL).NE.' '.AND. I.EQ.NX) GO TO 1108
                    IF(CCURVE(4,IEL).NE.' '.AND. I.EQ. 1) GO TO 1108
                   CALL MOVEC(XP(ind(I,J  ,1,IEL)),YP(ind(I,J  ,1,IEL)))
                   CALL DRAWC(XP(ind(I,J+1,1,IEL)),YP(ind(I,J+1,1,IEL)))
1108                CONTINUE
1109            CONTINUE
               DO 111 I=1,NX-1
                   DO 111 J = 1,NY
                     IF(CCURVE(1,IEL).NE.' '.AND. J.EQ. 1) GO TO 110
                     IF(CCURVE(3,IEL).NE.' '.AND. J.EQ.NY) GO TO 110
                  CALL MOVEC(XP(ind(I  ,J,1,IEL)),YP(ind(I  ,J,1,IEL)))
                  CALL DRAWC(XP(ind(I+1,J,1,IEL)),YP(ind(I+1,J,1,IEL)))
110                  CONTINUE
111            CONTINUE
C              Draw curved sides for hardcopy
               DO 130 IEDGE=1,4
                  IF(CCURVE(IEDGE,IEL).NE.' ')THEN
                     CALL MOVEC(X(IEL,IEDGE),Y(IEL,IEDGE))
                     CALL DRAWED(IEL,IEDGE,1)
                  ENDIF
130            CONTINUE
            ENDIF
          ENDIF
112      continue
         CALL NOHARD
         NCURVE=0
            DO 114 IEDGE=1,8
               IF(CCURVE(IEDGE,IEL).NE.' ') NCURVE=NCURVE+1
114      CONTINUE
       ENDIF
      return
      END
c-----------------------------------------------------------------------
      subroutine MESPOS
C     Generates mesh in postprocessor
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      DIMENSION DXCHEB(20),DYCHEB(20)
      REAL CSPACE(100),XCRVED(100),YCRVED(100),XFRAC(20)
      REAL XPTSEL(NXM,NYM),YPTSEL(NXM,NYM)
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
      NX=PARAM(20)
      NY=NX
      IF(IF3D)NZ=NX
      IF(.NOT.IF3D)NZ=1
      IF(NKTONV.EQ.1) THEN
         DO 104 I=1,NX
            XXPTS(I) = -COS(PI*FLOAT(I-1)/FLOAT(NX-1))
            YYPTS(I) = -COS(PI*FLOAT(I-1)/FLOAT(NX-1))
104      CONTINUE
      ELSE
C        Need legendre points for mesh drawn on screen
         CALL LEGEND(XXPTS,WGHT,NX)
         CALL LEGEND(YYPTS,WGHT,NY)
      ENDIF
      NSIDES=4
      IFCEN=.FALSE.
      IF(IF3D)NSIDES=6
C
      DO 105 I=1,NX
         XFRAC(I)=XXPTS(I)/2.0 + 0.5
105   CONTINUE
      NTOP=0
      IF(IF3D) NTOP=1
      DO 108 IEL = 1,NEL
      DO 108 ITOP = 0,NTOP
         IF(ITOP.EQ.0) K=1
         IF(ITOP.EQ.1) K=NZ
         DO 106 I = 1,NX
           DO 106 J = 1,NY
C           Calculate original xpts based on straight sides
 
            XP(IND(I,J,K,IEL)) =  1./4.* (
     $      X(IEL,1+4*ITOP)*(XXPTS(I)-1.0)*(YYPTS(J)-1.0)-
     $      X(IEL,2+4*ITOP)*(XXPTS(I)+1.0)*(YYPTS(J)-1.0)+
     $      X(IEL,3+4*ITOP)*(XXPTS(I)+1.0)*(YYPTS(J)+1.0)-
     $      X(IEL,4+4*ITOP)*(XXPTS(I)-1.0)*(YYPTS(J)+1.0))
            YP(IND(I,J,K,IEL)) =  1./4.* (
     $      Y(IEL,1+4*ITOP)*(XXPTS(I)-1.0)*(YYPTS(J)-1.0)-
     $      Y(IEL,2+4*ITOP)*(XXPTS(I)+1.0)*(YYPTS(J)-1.0)+
     $      Y(IEL,3+4*ITOP)*(XXPTS(I)+1.0)*(YYPTS(J)+1.0)-
     $      Y(IEL,4+4*ITOP)*(XXPTS(I)-1.0)*(YYPTS(J)+1.0))
            ZP(IND(I,J,K,IEL)) =
     $      Z(IEL,1+4*ITOP)
C           Initial values for temporary array
            XPTSEL(I,J) = XP(IND(I,J,K,IEL))
            YPTSEL(I,J) = YP(IND(I,J,K,IEL))
106       CONTINUE
           DO 165 IE=1,4
             IEDGE=IE+4*ITOP
             IF(CCURVE(IEDGE,IEL).NE.' ')ICRV(IEL)=ICRV(IEL)+IEDGE**2
165        CONTINUE
           IF(ICRV(IEL).NE.0)THEN
C             Shift internal points to account for curviness
              DO 107 IE=1,4
                IEDGE=IE+4*ITOP
                IF(CCURVE(IEDGE,IEL).NE.' ')THEN
                  NPOINT=NX
                  CALL GETPTS(NPOINT,XFRAC,IEL,IEDGE,XCRVED,YCRVED)
                  DO 117 I = 1,NX
                    IF(IE.EQ.1)THEN
                      II=I
                      JJ=1
                    ELSE IF(IE.EQ.2)THEN
                      II=NX
                      JJ=I
                    ELSE IF(IE.EQ.3)THEN
                      II=NX-I+1
                      JJ=NX
                    ELSE IF(IE.EQ.4)THEN
                      II=1
                      JJ=NX-I+1
                     ENDIF
                    DXCHEB(I)=XCRVED(I)-XP(IND(II,JJ,K,IEL))
                    DYCHEB(I)=YCRVED(I)-YP(IND(II,JJ,K,IEL))
117               CONTINUE
                  DO 120 I=1,NX
                    DO 120 J=1,NY
                      IF(IE.EQ.2)III=J
                      IF(IE.EQ.4)III=NX-J+1
                      IF(IE.EQ.1)III=I
                      IF(IE.EQ.3)III=NX-I+1
                      IF(IE.EQ.1) FAC = XFRAC(NX-J+1)
                      IF(IE.EQ.2) FAC = XFRAC(I)
                      IF(IE.EQ.3) FAC = XFRAC(J)
                      IF(IE.EQ.4) FAC = XFRAC(NX-I+1)
C
                      XPTSEL(I,J)=XPTSEL(I,J)+DXCHEB(III)*FAC
                      YPTSEL(I,J)=YPTSEL(I,J)+DYCHEB(III)*FAC
120               CONTINUE
                ENDIF
107           CONTINUE
C             Now that the points in the temporary array are moved, copy to xpts
              DO 109 I=1,NX
                DO 109 J=1,NY
                  XP(IND(I,J,K,IEL))=XPTSEL(I,J)
                  YP(IND(I,J,K,IEL))=YPTSEL(I,J)
109           CONTINUE
           endif
108   CONTINUE
      IF(IF3D)THEN
C        Fill in other points with Linear interpolation
         DO 200 IEL=1,NEL
            DO 200 I=1,NX
               DO 200 J=1,NY
                  DO 200 K=2,NZ-1
                     XP(IND(I,J,K ,IEL))=XP(IND(I,J,1,IEL))+XFRAC(K)*
     $              (XP(IND(I,J,NZ,IEL))-XP(IND(I,J,1,IEL)) )
                     YP(IND(I,J,K ,IEL))=YP(IND(I,J,1,IEL))+XFRAC(K)*
     $              (YP(IND(I,J,NZ,IEL))-YP(IND(I,J,1,IEL)) )
                     ZP(IND(I,J,K ,IEL))=ZP(IND(I,J,1,IEL))+XFRAC(K)*
     $              (ZP(IND(I,J,NZ,IEL))-ZP(IND(I,J,1,IEL)) )
200      CONTINUE
      ENDIF
C
C     Generate spherical meshes, if present.
C
      if (if3d) then
        do ie=1,nel
           call genxyz (xp,yp,zp,ie,nx,ny,nz) 
        enddo
      endif
c
c       DO 500 IE=1,NEL
c       IEOFF=NX*NY*NZ*(IE-1)+1
c       DO 400 IFACE=1,6
c         IF (CCURVE(IFACE,IE).EQ.'s'.or.
c    $        CCURVE(IFACE,IE).EQ.'e') THEN
c           ifgngm=.TRUE.
c           CALL GENXYZ (XP,YP,ZP,IE,NX,NY,NZ) 
c           GOTO 500
c         ENDIF
c 400  CONTINUE
c 500  CONTINUE
c
c        Use genxyz for any element not having 'C' boundaries.
c
c        IF (ifgngm) THEN
c           DO 700 IE=1,NEL
c              IEF=NX*NY*NZ*(IE-1)+1
c              DO 600 ISID=1,8
c                 IF (CCURVE(ISID,IE).EQ.'C') GOTO 700
c 600          CONTINUE
c           CALL GENXYZ (XP,YP,ZP,IE,NX,NY,NZ)
c 700       CONTINUE
c        ENDIF
c     ENDIF
C
      return
      END
c-----------------------------------------------------------------------
      subroutine vel
C
C     Subroutine that draws velocity arrow plots
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER KEY,STRING*5,ELE(4),CVEL*20
      PARAMETER (NXM2=NXM*NYM)
      COMMON /SCRTCH/ TPLOT(NXM2),XPLOT(NXM2),YPLOT(NXM2)
     $               ,ZPLOT(NXM2)
     $               ,XVEC(NXM2) ,YVEC(NXM2) ,ZVEC(NXM2)
      LOGICAL IFCRV(4),IFNX
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
      NXYZ=NX*NY*NZ
C
      IF( OCODE(3).NE.'UU'.OR.OCODE(4).NE.'VV') THEN
c              CALL PRS('No velocity data available.$')
C              return   !!?? SHORT-CURCUIT
      ENDIF
      CALL COLOR(1)
C     WHITE
      VELSCL = 10.* UVMAX/XFAC
C     BUG: XFAC,YFAC MAY BE DIFFERENT
      IF(VELSCL.EQ.0.0) THEN
              CALL PRS('***** Zero Velocity Field *****$')
              return
      ENDIF
      VELINV = 1.0/VELSCL
      UARROW = UVMAX/VELSCL
      CALL ARROW(XPHY(1.01),YPHY(0.80),UARROW,0.0,SARROW)
c
      if (ifvecout.and..not.ifauto) open(unit=47,file='vector.out')
c
      IF (IF3D) THEN
C
C        Sort the selected list of planes according to visibility
         CALL SORTL
C
C        Plot the selected planes
         DO 1100 IP=1,NLSTP
C
            LISTA=ABS(LISW(IP))
            im   =    lmir(ip)
            CALL DECOD(IPLANE,IPLN,IEL,IDUM,LISTA,NX,6,NELM)
            I1=1
            I2=NX
            J1=1
            J2=NY
            K1=1
            K2=NZ
            IF(DARROW.EQ.'HIGH')THEN
C              Plot them all
               I3=1
               J3=1
               K3=1
               NXSKIP=NX 
            ELSE IF(DARROW.EQ.'MEDIUM')THEN
C              Plot only odd ones
               I3=2
               J3=2
               K3=2
               NXSKIP=(NX+1)/2
            ELSE 
C              Plot only center ones
               I3=(NX+1)/2
               J3=(NX+1)/2
               K3=(NX+1)/2
               NXSKIP=2+MOD(NX,2)
            ENDIF
            IF (IPLN.LE.2) THEN
               I1=IPLANE
               I2=IPLANE
               I3=1
            ELSEIF (IPLN.LE.4) THEN
               J1=IPLANE
               J2=IPLANE
               J3=1
            ELSE
               K1=IPLANE
               K2=IPLANE
               K3=1
            ENDIF
            II=0
c           write(6,*) iel  ,' k:',k1,k2,k3
c           write(6,*) lista,' j:',j1,j2,j3
c           write(6,*) ipln,' i:',i1,i2,i3
            DO 1000 K=K1,K2,K3
            DO 1000 J=J1,J2,J3
            DO 1000 I=I1,I2,I3
               IPOINT=IND(I,J,K,IEL)
               II=II+1
               XPLOT(II)=XP(IPOINT)
               YPLOT(II)=YP(IPOINT)
               ZPLOT(II)=ZP(IPOINT)
C              Velocity Arrows
               XVEC(II) = WKV1(IPOINT)*VELINV
               YVEC(II) = WKV2(IPOINT)*VELINV
               ZVEC(II) = WKV3(IPOINT)*VELINV
               if (ifvecout) write(47,47) xplot(ii),yplot(ii),zplot(ii)
     $                          ,wkv1(ipoint),wkv2(ipoint),wkv3(ipoint)
   47          format(1p6e13.5)
 1000       CONTINUE
            fii = ii
            xskip = sqrt(fii)
            ii    = nxskip
            nxskip= xskip
C
C           We had to go to this format because of the 
C           projection required onto the planes.
            CALL AARROW(XPLOT,YPLOT,ZPLOT,XVEC,YVEC,ZVEC,NXSKIP)
C
C           Put border around element?
            IFCRV(1)=.TRUE.
            IFCRV(2)=.TRUE.
            IFCRV(3)=.TRUE.
            IFCRV(4)=.TRUE.
            IFNX=.TRUE.

c           Border not well defined for lower densities...
            IF (IFCFBD.and.darrow.eq.'HIGH') THEN
               IEOFF=NXYZ*(IEL-1)+1
               IFNX=.TRUE.
               CALL BORDER(XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,NX,IFNX)
            endif
 1100    CONTINUE
      ELSE
C        2-D
         DO 2000 IEL=1,NEL
            IF (IFSELE.AND..NOT.IFPLOT(IEL)) GOTO 2000
C
            DO 2001 K=1,NZ
            DO 2001 I=1,NX
            DO 2001 J=1,NY
               IPOINT=IND(I,J,K,IEL)
               IF(DARROW.EQ.'HIGH')THEN
C                 Plot them all
               ELSE IF(DARROW.EQ.'MEDIUM')THEN
C                 Plot only odd ones
                  IF(MOD(I+2,2).EQ.0) GO TO 2002
                  IF(MOD(J+2,2).EQ.0) GO TO 2002
                  IF(MOD(K+2,2).EQ.0) GO TO 2002
               ELSE IF(DARROW.EQ.'LOW')THEN
C                 Plot only center ones
                  IF(I.NE.(1+NX)/2 ) GO TO 2002
                  IF(J.NE.(1+NY)/2 ) GO TO 2002
                  IF(K.NE.(1+NZ)/2 ) GO TO 2002
               ENDIF
               UARROW =WKV1(IPOINT)*VELINV
               VARROW =WKV2(IPOINT)*VELINV
               WARROW =WKV3(IPOINT)*VELINV
               CALL ARROW3(XP(IPOINT),YP(IPOINT),ZP(IPOINT)
     $             ,UARROW,VARROW,WARROW,PLANE,SARROW)
               if (ifvecout) write(47,47) xp(ii),yp(ii)
     $                      ,wkv1(ipoint),wkv2(ipoint)
 2002       CONTINUE
 2001       CONTINUE
C
C           Put border around element?
            IF (IFCFBD) THEN
               CALL LFALSE(IFCRV,4)
               IF (CCURVE(1,IEL).NE.' ') IFCRV(1)=.TRUE.
               IF (CCURVE(2,IEL).NE.' ') IFCRV(2)=.TRUE.
               IF (CCURVE(3,IEL).NE.' ') IFCRV(3)=.TRUE.
               IF (CCURVE(4,IEL).NE.' ') IFCRV(4)=.TRUE.
               IEOFF=NXYZ*(IEL-1)+1
               IFNX=.TRUE.
               CALL BORDER(XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,NX,IFNX)
            ENDIF
 2000     CONTINUE
      ENDIF
      if (ifvecout.and..not.ifauto) close(unit=47)
C
      return
      END
c-----------------------------------------------------------------------
      subroutine DSPOBJ
C
C        Rapid redraw of saved move and draw commands.
C
         OPEN (unit=77,file='t.q',status='old',err=77)
C
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            call color(9)
            CALL MOVEC(X-.15,Y)
            CALL DRAWC(X+.15,Y)
            CALL MOVEC(X,Y-.15)
            CALL DRAWC(X,Y+.15)
C
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            CALL MOVEC(X-.15,Y)
            CALL DRAWC(X+.15,Y)
            CALL MOVEC(X,Y-.15)
            CALL DRAWC(X,Y+.15)
C
         DO 100 I=1,10000
            read (77,*,end=101) s,x3,y3,z3
            X  = XISO(X3,Y3,Z3)
            Y  = YISO(X3,Y3,Z3)
            ic = int(12.0*s)
            ic = mod(ic,12)+2
            if (s.eq.0) then
               CALL MOVEC(X,Y)
            else
               CALL COLOR(IC)
               CALL DRAWC(X,Y)
            endif
  100    CONTINUE
  101    CONTINUE
         close (unit=77)
   77    CONTINUE
C
      return
      END
c-----------------------------------------------------------------------
      subroutine FINDNAM(CHOICES,NCC)
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
C
      CHARACTER*26 CHOICES(1)
C
      nc=1
      CHOICES(1)=NAMELS(1)
C
      DO 200 I=2,NLSTP
         DO 100 k=1,nc
            if(NAMELS(i).eq.choices(k)) goto 200
  100    continue       
         nc=nc+1
         choices(nc)=namels(i)
  200 continue       
C
      NCC=NCC+nc
      return
      END
c-----------------------------------------------------------------------
      subroutine relite
      INCLUDE 'basics.inc'
      do 100 ihi=1,nhi
         ie = ieh(ihi)
         CALL HILIGHT(IE,9)
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine initstate
c
c     See if there is a state array...
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
      logical ifdrm
c
      ifstate=.false.
      open(unit=23,file='.neklast',status='old',err=999)
      ifstate=.true.
c
      nstate = 0
      do is=1,mxstat
         read (23,4,end=99)  ifdrm
         read (23,*,end=99)  nr
         read (23,*,end=99)  (rstat(ir,is),ir=1,nr)
         read (23,*,end=99)  nc
         read (23,5,end=99) (cstat(ic,is),ic=1,nc)
         read (23,*,end=99)  nl
         read (23,4,end=99)  (lstat(il,is),il=1,nl)
         read (23,*,end=99)  ni
         read (23,*,end=99)  (istat(ii,is),ii=1,ni)
         nstate = nstate+1
      enddo
   99 continue

      close(unit=23)
    4 format(20l4)
    5 format(4a20)
c
      call setstate(nstate,ifdrm,ir,ic,il,ii,'strt')

      write(6,*) 'hey!!',lstat(10,1)
      call outstate('.nekbkup')
      write(6,*) 'hey!!',lstat(10,1)
c
  999 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine dmpstate
c
c     dump out last state array
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
      logical ifdrm
      character*8 name
c
      name='.neklast'
      if (ifstate) then
         call prs('save current state? (y,n)$')
         call res (ans,1)
         if (ans.eq.'n') name='.nekbkup'
      endif
      call outstate(name)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outstate(name)
c
c     See if there is a state array...
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
      logical ifdrm
      character*8 name
c

      nstate=nstate+1
      call savstate(nstate,nstate,nr,nc,nl,ni,'quit')
      
c
      write(6,*) 'save to ',name,nr,nc,nl,ni
      ifdrm=.true.
c
      open(unit=23,file=name)
      do is=1,nstate
         write (23,4)  ifdrm
         write (23,*)  nr
         write (23,*)  (rstat(ir,is),ir=1,nr)
         write (23,*)  nc
         write (23,20) (cstat(ic,is),ic=1,nc)
         write (23,*)  nl
         write (23,4)  (lstat(il,is),il=1,nl)
         write (23,*)  ni
         write (23,*)  (istat(ii,is),ii=1,ni)
      enddo
      close(unit=23)
c
    4 format(20l4)
   20 format(4a20)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine INIT0
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,inewt
      REAL PN(NXM)
      CHARACTER*12 FSESSION
      CHARACTER*1 SRFQAL(20)
      CHARACTER*1 tfile(30)
      CHARACTER*30 tfile1
      CHARACTER*20 s20
      CHARACTER*3 chtmp3
      EQUIVALENCE (tfile1,tfile)
      LOGICAL IFFMAT
C
c     ifgngm=.FALSE.
      ifgngm=.true.
c
      return
      end
c-----------------------------------------------------------------------
      subroutine post_reset_window
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      call devini('GENERIC')
      CALL CLEAR
      CALL DRMESH
c
      return
      end
c-----------------------------------------------------------------------
      subroutine setloc_bdry(if_add_loc)
c
c     Set location based on boundary conditions
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,inewt
      INTEGER FCORNS (4,6)
      LOGICAL IFTMP,if_add_loc
C     surface qualifiers
      CHARACTER*1 SRFQAL(20)
      DATA FCORNS / 1,2,6,5,
     $              2,3,7,6,
     $              3,4,8,7,
     $              4,1,5,8,
     $              1,2,3,4,
     $              8,7,6,5 /
      logical ifnoncon
      save    ifnoncon
c
      call noncon_chk(ifnoncon)
      write(6,*) 'this is ifnoncon',ifnoncon
c
      IOLD=ILEVEL
C
                                         nchoic=1
      ITEM(nchoic)='MAIN MENU'
c                                        nchoic=nchoic+1
c     ITEM(nchoic)='UP MENU'
                                         nchoic=nchoic+1
      ITEM(nchoic)='ALL BOUNDARIES'
                                         nchoic=nchoic+1
      ITEM(nchoic)='PERIODIC'
                                         nchoic=nchoic+1
      ITEM(nchoic)='OUTFLOW'
c
      if (ifnoncon) then
                                         nchoic=nchoic+1
         ITEM(nchoic)='J  '
                                         nchoic=nchoic+1
         ITEM(nchoic)='SP '
c
      endif
c
      if (ifflow) then
                                         nchoic=nchoic+1
         ITEM(nchoic)='WALLS'
                                         nchoic=nchoic+1
         ITEM(nchoic)='SYMMETRY'
                                         nchoic=nchoic+1
         ITEM(nchoic)='VELOCITY (any)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='VELOCITY (func)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='VELOCITY (const)'
      endif
c
      if (ifheat) then
                                         nchoic=nchoic+1
         ITEM(nchoic)='INSULATED'
                                         nchoic=nchoic+1
         ITEM(nchoic)='TEMPERATURE (any)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='TEMPERATURE (func)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='TEMPERATURE (const)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='FLUX (any)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='FLUX (func)'
                                         nchoic=nchoic+1
         ITEM(nchoic)='FLUX (const)'
c
      endif
c
                                          nchoic=nchoic+1
      ITEM(nchoic)='FACE (number)'
c
c
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'BDRY LOCATION')
c
      if (choice.eq.'MAIN MENU') return
      if (choice.eq.'UP MENU') return
c
      if (.not.if_add_loc) nlstp = 0
      if (choice.eq.'ALL BOUNDARIES') then
         NQUAL=-2
         SRFQAL(1)='E'
         SRFQAL(2)='P'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'PERIODIC') then
         NQUAL=1
         SRFQAL(1)='P'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'OUTFLOW') then
         NQUAL=1
         SRFQAL(1)='O'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'WALLS') then
         NQUAL=1
         SRFQAL(1)='W'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'J  ') then
         NQUAL=1
         SRFQAL(1)='J'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'SP ') then
         NQUAL=1
         SRFQAL(1)='S'
         IFLD=2
         IF (IFFLOW) IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'SYMMETRY') then
         NQUAL=1
         SRFQAL(1)='S'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'VELOCITY (any)') then
         NQUAL=2
         SRFQAL(1)='V'
         SRFQAL(2)='v'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'VELOCITY (func)') then
         NQUAL=1
         SRFQAL(1)='v'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'VELOCITY (const)') then
         NQUAL=1
         SRFQAL(1)='V'
         IFLD=1
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'FACE (number)') then
         call prs('Enter face number [1:6]:$')
         call res(srfqal(1),1)
         NQUAL=1
         CALL SETSRFP(SRFQAL,NQUAL,1)
      elseif (choice.eq.'INSULATED') then
         NQUAL=1
         SRFQAL(1)='I'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'TEMPERATURE (any)') then
         NQUAL=2
         SRFQAL(1)='t'
         SRFQAL(2)='T'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'TEMPERATURE (func)') then
         NQUAL=1
         SRFQAL(1)='t'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'TEMPERATURE (const)') then
         NQUAL=1
         SRFQAL(1)='T'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'FLUX (any)') then
         NQUAL=2
         SRFQAL(1)='f'
         SRFQAL(2)='F'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'FLUX (func)') then
         NQUAL=1
         SRFQAL(1)='f'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      elseif (choice.eq.'FLUX (const)') then
         NQUAL=1
         SRFQAL(1)='F'
         IFLD=2
         CALL SETSRFP(SRFQAL,NQUAL,IFLD)
      endif
      call pris(nlstp,' surfaces found.$')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine noncon_chk(ifnc)
c
c     See if noncon is true
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
c
      logical ifnc
      logical ifnco
      save    ifnco
c
      integer icalld
      save    icalld
      data    icalld / 0 /
c
c
      if (icalld.gt.0) then
         ifnc = ifnco
         return
      endif
c
      icalld = 1
c
c
      jfld = 1
      if (ifheat) jfld=2
      ifnc  = .false.
c
      do je=1,nel
      do jf=1,2*ndim
         if (CBC(jf,je,jfld).eq.'J  '.or.CBC(jf,je,jfld).eq.'SP ') then
            ifnc  = .true.
            ifnco = .true.
            return
         endif
      enddo
      enddo
      ifnco = ifnc
c
      return
      end
c-----------------------------------------------------------------------
      subroutine TEMVEL
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER LABEL1*10,LABEL2*10,LABEL3*6
      LOGICAL IFCRV(4)
      common/Ttscal/ TX(16),CONTRS,CNTRLV,TMPMIN,TMPMAX
      PARAMETER (NXM2=NXM*NYM)
      COMMON /SCRTCH/ TPLOT(NXM2),XPLOT(NXM2),YPLOT(NXM2)
     $               ,ZPLOT(NXM2)
     $               ,XVEC(NXM2) ,YVEC(NXM2) ,ZVEC(NXM2)
      COMMON /ccdiag/ iee
      common /surfavg/ xsavg,ysavg,zsavg,wsavg
      CHARACTER C*20
      SAVE IFIRST
      DATA IFIRST /0/
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
C     Draw colorbar
C
      if (jhard.eq.3)        ifhard=.false.
      if (.not.ifportrait)   CALL COLRBAR
      if (jhard.eq.3)        ifhard=.true.
C
      CALL PENW(1)
C
      CALL COLOR(1) ! WHITE
      VELSCL = 10.* UVMAX/XFAC
C     BUG: XFAC,YFAC MAY BE DIFFERENT
      IF(VELSCL.EQ.0.0) THEN
              CALL PRS('***** Zero Velocity Field *****$')
              return
      ENDIF
      VELINV = 1.0/VELSCL
      UARROW = UVMAX/VELSCL
      CALL ARROW(XPHY(1.01),YPHY(0.80),UARROW,0.0,SARROW)
      if (ifvecout.and..not.ifauto) open(unit=47,file='vector.out')
C
C     Plot element by element
      IF (.NOT.IF3D) THEN
         DO 1000 IEL=1,NEL
C
C           Check for selection by element in 2D - pff 9 May 1988 17:47:22
C
            IF (IFSELE.AND..NOT.IFPLOT(IEL)) GOTO 1000
C
C           New plotting algorithms for color fill and fishnet - pff 7-1-90
C           (3D will come later - 10-10-90)
C
            NXYZ=NX*NY*NZ
            IEOFF=NX*NY*NZ*(IEL-1)
            WSCALE=1.0
            IF (WKMAX.NE.WKMIN)            WSCALE=1.0/(WKMAX-WKMIN)
            IF(IFUSCL.AND.USRMIN.NE.USRMAX)WSCALE=1.0/(USRMAX-USRMIN)
            CURMIN=WKMIN
            IF(IFUSCL) CURMIN=USRMIN
C           Normalize temperatures
            DO 300 I=1,NXYZ
               I1=IEOFF+I
               TPLOT(I)=WSCALE*(WORK(I1)-CURMIN)
c              IF (IFUSCL) THEN
c                 TPLOT(I)=MIN(TPLOT(I),1.0)
c                 TPLOT(I)=MAX(TPLOT(I),0.0)
c              ENDIF
  300       CONTINUE
C
C           New routine
C
            IEOFF=IEOFF+1
            CALL LFALSE(IFCRV,4)
            IF (CCURVE(1,IEL).NE.' ') IFCRV(1)=.TRUE.
            IF (CCURVE(2,IEL).NE.' ') IFCRV(2)=.TRUE.
            IF (CCURVE(3,IEL).NE.' ') IFCRV(3)=.TRUE.
            IF (CCURVE(4,IEL).NE.' ') IFCRV(4)=.TRUE.
            IF (SCALPT.EQ.'FISHNET GRID') THEN
               INRM=1
               CALL HEATF
     $         (TPLOT,XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,INRM)
            ELSE
               KTR=1
               IF (SCALPT.EQ.'CONTOUR LINES') KTR=0
                iee=iel
               CALL HEATC
     $         (TPLOT,XP(IEOFF),YP(IEOFF),ZP(IEOFF),IFCRV,KTR)
            ENDIF
 1000    CONTINUE
C
      ELSE
C     3-D
C        Sort the selected list of planes according to visibility
C
         CALL SORTL
C
         WORKMIN= 9.999E20
         WORKMAX=-9.999E20
         ilist = 0
         xsavg=0.
         ysavg=0.
         zsavg=0.
         wsavg=0.
         nsavg=0
         DO Imir=1,nmirror
         DO Ipla=1,NLSTP
            ilist = ilist+1
C
C           New plotting algorithms for color fill and fishnet - pff 7-1-90
C           (3D will come later - 10-10-90)
C
            lista=abs(lisw(ilist))
            im   =    lmir(ilist)
            CALL DECOD(IPLANE,IPLN,IE,IDUM,LISTA,NX,6,NELM)
c           write(6,*) 'str:',iplane,ipln,ie,lista,lisw(ilist),ilist
            I1=1
            I2=NX
            J1=1
            J2=NY
            K1=1
            K2=NZ
            IF (IPLN.LE.2) THEN
               I1=IPLANE
               I2=IPLANE
            ELSEIF (IPLN.LE.4) THEN
               J1=IPLANE
               J2=IPLANE
            ELSE
               K1=IPLANE
               K2=IPLANE
               ENDIF
            KTR=1
            IF (SCALPT.EQ.'CONTOUR LINES') KTR=0
            IFCRV(1)=.TRUE.
            IFCRV(2)=.TRUE.
            IFCRV(3)=.TRUE.
            IFCRV(4)=.TRUE.
            WSCALE=1.0
            IF (WKMAX.NE.WKMIN)            WSCALE=1.0/(WKMAX-WKMIN)
            IF(IFUSCL.AND.USRMIN.NE.USRMAX)WSCALE=1.0/(USRMAX-USRMIN)
            CURMIN=WKMIN
            IF(IFUSCL) CURMIN=USRMIN
            II=0
            DO 5001 IZ=K1,K2
            DO 5001 IY=J1,J2
            DO 5001 IX=I1,I2
               IPOINT=IND(IX,IY,IZ,IE)
               II=II+1
               XPLOT(II)=XP(IPOINT)
               YPLOT(II)=YP(IPOINT)
               ZPLOT(II)=ZP(IPOINT)
               WORKIP =WORK(IPOINT)
               TPLOT(II)=WSCALE*(WORKIP-CURMIN)
c
               xsavg=xsavg+xplot(ii)
               ysavg=ysavg+yplot(ii)
               zsavg=zsavg+zplot(ii)
               wsavg=wsavg+workip
               nsavg=nsavg+1
C
C              Find observed range:
C
               IF (WORKIP.LT.WORKMIN) THEN
                  WORKMIN=WORKIP
                  XWMIN=XP(IPOINT)
                  YWMIN=YP(IPOINT)
                  ZWMIN=ZP(IPOINT)
                  IEMIN=IE
               ENDIF
               IF (WORKIP.GT.WORKMAX) THEN
                  WORKMAX=WORKIP
                  XWMAX=XP(IPOINT)
                  YWMAX=YP(IPOINT)
                  ZWMAX=ZP(IPOINT)
                  IEMAX=IE
               ENDIF
C
               IF (IFUSCL) THEN
                  TPLOT(II)=MIN(TPLOT(II),1.0)
                  TPLOT(II)=MAX(TPLOT(II),0.0)
               ENDIF
 5001       CONTINUE
c
c           Check for mirroring
c
            if (im.gt.1) call vmirror(tplot,xplot,yplot,zplot,nx,im)
c
            IF (SCALPT.EQ.'FISHNET GRID') THEN
               INRM=1
c
c              lplane = lista
c              call signpl(lplane)
c              INRM=1
c              IF (lplane.LT.0) INRM=-1
c
               IF (lisw(i).LT.0) INRM=-1
               CALL HEATF(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,INRM)
            ELSE
               CALL HEATC(TPLOT,XPLOT,YPLOT,ZPLOT,IFCRV,KTR)
            ENDIF
c
c           New vector stuff  12/9/03, pff
c 
            I1=1
            I2=NX
            J1=1
            J2=NY
            K1=1
            K2=NZ
            IF(DARROW.EQ.'HIGH')THEN
C              Plot them all
               I3=1
               J3=1
               K3=1
               NXSKIP=NX
            ELSE IF(DARROW.EQ.'MEDIUM')THEN
C              Plot only odd ones
               I3=2
               J3=2
               K3=2
               NXSKIP=(NX+1)/2
            ELSE
C              Plot only center ones
               I3=(NX+1)/2
               J3=(NX+1)/2
               K3=(NX+1)/2
               NXSKIP=2+MOD(NX,2)
            ENDIF
            IF (IPLN.LE.2) THEN
               I1=IPLANE
               I2=IPLANE
               I3=1
            ELSEIF (IPLN.LE.4) THEN
               J1=IPLANE
               J2=IPLANE
               J3=1
            ELSE
               K1=IPLANE
               K2=IPLANE
               K3=1
            ENDIF
            II=0
            DO 6000 K=K1,K2,K3
            DO 6000 J=J1,J2,J3
            DO 6000 I=I1,I2,I3
               IPOINT=IND(I,J,K,ie)
               II=II+1
               XPLOT(II)=XP(IPOINT)
               YPLOT(II)=YP(IPOINT)
               ZPLOT(II)=ZP(IPOINT)
C              Velocity Arrows
               XVEC(II) = WKV1(IPOINT)*VELINV
               YVEC(II) = WKV2(IPOINT)*VELINV
               ZVEC(II) = WKV3(IPOINT)*VELINV
               if (ifvecout) write(47,47) xplot(ii),yplot(ii),zplot(ii)
     $                          ,wkv1(ipoint),wkv2(ipoint),wkv3(ipoint)
   47          format(1p6e13.5)
 6000       CONTINUE
            fii = ii
            xskip = sqrt(fii)
            nxskip= xskip
C           We had to go to this format because of the
C           projection required onto the planes.
            CALL COLOR(1) ! WHITE
            CALL AARROW(XPLOT,YPLOT,ZPLOT,XVEC,YVEC,ZVEC,NXSKIP)
C
         enddo
         enddo
         if (nsavg.gt.0) then
            xsavg=xsavg/nsavg
            ysavg=ysavg/nsavg
            zsavg=zsavg/nsavg
            wsavg=wsavg/nsavg
         endif
C
C        Range information:
C
            WRITE(S,5004) wsavg,xsavg,ysavg,zsavg,nsavg
            CALL PRS(S//'$')
         IF(PLFORM.ne.'VOLUME')THEN
            WRITE(S,5002) WORKMIN,XWMIN,YWMIN,ZWMIN,IEMIN
            CALL PRS(S//'$')
            WRITE(S,5003) WORKMAX,XWMAX,YWMAX,ZWMAX,IEMAX
            CALL PRS(S//'$')
 5002       FORMAT(' Local Min:',E13.5,5x,'X:',3E12.3,I7,'$')
 5003       FORMAT(' Local Max:',E13.5,5x,'X:',3E12.3,I7,'$')
 5004       FORMAT(' Local Avg:',E13.5,5x,'X:',3E12.3,I7,'$')
         ENDIF
C
      ENDIF
C
      return
      end
c-----------------------------------------------------------------------
      subroutine exitt
      write(6,*) 'stopping in exitt'
      stop
      end
c-----------------------------------------------------------------------
c     subroutine exit
c     stop
c     end
c-----------------------------------------------------------------------
c     subroutine systemx
c     return
c     end
c-----------------------------------------------------------------------
c     subroutine sleep
c     return
c     end
c-----------------------------------------------------------------------
c     subroutine rename
c     return
c     end
c-----------------------------------------------------------------------
      subroutine get_subset_list
      include 'basics.inc'
      include 'basicsp.inc'
      integer e,e0,e1,es

      call izero(isubset,mxel)

      call prsi ('Input start,stop,skip for elements: (nel:)$',nel)
      call reiii(e0,e1,es)
      nelq = 0
      do e=e0,e1,es
         nelq = nelq+1
         isubset(e) = nelq
      enddo
      nel = nelq

      do e=1,mxel
         if (isubset(e).eq.0) isubset(e)=nelm
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine copy4b(a,b,n,iel,nel)
      real   a(nel,1)
      real*4 b(1)
      do i = 1, n
         a(iel,i) = b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine copy8b(a,b,n,iel,nel)
      real   a(nel,1)
      real*8 b(1)
      do i = 1, n
         a(iel,i) = b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine copyi(a,b,n)
      integer a(1)
      real*8  b(1)
      do i = 1, n
         a(i) = b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine quickfill(pp)
      include 'basics.inc'
      real   pp(nx,ny,nz)
      do k = 1, nz
      do j = 1, ny
      do i = 1, nx
         pp(i,j,k) = i
      enddo
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
