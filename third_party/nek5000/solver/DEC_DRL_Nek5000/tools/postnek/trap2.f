c-----------------------------------------------------------------------
      subroutine tec_cell_gen(kcell,ncells,icell,mx,my,mz,nel)
c
      integer icell(mx,my,mz,nel)
      integer jcell(8)
      integer kcell(8,1)

      parameter (lvtk=2600000)
c
c
c     count number of valid cells (with non-zero vertex pointers)
c     and load into kcell
c     
      ncells = 0
      do ie=1,nel
      do k=1,mz-1
      do j=1,my-1
      do i=1,mx-1
         l = 0
         jcell_min = 1
         do k0 = 0,1
         do j0 = 0,1
         do i0 = 0,1
            kk = k+k0
            jj = j+j0
            ii = i+i0
            l  = l+1
            jcell(l)  = icell(ii,jj,kk,ie)
            jcell_min = min(icell(ii,jj,kk,ie),jcell_min)
c           write(6,111) jcell_min,ie,ii,jj,kk,jcell(l),l
  111       format(7i6,' JCELL')
         enddo
         enddo
         enddo
         if (jcell_min.gt.0) then   ! cell=0 --> inactive cell
c           Save cell data for checking in postnek
            if (ncells.lt.lvtk) then
               ncells = ncells+1
c
c              tec nodes are numbered from 1 to n  (pff 5/9/07)
c              tec takes the EB format
c
               kcell(1,ncells) = jcell(1)
               kcell(2,ncells) = jcell(2)
               kcell(3,ncells) = jcell(4)
               kcell(4,ncells) = jcell(3)
               kcell(5,ncells) = jcell(5)
               kcell(6,ncells) = jcell(6)
               kcell(7,ncells) = jcell(8)
               kcell(8,ncells) = jcell(7)
            else
               call prsii('kcell too small (trap.f)$',ncells,lvtk)
            endif
         endif
      enddo
      enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mltiplt_tec
c
c     For creating multiple tecplot files   pff 10/12/07;
c
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      INCLUDE 'state.inc'
      logical ifdrm,iftmp
c
      character*80  mv_command,rm_command
      character*112 xx_command
      character*56  xx_commnd1(2)
      equivalence  (xx_command,xx_commnd1)
c
      character quanty1*20,compon1*1,deriv1*5
C
      IFTMP=IFCLRB
      IFCLRB=.FALSE.
      nframe = 1
c
      call prs  ('.fld start, stop, and stride numbers?:$')
      call reiii(iframe0,iframe1,istrid)
c
      call prs  ('Input resolution:$')
      call rei  (nppcell)
c
      kframe = 0
      do iframe = iframe0,iframe1,istride
c
         DERIV = 'NO'
         QUANTY= 'VORTEX'

c        Get .fld_iframe ....

         ndumps = iframe
         call getfld(ndumps,ierr,.true.,.true.)

         call tec_out_xyz_sc(nppcell,vortex,p,iframe)

      enddo
      CALL PRS('WAKE UP!!  The file creation is over!!$')
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
      IFHARD=IFTMP
C
      return
      end
c-----------------------------------------------------------------------
      subroutine tec_out_xyz_sc(mx,v1,v2,iframe)
c
c     Generate an FEM input file based on Nekton .rea file
c
C
      include 'basics.inc'
      include 'basicsp.inc'

      parameter (lvtk=2600000)
      common /vtkwki/ loc(lvtk),ind(lvtk)
      common /vtkwka/ w1 (lvtk),wk (lvtk)
      common /vtkwsi/ mx_save,nvtk,nelclip,ncells
      common /vtkwkr/ xcmn,xcmx
      common /vtkwkl/ ifxin(nelm)
      logical         ifxin

      common /bigvtki/ jglob(lvtk),kcell(8,lvtk)
      common /bigvtkr/ vtkxyz(lvtk,3)
      common /bigvtkl/ ifclip,ifillt,iffmt_vtk,ifdouble
      logical          ifclip,ifillt,iffmt_vtk,ifdouble

      real xm(1),ym(1),zm(1)
      equivalence (xm,vtkxyz(1,1))
      equivalence (ym,vtkxyz(1,2))
      equivalence (zm,vtkxyz(1,3))

      real v1(1),v2(1)  ! input scalar fields

      integer ic(8)

      logical ifnew
      save    ifnew

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      include 'tecio.inc'

      INTEGER*4   Debug,I,J,K,L,III,NPts,NElm,DIsDouble,VIsDouble
      INTEGER*4   IMax,JMax,KMax,NM(4,12)   
      REAL*8      XP, YP, ZP, FH, LineSpacing, PatternLength
      REAL*8      BoxMargin, BoxLineThickness, TextAngle
      INTEGER*4   AttachToZone, Zone, Scope, PositionCoordSys
      INTEGER*4   Clipping
      INTEGER*4   FontType, HeightUnits, Anchor, BoxType
      INTEGER*4   IsFilled, GeomType, LinePattern, NumEllipsePts
      INTEGER*4   BoxColor, BoxFillColor, TextColor, Color, FillColor
      INTEGER*4   ArrowheadStyle, ArrowheadAttachment, NumSegments
      INTEGER*4   NumSegPts(1)
      REAL*8      LineThickness, ArrowheadSize, ArrowheadAngle
      REAL*4      XGeomData(1), YGeomData(1), ZGeomData(1)
      CHARACTER*1 NULCHAR
      INTEGER*4   Zero
      POINTER     (NullPtr,Null)
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      Debug     = 2
      VIsDouble = 0
      DIsDouble = 0
      NULCHAR   = CHAR(0)
      Zero      = 0
      NullPtr   = 0
C
C Open field.plt and write the header information.
C 
      I = TECINI100('DATASET WITH 1 ORDERED ZONE, 1 QUAD ZONE'//NULCHAR,
     &              'X Y P'//NULCHAR,
     &              'field.plt'//NULCHAR,
     &              '.'//NULCHAR,
     &               Debug,
     &               VIsDouble)
c
      ifnew = .false.
      if (mx.ne.mx_save) ifnew = .true.
      mx_save = mx
      my = mx
      mz = mx
      if (.not.if3d) mz=1

      if (ifnew) then           !        Clip output?
         ifclip = .true.
         call prs('Clip volume in x? (enter xmn,xmx or  0,0 for no)$')
         call rerr(xcmn,xcmx)
         if (xcmn.eq.0.  .and. xcmx.eq.0.) ifclip = .false.
         nx3     = nx*ny*nz
         nelclip = nel
         call find_xin(nelclip,ifclip,ifxin,xp,nx3,nel,xcmn,xcmx)
      endif

      nton=nx*ny*nz*nel
      ntot=mx*my*mz*nelclip
      nvtk=ntot

      write(6,*) 'This is iffmt_vtk:',iffmt_vtk
      if (ifclip) call prsrr('Clipping in effect!$',xcmn,xcmx)
      call prsii('nelclip $',nelclip,nel)
      call prsii('ntot    $',nvtk   ,nton)


c     Establish local to global node numbering for mx points
      call locglobm(jglob,nglb,mx,nel)
      write(6,*) 'this is nglb',nglb,ntot,mx,nel

c     Now, use jglob to sort data
      call icopy (loc ,jglob   ,ntot)
      call isort (loc ,ind     ,ntot)
c
c     Compress and output coordinate data
      call prsii('call mapreg3d mx,nx$',mx,nx)
      call mapreg3d(xm,mx,lvtk,xp,nx,nel,ifclip,ifxin,wk)
      call mapreg3d(ym,mx,lvtk,yp,nx,nel,ifclip,ifxin,wk)
      if (if3d) 
     $call mapreg3d(zm,mx,lvtk,zp,nx,nel,ifclip,ifxin,wk)

      call compress (ngpt,xm,loc,ind,ntot,w1)
      call compress (ngpt,ym,loc,ind,ntot,w1)
      call compress (ngpt,zm,loc,ind,ntot,w1)

C  
C  Generate the cell data and count number of cells
C  

      call tec_cell_gen(kcell,ncells,jglob,mx,my,mz,nel)
C  
C  Write the zone header information for the finite-element zone.
C  
      NPts      = 20 
      NElm      = 12 
      KMax      = 1  
      I = TECZNE100('Finite Zone'//NULCHAR,
     &              3,  ! FEQUADRILATERAL
     &              ngpt,
     &              ncells,
     &              KMax,
     &              0,
     &              0,
     &              0,
     &              1,
     &              0,
     &              0,
     &              Null,
     &              Null,
     &              0)
C  
C  Write out the field data for the finite-element zone.
C  
      DIsDouble = 0
      i = tecdat100(ngpt,xm,DIsDouble)
      i = tecdat100(ngpt,ym,DIsDouble)
      if (if3d) i = tecdat100(ngpt,zm,DIsDouble)


c  Generate and write out the connectivity list.
c  Note: The kcell array references cells starting with 1

      i = tecnod100(kcell)
c
c     Output cell data
c
      if (ifnew) call vtk_out_hex (jglob,mx,my,mz,nelclip,ifout)
      if (ifnew) call coef2
uuuu
c
c
c
c     Now, use loc,ind to sort data
c
c     Compress and output scalar work data
c
      m_scalar = 2
      do i=1,m_scalar

c        quick hack:  force quantity

         if (i.eq.1) QUANTY = 'VORTEX'
         if (i.eq.2) QUANTY = 'PRESSURE'
         call setwrk(.false.)

         call mapreg3d (wk,mx,lvtk,work,nx,nel,ifclip,ifxin,w1)
         call compress         (ngpt,wk,loc,ind,ntot,w1)
         call bin_out_sca      (wk,ngpt,ifout,i)
uuuu

      enddo
c
      return
      end
c-----------------------------------------------------------------------
uuuu
C 
C  Complex example FORTRAN program to write a
C  binary data file for Tecplot. This example
C  does the following:
C 
C    1.  Open a data file called "field.plt."
C    2.  Open a data file called "line.plt."
C    3.  Assign values for X, Y and P. These will be used
C        in both the ordered and FE data files.
C    4.  Write out an ordered zone dimensioned 4 x 5 to "field.plt."
C    5.  Assign values for XL and YL arrays.
C    6.  Write out data for line plot to "line.plt."  Make the data
C        use double precision.
C    7.  Write out a finite element zone to "field.plt."
C    8.  Write out a text record to "field.plt."
C    9.  Write out a geometry (circle) record to "field.plt."
C   10.  Close file 1.
C   11.  Close file 2.
C  
      Program ComplexTest

uuu
C  
C  Open line.plt and write the header information.
C  
      VIsDouble = 1
      I = TECINI100('DATASET WITH ONE I-ORDERED ZONE'//NULCHAR,
     &              'X Y'//NULCHAR,
     &              'line.plt'//NULCHAR,
     &             '.'//NULCHAR,
     &               Debug,
     &               VIsDouble)

C  
C  Calculate values for the field variables.
C  
      Do 10 J = 1,5
      Do 10 I = 1,4
          X(I,J) = I
          Y(I,J) = J
          P(I,J) = I*J
   10 Continue

C  
C  Make sure writing to file #1.
C  
      III = 1
      I = TECFIL100(III)

C  
C  Write the zone header information for the ordered zone.
C  
      IMax = 4
      JMax = 5
      KMax = 1
	I = TECZNE100('Ordered Zone'//NULCHAR,
     &               0, ! ZONETYPE
     &               IMax,
     &               JMax,
     &               KMax,
     &               0,
     &               0,
     &               0,
     &               1,     ! ISBLOCK
     &               0,     !  NumFaceConnections
     &               0,     ! FaceNeighborMode
     &               Null,  ! ValueLocation
     &               Null,  ! ShareVarFromZone
     &               0)     ! ShareConnectivityFromZone)

C  
C  Write out the field data for the ordered zone.
C  
      III = IMax*JMax
      I   = TECDAT100(III,X,DIsDouble)
      I   = TECDAT100(III,Y,DIsDouble)
      I   = TECDAT100(III,P,DIsDouble)

C   
C  Calculate values for the I-ordered zone.
C  

      Do 20 I = 1,50
         XL(I) = I
         YL(I) = sin(I/20.0)
   20 Continue
C  
C  Switch to the 'line.plt' file (file number 2)
C  and write out the line plot data.
C  
      III = 2
      I = TECFIL100(III)
C  
C  Write the zone header information for the XY-data.
C  
      IMax = 50
      JMax = 1
      KMax = 1
      I = TECZNE100('XY Line plot'//NULCHAR,
     &              0,
     &              IMax,
     &              JMax,
     &              KMax,
     &              0,
     &              0,
     &              0,
     &              1,
     &              0,
     &              0,
     &              Null,
     &              Null,
     &              0)
C  
C  Write out the line plot.
C  
      DIsDouble = 1
      III = IMax
      I   = TECDAT100(III,XLDummy,DIsDouble)
      I   = TECDAT100(III,YLDummy,DIsDouble)

C  
C  Switch back to the field plot file and write out
C  the finite-element zone.
C  
      III = 1
      I = TECFIL100(III)
C  
C  Write the zone header information for the finite-element zone.
C  
      NPts      = 20 
      NElm      = 12 
      KMax      = 1  
      I = TECZNE100('Finite Zone'//NULCHAR,
     &              3,  ! FEQUADRILATERAL
     &              NPts,
     &              NElm,
     &              KMax,
     &              0,
     &              0,
     &              0,
     &              1,
     &              0,
     &              0,
     &              Null,
     &              Null,
     &              0)
uuuu
C  
C  Write out the field data for the finite-element zone.
C  
      IMax      = 4
      JMax      = 5
      III       = IMax*JMax
      DIsDouble = 0
      I    = TECDAT100(III,X,DIsDouble)
      I    = TECDAT100(III,Y,DIsDouble)
      I    = TECDAT100(III,P,DIsDouble)

C  
C  Calculate and then write out the connectivity list.
C  Note: The NM array references cells starting with
      I = TECNOD100(NM)
C        offset of 1.
C  

      Do 30 I = 1,IMax-1
      Do 30 J = 1,JMax-1
          K = I+(J-1)*(IMax-1)
          L = I+(J-1)*IMax
          NM(1,K) = L
          NM(2,K) = L+1
          NM(3,K) = L+IMax+1
          NM(4,K) = L+IMax
   30 Continue

      I = TECNOD100(NM)

C  
C  Prepare to write out text record. Text is positioned
C  at 50, 50 in frame units and has a height 5 frame units.
C  
      XP               = 50 
      YP               = 50 
      FH               = 5
      Scope            = 1 
      Clipping         = 0
      PositionCoordSys = 1 
      FontType         = 1 
      HeightUnits      = 1 
      AttachToZone     = 0
      Zone             = 0
      BoxType          = 0 
      BoxMargin        = 5.0
      BoxLineThickness = 0.5
      BoxColor         = 3
      BoxFillColor     = 7
      TextAngle        = 0.0
      Anchor           = 0 
      LineSpacing      = 1.5
      TextColor        = 0 
    
      III =  TECTXT100(XP,
     &                 YP,
     &                 0.0d0,
     &                 PositionCoordSys,
     &                 AttachToZone,
     &                 Zone,
     &                 FontType,
     &                 HeightUnits,
     &                 FH,
     &                 BoxType,
     &                 BoxMargin,
     &                 BoxLineThickness,
     &                 BoxColor,
     &                 BoxFillColor,
     &                 TextAngle,
     &                 Anchor,
     &                 LineSpacing,
     &                 TextColor,
     &                 Scope,
     &                 Clipping,
     &                'Hi Mom'//NULCHAR,
     &                ''//NULCHAR)

C  
C  Prepare to write out geometry record (circle). Circle is 
C  positioned at 25, 25 in frame units and has a radius of 30.
C  Circle is drawn using a dashed line pattern.
C  


      XP                  = 25
      YP                  = 25
      ZP                  = 0.0
      IsFilled            = 0
      Color               = 0
      FillColor           = 7
      GeomType            = 2 
      LinePattern         = 1 
      LineThickness       = 0.3
      PatternLength       = 1
      NumEllipsePts       = 72
      ArrowheadStyle      = 0
      ArrowheadAttachment = 0
      ArrowheadSize       = 0.0
      ArrowheadAngle      = 15.0
      NumSegments         = 1
      NumSegPts(1)        = 1
    
      XGeomData(1) = 30
      YGeomData(1) = 0.0
      ZGeomData(1) = 0.0
    
    
      III =  TECGEO100(XP,
     &                 YP,
     &                 ZP,
     &                 PositionCoordSys,
     &                 AttachToZone,
     &                 Zone,
     &                 Color,
     &                 FillColor,
     &                 IsFilled,
     &                 GeomType,
     &                 LinePattern,
     &                 PatternLength,
     &                 LineThickness,
     &                 NumEllipsePts,
     &                 ArrowheadStyle,
     &                 ArrowheadAttachment,
     &                 ArrowheadSize,
     &                 ArrowheadAngle,
     &                 Scope,
     &                 Clipping,
     &                 NumSegments,
     &                 NumSegPts,
     &                 XGeomData,
     &                 YGeomData,
     &                 ZGeomData,
     &                 ''//NULCHAR)
          
C 
C  Close out file 1.
C 
      I = TECEND100() 

C  
C  Close out file 2.
C  
      III = 2
      I = TECFIL100(III)
      I = TECEND100() 
      STOP
      END
uuuu
