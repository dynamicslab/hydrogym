c-----------------------------------------------------------------------
      subroutine setup_topo
C
C     Parallel compatible routine to find 
C     connectivity of element structure.
C
C     On Processor 0:
C
C     .Verify right-handedness of elements.
C     .Verify element-to-element reciprocity of BC's
C     .Verify correlation between E-E BC's and physical coincidence
C     .Set rotations
C     .Determine multiplicity
C     .Set up direct stiffness summation arrays.
C
C     All Processors:
C
C     .Disperse/Receive BC and MULT temporary data read from preprocessor.
C
C
      include 'SIZE'
      include 'TOTAL'
      include 'NONCON'
      include 'SCRCT'
c
      COMMON /SCRUZ/ XM3 (LX3,LY3,LZ3,LELT)
     $ ,             YM3 (LX3,LY3,LZ3,LELT)
     $ ,             ZM3 (LX3,LY3,LZ3,LELT)
C
      common /c_is1/ glo_num(1*lx1*ly1*lz1*lelv)
      integer*8 glo_num
      common /ivrtx/ vertex ((2**ldim)*lelt)
      integer vertex

      if(nio.eq.0) write(6,*) 'setup mesh topology'
C
C     Initialize key arrays for Direct Stiffness SUM.
C
      NXL=3
      NYL=3
      NZL=1+2*(ldim-2)

      call initds
      call dsset (nxl,nyl,nzl)
      call setedge
C
C=================================================
C     Establish (global) domain topology
C=================================================
C
C     .Generate topologically correct mesh data.
C     .Set up element centers, face centers, etc. 
C     .Check  right handedness of elements.
C     .Check  element boundary conditions.
C     .Establish Element-Element rotations
C     .Construct the element to processor map and

      call genxyzl
      call setside
      call verify

      CALL SETCDOF 
      IF (IFAXIS            ) CALL SETRZER
      IF (IFMVBD            ) CALL CBCMESH
      CALL CHKAXCB
C
C========================================================================
C     Set up element-processor mapping and establish global numbering
C========================================================================
C
      mfield=2
      if (ifflow) mfield=1
      if (ifmvbd) mfield=0

      ncrnr = 2**ldim

      if (nelgv.eq.nelgt) then
         call get_vert
         call setupds(gsh_fld(1),lx1,ly1,lz1,nelv,nelgv,vertex,glo_num)
         gsh_fld(2)=gsh_fld(1)

c        call gs_counter(glo_num,gsh_fld(1))

      else

c
c        For conjugate heat transfer, it is assumed that fluid
c        elements are listed both globally and locally with lower
c        element numbers than the solid elements.
c
c        We currently assume that there is at least one fluid elem.
c        per processor.
c

         call get_vert
c        call outmati(vertex,4,nelv,'vrtx V')
         call setupds(gsh_fld(1),lx1,ly1,lz1,nelv,nelgv,vertex,glo_num)

c        call get_vert  (vertex, ncrnr, nelgt, '.mp2')  !  LATER !
c        call outmati(vertex,4,nelt,'vrtx T')
         call setupds(gsh_fld(2),lx1,ly1,lz1,nelt,nelgt,vertex,glo_num)

c
c        Feb 20, 2012:  It appears that we do not need this restriction: (pff)
c
c        check if there is a least one fluid element on each processor
c        do iel = 1,nelt
c           ieg = lglel(iel)
c           if (ieg.le.nelgv) goto 101 
c        enddo
c        if(nio.eq.0) write(6,*) 
c    &     'ERROR: each domain must contain at least one fluid element!'
c        call exitt
c 101   continue

      endif


c     if (ifmvbd) call setup_mesh_dssum ! Set up dssum for mesh

C========================================================================
C     Set up multiplicity and direct stiffness arrays for each IFIELD
C========================================================================

      ntotv = lx1*ly1*lz1*nelv
      ntott = lx1*ly1*lz1*nelt



      if (ifflow) then
         ifield = 1
         call rone    (vmult,ntotv)
         call dssum   (vmult,lx1,ly1,lz1)
         vmltmax=glmax(vmult,ntotv)
         ivmltmax=vmltmax
         if (nio.eq.0) write(6,*) ivmltmax,' max multiplicity'
         call invcol1 (vmult,ntotv)
      endif
      if (ifheat) then
         ifield = 2
         call rone    (tmult,ntott)
         call dssum   (tmult,lx1,ly1,lz1)
         call invcol1 (tmult,ntott)
      endif
      if (.not.ifflow) call copy(vmult,tmult,ntott)
      if (ifmvbd)  call copy (wmult,vmult,ntott)
      do ifield=3,nfield                  ! Additional pass. scalrs.
         if (nelg(ifield).eq.nelgv) then
            gsh_fld(ifield) = gsh_fld(1)
            call copy (tmult(1,1,1,1,ifield-1),vmult,ntotv)
         else
            gsh_fld(ifield) = gsh_fld(2)
            call copy (tmult(1,1,1,1,ifield-1),tmult,ntott)
         endif
      enddo

      ifgsh_fld_same = .true.
      do ifield=2,nfield
         if (gsh_fld(ifield).ne.gsh_fld(1)) then
            ifgsh_fld_same = .false.
            goto 100
         endif
      enddo
 100  continue

      if(nio.eq.0) then
        write(6,*) 'done :: setup mesh topology'
        write(6,*) ' '
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine initds
C
C          -- Direct Stiffness Initialization Routine --
C
C     Set up required data for packing data on faces of spectral cubes.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
C     Nominal ordering for direct stiffness summation of faces
C
      J=0
      DO 5 IDIM=1,ldim
      DO 5 IFACE=1,2
        J=J+1
         NOMLIS(IFACE,IDIM)=J
    5 CONTINUE
C
C     Assign Ed's numbering scheme to PF's scheme.
C
      EFACE(1)=4
      EFACE(2)=2
      EFACE(3)=1
      EFACE(4)=3
      EFACE(5)=5
      EFACE(6)=6
C
C     Assign inverse of Ed's numbering scheme to PF's scheme.
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
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine setedge
C
C     .Initialize EDGE arrays for face and edge specific tasks.
C
C     .NOTE: Sevaral arrays in common are initialized via 
C            BLOCKDATA EDGEC
C
C     Computed arrays: 
C
C     IEDGE  -  Minimal list of wire frame nodes.
C               Used to search for all physical
C               coincidences.
C
C
C     IEDGEF - .Ordered list of wire frame nodes 
C               associated with faces 1 through 6.
C              .Each of 4 sides of square frame stored 
C               individually so that rotations are 
C               readily handled.
C              .Two types of node orderings stored -
C               (0) is clockwise marching
C               (1) is counter-clockwise marching 
C                   for image face.
C                               
C
C     IFACE         - indicates the face number.  Two notations
C                     are currently in use:
C
C                     i) Preprocessor notation:
C                
C                                       +--------+     ^ S
C                                      /        /|     |
C                                     /    3   / |     |
C                               4--> /        /  |     |
C                                   +--------+ 2 +     +----> R
C                                   |        |  /     /
C                                   |    6   | /     /
C                                   |        |/     /
C                                   +--------+     T
C                                       1
C
C                    ii) Symmetric notation:
C                     
C                                       +--------+     ^ S
C                                      /        /|     |
C                                     /    4   / |     |
C                               1--> /        /  |     |
C                                   +--------+ 2 +     +----> R
C                                   |        |  /     /
C                                   |    6   | /     /
C                                   |        |/     /
C                                   +--------+     T
C                                       3
C
C     EFACE(IFACE)  -   Given face number IFACE in symmetric notation, 
C                       returns preprocessor notation face number.
C
C     EFACE1(IFACE) -   Given face number IFACE in preprocessor notation, 
C                       returns symmetric notation face number.
C
C  The following variables all take the symmetric 
C  notation of IFACE as arguments:
C
C     ICFACE(i,IFACE) -   Gives the 4 vertices which reside on face IFACE
C                         as depicted below, e.g. ICFACE(i,2)=2,4,6,8.
C
C                        3+-----+4    ^ Y
C                        /  2  /|     |
C     Edge 1 extends    /     / |     |
C       from vertex   7+-----+8 +2    +----> X
C       1 to 2.        |  4  | /     /
C                      |     |/     /
C                     5+-----+6    Z
C                         3
C
C     IEDGFC(i,IFACE) -   Gives the 4 edges which border the face IFACE
C                         Edge numbering is as follows:
C                            Edge = 1,2,3,4     run in +r direction
C                            Edge = 5,6,7,8     run in +s direction
C                            Edge = 9,10,11,12  run in +t direction
C
C                         Ordering of each edge is such that a monotonically
C                         increasing sequence of vertices is associated with
C                         the start point of a corresponding set of 
C                         monotonically increasing edge numbers, e.g.,
C
C     ICEDG(i,IEDGE)  -   Gives 3 variables for determining the stride along
C                         a given edge, IEDGE;  i=1 gives the starting vertex
C                                               i=2 gives the stopping vertex
C                                               i=3 gives the stride size.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
C
      COMMON /CTMP0/ ITMP(3,3,3)
      INTEGER ORDER
C
      NXL=3
      NYL=3
      NZL=1+2*(ldim-2)
      NXY   =NXL*NYL
      NXYZ  =NXL*NYL*NZL
      NFACES=2*ldim
C
C----------------------------------------------------------------------
C     Set up edge arrays (temporary - required only for defining DS)
C----------------------------------------------------------------------
C
C     Fill corners - 1 through 8.
C
      I3D=1
      IF (ldim.EQ.2) I3D=0
C
      I=0
      DO 10 I3=0,I3D
         IZ=1+(NZL-1)*I3
         DO 10 I2=0,1
            IY=1+(NYL-1)*I2
            DO 10 I1=0,1
               IX=1+(NXL-1)*I1
               I=I+1
               IEDGE(I)=IX+NXL*(IY-1)+NXY*(IZ-1)
   10 CONTINUE
C
C     Fill X-direction edges.
C
      DO 20 I3=0,I3D
         IZ=1+(NZL-1)*I3
         DO 20 I2=0,1
            IY=1+(NYL-1)*I2
            DO 20 IX=2,NXL-1
               I=I+1
               IEDGE(I)=IX+NXL*(IY-1)+NXY*(IZ-1)
   20 CONTINUE
C
C     Fill Y-direction edges.
C
      DO 30 I3=0,I3D
         IZ=1+(NZL-1)*I3
         DO 30 I1=0,1
            IX=1+(NXL-1)*I1
            DO 30 IY=2,NYL-1
               I=I+1
               IEDGE(I)=IX+NXL*(IY-1)+NXY*(IZ-1)
   30 CONTINUE
C
C     Fill Z-direction edges.
C
      IF (ldim.EQ.3) THEN
         DO 40 I2=0,1
            IY=1+(NYL-1)*I2
            DO 40 I1=0,1
               IX=1+(NXL-1)*I1
               DO 40 IZ=2,NZL-1
                  I=I+1
                  IEDGE(I)=IX+NXL*(IY-1)+NXY*(IZ-1)
   40    CONTINUE
      ENDIF
C
      CALL IZERO(INVEDG,27)
      DO 44 II=1,20
         IX=IEDGE(II)
         INVEDG(IX)=II
   44 CONTINUE
C
C
C     GENERAL FACE, GENERAL ROTATION EDGE NUMBERS.
C
      IF (ldim.EQ.3) THEN
C
C        Pack 3-D edge numbering:
C   
C        Fill temporary array with local index numbers:
C
         DO 50 IX=1,NXYZ
            ITMP(IX,1,1)=IX
   50    CONTINUE
C
C        Two sets are required, the base cube and the image cube 
C        which is being summed with it.
C
         DO 1000 IMAGE=0,1
C
C        Pack edges for each face, no rotation.
C
         DO 500 IFACE=1,NFACES
            JS1    = SKPDAT(1,IFACE)
            JF1    = SKPDAT(2,IFACE)
            JSKIP1 = SKPDAT(3,IFACE)
            JS2    = SKPDAT(4,IFACE)
            JF2    = SKPDAT(5,IFACE)
            JSKIP2 = SKPDAT(6,IFACE)
C
C           Choose proper indexing order according to face type and image.
C
            ORDER = (-1)**(GROUP(IFACE)+IMAGE)
            IF (ORDER.EQ.1) THEN
C
C              Forward ordering:
C
C            +-------------+    ^ v1
C            | --------->| |    |
C            | ^    2    | |    +-->
C            | |         | |      v2
C            | |1       3| |
C            | |    4    V |
C            | |<--------- |
C            F-------------I     F is fiducial node.
C
C                                I is location of fiducial node for
C                                     image face.
C
C           Load edge 1:
C
            J=0
            J2=JS2
            DO 100 J1=JS1,JF1-JSKIP1,JSKIP1
               J=J+1
               IEDGEF(J,1,IFACE,IMAGE)=ITMP(J1,J2,1)
  100       CONTINUE
C
C           Load edge 2:
C
            J=0
            J1=JF1
            DO 200 J2=JS2,JF2-JSKIP2,JSKIP2
               J=J+1
               IEDGEF(J,2,IFACE,IMAGE)=ITMP(J1,J2,1)
  200       CONTINUE
C
C
C           Load edge 3:
C
            J=0
            J2=JF2
            DO 300 J1=JF1,JS1+JSKIP1,-JSKIP1
               J=J+1
               IEDGEF(J,3,IFACE,IMAGE)=ITMP(J1,J2,1)
  300       CONTINUE
C
C           Load edge 4:
C
            J=0
            J1=JS1
            DO 400 J2=JF2,JS2+JSKIP2,-JSKIP2
               J=J+1
               IEDGEF(J,4,IFACE,IMAGE)=ITMP(J1,J2,1)
  400       CONTINUE
C
            ELSE
C
C           Reverse ordering:
C
C            +-------------+
C            | |<--------- |       ^ v2
C            | |    2    ^ |       |
C            | |         | |    <--+
C            | |3       1| |     v1
C            | V    4    | |
C            | --------->| |
C            I-------------F     F is fiducial node.
C
C                                I is location of fiducial node for
C                                     image face.
C
C           Load edge 1:
C   
            J=0
            J1=JS1
            DO 105 J2=JS2,JF2-JSKIP2,JSKIP2
               J=J+1
               IEDGEF(J,1,IFACE,IMAGE)=ITMP(J1,J2,1)
  105       CONTINUE
C
C           Load edge 2:
C
            J=0
            J2=JF2
            DO 205 J1=JS1,JF1-JSKIP1,JSKIP1
               J=J+1
               IEDGEF(J,2,IFACE,IMAGE)=ITMP(J1,J2,1)
  205       CONTINUE
C
C           Load edge 3:
C
            J=0
            J1=JF1
            DO 305 J2=JF2,JS2+JSKIP2,-JSKIP2
                J=J+1
               IEDGEF(J,3,IFACE,IMAGE)=ITMP(J1,J2,1)
  305       CONTINUE
C
C           Load edge 4:
C
            J=0
            J2=JS2
            DO 405 J1=JF1,JS1+JSKIP1,-JSKIP1
               J=J+1
              IEDGEF(J,4,IFACE,IMAGE)=ITMP(J1,J2,1)
  405       CONTINUE
            ENDIF
C
  500    CONTINUE
 1000    CONTINUE
      ELSE
C
C        Load edge information for 2-D case
C
         IEDGEF(1,1,1,0) = NXY - NXL + 1
         IEDGEF(1,2,1,0) = 1
         IEDGEF(1,1,2,0) = NXL
         IEDGEF(1,2,2,0) = NXY
         IEDGEF(1,1,3,0) = 1
         IEDGEF(1,2,3,0) = NXL
         IEDGEF(1,1,4,0) = NXY
         IEDGEF(1,2,4,0) = NXY - NXL + 1
C
         IEDGEF(1,1,1,1) = 1
         IEDGEF(1,2,1,1) = NXY - NXL + 1
         IEDGEF(1,1,2,1) = NXY
         IEDGEF(1,2,2,1) = NXL
         IEDGEF(1,1,3,1) = NXL
         IEDGEF(1,2,3,1) = 1
         IEDGEF(1,1,4,1) = NXY - NXL + 1
         IEDGEF(1,2,4,1) = NXY
      ENDIF
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine dsset(nx,ny,nz)
C
C     Set up arrays IXCN,ESKIP,SKPDAT,NEDG,NOFFST for new NX,NY,NZ
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TOPOL'
      INTEGER NXO,NYO,NZO
      SAVE    NXO,NYO,NZO
      DATA    NXO,NYO,NZO /3*0/
C
C     Check if element surface counters are already set from last call...
C
      IF (NXO.EQ.NX.AND.NYO.EQ.NY.AND.NZO.EQ.NZ) RETURN
C
C     else, proceed....
C
      NXO = NX
      NYO = NY
      NZO = NZ
C
C     Establish corner to elemental node number mappings
C
      IC=0
      DO 10 ICZ=0,1
      DO 10 ICY=0,1
      DO 10 ICX=0,1
C       Supress vectorization to 
c        IF(ICX.EQ.0)DUMMY=0        
c        IF(ICX.EQ.1)DUMMY=1
c        DUMMY2=DUMMY2+DUMMY
        IC=IC+1
        IXCN(IC)= 1 + (NX-1)*ICX + NX*(NY-1)*ICY + NX*NY*(NZ-1)*ICZ
   10   CONTINUE
C
C     Assign indices for direct stiffness summation of arbitrary faces.
C
C
C     Y-Z Planes (Faces 1 and 2)
C
      SKPDAT(1,1)=1
      SKPDAT(2,1)=NX*(NY-1)+1
      SKPDAT(3,1)=NX
      SKPDAT(4,1)=1
      SKPDAT(5,1)=NY*(NZ-1)+1
      SKPDAT(6,1)=NY
C
      SKPDAT(1,2)=1             + (NX-1)
      SKPDAT(2,2)=NX*(NY-1)+1   + (NX-1)
      SKPDAT(3,2)=NX
      SKPDAT(4,2)=1
      SKPDAT(5,2)=NY*(NZ-1)+1
      SKPDAT(6,2)=NY
C
C     X-Z Planes (Faces 3 and 4)
C
      SKPDAT(1,3)=1
      SKPDAT(2,3)=NX
      SKPDAT(3,3)=1
      SKPDAT(4,3)=1
      SKPDAT(5,3)=NY*(NZ-1)+1
      SKPDAT(6,3)=NY
C
      SKPDAT(1,4)=1           + NX*(NY-1)
      SKPDAT(2,4)=NX          + NX*(NY-1)
      SKPDAT(3,4)=1
      SKPDAT(4,4)=1
      SKPDAT(5,4)=NY*(NZ-1)+1
      SKPDAT(6,4)=NY
C
C     X-Y Planes (Faces 5 and 6)
C
      SKPDAT(1,5)=1
      SKPDAT(2,5)=NX
      SKPDAT(3,5)=1
      SKPDAT(4,5)=1
      SKPDAT(5,5)=NY
      SKPDAT(6,5)=1
C
      SKPDAT(1,6)=1           + NX*NY*(NZ-1)
      SKPDAT(2,6)=NX          + NX*NY*(NZ-1)
      SKPDAT(3,6)=1
      SKPDAT(4,6)=1
      SKPDAT(5,6)=NY
      SKPDAT(6,6)=1
C
C     Set up skip indices for each of the 12 edges 
C
C         Note that NXY = NX*NY even for 2-D since
C         this branch does not apply to the 2D case anyway.
C
C     ESKIP(*,1) = start location
C     ESKIP(*,2) = end 
C     ESKIP(*,3) = stride
C
      NXY=NX*NY
      ESKIP( 1,1) = IXCN(1) + 1
      ESKIP( 1,2) = IXCN(2) - 1
      ESKIP( 1,3) = 1
      ESKIP( 2,1) = IXCN(3) + 1
      ESKIP( 2,2) = IXCN(4) - 1
      ESKIP( 2,3) = 1
      ESKIP( 3,1) = IXCN(5) + 1
      ESKIP( 3,2) = IXCN(6) - 1
      ESKIP( 3,3) = 1
      ESKIP( 4,1) = IXCN(7) + 1
      ESKIP( 4,2) = IXCN(8) - 1
      ESKIP( 4,3) = 1
      ESKIP( 5,1) = IXCN(1) + NX
      ESKIP( 5,2) = IXCN(3) - NX
      ESKIP( 5,3) = NX
      ESKIP( 6,1) = IXCN(2) + NX
      ESKIP( 6,2) = IXCN(4) - NX
      ESKIP( 6,3) = NX
      ESKIP( 7,1) = IXCN(5) + NX
      ESKIP( 7,2) = IXCN(7) - NX
      ESKIP( 7,3) = NX
      ESKIP( 8,1) = IXCN(6) + NX
      ESKIP( 8,2) = IXCN(8) - NX
      ESKIP( 8,3) = NX
      ESKIP( 9,1) = IXCN(1) + NXY
      ESKIP( 9,2) = IXCN(5) - NXY
      ESKIP( 9,3) = NXY
      ESKIP(10,1) = IXCN(2) + NXY
      ESKIP(10,2) = IXCN(6) - NXY
      ESKIP(10,3) = NXY
      ESKIP(11,1) = IXCN(3) + NXY
      ESKIP(11,2) = IXCN(7) - NXY
      ESKIP(11,3) = NXY
      ESKIP(12,1) = IXCN(4) + NXY
      ESKIP(12,2) = IXCN(8) - NXY
      ESKIP(12,3) = NXY
C
C     Load reverse direction edge arrays for reverse mappings...
C
      DO 20 IED=1,12
      IEDM=-IED
      ESKIP(IEDM,1) =  ESKIP(IED,2)
      ESKIP(IEDM,2) =  ESKIP(IED,1)
      ESKIP(IEDM,3) = -ESKIP(IED,3)
   20 CONTINUE
C
C     Compute offset for global edge vector given current element
C     dimensions.....
C
C     NGSPED(ITE,ICMP) = number of global (ie, distinct) special edges
C                        of type ITE (1,2, or 3)  for field ICMP.
C
C                        ITE = 1 implies an "X" edge
C                        ITE = 2 implies an "Y" edge
C                        ITE = 3 implies an "Z" edge
C
C     Set up number of nodes along each of the 3 types of edges
C     (endpoints excluded).
C
      NEDG(1)=NX-2
      NEDG(2)=NY-2
      NEDG(3)=NZ-2
C
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine genxyzl
C
C     Generate xyz coordinates 
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'SCRCT'
      COMMON /CTMP0/ XCB(2,2,2),YCB(2,2,2),ZCB(2,2,2),H(3,3,2),INDX(8)
C
      NXL=3
      NYL=3
      NZL=1+2*(ldim-2)
      NTOT3=NXL*NYL*NZL*NELT
C
C   Preprocessor Corner notation:      Symmetric Corner notation:
C                
C           4+-----+3    ^ s                    3+-----+4    ^ s        
C           /     /|     |                      /     /|     |          
C          /     / |     |                     /     / |     |          
C        8+-----+7 +2    +----> r            7+-----+8 +2    +----> r   
C         |     | /     /                     |     | /     /           
C         |     |/     /                      |     |/     /            
C        5+-----+6    t                      5+-----+6    t             
C
      DO 10 IX=1,NXL
         H(IX,1,1)=0.5*FLOAT(3-IX)
         H(IX,1,2)=0.5*FLOAT(IX-1)
   10 CONTINUE
      DO 20 IY=1,NYL
         H(IY,2,1)=0.5*FLOAT(3-IY)
         H(IY,2,2)=0.5*FLOAT(IY-1)
   20 CONTINUE
      DO 30 IZ=1,NZL
         H(IZ,3,1)=0.5*FLOAT(3-IZ)
         H(IZ,3,2)=0.5*FLOAT(IZ-1)
   30 CONTINUE
C
      INDX(1)=1
      INDX(2)=2
      INDX(3)=4
      INDX(4)=3
      INDX(5)=5
      INDX(6)=6
      INDX(7)=8
      INDX(8)=7
C
      CALL RZERO(XML,NTOT3)
      CALL RZERO(YML,NTOT3)
      CALL RZERO(ZML,NTOT3)
      CALL RZERO(XCB,8)
      CALL RZERO(YCB,8)
      CALL RZERO(ZCB,8)
C
      DO 5000 IE=1,NELT
C
         ldim2 = 2**ldim
         DO 50 IX=1,ldim2
            I=INDX(IX)
            XCB(IX,1,1)=XC(I,IE)
            YCB(IX,1,1)=YC(I,IE)
            ZCB(IX,1,1)=ZC(I,IE)
   50    CONTINUE
C
C        Map R-S-T space into physical X-Y-Z space.
C
         DO 100 IZT=1,ldim-1
         DO 100 IYT=1,2
         DO 100 IXT=1,2
C
         DO 100 IZ=1,NZL
         DO 100 IY=1,NYL
         DO 100 IX=1,NXL
            XML(IX,IY,IZ,IE)=XML(IX,IY,IZ,IE)+
     $        H(IX,1,IXT)*H(IY,2,IYT)*H(IZ,3,IZT)*XCB(IXT,IYT,IZT)
            YML(IX,IY,IZ,IE)=YML(IX,IY,IZ,IE)+
     $        H(IX,1,IXT)*H(IY,2,IYT)*H(IZ,3,IZT)*YCB(IXT,IYT,IZT)
            ZML(IX,IY,IZ,IE)=ZML(IX,IY,IZ,IE)+
     $        H(IX,1,IXT)*H(IY,2,IYT)*H(IZ,3,IZT)*ZCB(IXT,IYT,IZT)
  100    CONTINUE
C
 5000 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine verify
C
C     .Verify right-handedness of elements.
C     .Verify element-to-element reciprocity of BC's
C     .Verify correlation between E-E BC's and physical coincidence
C
      include 'SIZE'
      include 'PARALLEL'
      include 'INPUT'
      include 'SCRCT'
 
      call verrhe

      return
      end
c-----------------------------------------------------------------------
      subroutine setside
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TOPOL'
      INCLUDE 'SCRCT'
C
C     SIDE(i,IFACE,IE) -  Physical (xyz) location of element side midpoint.
C                         i=1,2,3 gives x,y,z value, respectively.
C                         i=4  gives average dimension of face for setting
C                              tolerances.
C
      INDX(1)=1
      INDX(2)=2
      INDX(3)=4
      INDX(4)=3
      INDX(5)=5
      INDX(6)=6
      INDX(7)=8
      INDX(8)=7
C
C     Flip vertex array structure
C
c     write(6,*) nelv,nelt,if3d
      call rzero(xyz,24*nelt)
      if (if3d) then
         do ie=1,nelt
         do j=1,8
            ivtx = indx(j)
            xyz(1,ivtx,ie) = xc(j,ie)
            xyz(2,ivtx,ie) = yc(j,ie)
            xyz(3,ivtx,ie) = zc(j,ie)
c           write(6,1) ie,j,ivtx,xc(j,ie),yc(j,ie),zc(j,ie),' xcz'
c           write(6,1) ie,j,ivtx,(xyz(k,ivtx,ie),k=1,3),' vtx'
c   1       format(3i5,1p3e12.4,a4)
         enddo
         enddo
      else
         do ie=1,nelt
         do j=1,4
            ivtx = indx(j)
            xyz(1,ivtx,ie) = xc(j,ie)
            xyz(2,ivtx,ie) = yc(j,ie)
            xyz(3,ivtx,ie) = 0.0
         enddo
         enddo
      endif
C
C     Compute location of center and "diameter" of each element side.
C
      NFACES=ldim*2
      NCRNR =2**(ldim-1)
      CALL RZERO(SIDE,24*NELT)
      DO 500 ICRN=1,NCRNR
      DO 500 IFAC=1,NFACES
         IVTX = ICFACE(ICRN,IFAC)
         ICR1 = NCRNR+(ICRN-1)
         ICR1 = MOD1(ICR1,NCRNR)
         IVT1 = ICFACE(ICR1,IFAC)
         DO 400 IE=1,NELT
            DO 300 IDIM=1,ldim
               SIDE(IDIM,IFAC,IE)=SIDE(IDIM,IFAC,IE)+XYZ(IDIM,IVTX,IE)
               SIDE(   4,IFAC,IE)=SIDE(   4,IFAC,IE)+
     $                     ( XYZ(IDIM,IVTX,IE)-XYZ(IDIM,IVT1,IE) )**2
  300       CONTINUE
            SIDE(4,IFAC,IE)=SQRT( SIDE(4,IFAC,IE) )
  400    CONTINUE
  500 CONTINUE
      AVWGHT=1.0/FLOAT(NCRNR)
      CALL CMULT(SIDE,AVWGHT,24*NELT)
C
c     call exitt
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine verrhe
C
C     8 Mar 1989 21:58:26   PFF
C     Verify right-handedness of given elements. 
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'PARALLEL'
      INCLUDE 'SCRCT'
      INCLUDE 'TOPOL'
      LOGICAL IFYES,IFCSTT
C
      IFCSTT=.TRUE.
      IF (.NOT.IF3D) THEN
      DO 1000 IE=1,NELT
C
C        CRSS2D(A,B,O) = (A-O) X (B-O)
C
         C1=CRSS2D(XYZ(1,2,IE),XYZ(1,3,IE),XYZ(1,1,IE))
         C2=CRSS2D(XYZ(1,4,IE),XYZ(1,1,IE),XYZ(1,2,IE))
         C3=CRSS2D(XYZ(1,1,IE),XYZ(1,4,IE),XYZ(1,3,IE))
         C4=CRSS2D(XYZ(1,3,IE),XYZ(1,2,IE),XYZ(1,4,IE))
C
         IF (C1.LE.0.0.OR.C2.LE.0.0.OR.
     $       C3.LE.0.0.OR.C4.LE.0.0 ) THEN
C
            ieg=lglel(ie)
            WRITE(6,800) IEG,C1,C2,C3,C4
            call exitt
  800       FORMAT(/,2X,'WARNINGa: Detected non-right-handed element.',
     $      /,2X,'Number',I8,'  C1-4:',4E12.4)
            IFCSTT=.FALSE.
C           CALL QUERY(IFYES,'Proceed                                 ')
C           IF (.NOT.IFYES) GOTO 9000
         ENDIF
 1000 CONTINUE
C
C     Else 3-D:
C
      ELSE      
      DO 2000 IE=1,NELT
C
C        VOLUM0(A,B,C,O) = (A-O)X(B-O).(C-O)
C
         V1= VOLUM0(XYZ(1,2,IE),XYZ(1,3,IE),XYZ(1,5,IE),XYZ(1,1,IE))
         V2= VOLUM0(XYZ(1,4,IE),XYZ(1,1,IE),XYZ(1,6,IE),XYZ(1,2,IE))
         V3= VOLUM0(XYZ(1,1,IE),XYZ(1,4,IE),XYZ(1,7,IE),XYZ(1,3,IE))
         V4= VOLUM0(XYZ(1,3,IE),XYZ(1,2,IE),XYZ(1,8,IE),XYZ(1,4,IE))
         V5=-VOLUM0(XYZ(1,6,IE),XYZ(1,7,IE),XYZ(1,1,IE),XYZ(1,5,IE))
         V6=-VOLUM0(XYZ(1,8,IE),XYZ(1,5,IE),XYZ(1,2,IE),XYZ(1,6,IE))
         V7=-VOLUM0(XYZ(1,5,IE),XYZ(1,8,IE),XYZ(1,3,IE),XYZ(1,7,IE))
         V8=-VOLUM0(XYZ(1,7,IE),XYZ(1,6,IE),XYZ(1,4,IE),XYZ(1,8,IE))
C
         IF (V1.LE.0.0.OR.V2.LE.0.0.OR.
     $       V3.LE.0.0.OR.V4.LE.0.0.OR.
     $       V5.LE.0.0.OR.V6.LE.0.0.OR.
     $       V7.LE.0.0.OR.V8.LE.0.0    ) THEN
C
            ieg=lglel(ie)
            WRITE(6,1800) IEG,V1,V2,V3,V4,V5,V6,V7,V8
            call exitt
 1800       FORMAT(/,2X,'WARNINGb: Detected non-right-handed element.',
     $      /,2X,'Number',I8,'  V1-8:',4E12.4
     $      /,2X,'      ',4X,'       ',4E12.4)
            IFCSTT=.FALSE.
         ENDIF
 2000 CONTINUE
      ENDIF
C
 9000 CONTINUE
C
C     Print out results from right-handed check
C
      IF (.NOT.IFCSTT) WRITE(6,2001)
C
C     Check consistency accross all processors.
C
      CALL GLLOG(IFCSTT,.FALSE.)
C
      IF (.NOT.IFCSTT) THEN
         IF (NID.EQ.0) WRITE(6,2003) NELGT
         call exitt
      ELSE
         IF (NIO.EQ.0) WRITE(6,2002) NELGT
      ENDIF
C
 2001 FORMAT(//,'  Elemental geometry not right-handed, ABORTING'
     $      ,' in routine VERRHE.')
 2002 FORMAT('   Right-handed check complete for',I12,' elements. OK.')
 2003 FORMAT('   Right-handed check failed for',I12,' elements.'
     $      ,'   Exiting in routine VERRHE.')
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION VOLUM0(P1,P2,P3,P0)
C
C                           3
C     Given four points in R , (P1,P2,P3,P0), VOLUM0 returns
C     the volume enclosed by the parallelagram defined by the
C     vectors { (P1-P0),(P2-P0),(P3-P0) }.  This routine has
C     the nice feature that if the 3 vectors so defined are
C     not right-handed then the volume returned is negative.
C
      REAL P1(3),P2(3),P3(3),P0(3)
C
         U1=P1(1)-P0(1)
         U2=P1(2)-P0(2)
         U3=P1(3)-P0(3)
C
         V1=P2(1)-P0(1)
         V2=P2(2)-P0(2)
         V3=P2(3)-P0(3)
C
         W1=P3(1)-P0(1)
         W2=P3(2)-P0(2)
         W3=P3(3)-P0(3)
C
         CROSS1 = U2*V3-U3*V2
         CROSS2 = U3*V1-U1*V3
         CROSS3 = U1*V2-U2*V1
C
         VOLUM0  = W1*CROSS1 + W2*CROSS2 + W3*CROSS3
         
      RETURN
      END
c-----------------------------------------------------------------------
      FUNCTION CRSS2D(XY1,XY2,XY0)
      REAL XY1(2),XY2(2),XY0(2)
C
         V1X=XY1(1)-XY0(1)
         V2X=XY2(1)-XY0(1)
         V1Y=XY1(2)-XY0(2)
         V2Y=XY2(2)-XY0(2)
         CRSS2D = V1X*V2Y - V1Y*V2X
C
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
c      ifcase in preprocessor notation
       KX1=1
       KY1=1
       KZ1=1
       KX2=NX
       KY2=NY
       KZ2=NZ
       IF (IFACE.EQ.1) KY2=1
       IF (IFACE.EQ.2) KX1=NX
       IF (IFACE.EQ.3) KY1=NY
       IF (IFACE.EQ.4) KX2=1
       IF (IFACE.EQ.5) KZ2=1
       IF (IFACE.EQ.6) KZ1=NZ
      return
      end
c-----------------------------------------------------------------------
      subroutine facindr (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)

c     restricted index set, iface in preprocessor notation

      kx1=2
      ky1=2
      kz1=2
      kx2=nx-1
      ky2=ny-1
      kz2=nz-1

      if (iface.eq.1) ky1=1
      if (iface.eq.1) ky2=1

      if (iface.eq.2) kx1=nx
      if (iface.eq.2) kx2=nx

      if (iface.eq.3) ky1=ny
      if (iface.eq.3) ky2=ny

      if (iface.eq.4) kx1=1
      if (iface.eq.4) kx2=1

      if (iface.eq.5) kz1=1
      if (iface.eq.5) kz2=1

      if (iface.eq.6) kz1=nz
      if (iface.eq.6) kz2=nz

      return
      end
c-----------------------------------------------------------------------
      subroutine facev(a,ie,iface,val,nx,ny,nz)
C
C     Assign the value VAL to face(IFACE,IE) of array A.
C     IFACE is the input in the pre-processor ordering scheme.
C
      INCLUDE 'SIZE'
      DIMENSION A(NX,NY,NZ,LELT)
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX,NY,NZ,IFACE)
      DO 100 IZ=KZ1,KZ2
      DO 100 IY=KY1,KY2
      DO 100 IX=KX1,KX2
         A(IX,IY,IZ,IE)=VAL
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine ifacev(a,ie,iface,val,nx,ny,nz)
C
C     Assign the value VAL to face(IFACE,IE) of array A.
C     IFACE is the input in the pre-processor ordering scheme.
C
      include 'SIZE'
      integer a(nx,ny,nz,lelt),val
      call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
      do 100 iz=kz1,kz2
      do 100 iy=ky1,ky2
      do 100 ix=kx1,kx2
         a(ix,iy,iz,ie)=val
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine facec(a,b,ie,iface,nx,ny,nz,nel)
C
C     Copy the face (IFACE) of B to A.
C     IFACE is the input in the pre-processor ordering scheme.
C
      DIMENSION A(NX,NY,NZ,NEL)
      DIMENSION B(NX,NY,NZ,NEL)
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX,NY,NZ,IFACE)
      DO 100 IZ=KZ1,KZ2
      DO 100 IY=KY1,KY2
      DO 100 IX=KX1,KX2
         A(IX,IY,IZ,IE)=B(IX,IY,IZ,IE)
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine combin2(glnm1,glnm2,nglob)
c
      write(6,*) 'Hey, who called combin2???  ABORT'
      call exitt
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldio (x,txt10)
      INCLUDE 'SIZE'
      integer x(lx1,ly1,lz1,lelt)
      character*10 txt10
C
      do ie=1,nelv,2
         do iz=1,lz1,1
            if (iz.eq.1) write(6,106) txt10,iz,ie
            if (iz.gt.1) write(6,107) 
            i1 = ie+1
            do j=ly1,1,-1
               write(6,105) (x(i,j,iz,ie),i=1,lx1)
     $                    , (x(i,j,iz,i1),i=1,lx1)
            enddo
         enddo
      enddo
C
  107 FORMAT(' ')
  105 FORMAT(4i6,20x,4i6)
  106 FORMAT(  /,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','Plane = ',I2,'/',I2,/,
     $           5X,'       X            ')
C
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldi (x,txt10)
      INCLUDE 'SIZE'
      integer x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      character*6 s(20,20)
c
      if (lx1.ne.4 .or. nelv.gt.3) return
c
      write(6,106) txt10,ie,ie
  106 FORMAT(  /,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','elem. = ',I2,'/',I2,/,
     $           5X,'       X            ')
C
C
      call blank(s,6*20*20)
      do ie=1,3
         if (ie.eq.1) then
            jstart  = 1
            istart  = 1
            istride = 3
         else
            jstart  = 2 + lx1
            istart  = 1
            istride = 1
            if (ie.eq.2) istart = 7
         endif
c
         i=istart
         do iy=ly1,1,-1
            j=jstart
            do ix=1,lx1
               write(s(i,j),6) x(ix,iy,1,ie)
               j=j+1
            enddo
            i=i+istride
         enddo
    6    format(20i6)
      enddo
c
      do i=1,10
         write(6,7) (s(i,l),l=1,j-1)
      enddo
    7 format(20a6)
c
      write(6,*)
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldr (x,txt10)
      INCLUDE 'SIZE'
      real x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      character*6 s(20,20)
c
      if (lx1.ne.4 .or. nelv.gt.3) return
      write(6,106) txt10,ie,ie
  106 FORMAT(  /,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','elem. = ',I2,'/',I2,/,
     $           5X,'       X            ')
      call blank(s,6*20*20)
C
C
      do ie=1,3
         if (ie.eq.1) then
            jstart  = 1
            istart  = 1
            istride = 3
         else
            jstart  = 2 + lx1
            istart  = 1
            istride = 1
            if (ie.eq.2) istart = 7
         endif
c
         i=istart
         do iy=ly1,1,-1
            j=jstart
            do ix=1,lx1
               write(s(i,j),6) x(ix,iy,1,ie)
               j=j+1
            enddo
            i=i+istride
         enddo
    6    format(f6.2)
      enddo
c
      do i=1,10
         write(6,7) (s(i,l),l=1,j-1)
      enddo
    7 format(20a6)
c
      write(6,*)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine checkit(idum)
      write(6,*) 'continue?'
      read (5,*) idum
      return
      end
c-----------------------------------------------------------------------
      subroutine outfldro (x,txt10,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /3/
      if (idum.lt.0) return
c
C
      mtot = lx1*ly1*lz1*nelv
      if (lx1.gt.8.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)

      do ie=1,1
         ne = 1
         do k=1,1
            write(6,116) txt10,k,ie,xmin,xmax,istep,time
            write(6,117) 
            do j=ly1,1,-1
              if (lx1.eq.2) write(6,102) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.3) write(6,103) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.4) write(6,104) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.5) write(6,105) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.6) write(6,106) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.7) write(6,107) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
              if (lx1.eq.8) write(6,118) ((x(i,j,k,e+1),i=1,lx1),e=0,ne)
            enddo
         enddo
      enddo

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
      subroutine outfldrv (x,txt10,ichk) ! writes to unit=40+ifield
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /3/
      if (idum.lt.0) return
      m = 40 + ifield
c
C
      mtot = lx1*ly1*lz1*nelv
      if (lx1.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(m,116) txt10,k,ie,xmin,xmax,istep,time
            write(m,117) 
            do j=ly1,1,-1
              if (lx1.eq.2) write(m,102) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.3) write(m,103) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.4) write(m,104) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.5) write(m,105) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.6) write(m,106) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.7) write(m,107) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.8) write(m,108) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
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
      subroutine outfldrv0 (x,txt10,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(lx1,ly1,lz1,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /3/
      if (idum.lt.0) return
c
C
      mtot = lx1*ly1*lz1*nelv
      if (lx1.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(6,116) txt10,k,ie,xmin,xmax,istep,time
            write(6,117) 
            do j=ly1,1,-1
              if (lx1.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
              if (lx1.eq.8) write(6,108) ((x(i,j,k,e+l),i=1,lx1),e=1,ne)
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
      subroutine outfldrp0 (x,txt10,ichk)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(lx2,ly2,lz2,lelt)
      character*10 txt10
c
      integer idum,e
      save idum
      data idum /3/
      if (idum.lt.0) return
c
C
      mtot = lx2*ly2*lz2*nelv
      if (lx2.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(6,116) txt10,k,ie,xmin,xmax,istep,time
            write(6,117) 
            do j=ly2,1,-1
              if (lx2.eq.2) write(6,102) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.3) write(6,103) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.4) write(6,104) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.5) write(6,105) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.6) write(6,106) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.7) write(6,107) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.8) write(6,108) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
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
      subroutine outfldrp (x,txt10,ichk) ! writes out into unit = 40+ifield 
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      real x(lx2,ly2,lz2,lelt)
      character*10 txt10
c
      integer idum,e
      save    idum
      data    idum /3/
      if (idum.lt.0)   return
      m = 40 + ifield             ! unit #
c
c
C
      mtot = lx2*ly2*lz2*nelv
      if (lx2.gt.7.or.nelv.gt.16) return
      xmin = glmin(x,mtot)
      xmax = glmax(x,mtot)
c
      rnel = nelv
      snel = sqrt(rnel)+.1
      ne   = snel
      ne1  = nelv-ne+1
      do ie=ne1,1,-ne
         l=ie-1
         do k=1,1
            if (ie.eq.ne1) write(m,116) txt10,k,ie,xmin,xmax,istep,time
            write(6,117) 
            do j=ly2,1,-1
              if (lx2.eq.2) write(m,102) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.3) write(m,103) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.4) write(m,104) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.5) write(m,105) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.6) write(m,106) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.7) write(m,107) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
              if (lx2.eq.8) write(m,108) ((x(i,j,k,e+l),i=1,lx2),e=1,ne)
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
      subroutine outmatp(a,m,n,name6,ie)
      include 'SIZE'
      include 'PARALLEL'
      real a(m,n)
      character*6 name6
c
      if (nelv.gt.1) return
      do jid=0,np-1
        imax = iglmax(jid,1)
        if (jid.eq.nid) then
           write(6,*) 
           write(6,*) nid,ie,' matrix: ',name6,m,n
           n12 = min(n,12)
           do i=1,m
              write(6,6) ie,name6,(a(i,j),j=1,n12)
           enddo
    6      format(i3,1x,a6,12f9.5)
           write(6,*) 
           call flush_io()
         endif
        imax = iglmax(jid,1)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine gs_chkr(glo_num)
      include 'SIZE'
      include 'TOTAL'

      integer glo_num(lx1,ly1,lz1,lelt)
      integer e,eg

      do ipass = 1,2
         iquick = 0
         if (nid.eq.0) iquick=glo_num(ipass,1,1,1)
         iquick = iglmax(iquick,1)

         do e=1,nelv
         do k=1,lz1
         do j=1,ly1
         do i=1,lx1
           if (glo_num(i,j,k,e).eq.iquick) then
            eg = lglel(e)
            write(6,1) nid,i,j,k,e,eg,iquick,ipass
     $      ,xm1(i,j,k,e),ym1(i,j,k,e),zm1(i,j,k,e)
  1         format(i12,3i4,2i12,i12,i2,1p3e12.4,' iquick')
          endif
         enddo
         enddo
         enddo
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine setup_mesh_dssum ! Set up dssum for mesh

      include 'SIZE'
      include 'TOTAL'
      include 'NONCON'

      common /c_is1/ glo_num(1*lx1*ly1*lz1*lelv)
      integer*8 glo_num
      common /ivrtx/ vertex ((2**ldim)*lelt)
      integer vertex

      parameter(lxyz=lx1*ly1*lz1)
      common /scrns/ enum(lxyz,lelt)
     $             ,  rnx(lxyz,lelt) , rny(lxyz,lelt) , rnz(lxyz,lelt)
     $             ,  tnx(lxyz,lelt) , tny(lxyz,lelt) , tnz(lxyz,lelt)
      common /scruz/  snx(lxz) , sny(lxz) , snz(lxz) ,  efc(lxz)
      common /scrsf/  jvrtex((2**ldim),lelt)

      integer e,f,eg


      gsh_fld(0)=gsh_fld(1)
      if (iftmsh(0)) gsh_fld(0)=gsh_fld(2)

      ifield = 0
      nel    = nelfld(0)
      nxyz   = lx1*ly1*lz1
      nxz    = lx1*lz1
      n      = nel*nxyz
      nface  = 2*ldim


      iflag=0
      do e=1,nel
      do f=1,nface
         if (cbc(f,e,1).eq.'msi'.or.cbc(f,e,1).eq.'MSI') iflag=1
      enddo
      enddo
      iflag = iglmax(iflag,1)
      if (iflag.eq.0) return


c     We need to differentiate elements according to unit normal on msi face.

c     NOTE that this code assumes we do not have two adjacent faces on a given
c     element that both have cbc = msi!

      call rzero(rnx,n)
      call rzero(rny,n)
      call rzero(rnz,n)
      call rzero(tnx,n)
      call rzero(tny,n)
      call rzero(tnz,n)

      do e=1,nel
         re = lglel(e)
         call cfill(enum(1,e),re,nxyz)
      enddo

      call dsop(enum,'min')


      do e=1,nel
         eg = lglel(e)
         do f=1,nface
           if (cbc(f,e,1).eq.'msi'.or.cbc(f,e,1).eq.'MSI') then
             call facexs (efc,enum(1,e),f,0) ! enum-->efc
             do i=1,nxz
               jg = efc(i)+0.1
               if (jg.eq.eg) then         ! this is the controlling face
                  snx(i) = unx(i,1,f,e)
                  sny(i) = uny(i,1,f,e)
                  snz(i) = unz(i,1,f,e)
               endif
             enddo
             call facexv (snx,sny,snz,rnx(1,e),rny(1,e),rnz(1,e),f,1) ! s-->r
             call facexv (unx(1,1,f,e),uny(1,1,f,e),unz(1,1,f,e) ! Control
     $                   ,tnx(1,e),tny(1,e),tnz(1,e),f,1)        ! u-->r
           endif
         enddo
      enddo

      call opdssum(rnx,rny,rnz)

      nv = nel*(2**ldim)
      call icopy(jvrtex,vertex,nv)  ! Save vertex
      mvertx=iglmax(jvrtex,nv)


c     Now, check to see if normal is aligned with incoming normal

      nsx=lx1-1
      nsy=ly1-1
      nsz=max(1,lz1-1)

      do e=1,nel
         l=0
         do k=1,lz1,nsz
         do j=1,ly1,nsy
         do i=1,lx1,nsx
            l=l+1
            m=i + lx1*(j-1) + lx1*ly1*(k-1)
            dot = tnx(m,e)*rnx(m,e)+tny(m,e)*rny(m,e)+tnz(m,e)*rnz(m,e)
            if (dot.lt.-.5) jvrtex(l,e)=jvrtex(l,e)+mvertx
         enddo
         enddo
         enddo
      enddo


      call setupds(gsh_fld(0),lx1,ly1,lz1,nelv,nelgv,jvrtex,glo_num)

      return
      end
c-----------------------------------------------------------------------
      subroutine outfldnx(x,txt10,nx,ny)
      include 'SIZE'
      real x(nx,ny,4)
      character*10 txt10
      integer e,g

      write(6,106) txt10,nel,nx
  106 FORMAT(  /,5X,'     ^              ',/,
     $           5X,'   Y |              ',/,
     $           5X,'     |              ',A10,/,
     $           5X,'     +---->         ','elem. = ',I2,'/',I2,/,
     $           5X,'       X            ')

      do e=3,1,-2
         g=e+1
         i=istart
         do j=ny,1,-1
            if (nx.eq.3) write(6,3) (x(i,j,e),i=1,nx),(x(i,j,g),i=1,nx)
            if (nx.eq.4) write(6,4) (x(i,j,e),i=1,nx),(x(i,j,g),i=1,nx)
            if (nx.eq.5) write(6,5) (x(i,j,e),i=1,nx),(x(i,j,g),i=1,nx)
            if (nx.eq.6) write(6,6) (x(i,j,e),i=1,nx),(x(i,j,g),i=1,nx)
            if (nx.eq.7) write(6,7) (x(i,j,e),i=1,nx),(x(i,j,g),i=1,nx)
   3        format(3f8.4,3x,3f8.4)
   4        format(4f8.4,3x,4f8.4)
   5        format(5f8.4,3x,5f8.4)
   6        format(6f8.4,3x,6f8.4)
   7        format(7f8.4,3x,7f8.4)
         enddo
         write(6,*)
      enddo
      write(6,*)

      return
      end
c-----------------------------------------------------------------------
