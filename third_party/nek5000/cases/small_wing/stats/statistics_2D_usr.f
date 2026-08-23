!=======================================================================
! Name        : statistics_2DIO_usr
! Author      : Adam Peplinski
! Version     : last modification 2015.05.20
! Copyright   : GPL
! Description : This is a set of user provided routines to calculate 
!     2D statistics
!=======================================================================
c----------------------------------------------------------------------
!     user routine to calculate element cetre
      subroutine user_stat_init(ctrs,cell,lctrs1,lctrs2,nelsort)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'           ! [XYZ]C
cc MA:      include 'GEOM_DEF'
      include 'GEOM'            ! [XYZ]M1
      include 'STATS'           ! 2D statistics speciffic variables

!     argument list
      integer lctrs1,lctrs2     ! array sizes
      real ctrs(lctrs1,lctrs2)  ! 2D element centres  and diagonals 
      integer cell(lctrs2)      ! local element numberring
      integer nelsort           ! number of local 3D elements to sort

!     local variables
      integer ntot              ! tmp array size for copying
      integer e,i,j             ! loop indexes
      integer nvert             ! vertex number
      real rnvert               ! 1/nvert
      real xmid,ymid            ! 2D element centre
      real xmin,xmax,ymin,ymax  ! to get approximate element diagonal
      integer ifc               ! face number

!     dummy arrays
      real xcoord(8,LELT), ycoord(8,LELT) ! tmp vertex coordinates

c$$$!     for testing
c$$$      integer itl1, itl2
c$$$      character*2 str

!     set important parameters
!     uniform direction; should be taken as input parameter
!     x-> 1, y-> 2, z-> 3
      STAT_IDIR = 3

      if (.not.if3d) STAT_IDIR = 3        ! #2D
      
!     get element midpoints
!     vertex number
      nvert = 2**NDIM
      rnvert= 1.0/real(nvert)

!     eliminate uniform direction
      ntot = 8*NELV
      if (STAT_IDIR.EQ.1) then  ! uniform X
         call copy(xcoord,YC,ntot) ! copy y
         call copy(ycoord,ZC,ntot) ! copy z
      elseif (STAT_IDIR.EQ.2) then  ! uniform Y
         call copy(xcoord,XC,ntot) ! copy x
         call copy(ycoord,ZC,ntot) ! copy z
      elseif (STAT_IDIR.EQ.3) then  ! uniform Z
         call copy(xcoord,XC,ntot) ! copy x
         call copy(ycoord,YC,ntot) ! copy y
      endif

!     set initial number of elements to sort
      nelsort = 0
      call izero(cell,NELT)

!     mark all elements as unwanted
      call ifill(STAT_LMAP,-1,NELT)

!     for every element
      do e=1,NELV
!     element centre
         xmid = xcoord(1,e)
         ymid = ycoord(1,e)
!     element diagonal
         xmin = xmid
         xmax = xmid
         ymin = ymid
         ymax = ymid
         do i=2,nvert
            xmid=xmid+xcoord(i,e)
            ymid=ymid+ycoord(i,e)
            xmin = min(xmin,xcoord(i,e))
            xmax = max(xmax,xcoord(i,e))
            ymin = min(ymin,ycoord(i,e))
            ymax = max(ymax,ycoord(i,e))
         enddo
         xmid = xmid*rnvert
         ymid = ymid*rnvert

!     place to exclude unwanted elements 
!     if you want statistics in only part of the domain
!     right now I take all elements
!         if () then             ! exclude unwanted elements

!     count elements to sort
            nelsort = nelsort + 1
!     2D position
            ctrs(1,nelsort)=xmid
            ctrs(2,nelsort)=ymid
!     reference distance
            ctrs(3,nelsort)=sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
            if (ctrs(3,nelsort).eq.0.0) then
               if(NIO.eq.0) write(6,*)
     $              'Error: stat_init_local; bad elem.'
               call exitt
            endif
!     element index
            cell(nelsort) = e
!     mark element as needed
            STAT_LMAP(e) = 1
!     right now I take all elements
!         endif                  ! exclude unwanted elements

      enddo

!     fill in 2D element mesh arrays
      if (STAT_IDIR.EQ.1) then  ! uniform X
         ifc = 4
         do e=1,NELV
            if(STAT_LMAP(e).ne.-1) then
               call ftovec(STAT_XM1(1,1,e),YM1,e,ifc,NX1,NY1,NZ1)
               call ftovec(STAT_YM1(1,1,e),ZM1,e,ifc,NX1,NY1,NZ1)
            endif
         enddo
      elseif (STAT_IDIR.EQ.2) then  ! uniform Y
         ifc = 1
         do e=1,NELV
            if(STAT_LMAP(e).ne.-1) then
               call ftovec(STAT_XM1(1,1,e),XM1,e,ifc,NX1,NY1,NZ1)
               call ftovec(STAT_YM1(1,1,e),ZM1,e,ifc,NX1,NY1,NZ1)
            endif
         enddo
      elseif (STAT_IDIR.EQ.3) then  ! uniform Z
         ifc = 5
         do e=1,NELV
            if(STAT_LMAP(e).ne.-1) then
               call ftovec(STAT_XM1(1,1,e),XM1,e,ifc,NX1,NY1,NZ1)
               call ftovec(STAT_YM1(1,1,e),YM1,e,ifc,NX1,NY1,NZ1)
            endif
         enddo
      endif

c$$$!     testing
c$$$         write(str,'(i2.2)') NID
c$$$         open(unit=10001,file='usr_init.txt'//str)
c$$$         write(10001,*) NID, STAT_IDIR, NELV, nelsort
c$$$         do itl1=1,NELV
c$$$            write(10001,*) itl1
c$$$            do j=1,NX1
c$$$               do i=1,NX1
c$$$                  write(10001,*) i,j,cell(itl1),STAT_XM1(i,j,itl1),
c$$$     $                 STAT_YM1(i,j,itl1)
c$$$               enddo
c$$$            enddo
c$$$         enddo
c$$$         close(10001)
c$$$!     testing end

      return
      end
c----------------------------------------------------------------------
!     user routine to calculate vorticity and get velocity in new coordinates
      subroutine user_stat_trnsv(lvel,dudx,dvdx,dwdx,vort)
      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'               ! if3d

!     argument list
      real lvel(LX1,LY1,LZ1,LELT,3) ! velocity array
      real dudx(LX1,LY1,LZ1,LELT,3) ! velocity derivatives; U
      real dvdx(LX1,LY1,LZ1,LELT,3) ! V
      real dwdx(LX1,LY1,LZ1,LELT,3) ! W
      real vort(LX1,LY1,LZ1,LELT,3) ! vorticity

!     local variables
      integer itmp              ! dummy variable

!     Velocity transformation; simple copy
      itmp = NX1*NY1*NZ1*NELV
      call copy(lvel(1,1,1,1,1),VX,itmp)
      call copy(lvel(1,1,1,1,2),VY,itmp)
      call copy(lvel(1,1,1,1,3),VZ,itmp)

!     Derivative transformation
!     No transformation
      call gradm1(dudx(1,1,1,1,1),dudx(1,1,1,1,2),dudx(1,1,1,1,3),
     $      lvel(1,1,1,1,1))
      call gradm1(dvdx(1,1,1,1,1),dvdx(1,1,1,1,2),dvdx(1,1,1,1,3),
     $      lvel(1,1,1,1,2))
      call gradm1(dwdx(1,1,1,1,1),dwdx(1,1,1,1,2),dwdx(1,1,1,1,3),
     $      lvel(1,1,1,1,3))

!     get vorticity
      if (IF3D) then
!     curlx
         call sub3(vort(1,1,1,1,1),dwdx(1,1,1,1,2),
     $        dvdx(1,1,1,1,3),itmp)
!     curly
         call sub3(vort(1,1,1,1,2),dudx(1,1,1,1,3),
     $        dwdx(1,1,1,1,1),itmp)
      endif
!     curlz
      call sub3(vort(1,1,1,1,3),dvdx(1,1,1,1,1),dudx(1,1,1,1,2),itmp)

      return
      end subroutine user_stat_trnsv
c----------------------------------------------------------------------

      subroutine user_stat_compute(slvel,slp,tmpvel,tmppr,
     $              dudx,dvdx,dwdx,lnvar,npos,alpha,beta)

      implicit none
     
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
!      include 'SOLN_DEF'
!      include 'SOLN'
      include 'RTFILTER'                ! filtered forcing/energy
cc MA:      include 'INPUT_DEF'               ! if3d
      include 'INPUT'

!     Additional variables calculated by the user      

!     work arrays
      real slvel(LX1,LY1,LZ1,LELT,3)    ! reshuffled velocities 
      real slp(LX1,LY1,LZ1,LELT)        ! reshuffled pres. Mesh 1
      real tmpvel(LX1,LY1,LZ1,LELT,3)   ! reinitialize. Use this for custom variables
      real tmppr(LX1,LY1,LZ1,LELT)      ! uv ! reinitialize

      real dudx(LX1,LY1,LZ1,LELT,3) ! velocity derivatives; U
      real dvdx(LX1,LY1,LZ1,LELT,3) ! V
      real dwdx(LX1,LY1,LZ1,LELT,3) ! W

      integer npos                  ! position in STAT_RUAVG
      integer lnvar                 ! variable count
      real alpha                    ! Old time fraction
      real beta                     ! New time fraction


!    Template

!      call user_stat_trnsv(tmpvel,dudx,dvdx,dwdx,slvel)    ! transform if needed

!     Additional user defined variables
!-------------------------------------------------- 
!     include RTFILTER      
      if (if3d) then               ! #2D
      call stat_reshufflev(tmpvel(1,1,1,1,1),rtfx(1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,2),rtfy(1,1,1,1),NELV)
      call stat_reshufflev(tmpvel(1,1,1,1,3),rtfz(1,1,1,1),NELV)
!     copy
      else
            call opcopy(tmpvel(1,1,1,1,1),tmpvel(1,1,1,1,2),
     $      tmpvel(1,1,1,1,3),rtfx,rtfy,rtfz)
      endif

c------------------------------------------------------------ 
!     rtfx

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav1(tmpvel(1,1,1,1,1),npos,alpha,beta)
!npos==61
c------------------------------------------------------------ 
!     rtfy

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav1(tmpvel(1,1,1,1,2),npos,alpha,beta)
!npos==62
c------------------------------------------------------------ 
!     rtfz

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav1(tmpvel(1,1,1,1,3),npos,alpha,beta)
!npos==63
c------------------------------------------------------------ 
!     rtfx*u

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav2(tmpvel(1,1,1,1,1),slvel(1,1,1,1,1),
     $         npos,alpha,beta)
!npos==64
c------------------------------------------------------------ 
!     rtfy*v

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav2(tmpvel(1,1,1,1,2),slvel(1,1,1,1,2),
     $         npos,alpha,beta)
!npos==65
c------------------------------------------------------------ 
!     rtfz*w

      lnvar = lnvar + 1
      npos = lnvar

      call stat_compute_1Dav2(tmpvel(1,1,1,1,3),slvel(1,1,1,1,3),
     $         npos,alpha,beta)
!npos==66
c------------------------------------------------------------ 

      return
      end subroutine user_stat_compute          

c----------------------------------------------------------------------




