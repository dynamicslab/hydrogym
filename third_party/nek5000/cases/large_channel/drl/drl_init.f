c==============================================
c DRL subroutines for INITIALIZATION 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================

c------------------------------------------------------------------
        subroutine drl_init
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include 'PARALLEL'
c=============================================
c       Function
c=============================================
        
        
                if (NID.eq.0) then
                        print *,"=============================="
                        print *, "[DRL] STATE Initialisation"
                        print *,"=============================="
                endif 
cc STEP 1: Preprocessing for Finding the Wall Points by coordinates and B.C type
c---------------------------------------------
                call register_wall_pts          ! FIND WALL Points by B.C 
                ! call make_wall_mask          ! FIND WALL Points by B.C 
c---------------------------------------------

c---------------------------------------------
c      Bubble Sorting for coordinates
c--------------------------------------------- 
c: YW It is redundant in practise 
c: BUT gives a clean results to check the implementations 
                call sort_wall_pts
c----------------------------------------------

cc STEP 2: Locating the sensing plane (y+=15)
c----------------------------------------------
                call interp_sensing_plane       ! Use u_tau for calculating y+, find the sensing plane
                call register_ctrl_pts          ! Interpolating the sensing plane on the mesh grid
c----------------------------------------------
                if (NID.eq.0) then
                        print *,"---------------------------"
                        print *, "[DRL] NODE INFO READY"
                        print *,"---------------------------"
                endif 
        
                if (NID.eq.0) then
                        print *,"=============================="
                        print *, "[DRL] FINISH INIT"
                        print *,"=============================="
                endif 
        return 
        end subroutine drl_init
c------------------------------------------------------------------

c------------------------------------------------------------------
        subroutine register_wall_pts
cc YW: A subroutine for finding the wall points 
cc We only care about the wall points with wall boundary conditions! 
c=============================================
c       Define variable
c=============================================

        implicit none 
        include "SIZE"
        include "TOTAL"
        include "NEKUSE"
cc YW:
        include "DRL"

        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        
        integer ie,iface,ix,iy,iz,lgi ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer idx
        integer gll_unique
        real xf,yf,zf
        real xr,yr,zr
        character*3 bcb

        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================
        nfaces = 2*NDIM
        NEL    = NELFLD(IFIELD)
        ! From MA: Face midpoints on each direction
        im = (1+nx1)/2
        jm = (1+ny1)/2
        km = (1+nz1)/2
        ! print *, "im,jm,km=",im,jm,kmz
        ! From MA: Grid-point number within element corresponding to each face
        fmid(4) =   1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x = -1
        fmid(2) = nx1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x =  1
        fmid(1) =  im + nx1*(  1-1) + nx1*ny1*( km-1) !  y = -1
        fmid(3) =  im + nx1*(ny1-1) + nx1*ny1*( km-1) !  y =  1
        fmid(5) =  im + nx1*( jm-1) + nx1*ny1*(  1-1) !  z = -1
        fmid(6) =  im + nx1*( jm-1) + nx1*ny1*(nz1-1) !  z =  1
c########################################
c     DRL initialisation
c########################################
cc YW: The idea is to find the wall points by using the range and the wall boundary condition
cc YW: FOR DRL, we only find the face mid points

        ! call build_owner_mask_wall
        ! Note: Using NEK5000's built-in velocity masks (v1mask, v2mask, v3mask) for ownership
        ! instead of custom agent_own array. This leverages NEK5000's proven ownership system.

        gll_unique = UPARAM(8)
        if (NID.eq.0) print *, "YW: GLL UNIQUE=",gll_unique

        NUMCTRL = 0 
        do ie=1,NEL
        do iface=1,nfaces
                xr=xm1(fmid(iface),1,1,ie) ! X coord for face mid point
                yr=ym1(fmid(iface),1,1,ie) ! Y coord for face mid point
                zr=zm1(fmid(iface),1,1,ie) ! Y coord for face mid point
                bcb=CBC(iface,ie,1)
                if (bcb.eq.'W  ') then
                ! YW Modified here to avoid overlapping
                if (gll_unique.eq.1) then
                call facindf(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                else
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                endif
                  do iz=KZ1,KZ2
                  do iy=KY1,KY2
                  do ix=KX1,KX2
                  xf=xm1(ix,iy,iz,ie)
                  yf=ym1(ix,iy,iz,ie)
                  zf=zm1(ix,iy,iz,ie)        
                  NUMCTRL = NUMCTRL +1 ! if meet condition
                  lgi=lglel(ie)
                  pos_agt(1,NUMCTRL)=xf
                  pos_agt(2,NUMCTRL)=yf
                  pos_agt(3,NUMCTRL)=zf
                  info_agt(1,NUMCTRL)=lgi         ! Global indicies
                  info_agt(2,NUMCTRL)=iface       ! Number of face
                  info_agt(3,NUMCTRL)=ix         ! Number of x ploy
                  info_agt(4,NUMCTRL)=iy ! Number of y poly 
                  info_agt(5,NUMCTRL)=iz ! Number of z poly 
cc YW: Write the RANK info, for further interpolation
                  proc(NUMCTRL)=NID              
                  enddo ! do ix = KX1, KX2 
                  enddo ! do iy = KY1, KY2 
                  enddo ! do iz = KZ1, KZ2
                  CBC(iface,ie,1) = "v  "
                endif ! If bcb.eq."W  "
        enddo ! iface = 1, nfaces
        enddo ! ie = 1, NEL
        
        ! some field = element idama
        ! dssum_somefield(max)
        ! if elemenid .eq. max II am gll owner


        if (NID.eq.0) print *, "YW: FIND WALL PTS"

c$$$ TEST: Write down all the walll points that we have found
#ifdef YWDEBUG
        if (NUMCTRL.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  "
        do ilx = 1,NUMCTRL
                write(10001,*) proc(ilx),
     $         (pos_agt(ily,ilx), ily=1,NDIM), 
     $         (info_agt(ily,ilx), ily=1,5) 
        enddo
        close(10001)
        endif
        ! call outpost(xwf,ywf,zwf,zwf,zwf,'wpt')
#endif
c$$$ TEST END
        if (NUMCTRL.gt.TOTCTRL) then
                print *, "NUMCTRL=",NUMCTRL
                print *, "YW: ERROR AT NID=",NID,"BEYOND TOTCTRL"
                call exit
        endif
        return 
        end subroutine register_wall_pts 
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine sort_wall_pts
cc YW; A subroutine for sort out the wall points through descending order
cc Boubble sort Algorithm
cc We do this before we interpolate the sensing points
c=============================================
c       Define variable
c=============================================

        implicit none 
        include 'SIZE'
        include "NEKUSE"
        include "PARALLEL"
        include "DRL" ! NUMCTRL, pos_agt 

        ! Parameter
        real jx, jy ,jz ! For the second loop
        real jx1, jy1 ,jz1 ! For the second loop
        real vdm     ! Dummy value
        integer sid  ! A dummy variable for inidicies
        
        integer il,jl,kl ! Loop 
        logical swaped           ! if swaped happend
        integer totpts,totfeat,tot_ctrl ! for sake of copy
        integer iglsum 
        !For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================

        if (NUMCTRL.gt.0) then 
        
! step 1: Bubble sort 
!-------------------------------------
        ! Bubble sort         
        do il = 1, NUMCTRL-1
        do jl = 1, NUMCTRL-1
                ! compare the x-coordinate
                jx = pos_agt(1,jl)
                jx1 = pos_agt(1,jl+1)
                swaped =.FALSE.
                ! We sort by ascent order, i.e. large value at last
                ! Rearrange the array
                if (jx.gt.jx1) then 
                        ! Exchange coordinates info
                        do kl=1,LDIM
                        vdm=pos_agt(kl,jl)
                        pos_agt(kl,jl)=pos_agt(kl,jl+1)
                        pos_agt(kl,jl+1)=vdm
                        enddo
                        
                        ! Exchange ieg, iface, ix,iy,iz
                        do kl=1,nfeat
                        sid=info_agt(kl,jl)
                        info_agt(kl,jl)=info_agt(kl,jl+1)
                        info_agt(kl,jl+1)=sid
                        enddo
                        
                        ! Exchange NID info
                        sid = proc(jl)
                        proc(jl)=proc(jl+1)
                        proc(jl+1)=sid

                        ! Switch the flag
                        swaped = .TRUE.
                else
                        swaped = .FALSE.                        
                endif
                if (.NOT.swaped) continue
        enddo ! jl = 1, NUMCTRL-1 
        enddo ! il = 1, NUMCTRL-1
        ! print *,"INDICIES:", indices
                
c$$$ TEST: Write down all the walll points that we have found
#ifdef YWDEBUG
        write(str,"(i4.4)") NID
        open(10001,file="sortedWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "iel  ", "iface  ",
     $                  "ix  ", "iy  ", "iz  "
        do ilx = 1,NUMCTRL
                write(10001,*) proc(ilx),
     $         (pos_agt(ily,ilx), ily=1,NDIM), 
     $         (info_agt(ily,ilx), ily=1,5) 
        enddo
        close(10001)
#endif 
c$$$ TEST END
        endif ! if (NUMCTRL.gt.0) 
        
        tot_ctrl = iglsum(NUMCTRL,1)
        
        if (NID.eq.0) print *, "[DRL] STATE SORTING NUMCTRL=",tot_ctrl

        do il=1,TOTCTRL
        call icopy(iptctl(il),info_agt(1,il),1)
        enddo 

        
        return 
        end subroutine sort_wall_pts
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine interp_sensing_plane
cc YW:  A subroutine for interploation of the sensing points based on specificed y+ 
cc      We use the current wall point (xw,yw,z) to get (xc,yc,z)
cc      xc = xw + dx; yc = yw+dy
c=============================================
c       Define variable
c=============================================

        implicit none 
        include "SIZE"
        include "INPUT"
        include "DRL"
        include "NEKUSE"
        include "GEOM" !  The angle unx, uny, unz 
        include 'TOPOL'
        include 'PARALLEL'
        real xw,yw,zw
        real xct,yct,zct
        real snx,sny,snz
        real Ret,ypctrl ! Re_tau and prescribed sensing plane
        real utaux,ynorm,dyy,dyx
        integer iel, f                     ! For find the face normal 
        integer ie, iface, ix, iy, iz ! indices 
        integer il, jl, kl            ! Iteration
        ! For interploation
        integer nfail 
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================
        ! Calculate viscous velocity u_tau
        !YW: I pre-determine this value because the resolution is very low
        ! The actual value should be -0.917xxxx
        ypctrl=UPARAM(3)
        Ret=UPARAM(4)
        ! The wall-unit distance. 
        ynorm = ypctrl*(1.0/Ret) ! compute the norm between ctrl and sensing point
        if (NID.eq.0) then
           print *, "[DRL]: Retau, yplus, ys",Ret,ypctrl,ynorm
        endif 
        
        if (NUMCTRL.gt.0) then  ! Only works when wall point exists in this RANK
        do il=1,NUMCTRL
                ! Wall points coordinates
                xw = pos_agt(1,il)
                yw = pos_agt(2,il)
                zw = pos_agt(3,il)
                ! Only Y-dir changes 
                dyy = ynorm
                xct = xw 
                yct = yw + dyy
                zct = zw
                ! Store the position into array
                pos_obs(1,il) = xct
                pos_obs(2,il) = yct
                pos_obs(3,il) = zct

        enddo 
        endif ! IF NUMCTRL.gt.0 

#ifdef YWDEBUG
c$$$ TEST: Write Sensing Points
        if (NUMCTRL.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRb.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,NUMCTRL
                write(10001,*) iptctl(ilx),
     $         (pos_obs(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif 
c$$$ TEST END

        return 
        end subroutine interp_sensing_plane 
c--------------------------------------------------------------------


c--------------------------------------------------------------------
       subroutine register_ctrl_pts 
cc: Defining the sensing plane points and registered on the mesh
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "DRL"
        include 'PARALLEL'
        include 'GEOM'
        include 'NEKUSE'
        integer nfail
        ! real tolin ! The tolerence for interpolation 
        integer nmsh, nelm, ih
        integer il,jl,kl ! Iteration 
        integer iglsum
        
        integer ih_intp(2,20)
        common /intp_h/ ih_intp
        
        ! global memory access
        integer nidd,npp,nekcomm,nekgroup,nekreal
        common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal
        integer ntot,npt_max, nxf, nyf, nzf
        integer nctlpts
        ! parameter(nctlpts=2*TOTCTRL)
        real ltim
        real tol, bb_t            ! interpolation tolerance and relative size to expand bounding boxes by
        
        ! The problem is to make this bb_t larger for more inflation
        parameter (tol = 5.0E-13, bb_t = 0.1)
        integer totpts,totfail

        
        ! Interp initialization
        integer iwk_tot(2*TOTCTRL,3)
        real    rwk_tot(2*TOTCTRL,3)
        real    crd_tot(3,2*TOTCTRL)
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

c SetUp for the interploation
c----------------------------
        ! nctlpts=2*TOTCTRL

      ! initialise findpts
        ntot=lx1*ly1*lz1*lelt 
        npt_max=256
        nxf=2*lx1 ! fine mesh for bb-test
        nyf=2*ly1
        nzf=2*lz1
        

        ! call interp_free(inth_hpts1)
        call fgslib_findpts_setup(inth_hpts1,nekcomm,npp,ldim,
     $     xm1,ym1,zm1,lx1,ly1,lz1,lelt,nxf,nyf,nzf,bb_t,ntot,ntot,
     $     npt_max,tol)
     
        ! inth_hpts2 = inth_hpts1
       
c       Step 2: Register them on the map
        call fgslib_findpts(inth_hpts1,
     &   iwk(1,1),1,                      ! $ rcode 1
     &   iwk(1,3),1,                      ! $ proc,1
     &   iwk(1,2),1,                      ! $ elid, 1
     &   rwk(1,2),LDIM,                   ! $ rst, ndim
     &   rwk(1,1),1,                      ! $ dist,1
     &   pos_obs(1,1),ldim,                   ! $ x
     &   pos_obs(2,1),ldim,                   ! $ y
     &   pos_obs(3,1),ldim,NUMCTRL)           ! $ z     

! #ifdef YWDEBUG
!         print *,NID,"[DEBUGG ] FINISH FINDPTS!"
! #endif
   
c Examine the deviations after interoplation
c----------------------
        nfail = 0 
        ! do il=1,TOTCTRL
        do il=1,NUMCTRL
        ! check return code
        if(iwk(il,1).eq.1) then
          if(rwk(il,1).gt.10*tol) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     pos_obs(1,il),pos_obs(2,il),pos_obs(3,il),rwk(il,1)
          endif
        elseif(iwk(il,1).eq.2) then
          nfail = nfail + 1
          if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        pos_obs(1,il),pos_obs(2,il),pos_obs(3,il)
        endif
        enddo
        
        totfail=iglsum(nfail,1)
        if (NID.eq.0) print *,"[DRL] Sensing: Failed Interp:",totfail


c Initialise the Velocity tensor 
cc Idea is to mask the tensor with a constant but pretty large value 
cc So it can become a condition when imposing the velocity 
c-------------------------------------
        ! totpts =LX1*LY1*LZ1*LELT*6
        ! do il=1,totpts
        !         Vnfluct(il,1,1,1,1) = dumi
        ! enddo 
c-----------------------
#ifdef YWDEBUG
c$$$ TEST: Write down all the sensing points that we have found
        if (NUMCTRL.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRL.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,NUMCTRL
                write(10001,*) NID,
     $         (pos_obs(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif
c$$$ TEST END
c-----------------------
        return 
        end subroutine register_ctrl_pts
c--------------------------------------------------------------------

      subroutine facindf (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
c      ifcase in preprocessor notation, 
c     we avoid the boundary by using start index of 1 and end index of Ni-1
c     where Ni is the number of points in the i-direction
c     This is used for masking the agent of DRL 
       KX1=1
       KY1=1
       KZ1=1
       KX2=NX-1
       KY2=NY-1
       KZ2=NZ-1

       IF (IFACE.EQ.1) KY2=1
       IF (IFACE.EQ.1) KY1=1

       IF (IFACE.EQ.2) KX1=NX
       IF (IFACE.EQ.2) KX2=NX
      
       IF (IFACE.EQ.3) KY1=NY
       IF (IFACE.EQ.3) KY2=NY
       
       IF (IFACE.EQ.4) KX1=1
       IF (IFACE.EQ.4) KX2=1

       IF (IFACE.EQ.5) KZ1=1
       IF (IFACE.EQ.5) KZ2=1

       IF (IFACE.EQ.6) KZ1=NZ
       IF (IFACE.EQ.6) KZ2=NZ

      return
      end

c--------------------------------------------------------------------
      subroutine build_owner_mask_wall()
        implicit none
        include 'SIZE'
        include 'TOTAL'
        include 'DRL'
c=============================================
c       Define variable
c=============================================
        real own(lx1,ly1,lz1,lelt)
        integer e,f,i,j,k,i1,i2,j1,j2,k1,k2
        logical is_wall, tI, tJ, tK, edgeI, edgeJ, edgeK
c=============================================
c       Function
c=============================================
        call rzero(own, lx1*ly1*lz1*nelv)

        do e=1,nelv
          do f=1,2*ndim
            is_wall = (cbc(f,e,1).eq.'W  ')   ! your control wall(s)
            if (.not.is_wall) cycle
            call facind(i1,i2,j1,j2,k1,k2,lx1,ly1,lz1,f)
            ! Which directions vary on the face? (two tangential, one fixed)
            tI = (i2.gt.i1)
            tJ = (j2.gt.j1)
            tK = (k2.gt.k1)

            do k=k1,k2
              do j=j1,j2
                do i=i1,i2
                  ! interior-of-face is always owner
!                   if ( (.not.tI .or. (i.gt.i1 .and. i.lt.i2)) .and.
!      &               (.not.tK .or. (k.gt.k1 .and. k.lt.k2)) ) then
                   if ((i.gt.i1 .and. i.lt.i2) .and. 
     &                (k.gt.k1 .and. k.lt.k2)) then
                    own(i,j,k,e) = 1.0
                   else
                    ! on edges/corners: owner only if sitting on the "low" side in every
                    ! tangential direction that is at an edge
                    edgeI = tI .and. (i.eq.i1 .or. i.eq.i2)
                    edgeJ = tJ .and. (j.eq.j1 .or. j.eq.j2)
                    edgeK = tK .and. (k.eq.k1 .or. k.eq.k2)

    !                 if ( ( .not.edgeI .or. (i.eq.i1) ) .and.
    !  &             ( .not.edgeK .or. (k.eq.k1) ) ) then
                        if (i.eq.1 .or. k.eq.1) then
                          own(i,j,k,e) = 1.0     ! “1 wins” in every tangential dir
                        elseif (i.eq.NX1 .or. k.eq.NZ1) then
                          own(i,j,k,e) = 1.0     ! “N loses”
                        endif ! end if ( .not.edgeI .or. (i.eq.i1) ) .and.
                  endif ! end if (.not.tI .or. (i.gt.i1 .and. i.lt.i2)) .and.
                enddo ! end do i=i1,i2
              enddo ! end do j=j1,j2
            enddo ! end do k=k1,k2
          enddo ! end do f=1,2*ndim
        enddo ! end do e=1,nelv
      call copy(agent_own(1,1,1,1), own(1,1,1,1), lx1*ly1*lz1*nelv)
      end subroutine build_owner_mask_wall
