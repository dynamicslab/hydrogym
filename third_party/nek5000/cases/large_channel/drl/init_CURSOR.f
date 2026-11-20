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
        call build_owner_mask_wall
        NUMCTRL = 0 
        do ie=1,NEL
        do iface=1,nfaces
                xr=xm1(fmid(iface),1,1,ie) ! X coord for face mid point
                yr=ym1(fmid(iface),1,1,ie) ! Y coord for face mid point
                zr=zm1(fmid(iface),1,1,ie) ! Y coord for face mid point
                bcb=CBC(iface,ie,1)
                if (bcb.eq.'W  ') then
                ! YW Modified here to avoid overlapping
                ! call facindr(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                call facind_d(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
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
        ! include "NEKUSE"
        include "GEOM" !  The angle unx, uny, unz 
        include 'TOPOL'
        include 'PARALLEL'
        real xw,yw,zw
        real xct,yct,zct
        real snx,sny,snz
        real Ret,nu, h 
        parameter(Ret=267.0)
        parameter(nu=2.0e-5)
        parameter(h=1.0)
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
        ynorm = ypctrl*(1.0/Ret) ! compute the norm between ctrl and sensing point
        ! ynorm = -0.923879533
        if (NID.eq.0) print *, "[DRL] SENSING PLANE DISTANCE:",ynorm
        
        if (NUMCTRL.gt.0) then  ! Only works when wall point exists in this RANK
        do il=1,NUMCTRL
                ! Wall points coordinates
                xw = pos_agt(1,il)
                yw = pos_agt(2,il)
                zw = pos_agt(3,il)
                ! Only Y-dir changes 
                dyx = 0
                dyy = ynorm
                xct = xw + dyx
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
        real tolin ! The tolerence for interpolation 
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
          if(rwk(il,1).gt.10*tolin) then
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



      subroutine facind_d (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
c      ifcase in preprocessor notation, 
c     we avoid the boundary by using start index of 2
       KX1=1
       KY1=1
       KZ1=1
       KX2=NX
       KY2=NY
       KZ2=NZ
       IF (IFACE.EQ.1) KY2=1
       IF (IFACE.EQ.1) KX1=2
       IF (IFACE.EQ.1) KZ1=2

       IF (IFACE.EQ.2) KX1=NX
       IF (IFACE.EQ.2) KY1=2
       IF (IFACE.EQ.2) KZ1=2
      
       IF (IFACE.EQ.3) KY1=NY
       IF (IFACE.EQ.3) KX1=2
       IF (IFACE.EQ.3) KZ1=2
       
       IF (IFACE.EQ.4) KX2=1
       IF (IFACE.EQ.4) KY1=2
       IF (IFACE.EQ.4) KZ1=2

       IF (IFACE.EQ.5) KZ2=1
       IF (IFACE.EQ.5) KY1=2
       IF (IFACE.EQ.5) KX1=2

       IF (IFACE.EQ.6) KZ1=NZ
       IF (IFACE.EQ.6) KX1=2
       IF (IFACE.EQ.6) KY1=2

      return
      end

c--------------------------------------------------
        subroutine build_owner_mask_wall()
c Subroutine for building the agent_owner mask for the wall
c This will be used in the initialization of the Agent. 
c The ownership mask ensures no double-counting when computing spatial averages
c For ZNMF condition: each GLL node should be owned by exactly one element
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'NEKUSE'
            include 'DRL'
c=============================================
c       Define variable
c=============================================
            integer ie,iface,il,jl,kl,KX1,KX2,KY1,KY2,KZ1,KZ2
            integer ntot
            character*3 bcb
            real work_own(LX1,LY1,LZ1,LELT)
            real temp_own(LX1,LY1,LZ1,LELT)
c=============================================
c       Function
c=============================================
            
            ntot = lx1*ly1*lz1*lelt
            call rzero(work_own, ntot)
            call rzero(temp_own, ntot)
            
            if (NID.eq.0) then 
                print *, "[YW] Building ownership mask for ZNMF condition"
            endif

            ! Step 1: Mark all wall face nodes as potentially owned
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then 
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                
                do kl=KZ1,KZ2
                      do jl=KY1,KY2
                      do il=KX1,KX2
                      ! Mark all nodes on this wall face
                      temp_own(il,jl,kl,ie) = 1.0
                      enddo ! do il=KX1,KX2
                      enddo ! do jl=KY1,KY2
                enddo ! do kl=KZ1,KZ2
            endif ! if (bcb.eq.'W  ')
            enddo ! do iface=1,2*ndim
            enddo ! do ie=1,NELFLD(IFIELD)
            
            ! Step 2: Apply ownership rules to avoid double-counting
            call apply_ownership_rules_znmf(work_own, temp_own)
            
            ! Step 3: Copy to global array
            call copy(agent_own(1,1,1,1),work_own(1,1,1,1),ntot)
            
            ! Step 4: Synchronize ownership mask across MPI ranks
            call synchronize_ownership_mask()
            
            if (NID.eq.0) then
                print *, "[YW] Ownership mask built and synchronized"
            endif
            
            ! Debug: Print ownership pattern for verification
            if (NID.eq.0) then
                call print_ownership_pattern()
            endif
        end subroutine build_owner_mask_wall

c--------------------------------------------------
        subroutine apply_ownership_rules_znmf(work_own, temp_own)
c Apply ownership rules for ZNMF condition to avoid double-counting
c Key principle: Each GLL node should be owned by exactly one element
c This ensures correct spatial averages for mass conservation
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            real work_own(LX1,LY1,LZ1,LELT)
            real temp_own(LX1,LY1,LZ1,LELT)
            
            integer ie,iface,il,jl,kl,KX1,KX2,KY1,KY2,KZ1,KZ2
            integer ntot, nwall_elements
            character*3 bcb
            
            ntot = lx1*ly1*lz1*lelt
            call rzero(work_own, ntot)
            
            if (NID.eq.0) then
                print *, "[YW] Applying ZNMF ownership rules"
            endif
            
            ! Count wall elements for this rank
            nwall_elements = 0
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then
                  nwall_elements = nwall_elements + 1
            endif
            enddo
            enddo
            
            if (NID.eq.0) then
                print *, "[YW] Found ", nwall_elements, " wall elements on rank ", NID
            endif
            
            ! Apply ownership rules: Element with lower global ID wins for shared nodes
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then
            call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
            
            do kl=KZ1,KZ2
                  do jl=KY1,KY2
                  do il=KX1,KX2
                  if (temp_own(il,jl,kl,ie) .gt. 0.5) then
                        ! Interior nodes are always owned by this face
                        if ( (il.gt.KX1 .and. il.lt.KX2) .and.
     &                       (jl.gt.KY1 .and. jl.lt.KY2) .and.
     &                       (kl.gt.KZ1 .and. kl.lt.KZ2) ) then
                              work_own(il,jl,kl,ie) = 1.0
                        else
                              ! Boundary nodes: need sophisticated ownership based on node type
                              ! Determine if this is an edge node, corner node, or domain boundary node
                              call compute_boundary_node_ownership(il, jl, kl, KX1, KX2, KY1, KY2, KZ1, KZ2, 
     &                                                           iface, ie, work_own)
                        endif
                  endif
                  enddo ! do il=KX1,KX2
                  enddo ! do jl=KY1,KY2
            enddo ! do kl=KZ1,KZ2
            endif ! if (bcb.eq.'W  ')
            enddo ! do iface=1,2*ndim
            enddo ! do ie=1,NELFLD(IFIELD)
            
            if (NID.eq.0) then
                print *, "[YW] ZNMF ownership rules applied"
            endif
            
        end subroutine apply_ownership_rules_znmf

c--------------------------------------------------
        subroutine compute_boundary_node_ownership(il, jl, kl, KX1, KX2, KY1, KY2, KZ1, KZ2, iface, ie, work_own)
c Compute ownership for nodes on a wall face in 3D domain
c This handles edge nodes (weight=0.5) and corner nodes (weight=0.25) on the wall
c For 3D domain with one wall face, we need to check if nodes are at global domain boundaries
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer il, jl, kl, KX1, KX2, KY1, KY2, KZ1, KZ2, iface, ie
            real work_own(LX1,LY1,LZ1,LELT)
            
            real ownership_weight
            logical is_edge_x, is_edge_y, is_edge_z
            logical is_corner, is_edge, is_domain_boundary
            integer edge_count
            integer global_elem_id, global_x, global_y, global_z
            
            ! Determine if this node is at any edge of the face
            is_edge_x = (il.eq.KX1 .or. il.eq.KX2)
            is_edge_y = (jl.eq.KY1 .or. jl.eq.KY2)
            is_edge_z = (kl.eq.KZ1 .or. kl.eq.KZ2)
            
            ! Count how many directions are at edges
            edge_count = 0
            if (is_edge_x) edge_count = edge_count + 1
            if (is_edge_y) edge_count = edge_count + 1
            if (is_edge_z) edge_count = edge_count + 1
            
            ! Get global element position to check domain boundaries
            global_elem_id = lglel(ie)
            call get_global_domain_position_3d(ie, global_x, global_y, global_z)
            
            ! Check if this node is at global domain boundary
            is_domain_boundary = .false.
            if (is_edge_x .and. (global_x .eq. 1 .or. global_x .eq. xnel)) then
                  is_domain_boundary = .true.
            endif
            if (is_edge_z .and. (global_z .eq. 1 .or. global_z .eq. znel)) then
                  is_domain_boundary = .true.
            endif
            
            ! Determine ownership weight based on edge count and domain boundary
            if (edge_count .eq. 0) then
                  ! Interior node - fully owned by this face
                  ownership_weight = 1.0
                  is_corner = .false.
                  is_edge = .false.
            elseif (edge_count .eq. 1) then
                  ! Edge node - shared between 2 adjacent faces
                  if (is_domain_boundary) then
                        ! Domain boundary edge - shared between 2 elements
                        ownership_weight = 0.5
                  else
                        ! Internal edge - shared between 2 faces
                        ownership_weight = 0.5
                  endif
                  is_corner = .false.
                  is_edge = .true.
            elseif (edge_count .eq. 2) then
                  ! Corner node - shared between 4 adjacent faces
                  if (is_domain_boundary) then
                        ! Domain boundary corner - shared between 4 elements
                        ownership_weight = 0.25
                  else
                        ! Internal corner - shared between 4 faces
                        ownership_weight = 0.25
                  endif
                  is_corner = .true.
                  is_edge = .false.
            elseif (edge_count .eq. 3) then
                  ! Very rare case: node at corner of 3D element
                  ownership_weight = 0.125
                  is_corner = .true.
                  is_edge = .false.
            endif
            
            ! Apply ownership weight
            work_own(il,jl,kl,ie) = ownership_weight
            
            ! Debug output for edge and corner nodes
            if (is_edge .or. is_corner) then
            if (NID.eq.0) then
                print *, "[YW] Wall face node (", il, ",", jl, ",", kl, ") in element ", ie, " face ", iface
                print *, "  Global position: X=", global_x, ", Y=", global_y, ", Z=", global_z
                if (is_corner) then
                    print *, "  Type: CORNER (", edge_count, " edges), Weight: ", ownership_weight
                else
                    print *, "  Type: EDGE (", edge_count, " edges), Weight: ", ownership_weight
                endif
                if (is_domain_boundary) then
                    print *, "  *** DOMAIN BOUNDARY NODE ***"
                endif
            endif
            endif
            
        end subroutine compute_boundary_node_ownership

c--------------------------------------------------
        subroutine get_global_domain_position_3d(ie, global_x, global_y, global_z)
c Get the global domain position (X,Y,Z coordinates) of element ie in 3D domain
c This is needed to determine if nodes are at global domain boundaries
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer ie, global_x, global_y, global_z
            integer global_elem_id
            
            ! Get global element ID
            global_elem_id = lglel(ie)
            
            ! Convert global element ID to X-Y-Z coordinates
            ! Assuming elements are numbered in X-Y-Z order
            ! This depends on your mesh generation strategy
            global_z = ((global_elem_id - 1) / (xnel * ynel)) + 1
            global_y = ((global_elem_id - 1 - (global_z - 1) * xnel * ynel) / xnel) + 1
            global_x = mod(global_elem_id - 1 - (global_z - 1) * xnel * ynel - (global_y - 1) * xnel, xnel) + 1
            
            ! Ensure bounds
            if (global_x .lt. 1) global_x = 1
            if (global_x .gt. xnel) global_x = xnel
            if (global_y .lt. 1) global_y = 1
            if (global_y .gt. ynel) global_y = ynel
            if (global_z .lt. 1) global_z = 1
            if (global_z .gt. znel) global_z = znel
            
        end subroutine get_global_domain_position_3d

c--------------------------------------------------
        subroutine verify_ownership_mask_znmf()
c Verify that the ownership mask is properly constructed for ZNMF condition
c This helps debug ownership issues and ensures mass conservation
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer ie,iface,il,jl,kl,KX1,KX2,KY1,KY2,KZ1,KZ2
            integer ntot, owned_nodes, total_wall_nodes
            character*3 bcb
            real work_own(LX1,LY1,LZ1,LELT)
            
            ntot = lx1*ly1*lz1*lelt
            call copy(work_own, agent_own, ntot)
            
            owned_nodes = 0
            total_wall_nodes = 0
            
            ! Count owned nodes
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then
            call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
            do kl=KZ1,KZ2
            do jl=KY1,KY2
            do il=KX1,KX2
                  total_wall_nodes = total_wall_nodes + 1
                  if (work_own(il,jl,kl,ie) .gt. 0.01) then
                        ! Count nodes with any ownership (including fractional)
                        owned_nodes = owned_nodes + 1
                  endif
            enddo
            enddo
            enddo
            endif
            enddo
            enddo
            
            if (NID.eq.0) then
                print *, "[YW] ZNMF Ownership mask verification:"
                print *, "  - Total wall nodes: ", total_wall_nodes
                print *, "  - Nodes with ownership: ", owned_nodes
                if (total_wall_nodes .gt. 0) then
                    print *, "  - Coverage ratio: ", real(owned_nodes)/real(total_wall_nodes)
                endif
            endif
            
            ! For ZNMF with fractional ownership, we need to check total weight
            call verify_znmf_weight_distribution()
            
        end subroutine verify_ownership_mask_znmf

c--------------------------------------------------
        subroutine synchronize_ownership_mask()
c Synchronize ownership mask across MPI ranks to ensure consistency
c This is crucial for ZNMF condition when computing spatial averages
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer ntot
            
            ntot = lx1*ly1*lz1*lelt
            
            if (NID.eq.0) then
                print *, "[YW] Synchronizing ownership mask across MPI ranks"
            endif
            
            ! Use DSSUM to synchronize ownership mask across ranks
            ! This ensures shared nodes have consistent ownership
            call dssum(agent_own, lx1, ly1, lz1)
            
            if (NID.eq.0) then
                print *, "[YW] Ownership mask synchronization completed"
            endif
            
        end subroutine synchronize_ownership_mask

c--------------------------------------------------
        subroutine print_ownership_pattern()
c Print ownership pattern for debugging and verification
c This shows which nodes are owned by which faces
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer ie,iface,il,jl,kl,KX1,KX2,KY1,KY2,KZ1,KZ2
            integer ntot, owned_count, total_count
            character*3 bcb
            
            ntot = lx1*ly1*lz1*lelt
            owned_count = 0
            total_count = 0
            
            print *, "[YW] === Ownership Pattern Summary ==="
            
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                
                print *, "[YW] Element ", ie, " Face ", 
     &             iface, " (", bcb, ")"
                print *, "  Face indices: X[", KX1, ":", KX2, 
     &             "], Y[", KY1, ":", KY2, "], Z[", 
     &             KZ1, ":", KZ2, "]"
                
                do kl=KZ1,KZ2
                do jl=KY1,KY2
                do il=KX1,KX2
                      total_count = total_count + 1
                      if (agent_own(il,jl,kl,ie) .gt. 0.01) then
                            owned_count = owned_count + 1
                            print *, "Owned: (", il, ",",
     &             jl, ",", kl, ") = ", agent_own(il,jl,kl,ie)
                      else
                            print *, "Not owned: (", il, ",",
     &             jl, ",", kl, ") = 0.0"
                      endif
                enddo
                enddo
                enddo
            endif
            enddo
            enddo
            
            print *, "[YW] === Ownership Statistics ==="
            print *, "  Total nodes: ", total_count
            print *, "  Owned nodes: ", owned_count
            if (total_count .gt. 0) then
                print *, "  Ownership ratio: ", 
     &             real(owned_count)/real(total_count)
            endif
            print *, "[YW] ================================"
            
        end subroutine print_ownership_pattern

c--------------------------------------------------
        subroutine verify_znmf_weight_distribution()
c Verify ZNMF weight distribution for proper mass conservation
c This checks that the sum of ownership weights equals the expected total
            implicit none
            include 'SIZE'
            include 'TOTAL'
            include 'DRL'
            
            integer ie,iface,il,jl,kl,KX1,KX2,KY1,KY2,KZ1,KZ2
            integer ntot, total_wall_nodes
            character*3 bcb
            real work_own(LX1,LY1,LZ1,LELT)
            real total_weight, expected_weight
            real weight_1_0, weight_0_5, weight_0_25
            
            ntot = lx1*ly1*lz1*lelt
            call copy(work_own, agent_own, ntot)
            
            total_wall_nodes = 0
            total_weight = 0.0
            weight_1_0 = 0.0
            weight_0_5 = 0.0
            weight_0_25 = 0.0
            
            ! Analyze weight distribution
            do ie=1,NELFLD(IFIELD)
            do iface=1,2*ndim
            bcb = CBC(iface,ie,1)
            if (bcb.eq.'W  ') then
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                do kl=KZ1,KZ2
                do jl=KY1,KY2
                do il=KX1,KX2
                      total_wall_nodes = total_wall_nodes + 1
                      total_weight = total_weight + work_own(il,jl,kl,ie)
                      
                      ! Count nodes by weight type
                      if (abs(work_own(il,jl,kl,ie) - 1.0) .lt. 0.01) then
                            weight_1_0 = weight_1_0 + 1.0
                      elseif (abs(work_own(il,jl,kl,ie) - 0.5) .lt. 0.01) then
                            weight_0_5 = weight_0_5 + 1.0
                      elseif (abs(work_own(il,jl,kl,ie) - 0.25) .lt. 0.01) then
                            weight_0_25 = weight_0_25 + 1.0
                      endif
                enddo
                enddo
                enddo
            endif
            enddo
            enddo
            
            ! Expected weight: each node should contribute exactly 1.0 to the total
            expected_weight = real(total_wall_nodes)
            
            if (NID.eq.0) then
                print *, "[YW] === ZNMF Weight Distribution Analysis (Wall Face) ==="
                print *, "  - Total wall nodes: ", total_wall_nodes
                print *, "  - Total ownership weight: ", total_weight
                print *, "  - Expected total weight: ", expected_weight
                print *, "  - Weight balance: ", total_weight - expected_weight
                print *, "  - Weight distribution:"
                print *, "    * Weight 1.0 (interior): ", weight_1_0
                print *, "    * Weight 0.5 (edge): ", weight_0_5
                print *, "    * Weight 0.25 (corner): ", weight_0_25
                
                ! Critical ZNMF check
                if (abs(total_weight - expected_weight) .gt. 0.01) then
                    print *, "[WARNING] ZNMF weight balance may be violated!"
                    print *, "  - Expected: ", expected_weight, ", Got: ", total_weight
                else
                    print *, "[SUCCESS] ZNMF weight balance satisfied!"
                endif
                print *, "[YW] =========================================="
            endif
            
        end subroutine verify_znmf_weight_distribution
