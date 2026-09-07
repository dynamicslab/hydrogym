c==============================================
c Collection of subroutines for implementing the Opposition control 
c Yuning Wang 
c==============================================



c------------------------------------------------------------------
        subroutine oppo_ctrl 
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
        
        if (ISTEP.eq.0) then
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
                print *,"=============================="
                print *, "YW: Complete OppoCtrl Initialisation"
                print *,"=============================="
                endif 

cc STEP 3: Computing Fluctuation and Actuating on the wall 
c-----------------------------------------------
        else ! When the simulation is running: 
                call wall_aft_vel               ! YWDEBUG the actutation on the wall 
                call sensing_pts_compute        ! Get sensing plane velocity and compute spatial fluctutations
        
        
        endif ! if (ISTEP.eq.0)

        return 
        end subroutine oppo_ctrl
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
        include "OPPO_CTL"

        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        
        integer ie,iface,ix,iy,iz ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer idx
        real xf,yf,zf
        real xr,yr,zr
        character*3 bcb

c$$$        ! YWDEBUG ONLY 
c$$$        ! For a output file for YWDEBUG, write down a MASK to show if we find the actual wall points
c        real xwf(LX1,LY1,LZ1,LELT)
c        real ywf(LX1,LY1,LZ1,LELT),zwf(LX1,LY1,LZ1,LELT)
c        integer ntot
c$$$        
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

c$$$     YWDEBUG 
c        ntot = LX1*LY1*LZ1*LELT
c        call rzero(xwf,ntot)
c        call rzero(ywf,ntot)
c        call rzero(zwf,ntot)
c$$$
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
c     Opposition control initialisation
c########################################
cc YW: The idea is to find the wall points by using the range and the wall boundary condition
        numctrl = 0 
        do ie=1,NEL
        do iface=1,nfaces
                xr=xm1(fmid(iface),1,1,ie) ! X coord for face mid point
                yr=ym1(fmid(iface),1,1,ie) ! Y coord for face mid point
                bcb=CBC(iface,ie,1)
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                do iz=KZ1,KZ2
                do iy=KY1,KY2
                do ix=KX1,KX2
                xf=xm1(ix,iy,iz,ie)
                yf=ym1(ix,iy,iz,ie)
                zf=zm1(ix,iy,iz,ie)        
                if (bcb.eq.'W  ') then
                numctrl = numctrl +1 ! if meet condition
                crdwall(1,numctrl)=xf
                crdwall(2,numctrl)=yf
                crdwall(3,numctrl)=zf
cc YW: I write the global grid information here for find index in USERBC
                grdwall(1,numctrl)=ie         ! Local indicies
                grdwall(2,numctrl)=iface      ! Number of face
                grdwall(3,numctrl)=ix! Number of x ploy
                grdwall(4,numctrl)=iy ! Number of y poly 
                grdwall(5,numctrl)=iz ! Number of z poly 
cc YW: Write the RANK info, for further interpolation
                proc(numctrl)=NID
cc YW: Change the BC to Dirichlet                 
                CBC(iface,ie,1) = "v  "

c$$$            YWDEBUG fill the found point by a non-zero value 
c                xwf(ix,iy,iz,ie) = 10
c                ywf(ix,iy,iz,ie) = 10
c                zwf(ix,iy,iz,ie) = 10
c$$$
                endif ! If bcb.eq."W  " 
                enddo ! do ix = KX1, KX2 
                enddo ! do iy = KY1, KY2 
                enddo ! do iz = KZ1, KZ2
        enddo ! iface = 1, nfaces
        enddo ! ie = 1, NEL

        if (NID.eq.0) print *, "YW: FIND WALL PTS"

c$$$ YWDEBUG: Write down all the walll points that we have found
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdwall(ily,ilx), ily=1,NDIM), 
     $         (grdwall(ily,ilx), ily=1,5) 
        enddo
        close(10001)
        endif
        ! call outpost(xwf,ywf,zwf,zwf,zwf,'wpt')
c$$$ YWDEBUG END

        if (numctrl.gt.totctrl) then
                print *, "NUMCTRL=",numctrl
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
        include "OPPO_CTL" ! numctrl, crdwall 

        ! Parameter
        real jx, jy ,jz ! For the second loop
        real jx1, jy1 ,jz1 ! For the second loop
        real vdm     ! Dummy value
        integer sid  ! A dummy variable for inidicies
        
        integer il,jl,kl ! Loop 
        logical swaped           ! if swaped happend
        integer totpts, totfeat ! for sake of copy

        !For YWDEBUG 
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
                jx = crdwall(1,jl)
                jx1 = crdwall(1,jl+1)
                swaped =.FALSE.
                ! We sort by ascent order, i.e. large value at last
                ! Rearrange the array
                if (jx.gt.jx1) then 
                        ! Exchange coordinates info
                        do kl=1,LDIM
                        vdm=crdwall(kl,jl)
                        crdwall(kl,jl)=crdwall(kl,jl+1)
                        crdwall(kl,jl+1)=vdm
                        enddo
                        
                        ! Exchange ieg, iface, ix,iy,iz
                        do kl=1,nfeat
                        sid=grdwall(kl,jl)
                        grdwall(kl,jl)=grdwall(kl,jl+1)
                        grdwall(kl,jl+1)=sid
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
        
        print *,"NID=",NID,"WALL PTS=",numctrl,"YW: Bubble Sorting END"
        
c$$$ YWDEBUG: Write down all the walll points that we have found
#ifdef YWDEBUG
        write(str,"(i4.4)") NID
        open(10001,file="sortedWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "iel  ", "iface  ",
     $                  "ix  ", "iy  ", "iz  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdwall(ily,ilx), ily=1,NDIM), 
     $         (grdwall(ily,ilx), ily=1,5) 
        enddo
        close(10001)
#endif 
c$$$ YWDEBUG END
        endif ! if (numctrl.gt.0) 

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
        include "OPPO_CTL"
        ! include "NEKUSE"
        include "GEOM" !  The angle unx, uny, unz 
        include 'TOPOL'
        include 'PARALLEL'
        real xw,yw,zw
        real xct,yct,zct
        real snx,sny,snz
        real Ret,nu, h 
        real utaux,ynorm,dyy,dyx
        integer iel, f                     ! For find the face normal 
        integer ie, iface, ix, iy, iz ! indices 
        integer il, jl, kl            ! Iteration
        ! For interploation
        integer nfail 
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================

        if (numctrl.gt.0) then  ! Only works when wall point exists in this RANK
        
        do il=1,numctrl
                ! Wall points coordinates
                xw = crdwall(1,il)
                yw = crdwall(2,il)
                zw = crdwall(3,il)

                nu    = 2e-5 
                Ret   = 180.0 
                h     = 1.0
                utaux  = Ret * nu / h
                ynorm = ypctrl*nu/utaux ! compute the norm between ctrl and sensing point
                
                ! Only Y-dir changes 
                dyx = 0
                dyy = ynorm
                
                
                xct = xw + dyx
                yct = yw + dyy
                zct = zw
                
                ! Store the position into array
                crdctl(1,il) = xct
                crdctl(2,il) = yct
                crdctl(3,il) = zct
        enddo 
        endif ! IF NUMCTRL.gt.0 

#ifdef YWDEBUG
c$$$ YWDEBUG: Write Sensing Points
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRb.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdctl(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif 
c$$$ YWDEBUG END

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
        include "OPPO_CTL"
        include 'PARALLEL'
        integer nfail
        real tolin ! The tolerence for interpolation 
        integer nmsh, nelm, ih
        integer il,jl,kl ! Iteration 
        integer ih_intp(2,20)
        
        integer totpts
        real dumi 
        parameter(dumi=100.0)
        common /intp_h/ ih_intp
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1


c=============================================
c       Function
c=============================================
        ! Register sesning points onto the mesh 
        ! The points are stored in crdctl 


c SetUp for the interploation !
c----------------------------
        tolin = 1e-2
        nmsh  = LX1 -1 
        nelm  = LELT
        call interp_setup(ih,tolin,nmsh,nelm)
        
        inth_hpts1 = ih_intp(1,ih)
        inth_hpts2 = ih_intp(2,ih)

c Interpolation, the same procedure in time-series 
c-----------------------
        call fgslib_findpts(inth_hpts2,
     &     iwk(1,1),1,
     &     iwk(1,3),1,
     &     iwk(1,2),1,
     &     rwk(1,2),LDIM,
     &     rwk(1,1),1,
     &     crdctl(1,1),1,
     &     crdctl(2,1),1,
     &     crdctl(3,1),1,totctrl)

c Examine the deviations after interoplation
c----------------------
        print *, "NID=",NID,'Checking Interpolation'
        nfail = 0 
        do il=1,totctrl
        ! check return code
        if(iwk(il,1).eq.1) then
          if(rwk(il,1).gt.10*tolin) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     crdctl(1,il),crdctl(2,il),crdctl(3,il),rwk(il,1)
          endif
        elseif(iwk(il,1).eq.2) then
          nfail = nfail + 1
          if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        crdctl(1,il),crdctl(2,il),crdctl(3,il)
        endif
        enddo
        print *, "NID=",NID,'NFAIL=',nfail

        if (NID.eq.0) print *,"YW: SENSING PLANE interpolated!"


c Initialise the Velocity tensor 
cc Idea is to mask the tensor with a constant but pretty large value 
cc So it can become a condition when imposing the velocity 
c-------------------------------------
        totpts =LX1*LY1*LZ1*LELT*6
        do il=1,totpts
                Vnfluct(il,1,1,1,1) = dumi
        enddo 


c-----------------------
#ifdef YWDEBUG
c$$$ YWDEBUG: Write down all the sensing points that we have found
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRL.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdctl(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif
c$$$ YWDEBUG END
c-----------------------
        return 
        end subroutine register_ctrl_pts
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine sensing_pts_compute
cc 1.Get the velocity component at the sensing plane points and 
cc 2.Compute the sptial fluctuation compoenets
cc 3.Find quantities on y+=15
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        
        include "OPPO_CTL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        include 'PARALLEL'
        
        ! velocity field solution
        real ctrlV(LX1,LY1,LZ1,LELT)
        
        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array

        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
        ! Communication to save memory 
        ! COMMON / CTRL / ctrlV

c=============================================
c       Function
c=============================================

c---------------------------------
cc STEP 1: Copy velocity arries from solution
cc YW: We need vx and vy for projecting onto the V 
        call normal_V_project(ctrlV)
c----------------------------------


c----------------------------------
cc: YWDEBUG ONLY:
cc YW: I set a extra step to check the velocity on sensing plane before computing fluctuation 
        call vel_pts_exam(ctrlV)
c----------------------------------


c----------------------------------
cc STEP 2: Compute z-avg and substract the mean value from vel
        call vel_fluct_compute(ctrlV)
c----------------------------------


c----------------------------------
cc STEP 3: Interpolating the quantities for sensing points
        call vel_pts_find(ctrlV)
c----------------------------------

c----------------------------------
cc Print in the output 
        if (NIO.eq.0) then  
        print *, "----------------------------------------"
        print *, "Opposition Control: VEL FLUCT COMPUTED"
        print *, "---------------------------------------"
        endif 
c----------------------------------

        return 
        end subroutine sensing_pts_compute
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine normal_V_project(ctrlV)
cc: Project the x-dir and y-dir velocity from Cartesian coordinates onto the Wall-normal velocity 
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "NEKUSE"
        include "TOTAL" ! Use "GEOM", "SOLN", "TOPOL" and NELFLD 

        ! Wall-normal Velocity arrary
        real ctrlV(LX1,LY1,LZ1,LELT)
        
        ! Iteration indices
        integer ie,iface,ix,iy,iz ! Iteration for all elements 
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer f        ! f = eface1(iface)
        integer ntot
        ! For projection 
        real vxi,vyi ! The un-projected vx vy  
        real snx,sny     ! The face normal: sin(theta)
        real vux,vun ! The projection of vx and vy and the normal velocity  
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

        ! Compute the Projection
        ctrlV = vy

c=============================================
c       YWDEBUGing
c=============================================
c$$$ YWDEBUG 
c        !Write down the face normal 
        ! if (ISTEP.eq.2) then
        ! call outpost(body_cos,body_sin,ctrlV,pr,t,'ang')
        ! if (NID.eq.0) print *,"At",ISTEP,"YW: WIRTE ANGLE FOR YWDEBUG"
        ! endif 

c$$$ YWDEBUG End

        return 
        end subroutine normal_V_project

c--------------------------------------------------------------------
        subroutine vel_fluct_compute(ctrlV)
cc: Leverage the subroutine "z_distribute(u)" to compute the average along z_dir  
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "TSTEP"
        include "GEOM"
        include "WZ"
        include "ZPER"
        include "SOLN"

        integer il,jl,kl,ll,fl ! Iteration
        integer nxy, ntot, nxyz ! Size of array

        ! Array used for computing the Mean 
        real ctrlV(LX1,LY1,LZ1,LELT),
     $   avgVZ(LX1,LY1,xnel,ynel), velV(LX1,LY1,LZ1,LELT), ! Z-average and the reshaped field 
     $   avgVX(LY1,LZ1,ynel,znel) ! X-average 
        ! Element in the array, calculate explictly 
        real ufluct, uorg, umean
        
        ! Arrary for YWDEBUG
        ! real w1r(LX1,LY1,LZ1,LELT), Vcpy(LX1,LY1,LZ1,LELT)
        ! real Vfluct(LX1,LY1,LZ1,LELT)
        ! real w1copy(LX1,LY1,xnel,ynel),ws1(LX1,LY1,LZ1,LELT)
        ! COMMON / CTRL / w1copy, ws1

        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
        nxyz  = LX1*LY1*LZ1
        ntot  = nxyz*LELT
        icount = 0
c=============================================
c       Function
c=============================================

c-----------------------------------------------
c: Step 1: copy the current array into a new array for computing the Mean
c-----------------------------------------------
        ! Make a copy of wall-normal velocity 
        call rzero(velV,ntot)
        call copy(velV(1,1,1,1),ctrlV(1,1,1,1),ntot)
        ! For YWDEBUG, copy another Wall-normal vel 
        ! call copy(Vcpy(1,1,1,1),velV(1,1,1,1),ntot) 
        

c-----------------------------------------------
c: Step 2: Compute the spatial Mean 
c-----------------------------------------------
        
        ! Z-dir average
        call z_averaging(velV,avgVZ) ! Modified Avg Subroutine 
        call z_avg_reshape(velV,avgVZ) ! Now the avgV is transposed and pass the value to velV
        ! print *, "Z-AVG Computed"
        ! X-dir average 
        call x_averaging(velV,avgVX) ! Now velV contains the z-average
        call x_avg_reshape(velV,avgVX) ! Reshape the 2D avgVX to 3D field 
        ! print *, "X_AVG Computed"
        ! Add FOR YWDEBUG 
        ! call z_weight_reshape(w1r,w1copy) ! Reshape the un-uniformed weight along z-dir to check the reuslts 
c-----------------------------------------------

c-----------------------------------------------
c: Step 3: Substract the Mean, writting it explicitly to check 
c-----------------------------------------------
        do ll=1,LELT
        do kl=1,LZ1 
        do jl=1,LY1
        do il=1,LX1               
        ctrlV(il,jl,kl,ll)=ctrlV(il,jl,kl,ll)-velV(il,jl,kl,ll)
        enddo
        enddo
        enddo
        enddo  ! do ll=1,LELT

        ! call copy(Vfluct(1,1,1,1),ctrlV(1,1,1,1),ntot) 

        ! For YWDEBUG, write down Org, Fluct, and Avg
        ! call outpost(Vcpy,Vfluct,velV,pr,t,'avg')
        ! call outpost(ws1,w1r,velV,pr,t,'wz1')

c-----------------------------------------------
        return
        end subroutine vel_fluct_compute
c--------------------------------------------------------------------


c-----------------------------------------------------------------------
        subroutine z_averaging(velV,avgV)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        include 'OPPO_CTL'

        ! YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to aviod error
        real avgV(LX1,LY1,xnel,ynel), velV(LX1,LY1,LZ1,LELT),
     $   w1(LX1,LY1,xnel,ynel),w2(LX1,LY1,xnel,ynel)
        
        ! This is a copy of w1 for YWDEBUG 
        ! real w1copy(LX1,LY1,xnel,ynel),ws1(LX1,LY1,LZ1,LELT)
        ! COMMON / CTRL / w1copy, ws1

        ! Iterator 
        integer i,j,k, il, jl 
        integer e,eg,ex,ey,ez,mxy
        real umean, dz2

c=============================================
c       Function 
c=============================================
        dz2 = 1.0  !  Assuming uniform in "z" direction

        mxy = LX1*LY1*xnel*ynel
        call rzero(avgV,mxy)
        call rzero(w1,mxy)
        call rzero(w2,mxy)

c Computing the weighted average along z-dir 
c--------------------------------------
        do e=1,lelt
        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,nelz)
        do j=1,ly1
        do i=1,lx1
        do k=1,lz1
        avgV(i,j,ex,ey)=avgV(i,j,ex,ey)
     $              +dz2*wzm1(k)*velV(i,j,k,e)

        w1(i,j,ex,ey)=w1(i,j,ex,ey)+dz2*wzm1(k)
cc  YW: YWDEBUG, make a copy of w1 
        ! w1copy(i,j,ex,ey)=w1(i,j,ex,ey)
        ! ws1(i,j,k,e)=wzm1(k)
        enddo
        enddo
        enddo
        enddo ! do e=1,nelt 
c--------------------------------------


c-------------------------------------
c For global sum-up, communication
c-------------------------------------
        call gop(avgV,w2,'+  ',mxy)
        call gop(w1,w2,'+  ',mxy)
c-------------------------------------

c-------------------------------------
c Normalisation by the weight array
c-------------------------------------
        do i=1,mxy
        avgV(i,1,1,1)=avgV(i,1,1,1)/w1(i,1,1,1)   ! Normalize
        enddo 
c-------------------------------------

c-------------------------------------
        return
        end subroutine z_averaging
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine z_avg_reshape(velV,avgV)
c Reshape the array of averaged value back to the regular shape of velocity array
c=============================================
c       Define variable
c=============================================
        implicit none
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        include 'OPPO_CTL'
cc YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to implement z-avg
        real avgV(LX1,LY1,xnel,ynel), velV(LX1,LY1,LZ1,LELT)
        integer e,eg,ex,ey,ez
        integer i,j,k
c=============================================
c       Function 
c=============================================

cc: For Loop for transpose element
c-----------------------------------
        do e=1,lelt

        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,nelz)
        do j=1,ly1
        do i=1,lx1
        do k=1,lz1
        velV(i,j,k,e) = avgV(i,j,ex,ey) ! Transpose 
        enddo
        enddo
        enddo

        enddo ! do e=1,nelt
c-----------------------------------
        return
        end subroutine z_avg_reshape
c--------------------------------------------------------------------




c-----------------------------------------------------------------------
        subroutine x_averaging(velV,avgVX)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        include 'OPPO_CTL'
        
        ! YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to aviod error
        real avgVX(LY1,LZ1,ynel,znel), velV(LX1,LY1,LZ1,LELT),
     $   w1(LY1,LZ1,ynel,znel),w2(LY1,LZ1,ynel,znel)
        
        ! This is a copy of w1 for YWDEBUG 
        ! real w1copy(LX1,LY1,xnel,ynel),ws1(LX1,LY1,LZ1,LELT)
        ! COMMON / CTRL / w1copy, ws1
        
        ! Iterator 
        integer i,j,k, il, jl 
        integer e,eg,ex,ey,ez,myz,nelyz
        real umean, dx2
        
c=============================================
c       Function 
c=============================================
        
        nelyz = ynel*znel
        ! print *, 'nelyz=',nelyz,'lely*lelz=',lely*lelz

!         if (nelyz.gt.lely*lelz) call exitti
!      $  ('ABORT IN x_average. Increase lely*lelz in SIZE:$',nelyz)


        dx2 = 1.0  !  Assuming uniform in "x" direction
        myz = LY1*LZ1*ynel*znel
        call rzero(avgVX,myz)
        call rzero(w1,myz)
        call rzero(w2,myz)
        
c Computing the weighted average along z-dir 
c--------------------------------------
        do e=1,lelt
        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,znel)
        do k=1,nz1
        do j=1,ny1
        do i=1,nx1
        avgVX(j,k,ey,ez)=avgVX(j,k,ey,ez)
     $                    +dx2*wxm1(i)*velV(i,j,k,e)
        w1(j,k,ey,ez)=w1(j,k,ey,ez)+dx2*wxm1(i) ! redundant but clear
        enddo
        enddo
        enddo
        enddo

        ! print *, "X-avg: Sum UP"
        
        call gop(avgVX,w2,'+  ',myz)
        call gop(w1,w2,'+  ',myz)
        ! print *, "X-avg: GOP"

        do i=1,myz
        avgVX(i,1,1,1) = avgVX(i,1,1,1) / (w1(i,1,1,1)+1e-16)   ! Normalize
        enddo
        
        ! print *, "X-avg: Normalise"

c--------------------------------------

        return
        end subroutine x_averaging
c--------------------------------------------------------------------
        
        


c--------------------------------------------------------------------
        subroutine x_avg_reshape(velV,avgVX)
c Reshape the array of averaged value back to the regular shape of velocity array
c=============================================
c       Define variable
c=============================================
        implicit none
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        include 'OPPO_CTL'
cc YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to implement z-avg
        real avgVX(LY1,LZ1,ynel,znel), velV(LX1,LY1,LZ1,LELT)
        integer e,eg,ex,ey,ez
        integer i,j,k
c=============================================
c       Function 
c=============================================
        
cc: For Loop for transpose element
c-----------------------------------
        do e=1,lelt
        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,znel)
        do j=1,lz1
        do i=1,ly1
        do k=1,lx1
        velV(i,j,k,e) = avgVX(i,j,ey,ez) ! Transpose 
        enddo
        enddo
        enddo
        enddo ! do e=1,nelt
c-----------------------------------
        return
        end subroutine x_avg_reshape
c--------------------------------------------------------------------
        
        
        




c-----------------------------------------------------------------------
        subroutine get_exyz_usr(ex,ey,ez,eg,nelx,nely,nelz)
c
c       By the global indices and number of 2D elements to locate the   
c 
c=============================================
c       Define variable
c=============================================
        integer ex,ey,ez,eg,nelx,nely,nelz
        integer nelxy
c
c=============================================
c       Function 
c=============================================
        ex = 0 
        ey = 0
        ez = 0

        nelxy = nelx*nely
c
        ez = 1+(eg-1)/nelxy
        ey = mod1(eg,nelxy)
        ey = 1+(ey-1)/nelx
        ex = mod1(eg,nelx)
c

        return
        end subroutine get_exyz_usr
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine vel_pts_find(ctrlV)
c: Get the velocity component at the sensing plane points and registered on the mesh
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        include 'PARALLEL'

        ! velocity field solution
        real ctrlV(LX1,LY1,LZ1,LELT)
        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array
        integer ifld       ! Field count 
        integer totpts 
        integer wel,wface,wx,wy,wz ! indicies for the wall
        real vf
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
        
c=============================================
c       Function
c=============================================
        ntot = LX1*LY1*LZ1*LELT
        ! call rzero(vwall(1,1),totctrl)
        ! call rzero(vctl(1,1),totctrl)
!     Velocity interpolation 
        
        call fgslib_findpts_eval(inth_hpts1,vctl(1,1),1,
     &                       iwk(1,1),1,
     &                       iwk(1,3),1,
     &                       iwk(1,2),1,
     &                       rwk(1,2),NDIM,totctrl,
     &                       ctrlV(1,1,1,1))
        
!         call interp_nfld(vctl(1,*),ctrlV,1,
!      $                  crdctrl(1,*),crdctrl(2,*),crdctrl(3,*),
!      $                  crdctrl(1,*),crdctrl(2,*),crdctrl(3,*))
cc YW: Added for optimize the performance to prevent traversing 
! Step 2 "Reshape" the velocity from a 1D array to a tensor 
! by the inidicies into a field, so for the rest value should be ZERO
        totpts=LX1*LY1*LZ1*LELT*6
        ! Traversing on all the points we find in 1D array 
        if (numctrl.gt.0) then 
                do il=1,numctrl
                        wel   = grdwall(1,il)
                        wface = grdwall(2,il)
                        wx    = grdwall(3,il)
                        wy    = grdwall(4,il)
                        wz    = grdwall(5,il)
                        ! YW: Here as long as it is no longer 
                        vf    = vctl(1,il)
                        Vnfluct(wx,wy,wz,wface,wel)=vf ! Rest of the elements will == dumi
                enddo ! il=numctrl 
        endif ! if (numctrl.gt.0)

c$$$ YWDEBUG: Write down all the sensing points that we have found and the averaged velocity
#ifdef YWDEBUG 
        if (numctrl.gt.0 .and. ISTEP.eq.2) then 
                write(str,"(i4.4)") NID
                write(str1,"(i4.4)") ISTEP
                open(10001,file="Fluct_Vel.txt"//str//str1)
                write(10001,*) "NID  ","T  ", 
     $                          "ie  ","iface  ","ix  ","iy  ","iz  ",
     $                          "X  ", "Y  ", "Z  ",
     $                          "U  "

        do ilx = 1,numctrl
        write(10001,*) proc(ilx),
     $     ISTEP,
     $     (grdwall(ily,ilx),ily=1,nfeat),
     $     (crdwall(ily,ilx), ily=1,NDIM),
     $     (vctl(1,ilx), ily = 1,nfldc)
        enddo

               close(10001)
        endif
#endif
c$$$ YWDEBUG END        
        return 
        end subroutine vel_pts_find
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine actuate_vel(ix,iy,iz,iside,iel,vf,isfind)
        ! subroutine actuate_vel(xi,yi,zi,iside,iel,vf,isfind)
cc: 1. Find the corresponding wall points to actuate the velcoity flucutaion
cc: 2. Return it to the userbc and change BC with a flag
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "TSTEP"
        include "PARALLEL"
        include "GEOM"
        include "NEKUSE"
        ! include "TOTAL"
        integer il,jl,kl ! Iteration
        
        !!Args:
        integer ix, iy, iz ! Element indicies
        real xi,yi,zi
        integer iside     ! Faceid 
        integer iel       ! Global order
        
        integer iloc
        !!Return
        real vf ! The return velocity 
        logical isfind ! Is founded the points
        !! Parameters 
        integer cx,cy,cz ! Current in lib 
        real rwx,rwy,rwz ! Current in lib 
        integer cside    ! Current faceid 
        integer cieg     ! Current global order 
        character*3 cbi

        ! Debugging the implementation of actuation 
        ! Velocity array, comm with the actuation subroutine 
        integer leng
        parameter(leng=1000)
        real postV(1,leng)
        real postC(LDIM,leng)
        integer postF(5,leng)
        integer itimes 
        real dumi 
        parameter(dumi=100)
        
        COMMON / CTRLP / postV,postC,postF,itimes

        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1

c=============================================
c       Function
c=============================================
! idea: We find them by pre-stored information
        
        if (numctrl.gt.0) then 
                vf   = Vnfluct(ix,iy,iz,iside,iel) ! Find the vel fluct according to the indices
                
                if (vf.ne.dumi) then ! only if the value is non-zero we impose it  
                        isfind = .TRUE.
                        goto 5000
                endif
        endif ! if (numctrl .gt.0)

5000    return 
        end subroutine actuate_vel
c--------------------------------------------------------------------



c############################################
c Subroutines BELOW:  JUST FOR YWDEBUG during develop  
c############################################

c--------------------------------------------------------------------
        subroutine vel_pts_exam(ctrlV)
c: Get the velocity component at the sensing plane points and registered on the mesh
c: WRITE INTO array vwall instead of vctrl for possible YWDEBUG 
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "TSTEP"
        include 'PARALLEL'

        ! velocity field solution
        real ctrlV(LX1,LY1,LZ1,LELT)
        ! COMMON / CTRL / ctrlV

        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array
        integer ifld       ! Field count 
        
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1
        
c=============================================
c       Function
c=============================================
        
!     Velocity interpolation 
c----------------------------------------

        call rzero(vwall,totctrl)
        call fgslib_findpts_eval(inth_hpts1,vwall(1,1),1,
     &                       iwk(1,1),1,
     &                       iwk(1,3),1,
     &                       iwk(1,2),1,
     &                       rwk(1,2),NDIM,totctrl,
     &                       ctrlV(1,1,1,1))
c----------------------------------------

#ifdef YWDEBUG
cc$$$ YWDEBUG: Write down all the sensing points that we have found and the averaged velocity 
!         if (numctrl.gt.0 .and. ISTEP.eq.2) then 
!                 write(str,"(i4.4)") NID
!                 write(str1,"(i4.4)") ISTEP
!                 open(10001,file="Org_Vel.txt"//str//str1)
!                 write(10001,*) "NID  ","T  ", 
!      $                          "ie  ","iface  ","ix  ","iy  ","iz  ",
!      $                          "X  ", "Y  ", "Z  ",
!      $                          "U  "

!                 do ilx = 1,numctrl
!                         write(10001,*) proc(ilx),
!      $                  ISTEP,
!      $                  (grdwall(ily,ilx),ily=1,nfeat)
!                         write(10001,'(4F16.6)')
!      $                 (crdwall(ily,ilx), ily=1,NDIM),
!      $                 vwall(ilx)
!                 enddo
!                 close(10001)
!         endif

        ! Clean-up the array after use it 
        call rzero(vwall,totctrl)
#endif
cc$$$ YWDEBUG END        
        return 
        end subroutine vel_pts_exam
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine record_actuate(ix,iy,iz,iel,iface,xi,yi,zi,vf)
c: Record the imposed velocity in the USERBC
c: We register the coordinates on the mesh and find the velocity components
c=============================================
c       Define variable
c=============================================
        ! implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        ! include "NEKUSE" 

        include 'PARALLEL'
        
        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array
        integer ifld       ! Field count 
        
        ! Inputs Arg:
        integer ix,iy,iz,iel,iface
        real xi,yi,zi,vf
        ! Debugging the implementation of actuation 
        ! Velocity array, comm with the actuation subroutine 
        integer leng
        parameter(leng=1000)
        real postV(1,leng)
        real postC(LDIM,leng)
        integer postF(5,leng)
        integer itimes 
        COMMON / CTRLP / postV,postC,postF,itimes
        
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1

c=============================================
c       Function
c=============================================
        nxyz  = LX1*LY1*LZ1
        ntot  = nxyz*NELT
        ntot = LX1*LY1*LZ1*LELT

        
!---------------------------
c$$     ! For YWDEBUG the implementation
        itimes = itimes+1       ! Counting 
        postV(1,itimes)=-vf     ! Pass Wall-normal Vel
        
        ! Write the coordinates 
        postC(1,itimes)=xi
        postC(2,itimes)=yi
        postC(3,itimes)=zi
        
        ! Write the indices 

        postF(1,itimes)= iel
        postF(2,itimes)= iface
        postF(3,itimes)= ix
        postF(4,itimes)= iy
        postF(5,itimes)= iz
        
c$$
!---------------------------

        
        return 
        end subroutine record_actuate
c--------------------------------------------------------------------




c--------------------------------------------------------------------
        subroutine wall_aft_vel
c: Get the wall velocity component 
c: We register the coordinates on the mesh and find the velocity components
c=============================================
c       Define variable
c=============================================
        ! implicit none 
        include 'SIZE'
        include "OPPO_CTL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        include "NEKUSE" 

        include 'PARALLEL'
        
        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array
        integer ifld       ! Field count 
        
        ! Debugging the implementation of actuation 
        ! Velocity array, comm with the actuation subroutine 
        integer leng
        parameter(leng=1000)
        real postV(1,leng)
        real postC(LDIM,leng)
        integer postF(5,leng)
        integer itimes 
        COMMON / CTRLP / postV,postC,postF,itimes
        
        ! For YWDEBUG 
        integer ilx, ily
        character*4 str, str1

c=============================================
c       Function
c=============================================
        nxyz  = LX1*LY1*LZ1
        ntot  = nxyz*NELT
        ntot = LX1*LY1*LZ1*LELT

c----------------------------------------
!     Write down the velocity 
c----------------------------------------
#ifdef YWDEBUG
        if (numctrl.gt.0 .and. ISTEP.eq.3) then 
                write(str,"(i4.4)") NID
                write(str1,"(i4.4)") ISTEP

                open(10001,file="Aft_Vel.txt"//str//str1)
                write(10001,*) "NID  ","T  ",  
     $                          "ie  ","iface  ","ix  ","iy  ","iz  ",
     $                          "X  ", "Y  ", "Z  ",
     $                          "U  "
                           do ilx = 1,itimes
                write(10001,*) NID,
     $                  ISTEP,
     $                  (postF(ily,ilx), ily=1,5)

                write(10001,'(4F16.6)')
     $                 (postC(ily,ilx), ily=1,NDIM),
     $                  postV(1,ilx)
                enddo
                close(10001)
        endif
#endif
        itimes=0 ! reset for counting the actuation 
        return 
        end subroutine wall_aft_vel
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine z_weight_reshape(w1r,w1copy)
c Reshape the array of averaged value back to the regular shape of velocity array
c=============================================
c       Define variable
c=============================================
        ! implicit none
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        include 'OPPO_CTL'
                
cc YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to aviod 
        ! integer xnel, ynel 
        ! parameter (xnel=elem2d)
        ! parameter (ynel=1)
        ! parameter 
        real w1copy(LX1,LY1,xnel,ynel), w1r(LX1,LY1,LZ1,LELT)
        integer e,eg,ex,ey,ez
        integer i,j,k
c=============================================
c       Function 
c=============================================
        
cc: For Loop for transpose element
c-----------------------------------
        do e=1,nelt
        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,nelz)
        do j=1,ny1
        do i=1,nx1
        do k=1,nz1
        w1r(i,j,k,e) = w1copy(i,j,ex,ey) ! Transpose 
        enddo
        enddo
        enddo
        enddo ! do e=1,nelt
c-----------------------------------
        return
        end subroutine z_weight_reshape
c--------------------------------------------------------------------