c==============================================
c DRL subroutines for sensing data 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================


c------------------------------------------------------------------
        subroutine drl_state
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
        
                ! call wall_aft_vel               ! TEST the actutation on the wall 
                
        ! if (NID.eq.0) then
        !         print *,"--------------------"
        !         print *, "[STATE] INQURY"
        !         print *,"--------------------"
        ! endif

        call sensing_pts_compute        ! Get sensing plane velocity and compute spatial fluctutations
        
        ! MPI SEND 
        call drl_state_out
        
        return 
        end subroutine drl_state
c------------------------------------------------------------------




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
        
        include "DRL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        include 'PARALLEL'
        
        ! velocity field solution
        ! Multi-variables
        real ctrlV(NFLDC,LX1,LY1,LZ1,LELT)
        real ctrlVx(LX1,LY1,LZ1,LELT)
        real ctrlVy(LX1,LY1,LZ1,LELT)
        
        integer nfail 
        integer il,jl,kl ! Iteration
        integer ntot, nxyz ! Size of array

        ! For test 
        integer ilx, ily
        character*4 str, str1
        ! Communication to save memory 
        ! COMMON / CTRL / ctrlV

c=============================================
c       Function
c=============================================
        ! if (NID.eq.0) print *, '[DRL] Start Sensing'
c---------------------------------
cc STEP 1: Copy velocity arries from solution
cc YW: We need vx and vy for projecting onto the V 
        call normal_V_project(ctrlVx,ctrlVy)
c----------------------------------
        ! if (NID.eq.0) print *, '[DRL] Projection'
c----------------------------------
cc STEP 2: Compute z-avg and substract the mean value from vel
        ! YW: We DO NOT NEED FLUCTUATION HERE/Check the DRL
        call vel_fluct_compute(ctrlVx)
        call vel_fluct_compute(ctrlVy)
        ! if (NID.eq.0) print *, '[DRL] Fluctuation'
c----------------------------------

c----------------------------------
cc STEP 3: Interpolating the quantities for sensing points
        ! call vel_pts_find(ctrlV)
        call vel_pts_find(ctrlVx,ctrlVy)
c----------------------------------
        ! if (NID.eq.0) print *, '[DRL] Sensing Plane'

c----------------------------------
cc STEP 4: Output the state files
        ! call drl_state_out()
c----------------------------------

c----------------------------------
cc Print in the output 
        if (NIO.eq.0) then  
        print *, "--------------------"
        print *, "[STATE] SENSING PLANE GET"
        print *, "--------------------"
        endif 
c----------------------------------

        return 
        end subroutine sensing_pts_compute
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine normal_V_project(ctrlVx,ctrlVy)
cc: Project the x-dir and y-dir velocity from Cartesian coordinates onto the Wall-normal velocity 
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "DRL"
        include "NEKUSE"
        ! include "TOTAL" ! Use "GEOM", "SOLN", "TOPOL" and NELFLD 
        include "GEOM"
        include "SOLN"
        include "TSTEP"
        include 'PARALLEL'
        ! Wall-normal Velocity arrary
        real ctrlVx(LX1,LY1,LZ1,LELT)
        real ctrlVy(LX1,LY1,LZ1,LELT)
        real wrk_buff1(LX1,LY1,LZ1,LELT)
        real wrk_buff2(LX1,LY1,LZ1,LELT)
        ! Iteration indices
        integer ie,iface,ix,iy,iz ! Iteration for all elements 
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer f        ! f = eface1(iface)
        integer ntot
        ! For projection 
        real vxi,vyi ! The un-projected vx vy  
        real snx,sny     ! The face normal: sin(theta)
        real vux,vun ! The projection of vx and vy and the normal velocity  
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

        ! Compute the Projection
        ntot = LX1*LY1*LZ1*LELT
        ! The first variable is the Streamwise velocity 
        call copy(wrk_buff1(1,1,1,1),vx(1,1,1,1),ntot)
        call copy(wrk_buff2(1,1,1,1),vy(1,1,1,1),ntot)
        call copy(ctrlVx(1,1,1,1),wrk_buff1(1,1,1,1),ntot)
        ! The 2nd is the Wall-normal velocity 
        call copy(ctrlVy(1,1,1,1),wrk_buff2(1,1,1,1),ntot)

c=============================================
c       Testing
c=============================================
c$$$ TEST 
c        !Write dow1 the face normal 
        if (ISTEP.eq.1) then
        call outpost(ctrlVx,ctrlVy,vx,pr,t,'ang')
        if (NID.eq.0) print *,"At",ISTEP,"YW: WIRTE ANGLE FOR TEST"
        endif 

c$$$ TEST End

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
        include "DRL"
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
     $    velV(LX1,LY1,LZ1,LELT),   
     $    wrk_buff(LX1,LY1,LZ1,LELT),   
     $    avgV(LX1,LY1,LZ1,LELT) 
        ! Element in the array, calculate explictly 
        real ufluct, uorg, umean
        
        ! Plane Average handles
        integer igs_x,igs_z
        save igs_x,igs_z
        
        ! For test 
        integer ilx, ily
        character*4 str, str1
        nxyz  = LX1*LY1*LZ1
        ntot  = nxyz*LELT
        

c=============================================
c       Function
c=============================================

c-----------------------------------------------
c: Step 1: copy the current array into a new array for computing the Mean
c-----------------------------------------------
        ! initialize the tensor anyway
        ! If we need average, then initialize the handle
        if (obs_xavg .or. obs_zavg) then 
        if (igs_z.eq.0.and.igs_x.eq.0) then 
        ! call interp_wall_pts
        call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
        call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
        if (NID.eq.0) print *, "[STATE] AVG HANDLE INIT!",igs_z,igs_x
        endif ! if igs_z eq.0 and igs_x.eq.0
        
        ! DO this feature by featur
        call copy(velV(1,1,1,1),ctrlV(1,1,1,1),ntot)
        call copy(wrk_buff(1,1,1,1),ctrlV(1,1,1,1),ntot)
        
c-----------------------------------------------
c: Step 2: Compute the spatial Mean 
c-----------------------------------------------
        if (obs_zavg) then 
        ! Z-dir average
        call rzero(avgV(1,1,1,1),ntot)
        call planar_avg(avgV,velV,igs_z)
        call copy(velV(1,1,1,1),avgV(1,1,1,1),ntot)

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] Z-AVG!"
#endif
        endif 
        
        if (obs_xavg) then 
        ! X-dir average
        call rzero(avgV(1,1,1,1),ntot)
        call planar_avg(avgV,velV,igs_x)
        call copy(velV(1,1,1,1),avgV(1,1,1,1),ntot)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] X-AVG!"
#endif
        endif

c-----------------------------------------------

c-----------------------------------------------
c: Step 3: Substract the Mean, writting it explicitly to check
c: Corrected on Apr 1st, 2025 
c-----------------------------------------------
        wrk_buff(:,:,:,:) = wrk_buff(:,:,:,:) 
     $                      - velV(:,:,:,:)

        call copy(ctrlV(1,1,1,1),wrk_buff(1,1,1,1),ntot)

        ! enddo ! do fl=1,NFLDC
        if(NID.eq.0) print *, "[DRL] STATE AVG SUBSTRACT!"
        else
        goto 119 
        endif 
c-----------------------------------------------
119     return
        end subroutine vel_fluct_compute
c--------------------------------------------------------------------





c--------------------------------------------------------------------
        subroutine vel_pts_find(ctrlVx,ctrlVy)
c: Get the velocity component at the sensing plane points and registered on the mesh
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "DRL"
        include "INPUT"
        include "SOLN"
        include "TSTEP"
        include 'PARALLEL'

        ! velocity field solution
        real ctrlVx(LX1,LY1,LZ1,LELT)
        real ctrlVy(LX1,LY1,LZ1,LELT)
#ifdef YWDEBUG
        real ctrl_test1(LX1,LY1,LZ1,LELT)
        real ctrl_test2(LX1,LY1,LZ1,LELT)
#endif
        real buffV(LX1*LY1*LZ1*LELT)
        integer nfail 
        integer il,jl,kl 
        integer iel,iface,ix,iy,iz ! Iteration for all elements 
        integer ntot, nxyz ! Size of array
        integer ifld       ! Field count 
        integer totpts 
        integer wel,wface,wx,wy,wz ! indicies for the wall
        real vf
        ! For test 
        integer ilx, ily
        character*4 str, str1
        
c=============================================
c       Function
c=============================================
        ntot = LX1*LY1*LZ1*LELT
        ! call rzero(vwall(1,1),totctrl)
#ifdef YWDEBUG
        if (ISTEP.eq.1) then 
        call copy(ctrl_test1(1,1,1,1),ctrlVx(1,1,1,1),ntot)
        call copy(ctrl_test2(1,1,1,1),ctrlVy(1,1,1,1),ntot)
        call outpost(ctrl_test1,ctrl_test2,
     &          ctrl_test1,pr,t,'flu')
        endif 
#endif

        call copy(buffV,ctrlVx(1,1,1,1),ntot)
        call fgslib_findpts_eval(inth_hpts1,val_obs(1,1),NFLDC,
     &                       iwk(1,1),1,
     &                       iwk(1,3),1,
     &                       iwk(1,2),1,
     &                       rwk(1,2),NDIM,NUMCTRL,
     &                       buffV(1))
        call copy(buffV,ctrlVy(1,1,1,1),ntot)
        call fgslib_findpts_eval(inth_hpts1,val_obs(2,1),NFLDC,
     &                       iwk(1,1),1,
     &                       iwk(1,3),1,
     &                       iwk(1,2),1,
     &                       rwk(1,2),NDIM,NUMCTRL,
     &                       buffV(1))

#ifdef YWDEBUG
        if (NID.eq.0) print *,"[DRL] GET FINDPTS"
#endif YWDEBUG


c$$$ TEST: Write down all the sensing points that we have found and the averaged velocity
#ifdef YWDEBUG 
        if (numctrl.gt.0 .and. ISTEP.le.3) then 
                write(str,"(i4.4)") NID
                write(str1,"(i4.4)") ISTEP
                open(10001,file="Fluct_Vel.txt"//str//str1)
                write(10001,*) "NID  ","T  ", 
     $                          "ie  ","iface  ","ix  ","iy  ","iz  ",
     $                          "X  ", "Y  ", "Z  ",
     $                          "U  ","V  "

        do ilx = 1,numctrl
        write(10001,*) iptctl(ilx),
     $     ISTEP,
     $     (info_agt(ily,ilx),ily=1,nfeat),
     $     (pos_obs(ily,ilx), ily=1,NDIM),
     $     (val_obs(ily,ilx), ily = 1,nfldc)
        enddo
        close(10001)
        endif
        if (ISTEP.eq.1) then 
        call rzero(ctrl_test1(1,1,1,1),ntot)
        call rzero(ctrl_test2(1,1,1,1),ntot)
        
        do ilx = 1,NUMCTRL
         iel = info_agt(1,ilx)
         iface = info_agt(2,ilx)
         iel = gllel(iel)
         ix = info_agt(3,ilx)
         iy = info_agt(4,ilx)
         iz = info_agt(5,ilx)
         ctrl_test1(ix,iy,iz,iel)=val_obs(1,ilx)
         ctrl_test2(ix,iy,iz,iel)=val_obs(2,ilx)
        enddo
        call outpost(ctrl_test1,ctrl_test2,
     &          ctrl_test1,pr,t,'fli')
        endif 
#endif
c$$$ TEST END        
        return 
        end subroutine vel_pts_find
c--------------------------------------------------------------------




c-----------------------------------------------------------------------
c               Utilities for AVG on x/z DIRECTIONS 
c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
        subroutine z_averaging(velV,avgVZ)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include 'GEOM'
        include 'PARALLEL'
        include 'WZ'
        include 'ZPER'
        ! include 'DRL'

        ! YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to aviod error
        real avgVZ(LX1,LY1,xnel,ynel), velV(LX1,LY1,LZ1,LELT),
     $   w1(LX1,LY1,xnel,ynel),w2(LX1,LY1,xnel,ynel)
        
        ! This is a copy of w1 for test 
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
        call rzero(avgVZ,mxy)
        call rzero(w1,mxy)
        call rzero(w2,mxy)
! #ifdef YWDEBUG
!         if (NID.eq.0) print *, "[REWARD] ZERO!"
! #endif
c Computing the weighted average along z-dir 
c--------------------------------------
        do e=1,lelt
        eg = lglel(e)
        call get_exyz_usr(ex,ey,ez,eg,xnel,ynel,nelz)
        do j=1,ly1
        do i=1,lx1
        do k=1,lz1
        avgVZ(i,j,ex,ey)=avgVZ(i,j,ex,ey)
     $              +dz2*wzm1(k)*velV(i,j,k,e)

        w1(i,j,ex,ey)=w1(i,j,ex,ey)+dz2*wzm1(k)
cc  YW: test, make a copy of w1 
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
        call gop(avgVZ,w2,'+  ',mxy)
        call gop(w1,w2,'+  ',mxy)
c-------------------------------------

c-------------------------------------
c Normalisation by the weight array
c-------------------------------------
        do i=1,mxy
        avgVZ(i,1,1,1)=avgVZ(i,1,1,1)/w1(i,1,1,1)   ! Normalize
        enddo 
c-------------------------------------

c-------------------------------------
        return
        end subroutine z_averaging
c--------------------------------------------------------------------


c--------------------------------------------------------------------
        subroutine z_avg_reshape(velV,avgVZ)
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
        ! include 'DRL'
cc YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to implement z-avg
        real avgVZ(LX1,LY1,xnel,ynel), velV(LX1,LY1,LZ1,LELT)
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
        velV(i,j,k,e) = avgVZ(i,j,ex,ey) ! Transpose 
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
        include 'DRL'
        
        ! YW: I force the nelx = 2D element and nely as the number of 2D elements and 1 here to aviod error
        real avgVX(LY1,LZ1,ynel,znel), velV(LX1,LY1,LZ1,LELT),
     $   w1(LY1,LZ1,ynel,znel),w2(LY1,LZ1,ynel,znel)
        
        ! This is a copy of w1 for test 
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
        include 'DRL'
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
        include 'DRL'
                
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