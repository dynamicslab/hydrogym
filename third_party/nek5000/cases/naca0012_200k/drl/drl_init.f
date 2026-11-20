
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
        ! include "NEKUSE"
cc YW:
        include "DRL"

        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        integer tot_ctrl
        integer ie,iface,ix,iy,iz,lgi ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer idx
        real xf,yf,zf
        real xr,yr,zr
        character*3 bcb
        integer iglsum
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

        if(nid.eq.0) print *, "[NEK] CTRL AREA:",ctrlxs,ctrlxe
c########################################
c     DRL initialisation
c########################################
cc YW: The idea is to find the wall points by using the range and the wall boundary condition
cc YW: FOR DRL, we only find the face mid points
        NUMCTRL = 0 
        do ie=1,NEL
        do iface=1,nfaces
                xr=xm1(fmid(iface),1,1,ie) ! X coord for face mid point
                yr=ym1(fmid(iface),1,1,ie) ! Y coord for face mid point
                zr=zm1(fmid(iface),1,1,ie) ! Y coord for face mid point
                bcb=CBC(iface,ie,1)
                ! call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                ! YW Modified here to avoid overlapping
                ! Try use facind 
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                do iz=KZ1,KZ2
                do iy=KY1,KY2
                do ix=KX1,KX2
                xf=xm1(ix,iy,iz,ie)
                yf=ym1(ix,iy,iz,ie)
                zf=zm1(ix,iy,iz,ie)        

                if (xr.gt.ctrlxs .and. xr.lt.ctrlxe .and.! within range of ctrl  Reth
     $              yr.gt.ctrlys .and. yr.lt.ctrlye .and.
     $              bcb.eq.'W  ') then
      
                NUMCTRL = NUMCTRL +1 ! if meet condition
                lgi=lglel(ie)         
                pos_agt(1,NUMCTRL)=xf ! Coordinate X
                pos_agt(2,NUMCTRL)=yf
                pos_agt(3,NUMCTRL)=zf
cc YW: I write the global grid information here for find index in USERBC
                info_agt(1,NUMCTRL)=lgi         ! Global indicies
                info_agt(2,NUMCTRL)=iface       ! Number of face
                info_agt(3,NUMCTRL)=ix         ! Number of x ploy
                info_agt(4,NUMCTRL)=iy ! Number of y poly 
                info_agt(5,NUMCTRL)=iz ! Number of z poly 
cc YW: Write the RANK info, for further interpolation
                proc(NUMCTRL)=NID
cc YW: Change the BC to Dirichlet LATER
                CBC(iface,ie,1) = "v  "
                endif ! If bcb.eq."W  "
                enddo ! do ix = KX1, KX2 
                enddo ! do iy = KY1, KY2 
                enddo ! do iz = KZ1, KZ2
        enddo ! iface = 1, nfaces
        enddo ! ie = 1, NEL
        
        tot_ctrl = iglsum(NUMCTRL,1)
        if (NID.eq.0) print *, "[NEK] TOTAL AGENT PTS = ",tot_ctrl

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
        subroutine calculate_norm_angle
cc YW:  A subroutine for calculate the angle used to project for wall-normal velocit
cc YW: Note that this function will be put in subroutine usrdat3()  
cc The implementation used in Body-force from MA 
c=============================================
c       Define variable
c=============================================
        
        implicit none 
        include "SIZE"
        include "INPUT"
        include "DRL"
        ! include "NEKUSE"
        include "SOLN" ! vy and vz to use 
        include "GEOM" !  The angle unx, uny, unz 
        include 'TOPOL'
        include 'PARALLEL'
        real xpts,ypts
        real xnn, ynn
        real bdx,bdy,delta2,bsin,bcos
        integer ie, ntot       
        
        character*132 field
        
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================
        ntot = LX1*LY1*LZ1*LELT
        if (NID.eq.0) then
        print *,"=============================="
        print *, " ROTATION WING"
        print *,"=============================="
        endif 
c------------------------------------------
c  Step1: Load the extruded field to get the correct angle

        field='mask_naca_wing0.f00002' ! Remember I rename it as 2, in dardel it is #3
        
        call load_fld(field)  ! After call this, the vx,vy,vz are replaced by the loaded data
        if (NID.eq.0) print *, "[IO] ANGLE DATA Loaded"
c-----------------------------------------
c------------------------------------------
c  Step2: Compute the real angle for projecting wall-normal direction at OFF-WALL 
        do ie=1,ntot
                xpts=xm1(ie,1,1,1)
                ypts=ym1(ie,1,1,1)
                xnn =vy(ie,1,1,1)
                ynn =vz(ie,1,1,1)
                
                bdx=abs(xpts-xnn)
                bdy=abs(ypts-ynn)
                delta2=bdx*bdx+bdy*bdy

                if (delta2.gt.0) then
                bsin = bdy/sqrt(delta2)
                bcos = bdx/sqrt(delta2)
                else ! if delta2.gt.0
                bsin = 0
                bcos = 0
                endif ! if delta2.le.0

!!!! after maximum chamber and over the suction side (A)
                if((xpts.gt.xnn).AND.(ypts.gt.0)) then
                body_sin(ie,1,1,1)=bsin
                body_cos(ie,1,1,1)=bcos
                endif
!!!! before maximum chamber and over the suction side (B)
                if((xpts.le.xnn).AND.(ypts.gt.0)) then
                body_sin(ie,1,1,1)=bsin
                body_cos(ie,1,1,1)=-bcos
                endif
!!!! before maximum chamber and over the pressure side (C)
                if((xpts.le.xnn).AND.(ypts.le.0)) then
                body_sin(ie,1,1,1)=bsin
                body_cos(ie,1,1,1)=bcos
                endif
!!!! after maximum chamber and over the pressure side (D)
                if((xpts.gt.xnn).AND.(ypts.le.0)) then
                body_sin(ie,1,1,1)=bsin
                body_cos(ie,1,1,1)=bcos
                endif
        enddo ! ie=1,ntot 
c-----------------------------------------------
        if (NID.eq.0) print *, "[NEK] ROTATION DONE"
        
        return 
        end subroutine calculate_norm_angle
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
        real utaux,ynorm,dyy,dyx
        ! Setup as parameter 
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

        if (NUMCTRL.gt.0) then  ! Only works when wall point exists in this RANK
        
        do il=1,NUMCTRL
                ! Wall points coordinates
                xw = pos_agt(1,il)
                yw = pos_agt(2,il)
                zw = pos_agt(3,il)

                ! Inidicies info to substract the angle 
                ie = info_agt(1,il)
                ! Convert  the global indicies into local 
                ie = gllel(ie)
                ix = info_agt(3,il)
                iy = info_agt(4,il)
                iz = info_agt(5,il)
                
                ! Use the wall normal given by the extrution 
                snx = body_cos(ix,iy,iz,ie)
                sny = body_sin(ix,iy,iz,ie)

                call X2Utau(xw,utaux)       ! Interploation to achieve the friction velocity

c------------------------------------------
c  CASE 1        The benchmark case: 
c------------------------------------------
                ynorm = ypctrl * nuctrl / utaux 
c------------------------------------------
                ! Inclination due to the surface curvature
                dyx = ynorm*snx
                dyy = ynorm*sny
                
                ! Take the inclination on x,y dir into account 
                if (yw.gt.0) then 
                  xct = xw + dyx 
                  yct = yw + dyy 
                else
                  xct = xw - dyx 
                  yct = yw - dyy 
                endif
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
                write(10001,*) proc(ilx),
     $         (pos_obs(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
c$$$ TEST END
#endif 

        return 
        end subroutine interp_sensing_plane 
c--------------------------------------------------------------------



! cc: NOTE: This libaray is ought to be updated once you have High-Fidelity Results
! c--------------------------------------------------------------------
         subroutine X2Utau(xw,utaux)
! cc YW: A subroutine used for interoplation the value of u_tau from the x corrdinates
! cc NOTE Suction Side ONLY
! cc this is based on the prior knowledge 
! c=============================================
! c       Define variable
! c=============================================
         implicit none 
         include "SIZE"
         include "DRL"
         integer ix ! Iteration
         real xw    ! input
         real utaux ! output utau based on input x
         real pl(10)! Polynominal   
! c=============================================
! c       Function
! c=============================================
cc Statistics obtained from A Tanarro
cc YW Modified here for NACA0012, 2024.04.29
cc---------------------------------------
        pl(1) =	0.04327887538008919
        pl(2) =	0.5248714861235598
        pl(3) =	-4.871857463913461
        pl(4) =	22.46151360065518
        pl(5) =	-61.58950584522501
        pl(6) =	104.56101035004765
        pl(7) =	-108.79482929066342
        pl(8) =	65.14084910886064
        pl(9) =	-18.9743110707296
        pl(10)=	1.5255420287605719


! c Adding up the polynominal 
! c-------------------------------------------
!         ! YW: Here we use the parallel flow assumption! 

         utaux = 0.0
          do ix=1,10 
          utaux = utaux + pl(ix)*xw**(ix-1)
          enddo 
         !utaux = 0.052

         return 
         end subroutine X2Utau
! c--------------------------------------------------------------------


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

        ! include 'GEOM'
        ! include 'NEKUSE'

        integer nfail,totfail
        real tolin ! The tolerence for interpolation 
        integer nmsh, nelm, ih
        integer il,jl,kl ! Iteration 
        integer iglsum
        parameter(tolin=-1.0)
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

c SetUp for the interploation
c----------------------------

        call intpts_setup(tolin,inth_hpts1)

#ifdef YWDEBUG
        if (NID.eq.0) print *,"[DEBUGG] INTPTS SETUP!"
#endif

c       Step 2: Register them on the map
        call findpts(inth_hpts1,
     &   rcode,1,                      ! $ rcode 1
     &   proc,1,                      ! $ proc,1
     &   elid,1,                      ! $ elid, 1
     &   rst,NDIM,                   ! $ rst, ndim
     &   dist,1,                      ! $ dist,1
     &   pos_obs(1,1),NDIM,                   ! $ x
     &   pos_obs(2,1),NDIM,                   ! $ y
     &   pos_obs(3,1),NDIM,TOTCTRL)           ! $ z     

#ifdef YWDEBUG
        if (NID.eq.0) print *,"[DEBUGG] INTPTS SETUP!"
#endif

c Examine the deviations after interoplation
c----------------------
        nfail = 0 
        ! do il=1,TOTCTRL
        do il=1,NUMCTRL
        ! check return code
        if(rcode(il).eq.1) then
          if(dist(il).gt.1e-12) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     pos_obs(1,il),pos_obs(2,il),pos_obs(3,il),dist(il)
          endif
        elseif(rcode(il).eq.2) then
          nfail = nfail + 1
          if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        pos_obs(1,il),pos_obs(2,il),pos_obs(3,il)
        endif
        enddo
        
        totfail=iglsum(nfail,1)
        if (NID.eq.0) print *,"[DRL] Sensing: Failed Interp:",totfail

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
c---------------------------------------------------------------

