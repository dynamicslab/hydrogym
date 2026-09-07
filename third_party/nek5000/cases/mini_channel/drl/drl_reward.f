c-------------------------------------
! All subroutines for the rewards func
! TODO: Need to be eventually adopted to the wing  
c-------------------------------------

! SOME PREDEFINED PARAM
#define INTP_NMAX 200 /* number of sample points */
#define XCINT 0.0     /* x coordinate of 1D line*/
#define ZCINT 0.0     /* z coordinate of 1D line */

c mesh dimensions
#define BETAM 1.0     /* wall normal stretching parameter */
#define PI (4.*atan(1.))

c------------------------------------------------------------------
        subroutine drl_reward(i_evolv)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include "INPUT"
        include 'PARALLEL'
        integer i_evolv
c=============================================
c       Function
c=============================================
        
        ! if (ISTEP.ne.0) then
        
        if (NID.eq.0) then
                print *,"--------------------------------"
                print *, "[REWARD] INQURY"
                print *,"--------------------------------"
        endif

        ! YW: OCT15 I got a issue regarding the MEMORY
        ! I comment this and will test it on cluster in the future.
        !-----------------
        call compute_dudy(i_evolv)
        !-----------------
        call drl_reward_out(i_evolv)
        
        if (NIO.eq.0) then  
            print *, "----------------------"
            print *, "[REWARD] GET!"
            print *, "----------------------"
        endif 
        
        ! endif 
        end subroutine
c------------------------------------------------------------------

c------------------------------------------------------------------
        subroutine compute_dudy(i_evolv)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include 'SOLN'     ! vx,vy,vz
        include 'PARALLEL'
        ! include 'GEOM'     ! rxm1
        include 'NEKUSE'
        include 'TSTEP'
        include 'DRL'
        ! For calculating Derivative
        real duidxj(LX1,LY1,LZ1,lelt,3)
        ! real devU2(LX1*LY1*LZ1*lelt,1)
        real devU1(LX1,LY1,LZ1,lelt)

        ! Doing average
        real velV(LX1,LY1,LZ1,LELT),avgV(LX1,LY1,LZ1,LELT)
        real avgVZ(LX1,LY1,xnel,ynel)   
        real avgVX(LX1,LY1,ynel,znel)
        
        ! Iteration 
        integer i_evolv
        real    rwd_i, rwd_c ! input and current reward in the buffer
        ! real rewrd(TOTCTRL)
        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        integer ie,iface,ix,iy,iz,lgi ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer ntot,nxyz
        integer igs_x,igs_z
        save igs_x,igs_z
        integer il,jl,kl,ll,ilx,ily
        character*4 str, str1
c=============================================
c       Function
c=============================================
c-----------------------------------------------
c: Step 1: copy the current array into a new array for computing the Mean
c-----------------------------------------------
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] XNEL,YNEL:",XNEL,YNEL
#endif

        nxyz=LX1*LY1*LZ1
        ntot=LX1*LY1*LZ1*LELT

        if (igs_z.eq.0.and.igs_x.eq.0) then 
        ! call interp_wall_pts
        call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
        call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
        if (NID.eq.0) print *, "[REWARD] AVG HANDLE INIT!",igs_z,igs_x
        endif 

c-----------------------------------------------
c: Step 2: Compute the Derivatives 
c-----------------------------------------------
        ! Copy the U into devU1
        call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)
        call gradm1(duidxj(1,1,1,1,1),
     $              duidxj(1,1,1,1,2),
     $              duidxj(1,1,1,1,3),
     $              devU1(1,1,1,1))

c-----------------------------------------------
c: Step 3: Compute the spatial Mean 
c-----------------------------------------------

        ! We know dUdy is at 2-th indicies!
        call copy(velV(1,1,1,1),duidxj(1,1,1,1,2),ntot)

!------- NOTE  THIS ONLY WORKED FOR NEW version!-----------
        if (rwd_zavg) then 
        ! call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
        call planar_avg(avgV,velV,igs_z)
        call copy(velV,avgV,ntot)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] Z-AVG!"
#endif
        endif ! if (rwd_zavg)

        ! Do streamwise average if it allowed/defined
        if (rwd_xavg) then 
        ! call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
        call planar_avg(avgV,velV,igs_x)
        call copy(velV,avgV,ntot)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] X-AVG!"
#endif
        endif 


c-----------------------------------------------
c: Step 4: Get the value at the Agent 
c-----------------------------------------------
        ! Initialize the array: 
        if (i_evolv.eq.1) call rzero(rwd_agt,TOTCTRL)

        ! Moving average
        do il=1,NUMCTRL
        ie=info_agt(1,il)
        ie=gllel(ie)
        iface=info_agt(2,il)
        ix=info_agt(3,il)
        iy=info_agt(4,il)
        iz=info_agt(5,il)
        rwd_i=velV(ix,iy,iz,ie)
        rwd_c=rwd_agt(il)
        
        ! Moving Average
        if (i_evolv.eq.1) then 
        rwd_agt(il)=rwd_i
        else
        rwd_agt(il)=(rwd_c*i_evolv+rwd_i)/(i_evolv+1) 
        endif 

        enddo ! do il=1,NUMCTRL

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy ASSIGNED!"
#endif

#ifdef YWDEBUG
        if (ISTEP.le.3) then 
        if (NUMCTRL.ne.0) then 

        write(str,"(i4.4)") NID
        open(51001,file="dUdy_Wall.txt"//str)
        write(51001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  ",
     $                  "dUdy  "
        do ilx = 1,numctrl
                write(51001,*) proc(ilx),
     $         (pos_agt(ily,ilx), ily=1,NDIM), 
     $         (info_agt(ily,ilx), ily=1,5),
     $         rwd_agt(ilx) 
        enddo
        close(51001)
        endif ! if (NUMCTRL.ne.0)
        endif 
#endif

        end subroutine compute_dudy



c------------------------------------------------------------------
c Calculate the dUidxj from OLD Statistics 
      subroutine comp_derivat(duidxj,u,v,w,ur,us,ut,vr,vs,vt,wr,ws,wt)
      include 'SIZE'
      include 'TOTAL'

      integer e

      real duidxj(lx1*ly1*lz1,lelt,3*ldim)    ! 9 terms
      real u  (lx1*ly1*lz1,lelt)
      real v  (lx1*ly1*lz1,lelt)
      real w  (lx1*ly1*lz1,lelt)
      real ur (1) , us (1) , ut (1)
      real vr (1) , vs (1) , vt (1)
      real wr (1) , ws (1) , wt (1)
c
c      common /dudxyj/ jacmi(lx1*ly1*lz1,lelt)
c      real jacmi
c
      n    = nx1-1                          ! Polynomial degree
      nxyz = nx1*ny1*nz1

      do e=1,nelv
         call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
         call local_grad3(vr,vs,vt,v,N,e,dxm1,dxtm1)
         call local_grad3(wr,ws,wt,w,N,e,dxm1,dxtm1)

!     Derivative tensor computed by using the inverse of 
!     the Jacobian array jacmi
      do k=1,nxyz
        ! dudx
         duidxj(k,e,1) = jacmi(k,e)*(ur(k)*rxm1(k,1,1,e)+
     $        us(k)*sxm1(k,1,1,e)+
     $        ut(k)*txm1(k,1,1,e))
        ! dvdy
         duidxj(k,e,2) = jacmi(k,e)*(vr(k)*rym1(k,1,1,e)+
     $        vs(k)*sym1(k,1,1,e)+
     $        vt(k)*tym1(k,1,1,e))
        ! dwdz
         duidxj(k,e,3) = jacmi(k,e)*(wr(k)*rzm1(k,1,1,e)+
     $        ws(k)*szm1(k,1,1,e)+
     $        wt(k)*tzm1(k,1,1,e))
        !dudy
         duidxj(k,e,4) = jacmi(k,e)*(ur(k)*rym1(k,1,1,e)+
     $        us(k)*sym1(k,1,1,e)+
     $        ut(k)*tym1(k,1,1,e))
        !dvdz
         duidxj(k,e,5) = jacmi(k,e)*(vr(k)*rzm1(k,1,1,e)+
     $        vs(k)*szm1(k,1,1,e)+
     $        vt(k)*tzm1(k,1,1,e))
        ! dwdx
         duidxj(k,e,6) = jacmi(k,e)*(wr(k)*rxm1(k,1,1,e)+
     $        ws(k)*sxm1(k,1,1,e)+
     $        wt(k)*txm1(k,1,1,e))
        ! dudz
         duidxj(k,e,7) = jacmi(k,e)*(ur(k)*rzm1(k,1,1,e)+
     $        us(k)*szm1(k,1,1,e)+
     $        ut(k)*tzm1(k,1,1,e))
        ! dvdx
         duidxj(k,e,8) = jacmi(k,e)*(vr(k)*rxm1(k,1,1,e)+
     $        vs(k)*sxm1(k,1,1,e)+
     $        vt(k)*txm1(k,1,1,e))
        ! dwdy
         duidxj(k,e,9) = jacmi(k,e)*(wr(k)*rym1(k,1,1,e)+
     $        ws(k)*sym1(k,1,1,e)+
     $        wt(k)*tym1(k,1,1,e))
      enddo
      enddo

      return
      end

c------------------------------------------------------------------




c------------------------------------------------------------------
        subroutine switch_BC_2Wall
c Change Wall-BC According to the indicies obtained before 
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
        
        integer ie,iface,ix,iy,iz ! Iteration
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
        if (NUMCTRL.ne.0) then 
            do idx=1,NUMCTRL
            ie=info_agt(1,idx)         ! Local indicies
            iface=info_agt(2,idx)      ! Number of face
            CBC(iface,ie,1)='W  '
            enddo
        endif
        if (NID.eq.0) print*,"[DRL] Switch B.C to WALL FOR REWARD"
        
        end subroutine switch_BC_2Wall
c------------------------------------------------------------------

c------------------------------------------------------------------
        subroutine switch_BC_2Drichlet
c Change Wall-BC According to the indicies obtained before 
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
        
        integer ie,iface,ix,iy,iz ! Iteration
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
        if (NUMCTRL.ne.0) then 
            do idx=1,NUMCTRL
            ie=info_agt(1,idx)         ! Local indicies
            iface=info_agt(2,idx)      ! Number of face
            CBC(iface,ie,1)='v  '
            enddo
        endif
        if (NID.eq.0) print*,"[DRL] Switch B.C to Dirichlet FOR CTRL"
        
        end subroutine switch_BC_2Drichlet
c------------------------------------------------------------------





c------------------------------------------------------------------
        subroutine compute_utau
c=============================================
c       Define variable
c=============================================
        include 'SIZE'
        include 'TOTAL'

        real x0(3)
        data x0 /0.0, 0.0, 0.0/ 
        save x0

        integer icalld
        save    icalld
        data    icalld /0/

        real atime,timel
        save atime,timel

        integer ntdump
        save    ntdump

        real    rwk(INTP_NMAX,ldim+1) ! r, s, t, dist2
        integer iwk(INTP_NMAX,3)      ! code, proc, el 
        save    rwk, iwk

        integer nint, intp_h
        save    nint, intp_h

        logical iffpts
        save iffpts
        
        real XLEN,ZLEN

        real xint(INTP_NMAX),yint(INTP_NMAX),zint(INTP_NMAX)
        real yi
        save xint, yint, zint
        save igs_x, igs_z

        parameter(nstat=9)
        real ravg(lx1*ly1*lz1*lelt,nstat)
        real stat(lx1*ly1*lz1*lelt,nstat)
        real stat_y(INTP_NMAX*nstat)
        save ravg, stat, stat_y

        save dragx_avg

        logical ifverbose,ifexist
        common /gaaa/  wo1(lx1,ly1,lz1,lelv)
     &              ,  wo2(lx1,ly1,lz1,lelv)
     &              ,  wo3(lx1,ly1,lz1,lelv)

        real tplus
        real tmn, tmx

        integer bIDs(1)
        save iobj_wall

        !-------------------------------
        n     = nx1*ny1*nz1*nelv
        nelx  = XNEL
        nely  = YNEL
        nelz  = ZNEL    
        ! NOTE: THIS have to be modified when it comes to WING
        XLEN = glmax(xm1,n)
        ZLEN = glmax(zm1,n)
        !-------------------------------


        if (istep.eq.0) then
            bIDs(1) = 1
            call create_obj(iobj_wall,bIDs,1)
            nm = iglsum(nmember(iobj_wall),1)
            if(nid.eq.0) write(6,*) 'obj_wall nmem:', nm 
        !     call prepost(.true.,'  ')
            call set_obj
        endif
        ubar = glsc2(vx,bm1,n)/volvm1
        e2   = glsc3(vy,bm1,vy,n)+glsc3(vz,bm1,vz,n)
        e2   = e2/volvm1
        if (nfield.gt.1) then
            tmn  = glmin(t,n)
            tmx  = glmax(t,n)
        endif
        if(nid.eq.0) write(6,2) time,ubar,e2,tmn,tmx
    2               format(1p5e13.4,' monitor')

        if (time.lt.tSTATSTART) return

c     What follows computes some statistics ...
        if(icalld.eq.0) then
        if(nid.eq.0) write(6,*) 'Start collecting statistics ...'

        nxm = 1 ! mesh is linear
        ! call interp_setup(intp_h,0.0,nxm,nelt)
        nint = 0
        ! if (nid.eq.0) then
        ! nint = INTP_NMAX
        ! ! Use the list XCINT to fill the xint
        ! call cfill(xint,XCINT,size(xint))
        
        ! do i = 1,INTP_NMAX 
        !     yi = (i-1.)/(INTP_NMAX-1)
        !     yint(i) = tanh(BETAM*(2*yi-1))/tanh(BETAM)
        ! enddo
        
        ! call cfill(zint,ZCINT,size(zint))
        
        ! endif
        iffpts = .true. ! dummy call to find points
        ! Now we use interp_nfld to find those points for statistics
!         call interp_nfld(stat_y,ravg,1,xint,yint,zint,nint,
!      $                   iwk,rwk,INTP_NMAX,iffpts,intp_h)
!         iffpts = .false.
!         call gtpp_gs_setup(igs_x,nelx     ,nely,nelz,1) ! x-avx
!         call gtpp_gs_setup(igs_z,nelx*nely,1   ,nelz,3) ! z-avg
        call rzero(ravg,size(ravg))
        dragx_avg = 0
        atime     = 0
        timel     = time
        ntdump    = int(time/tSTATFREQ)
        icalld = 1
        endif

        dtime = time - timel
        atime = atime + dtime

        ! averaging over time
        if (atime.ne.0. .and. dtime.ne.0.) then
        beta      = dtime / atime

        alpha     = 1. - beta
        ifverbose = .false.
        ! call avg1(ravg(1,1),vx   ,alpha,beta,n,'uavg',ifverbose)
        ! call avg2(ravg(1,2),vx   ,alpha,beta,n,'urms',ifverbose)
        ! call avg2(ravg(1,3),vy   ,alpha,beta,n,'vrms',ifverbose)
        ! call avg2(ravg(1,4),vz   ,alpha,beta,n,'wrms',ifverbose)
        ! call avg3(ravg(1,5),vx,vy,alpha,beta,n,'uvmm',ifverbose)
        ! call avg1(ravg(1,6),t    ,alpha,beta,n,'tavg',ifverbose)
        ! call avg2(ravg(1,7),t    ,alpha,beta,n,'trms',ifverbose)
        ! call avg3(ravg(1,8),vx,t ,alpha,beta,n,'utmm',ifverbose)
        ! call avg3(ravg(1,9),vy,t ,alpha,beta,n,'vtmm',ifverbose)
        
        call torque_calc(1.0,x0,.false.,.false.) ! compute wall shear
        dragx_avg = alpha*dragx_avg + beta*dragx(iobj_wall)
        endif

        timel = time

        ! write statistics to file
        if(istep.gt.0 .and. time.gt.(ntdump+1)*tSTATFREQ) then
        ! averaging over statistical homogeneous directions (x-z)
        !     do i = 1,nstat
        !         call planar_avg(wo1      ,ravg(1,i),igs_x)
        !         call planar_avg(stat(1,i),wo1      ,igs_z)
        !     enddo

!             ! extract data along wall normal direction (1D profile)
!             call interp_nfld(stat_y,stat,nstat,xint,yint,zint,nint,
!      $                    iwk,rwk,INTP_NMAX,iffpts,intp_h)

            ntdump = ntdump + 1
            if (nid.ne.0) goto 998 

            rho    = param(1)
            dnu    = param(2)
            A_w    = XLEN * ZLEN
            tw     = dragx_avg / A_w
            u_tau  = sqrt(tw / rho)
            Re_tau = u_tau / dnu
            tplus  = time * u_tau**2 / dnu

            write(6,*) "[DRL] REWARD: u_tau",u_tau, "Re_tau", Re_tau
            
            ! Write Down the Reward from calculation
            inquire(file='reward.dat',exist=ifexist)
            if (ifexist) then
            open(unit=55,file='reward.dat',status='old',
     &         position='append',action='write')
            else
            open(unit=55,file='reward.dat',status='new',action='write')
            write(55,'(A)') 
     $    '  time ut Ret t+ mu'

            endif 
            write(55,3)
     &        time,
     &        u_tau,
     &        Re_tau,
     &        tplus,
     &        dnu
            close(55)
!             open(unit=56,file='vel_fluc_prof.dat')
!             write(56,'(A,1pe14.7)') '#time = ', time
!             write(56,'(A)') 
!      $    'y    y+    uu    vv    ww    uv'

!             open(unit=57,file='mean_prof.dat')
!             write(57,'(A,1pe14.7)') '#time = ', time
!             write(57,'(A)') 
!      $    '#  y    y+    Umean'

!             do i = 1,nint
!                 yy = 1+yint(i)
!                 write(56,3) 
!      &           yy,
!      &           yy*Re_tau,
!      &           (stat_y(1*nint+i)-(stat_y(0*nint+i))**2)/u_tau**2,
!      &           stat_y(2*nint+i)/u_tau**2,
!      &           stat_y(3*nint+i)/u_tau**2,
!      &           stat_y(4*nint+i)/u_tau**2

!                 write(57,3) 
!      &           yy,
!      &           yy*Re_tau, 
!      &           stat_y(0*nint+i)/u_tau

!             enddo
!             close(56)
!             close(57)
    3        format(1p15e17.9)

  998       endif

            return
            end subroutine compute_utau
c------------------------------------------------------------------


c------------------------------------------------------------------
            subroutine set_obj  ! define objects for surface integrals
c=============================================
c       Define variable
c=============================================
            include 'SIZE'
            include 'TOTAL'

            integer e,f,eg

c=============================================
c       Functions
c=============================================
            nobj = 1
            iobj = 0
            do ii=nhis+1,nhis+nobj
                iobj = iobj+1
                hcode(10,ii) = 'I'
                hcode( 1,ii) = 'F'
                hcode( 2,ii) = 'F'
                hcode( 3,ii) = 'F'
                lochis(1,ii) = iobj
            enddo
            nhis = nhis + nobj

            if (maxobj.lt.nobj) then 
            call exitti('increase maxobj in SIZE$',nobj)
            endif
            
            nxyz  = nx1*ny1*nz1
            nface = 2*ndim

            do e=1,nelv
            ! We need to tune this
            if (abs(ym1(1,1,1,e)) .gt. 0.9) then
            do f=1,nface
                if (cbc(f,e,1).eq.'W  ') then
                    iobj  = 1
                    if (iobj.gt.0) then
                    nmember(iobj) = nmember(iobj) + 1
                    mem = nmember(iobj)
                    eg  = lglel(e)
                    object(iobj,mem,1) = eg
                    object(iobj,mem,2) = f
                    print*, iobj,mem,f,eg,e,nid,' OBJ'
                    endif
                endif ! if (cbc.eq.W)
            enddo
            endif
            enddo

            return
            end
c------------------------------------------------------------------

