c==============================================
c DRL subroutines for ACTION 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================



c------------------------------------------------------------------
        subroutine drl_action
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include 'PARALLEL'
        include 'INPUT'
c=============================================
c       Function
c=============================================
        
      !   if (ISTEP.eq.0) then
cc STEP 1: Do nothing 
            if (NID.eq.0) then 
                  print *,"-------------------------"
                  print *,"[ACTION] STANDING BY"
                  print *,"-------------------------"
            endif 
cc STEP 3: Computing Fluctuation and Actuating on the wall 
c-----------------------------------------------
      !   else ! When the simulation is running: 
            ! call wall_aft_vel               ! TEST the actutation on the wall 
            call recv_Actions() 
            call nekgsync()
      
      ! Zero-net-mass-flux 
      if (int(PARAM(90)).gt.0) then 
            call znmf_avg()
      endif 

      !   endif ! if (ISTEP.eq.0)

        return 
        end subroutine drl_action
c------------------------------------------------------------------



c------------------------------------------------------------------
      subroutine recv_Actions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'PARALLEL'
      include 'DRL'         
      include 'mpif.h'
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      !-----------------------
      integer totLine,ntot
      ! integer gllel
      integer glbid,fceid,nidid,nididnek,lclid
      integer ix,iy,iz
      real    jetval
      character*4 str, str1
      !-----------------------
      
      !--------------------------
      ! MPI 
      integer parent_comm, ierr
      integer idx_buffer(TOTCTRL)
      integer fce_buffer(TOTCTRL)
      real    act_buffer(TOTCTRL)
      integer sendLen 
      integer request_send, request_recv, status(mpi_status_size)
      !--------------------------
      logical ifexist
      character*13 fNAME
      real act_buff(LX1,LY1,LZ1,LELT),act_i

c=============================================
c       Function
c=============================================

      if (NUMCTRL.ne.0) then 

      call MPI_RECV(act_buffer,TOTCTRL,MPI_DOUBLE,
     &            0,NID+90000,DRL_COMM,
     &            MPI_STATUS_IGNORE,ierr)

      ! print *,"[ACTION] UPDATE NID=",NID
      call nekgsync()
      else
      ! print *,"[ACTION] UPDATE NID=",NID
      call nekgsync()
      endif
! Update 
!----------------------------------------
      ntot=LX1*LY1*LZ1*LELT
      ! INIT THE buffer with dumi value YW: NOT Needed!
      ! call cfill(ACTIONS(1,1,1,1),dumi,ntot)
      call rzero(act_buff(1,1,1,1),ntot)
      call ifill(msk_act(1,1,1,1),0,ntot)
      ! Update 
      do il=1,NUMCTRL 
            glbid=info_agt(1,il)
            lclid=gllel(glbid)
            fceid=info_agt(2,il)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            act_i=act_buffer(il)
            ! ACTIONS(ix,iy,iz,fceid,lclid)=act_buffer(il)
            ! YW Modified Nov 4th 2024, NO NEED of FACE!
            act_buff(ix,iy,iz,lclid)=act_i
            if (int(PARAM(90)).le.0) then ! If ZNMF is not used 
            msk_act(ix,iy,iz,lclid)=1
            endif ! 
      enddo 

      call copy(ACTIONS(1,1,1,1),act_buff(1,1,1,1),NTOT)

c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      if (ISTEP.le.2) then 
      if (NUMCTRL.gt.0) then 
      write(str,"(i4.4)") NID
      write(str1,"(i4.4)") ISTEP
      open(10001,file="RECV-ACTION.txt"//str//str1)
      write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, "
      do il = 1,NUMCTRL
      lclid=info_agt(1,il)
      lclid=gllel(lclid)
      ix=info_agt(3,il)
      iy=info_agt(4,il)
      iz=info_agt(5,il)
      write(10001,*) lclid,
     $      (pos_agt(jl,il), jl=1,NDIM),
     $       ACTIONS(ix,iy,iz,lclid) 
      enddo
      close(10001)
      endif ! if NUMCTRL .ne. 0 
      endif ! if ISTEP .le. 2
#endif 
!----------------------------------------

      if (NID.eq.0) then
            print *, "-------------------------"
            print *, "[ACTION] UPDATED"
            print *, "-------------------------"
      endif
      
      end subroutine recv_Actions 
c------------------------------------------------------------------

c------------------------------------------------------------------
      subroutine ACTUATE_JET(vf,isfind,ix,iy,iz,iside,eg)
! Used for userbc: Impose velocity 
! c=============================================
! c       Define variable
! c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,iside,eg,iel
      integer ierr,IO_STATUS,rface
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      real vf
      logical isfind
! c=============================================
! c       Function
! c=============================================
      iel = gllel(eg)
      vf =0.0
      isfind =.FALSE.
      
      if (NUMCTRL.ne.0) then 
            ! We know there is only one face! 
            rface=info_agt(2,1)
            recctrl=msk_act(ix,iy,iz,iel)
            if (recctrl.ne.0) then 
                  vf=ACTIONS(ix,iy,iz,iel)
                  isfind=.TRUE.
            endif
      endif 

      end subroutine ACTUATE_JET
c--------------------------------------------------


c--------------------------------------------------

      subroutine znmf_avg() 
c Subroutine for average the Actions for ZNMF conditions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real velV(LX1,LY1,LZ1,LELT)
      real actV(LX1,LY1,LZ1,LELT)
      real avgVZ(LX1,LY1,xnel,ynel)
      real avgVX(LX1,LY1,ynel,znel)
      real wrk_buff(LX1,LY1,LZ1,LELT)
      real vf,va,vo
      real xi, uti 
      
      ! Handlers for gop average
      integer igs_x,igs_z
      save igs_x,igs_z
      ! For test 
      integer ilx, ily
      character*4 str, str1
c=============================================
c       Functions
c=============================================
      ntot=LX1*LY1*LZ1*LELT

      ! STEP 1: Spatial Average based on X- Z-dir 
      call copy(velV(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      call rzero(actV,ntot) 
      ! Rescale the actions by local u_tau
#ifdef UTAU
      do il = 1, NUMCTRL
        xi = pos_agt(1,il) 
        call X2Utau(xi,uti) 
        iel = info_agt(1,il) 
        iel = gllel(iel)
        ix = info_agt(3,il) 
        iy = info_agt(4,il) 
        iz = info_agt(5,il) 
        vf = velV(ix,iy,iz,iel) 
        va = vf * uti 
        actV(ix,iy,iz,iel) = va 
      enddo
      call copy(velV(1,1,1,1),actV(1,1,1,1),ntot)
      if (NID.eq.0) print *, "[ACTION] Rescale"
#endif 

      call copy(actV(1,1,1,1),velV(1,1,1,1), ntot)    
        
      ! in deterministic, it should be the same of using facind without
      ! dssum 
      !call dssum(velV,LX1,LY1,LZ1)
!#ifdef YWDEBUG
!        if (NID.eq.0) print *, "[ACTION] DSSUM" 
!#endif 
      if (rwd_zavg) then 
            call z_averaging(velV,avgVZ)
            call z_avg_reshape(velV,avgVZ)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] Z-AVG!"
#endif
            endif ! if (rwd_zavg)

            ! Do streamwise average if it allowed/defined
            if (rwd_xavg) then 
            call z_averaging(velV,avgVX)
            call z_avg_reshape(velV,avgVX)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] X-AVG!"
#endif
            endif 

      ! Step 2: Subtract the mean of the action 
            do ilx = 1,NUMCTRL
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            ifs=info_agt(2,ilx)
            call impose_ivalue(1,msk_act,iel,ifs) 
            enddo
            call rzero(wrk_buff(1,1,1,1),ntot)
            call sub3(wrk_buff,actV,velV,ntot)       
            call copy(ACTIONS(1,1,1,1),wrk_buff(1,1,1,1),ntot) 
      ! Save the result for testing
!#ifdef YWDEBUG  
!      if (ISTEP.eq.1) then 
!      call outpost(velV,actV,wrk_buff,
!     $             velV,t,'act')
!      if (NID.eq.0) print *, "[ACTION] SAVED BUFFER FOR TEST"
!      endif 
!#endif 

c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      if (NID.eq.0) print *, "[ACTION] ZNMF AVERAGED"
      if (ISTEP.eq.1) then 
            if (NUMCTRL.gt.0) then 
            write(str,"(i4.4)") NID
            write(str1,"(i4.4)") ISTEP
            open(10001,file="ZNMF-ACTION.txt"//str//str1)
            write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, ","MEAN, ","BEFORE" 
            do ilx = 1,NUMCTRL
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            ix=info_agt(3,ilx)
            iy=info_agt(4,ilx)
            iz=info_agt(5,ilx)
            write(10001,*) iel,
     $      (pos_agt(ily,ilx), ily=1,NDIM),
     $       ACTIONS(ix,iy,iz,iel), 
     $       velV(ix,iy,iz,iel),
     $       actV(ix,iy,iz,iel)
            enddo
            close(10001)
            endif
      endif 
#endif 

      end subroutine znmf_avg
c--------------------------------------------------



c--------------------------------------------------
      subroutine moving_smooth_action(i_evolv,ndrl)
c Subourtine for Moving average for smoothing the action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real ctrl_val(LX1,LY1,LZ1,LELT)
      real old_ctrl_val(LX1,LY1,LZ1,LELT)
      real vf,va,vo,reduce_r
      common /ctrl_cache/ ctrl_val,old_ctrl_val
      ! save ctrl_val,old_ctrl_val
c=============================================
c       Function
c=============================================
      ntot = LX1*LY1*LZ1*LELT

      ! if (NID.eq.0) print *, "[ACTION] SMOOTHING",i_evolv,ndrl

      ! Copy for the worker array
      call copy(ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      
      ! Initialize the reference, 
      ! which means the first NDRL we do not decay the action
      if (ISTEP.eq.1) then 
      call copy(old_ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      endif 
      ! endif


      do il=1,NUMCTRL
      iel=info_agt(1,il)
      iel=gllel(iel)
      ix=info_agt(3,il)
      iy=info_agt(4,il)
      iz=info_agt(5,il)
      vo = old_ctrl_val(ix,iy,iz,iel)
      vf = ctrl_val(ix,iy,iz,iel)

#ifdef YWDEBUG
      if (NUMCTRL.ne.0 .and. il.le.5) then 
      print *,'VO AND VF',vo,vf
      endif 
#endif

      ! ------------ Update Action using moving average ------------
      ! if (vf.ne.vo) then
      reduce_r = real(i_evolv)/real(ndrl)
      vf=vo+(vf-vo)*reduce_r
      ctrl_val(ix,iy,iz,iel)=vf
      ! endif 
      !------------------------------------------------------

#ifdef YWDEBUG
      if (NUMCTRL.ne.0 .and. il.le.5) then 
      print *,'Smooth VO from VF',vo,vf,reduce_r
      endif 
#endif 
      
      enddo ! do il=1,NUMCTRL
      

      ! At the end of evolv also update reference 
      if (i_evolv.eq.ndrl) then 
            call copy(old_ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      endif 

      ! ! Update actions 
      call copy(ACTIONS(1,1,1,1),ctrl_val(1,1,1,1),ntot)
      

      end subroutine moving_smooth_action
c--------------------------------------------------


c--------------------------------------------------
      subroutine exp_smooth_action(i_evolv,ndrl)
c Exponentially Smoothing the Action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real ctrl_val(LX1,LY1,LZ1,LELT)
      real vf,va,vo,reduce_r
      ! save ctrl_val,old_ctrl_val
c=============================================
c       Function
c=============================================
      ntot = LX1*LY1*LZ1*LELT

      ! if (NID.eq.0) print *, "[ACTION] SMOOTHING",i_evolv,ndrl

      ! Copy for the worker array
      call copy(ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      
      do il=1,NUMCTRL
            iel=info_agt(1,il)
            iel=gllel(iel)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            vf = ctrl_val(ix,iy,iz,iel)
            vo = vf
            ! ------- UPDATE ACTION -------------
            reduce_r = real(i_evolv)/real(ndrl)
            call smooth_step(vf,reduce_r)
            ctrl_val(ix,iy,iz,iel) = vf
            !------------------------------------
#ifdef YWDEBUG
            if (NUMCTRL.ne.0 .and. il.le.5) then 
            print *,NID,'Smooth VO from VF',vo,vf,reduce_r
            endif 
#endif 
      enddo 
      ! ! Update actions 
      call copy(ACTIONS(1,1,1,1),ctrl_val(1,1,1,1),ntot)
      

      end subroutine exp_smooth_action
c--------------------------------------------------





c--------------------------------------------------
      subroutine smooth_step(step,x)
c
c     Smooth step function:
c     x<=0 : step(x) = 0
c     x>=1 : step(x) = 1
c     Non-continuous derivatives at x=0.02 and x=0.98
c
      implicit none

      real x,step

      if (x.le.0.02) then
            step = 0.0
      else
            if (x.le.0.98) then
            step = 1./( 1. + exp(1./(x - 1.) + 1./x) )
            else
            step = 1.
            end if
      end if

      return  
      end subroutine smooth_step
c--------------------------------------------------

c--------------------------------------------------
      subroutine impose_ivalue(vf,buffer,iel,iface)
c Subroutine for imposing the action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      integer iel,iface
      integer ix,iy,iz
      integer KX1,KX2,KY1,KY2,KZ1,KZ2
      integer buffer(LX1,LY1,LZ1,LELT)
      integer vf
      call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
      do iz=KZ1,KZ2
      do iy=KY1,KY2
      do ix=KX1,KX2
      buffer(ix,iy,iz,iel) = vf
      enddo
      enddo
      enddo

      end subroutine impose_ivalue
c--------------------------------------------------




