#ifndef CVODE
      subroutine cv_setsize

      include 'SIZE'
      if(nio.eq.0) write(6,*) 'ABORT: not compiled with CVODE support!'
      call exitt

      return
      end
c----------------------------------------------------------------------
      subroutine cv_init

      include 'SIZE'
      if(nio.eq.0) write(6,*) 'ABORT: not compiled with CVODE support!'
      call exitt

      return
      end
c----------------------------------------------------------------------
      subroutine cdscal_cvode

      include 'SIZE'
      if(nio.eq.0) write(6,*) 'ABORT: not compiled with CVODE support!'
      call exitt

      return
      end
c----------------------------------------------------------------------
#else
      subroutine cv_setsize

      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      integer*8 i8glsum

      nxyz = lx1*ly1*lz1

      ! set local ODE size
      cv_nlocal = 0
      do i = 2,nfield
         if (ifcvfld(i)) then
           ntot = nxyz*nelfld(i)
           cv_nlocal = cv_nlocal + ntot
         endif
      enddo

      ! check array size is large enough
      if (cv_nlocal .gt. cv_lysize) then
        if(nio.eq.0) write(6,*)
     &  'ABORT cv_setsize(): workspace too small, check SIZE! ' 
        call exitt
      endif 

      ! determine global ODE size
      cv_nglobal = i8glsum(cv_nlocal,1)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_init

      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      common /CVWRK1/  y0(cv_lysize) 

      ! cvode will not allocate these arrays
      common /cv_rout/ rout(6),rpar(1)

      integer*8 iout,ipar
      integer cvcomm
      common /cv_iout/ iout(21),ipar(1),cvcomm

      real cv_atol_(lx1,ly1,lz1,lelt,ldimt)
      common /CV_YDOT/ cv_atol_ ! used as scratch

      integer cv_meth

      real*8 etime1
      character*15 txt_meth,txt_itask

      real atol_t(ldimt)


      nxyz = lx1*ly1*lz1
      ifcvodeinit   = .false.

      if(nio.eq.0) write(*,*) 'Initializing CVODE ...'
      etime1=dnekclock_sync()

      cv_timel = -1
      call cv_rstat

      ! set solver parameters
      cv_itask    = param(160)
      cv_meth     = param(161) ! AM or BDF
      cv_rtol     = param(163) 
      cv_dtmax    = param(164)
      cv_sigs     = param(165) 
      cv_delt     = param(166)
      cv_ipretype = param(167) ! 0: no, 1:left, 2: right
      cv_maxl     = 20         ! max dimension of Krylov subspace
      cv_iatol    = 2          ! 1: scalar 2: vector

      ! setup absolute tolerances
      if (cv_iatol.eq.1) then
         cv_atol(1) = param(162)
      else if (cv_iatol.eq.2) then
         do i = 2,nfield
            if (ifcvfld(i)) then
               ntot = nxyz*nelfld(i)
               call cfill(cv_atol_(1,1,1,1,i-1),atol(i),ntot)
            endif
         enddo
         call cvpack(cv_atol,cv_atol_,.false.)
      endif

      ! initialize vector module
      call create_comm(cvcomm)
#ifdef MPI
      call fnvinitp(cvcomm, 1, cv_nlocal, cv_nglobal, ier)
#else
      call fnvinits(1, cv_nlocal, ier)
#endif
      if (ier.ne.0) then
        write(*,'(a,i3)') 'ABORT cv_init(): fnvinitp ier=', ier
        call exitt
      endif

      ! initialize cvode
      call cvpack(y0,t,.false.)
      call fcvmalloc(time, y0, cv_meth, itmeth, cv_iatol,
     &               cv_rtol, cv_atol, iout, rout, ipar, rpar, ier)
      if (ier.ne.0) then
        write(*,'(a,i3)') 'ABORT cv_init(): fcvmalloc ier=', ier
        call exitt
      endif

      ! initialize linear solver
      call fcvspGMR(cv_ipretype,igstype,cv_maxl,cv_delt,ier)
      if (ier .ne. 0) then
        write(*,'(a,i3)') 'ABORT cv_init(): fcvspgmr ier=', ier
        call exitt
      endif

      ! preconiditoner
      if(cv_ipretype.gt.0) call fcvspilssetprec(1, ier)

      ! enable user-supplied Jacobian routine
      call fcvspilssetjac(1, ier)

      ! print solver settings
      etime1 = dnekclock_sync() - etime1
      if(nio.eq.0) then
        if(cv_meth.eq.1) txt_meth='AM'      
        if(cv_meth.eq.2) txt_meth='BDF'      
        write(6,'(A,A)')         '   integration method             : ',
     &                         txt_meth
        if(cv_itask.eq.1) txt_itask='NORMAL'      
        if(cv_itask.eq.3) txt_itask='NORMAL_TSTOP'      
        write(6,'(A,A)')         '   integration mode               : ',
     &                         txt_itask
        write(6,'(A,i10)')       '   Nglobal                        : ',
     &                         cv_nglobal
        write(6,'(A,1pe8.1)')    '   relative tolerance             : ',
     &                         cv_rtol

        do i = 2,nfield
           if (ifcvfld(i)) then
              if (cv_iatol.eq.1) then
                 dd = cv_atol(1)
              else
                 ntot = nxyz*nelfld(i)
                 dd = vlmax(cv_atol_(1,1,1,1,i-1),ntot)
              endif

              write(6,1000) i, dd
 1000         format(3x,'absolute tolerance field ',i3,'   : ',1pe8.1)
           endif
        enddo

        write(6,'(A,i3)')     '   krylov dimension               : ',
     &                         cv_maxl
        write(6,'(A,f5.3)')   '   ratio linear/non-linear tol    : ',
     &                         cv_delt
        write(6,'(A,i3)')     '   preconditioner                 : ',
     &                         int(param(167))
        write(6,'(A,1pe8.1)') '   dt_max                         : ',
     &                         cv_dtmax
        write(6,'(A,f5.3)')  '   increment factor         DQJ   : ',
     &                         cv_sigs
        write(6,'(A,g15.3,A,/)') ' done :: initializing CVODE',
     &                         etime1, ' sec'
      endif

      ifcvodeinit = .true.

      return
      end
c----------------------------------------------------------------------
      subroutine cdscal_cvode
c
c     Top level driver for CVODE
c     webpage: https://computation.llnl.gov/casc/sundials 
c
c     Integrate the IVP d/dt[y] = f(y(t),t); y(t=t0) := f0
c     using BDF(stiff) or AM(non-stiff).
c     f denotes the RHS function and is evaluated in fcvfun().
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      common /CVWRK1/  y(cv_lysize) 

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      integer*8 iout,ipar
      integer cvcomm
      common /cv_iout/ iout(21),ipar(1),cvcomm

      nxyz = lx1*ly1*lz1
      ntot = nxyz * nelv

      if (.not.ifcvodeinit) then
        write(6,*) 'ABORT: CVODE was not initialized!'
        call exitt
      endif 

      ! save old output values
      if (cv_itask.eq.1) then
        do i=1,21
           iout_save(i) = iout(i)
        enddo
      endif

      ! save coord and mesh vel
      if(ifmvbd) then
        call copy(xm1_,xm1,ntot)           
        call copy(ym1_,ym1,ntot)           
        if (if3d) call copy(zm1_,zm1,ntot) 
        call copy(wx_,wx,ntot)           
        call copy(wy_,wy,ntot)           
        if (if3d) call copy(wz_,wz,ntot) 
      endif

      ! save velocities                  
      call copy(vx_,vx,ntot)             
      call copy(vy_,vy,ntot)            
      if (if3d) call copy(vz_,vz,ntot)  

      ! MAIN solver call
      time_ = time
      call fcvsetrin('MAX_STEP',cv_dtmax,ier)
c      call fcvsetiin('MAX_ORD' ,3       ,ier)

      if (cv_itask.eq.3) then
        if(istep.gt.1) then 
          call cvpack(y,t,.false.)
          call fcvreinit(timef,y,cv_iatol,cv_rtol,cv_atol,ier)
          if (ier .ne. 0) then
            write(*,'(a,i3)') 'ABORT: fcvsetrin ier=', ier
            call exitt
          endif
          cv_timel = 0
          call cv_rstat
        endif
        call fcvsetrin('STOP_TIME',time,ier)
      endif

      call fcvode(time,tout,y,cv_itask,ier)
      time = time_

      if (ier.lt.0) then
         if (nid.eq.0) then
            write(*,'(a)') ' Restart integrator and try again  ...'
         endif
         call cvpack(y,t,.false.)
         call fcvreinit(timef,y,cv_iatol,cv_rtol,cv_atol,ier)
         cv_timel = 0
         call cv_rstat
         call fcvode(time,tout,y,cv_itask,ier)
         time = time_
      endif
        
      if (ier.lt.0) then
         if (nid.eq.0) then
            write(*,'(a,i3)') 'ABORT: fcvode returned ier =', ier
         endif
         call exitt
      endif

      cv_istep = cv_istep + 1 
      call cvunpack(t,y)

      ! restore velocities               
      call copy(vx,vx_,ntot)             
      call copy(vy,vy_,ntot)             
      if (if3d) call copy(vz,vz_,ntot) 
      if (param(99).gt.0) call set_convect_new(vxd,vyd,vzd,vx,vy,vz)
 
      ! restore coord and mesh vel 
      if(ifmvbd) then        
        call copy(xm1,xm1_,ntot)           
        call copy(ym1,ym1_,ntot)          
        if (if3d) call copy(zm1,zm1_,ntot) 
        call copy(wx,wx_,ntot)           
        call copy(wy,wy_,ntot)           
        if (if3d) call copy(wz,wz_,ntot) 

        call cv_eval_geom                      
      endif

      ! print solver statistics 
      if (nid.eq.0) call cv_pstat

      return
      end
c----------------------------------------------------------------------
      subroutine cv_upd_v
c
      include 'SIZE'
      include 'TSTEP'
      include 'DEALIAS'
      include 'SOLN'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(vx,vx_,vxlag,ntot,cv_ab,nbd)
      call sumab(vy,vy_,vylag,ntot,cv_ab,nbd)
      if(if3d) call sumab(vz,vz_,vzlag,ntot,cv_ab,nbd)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_upd_w
c
      include 'SIZE'
      include 'TSTEP'
      include 'MVGEOM'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(wx,wx_,wxlag,ntot,cv_ab,nbd)
      call sumab(wy,wy_,wylag,ntot,cv_ab,nbd)
      if(if3d) call sumab(wz,wz_,wzlag,ntot,cv_ab,nbd)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_upd_coor
c
      include 'SIZE'
      include 'TSTEP'
      include 'GEOM'
      include 'MVGEOM'
      include 'INPUT'
      include 'CVODE'

      common /CVWRK2/  vx_ (lx1,ly1,lz1,lelv) 
     &                ,vy_ (lx1,ly1,lz1,lelv)
     &                ,vz_ (lx1,ly1,lz1,lelv)
     &                ,xm1_(lx1,ly1,lz1,lelv)
     &                ,ym1_(lx1,ly1,lz1,lelv)
     &                ,zm1_(lx1,ly1,lz1,lelv)
     &                ,wx_ (lx1,ly1,lz1,lelv)
     &                ,wy_ (lx1,ly1,lz1,lelv)
     &                ,wz_ (lx1,ly1,lz1,lelv)

      COMMON /SCRSF/ dtmp(lx1*ly1*lz1*lelv)

      ntot = lx1*ly1*lz1*nelv

      call sumab(dtmp,wx_,wxlag,ntot,cv_abmsh,nabmsh)
      call add3 (xm1,xm1_,dtmp,ntot)

      call sumab(dtmp,wy_,wylag,ntot,cv_abmsh,nabmsh)
      call add3 (ym1,ym1_,dtmp,ntot)

      if(if3d) then
        call sumab(dtmp,wz_,wzlag,ntot,cv_abmsh,nabmsh)
        call add3 (zm1,zm1_,dtmp,ntot)
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine fcvfun (time_, y, ydot, ipar, rpar, ier)
c
c     Compute RHS function f (called within cvode)
c     CAUTION: never touch y! 
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
      include 'CVODE'

      real time_,y(*),ydot(*),rpar(*)
      integer*8 ipar(*)

      real w1(lx1,ly1,lz1,lelt),
     $     w2(lx1,ly1,lz1,lelt),
     $     w3(lx1,ly1,lz1,lelt)

      real ydott(lx1,ly1,lz1,lelt,ldimt)
      common /CV_YDOT/ ydott

      ifcvfun = .true.
      etime1  = dnekclock()
      time    = time_   
      nxyz    = lx1*ly1*lz1
      ntotv   = nxyz*nelv
       
      if (time.ne.cv_timel) then
         call cv_settime     
 
         if(nio.eq.0) write(6,10) istep,time,time-cv_timel
  10       format(4x,i7,2x,'t=',1pE14.7,'  stepsize=',1pE13.4)

         call cv_upd_v
         call copy(w1,vx,ntotv)
         call copy(w2,vy,ntotv)
         if (if3d) call copy(w3,vz,ntotv)

         if (ifmvbd) then
            call cv_upd_coor 
            call cv_eval_geom
            call cv_upd_w
            call sub2(vx,wx,ntotv)
            call sub2(vy,wy,ntotv)
            if (if3d) call sub2(vz,wz,ntotv)
         endif
        
         if (param(99).gt.0) call set_convect_new(vxd,vyd,vzd,vx,vy,vz)

         call copy(vx,w1,ntotv)
         call copy(vy,w2,ntotv)
         if (if3d) call copy(vz,w3,ntotv)

         cv_timel = time          
      endif

      call cvunpack(t,y)          

      if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'fcvfun',
     $                                 ifdqj

      ifield = 1
      call vprops ! we may use fluid properties somewhere
      do ifield=2,nfield
         if (ifcvfld(ifield)) call vprops
      enddo  

      do ifield=2,nfield
         if (ifcvfld(ifield)) then
           ntot = nxyz*nelfld(ifield)
           call makeq

           if (iftmsh(ifield)) then                                
              call dssum(bq(1,1,1,1,ifield-1),lx1,ly1,lz1)
              call col2(bq(1,1,1,1,ifield-1),bintm1,ntot)

              call col3(w1,vtrans(1,1,1,1,ifield),bm1,ntot)      
              call dssum(w1,lx1,ly1,lz1)                        
              call col2(w1,bintm1,ntot)                           
           else                                                    
              call copy(w1,vtrans(1,1,1,1,ifield),ntot)
           endif    

           call invcol3(ydott(1,1,1,1,ifield-1),bq(1,1,1,1,ifield-1),
     &                  w1,ntot)           
         endif
      enddo

      if (ifgsh_fld_same) then ! all fields are on the v-mesh
         istride = lx1*ly1*lz1*lelt
         call nvec_dssum(ydott,istride,nfield-1,gsh_fld(1))
      else
         do ifield = 2,nfield
            if (ifcvfld(ifield) .and. gsh_fld(ifield).ge.0) then
               if(.not.iftmsh(ifield))       
     &         call dssum(ydott(1,1,1,1,ifield-1),lx1,ly1,lz1)
            endif
         enddo
      endif

      do ifield = 2,nfield
         if (ifcvfld(ifield)) then                                
           ntot = nxyz*nelfld(ifield)
           if (.not.iftmsh(ifield) .and. gsh_fld(ifield).ge.0) then
              call col2(ydott(1,1,1,1,ifield-1),binvm1,ntot)
           endif
         endif
      enddo

      call cvpack(ydot,ydott,.true.)

      tcvf = tcvf + dnekclock()-etime1 
      ncvf = ncvf + 1 

      ier = 0
      ifcvfun = .false.

      return
      end

#endif
c----------------------------------------------------------------------
      subroutine cv_eval_geom

      call glmapm1
      call geodat1
      call geom2
      call volume
      call setinvm
      call setdef

      return
      end

c----------------------------------------------------------------------
      subroutine cv_settime

      include 'SIZE'
      include 'TSTEP'
      include 'CVODE'

      cv_dtNek = time - timef ! stepsize between nek and cvode    

      cv_dtlag(1) = cv_dtNek 
      cv_dtlag(2) = dtlag(2)
      cv_dtlag(3) = dtlag(3)

      call rzero(cv_abmsh,3)
      call setabbd(cv_abmsh,cv_dtlag,nabmsh,1)
      do i = 1,3
         cv_abmsh(i) = cv_dtNek*cv_abmsh(i) 
      enddo

      call rzero(cv_ab,3)
      call setabbd(cv_ab,cv_dtlag,nbd,nbd)

      call rzero(cv_bd,4)
      call setbd(cv_bd,cv_dtlag,nbd)

      return
      end
c----------------------------------------------------------------------
      subroutine cv_pstat

      include 'SIZE'
      include 'CVODE'

      integer*8 iout,ipar
      integer cvcomm
      common /cv_iout/ iout(21),ipar(1),cvcomm

      real nli_nni

      nstp        = iout(3) - iout_save(3)
      nfe         = iout(4) - iout_save(4) + iout(20)-iout_save(20)
      nni         = iout(7) - iout_save(7)
      nli         = iout(20)- iout_save(20)
      nli_nni     = float(nli)/max(1,nni)

      nfe_avg     = (1.-1./cv_istep)*nfe_avg     + nfe/cv_istep
      nli_nni_avg = (1.-1./cv_istep)*nli_nni_avg + nli_nni/cv_istep      

      write(*,'(13x,a8,i8,3x,a8,i8,3x,a10,f5.1)')
     &    'nsteps: ',   nstp     ,
     &    'nfe:    ',   nfe      ,
     &    '<nfe>:    ', nfe_avg
      write(*,'(13x,a8,i8,3x,a8,i8,3x,a10,f5.1)')  
     &    'nni:    ',   nni      ,
     &    'nli:    ',   nli      ,
     &    '<nli/nni>:', nli_nni_avg
      write(*,'(13x,a8,i8,3x,a8,i8,3x,a10,i5)') 
     &    'ncfn:   '  ,iout(6)-iout_save(6),
     &    'ncfl:   '  ,iout(21)-iout_save(21),  
     &    'netf:     ',iout(5)-iout_save(5) 

      return
      end
c----------------------------------------------------------------------
      subroutine cv_rstat

      include 'SIZE'
      include 'CVODE'

      do i=1,21
         iout_save(i) = 0.0
      enddo

      nfe_avg     = 0
      nli_nni_avg = 0
      cv_istep    = 0
 
      return
      end
c----------------------------------------------------------------------
