!> @file mntrtmr.f
!! @ingroup monitor
!! @brief Set of timer database routines for KTH framework
!! @author Adam Peplinski
!! @date Oct 13, 2017
!=======================================================================
!> @brief Register new timer
!! @ingroup monitor
!! @param[out] mid      new timer id
!! @param[in]  pmid     parent timer id
!! @param[in]  modid    registerring module id
!! @param[in]  mname    timer name
!! @param[in]  mdscr    timer description
!! @param[in]  ifsum    add timer to parent
      subroutine mntr_tmr_reg(mid,pmid,modid,mname,mdscr,ifsum)
      implicit none

      include 'SIZE'
      include 'PARALLEL'        ! ISIZE
      include 'MNTRLOGD'
      include 'MNTRTMRD'
      include 'FRAMELP'

      ! argument list
      integer mid, pmid, modid
      character*(*) mname, mdscr
      logical ifsum

      ! local variables
      character*10  lname
      character*132 ldscr
      integer slen,slena

      integer il, ipos
!-----------------------------------------------------------------------
      ! check name length
      slena = len_trim(adjustl(mname))
      ! remove trailing blanks
      slen = len_trim(mname) - slena + 1
      if (slena.gt.mntr_lstl_mnm) then
         call mntr_log(mntr_id,lp_deb,
     $        'too long timer name; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(lname,mntr_lstl_mnm)
      lname= mname(slen:slen+slena- 1)
      call capit(lname,slena)

      ! check description length
      slena = len_trim(adjustl(mdscr))
      ! remove trailing blanks
      slen = len_trim(mdscr) - slena + 1
      if (slena.ge.mntr_lstl_mds) then
         call mntr_log(mntr_id,lp_deb,
     $        'too long timer description; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(ldscr,mntr_lstl_mds)
      ldscr= mdscr(slen:slen + slena - 1)

      ! find empty space
      ipos = 0

      ! to ensure consistency I do it on master and broadcast result
      if (nid.eq.mntr_pid0) then

         ! check if module is already registered
         do il=1,mntr_tmr_mpos
            if (mntr_tmr_id(mntr_tmr_mark,il).ge.0.and.
     $         mntr_tmr_name(il).eq.lname) then
               ipos = -il
               exit
            endif
         enddo

         ! find empty spot
         if (ipos.eq.0) then
            do il=1,mntr_tmr_id_max
               if (mntr_tmr_id(mntr_tmr_mark,il).eq.-1) then
                  ipos = il
                  exit
               endif
            enddo
         endif
      endif

      ! broadcast mid
      call bcast(ipos,isize)

      ! error; no free space found
      if (ipos.eq.0) then
         mid = ipos
         call mntr_abort(mntr_id,
     $        'timer ['//trim(lname)//'] cannot be registered')
      ! module already registered
      elseif (ipos.lt.0) then
         mid = abs(ipos)
         call mntr_abort(mntr_id,
     $    'timer ['//trim(lname)//'] is already registered')
      ! new module
      else
         mid = ipos
         ! check if parent timer is registered
         if (pmid.gt.0) then
            if (mntr_tmr_id(mntr_tmr_mark,pmid).ge.0) then
               mntr_tmr_id(mntr_tmr_mark,ipos) = pmid
            else
               mntr_tmr_id(mntr_tmr_mark,ipos) = 0
               call mntr_log(mntr_id,lp_inf,
     $       "timer's ["//trim(lname)//"] parent not registered.")
            endif
         else
            mntr_tmr_id(mntr_tmr_mark,ipos) = 0
         endif

         ! check if registerring module is registered
         if (modid.gt.0) then
            if (mntr_mod_id(modid).ge.0) then
               mntr_tmr_id(mntr_tmr_mod,ipos) = modid
            else
               mntr_tmr_id(mntr_tmr_mod,ipos) = 0
               call mntr_log(mntr_id,lp_inf,
     $       "timer's ["//trim(lname)//"] module not registered.")
            endif
         else
            mntr_tmr_id(mntr_tmr_mod,ipos) = 0
         endif

         mntr_tmr_name(ipos)=lname
         mntr_tmr_dscr(ipos)=ldscr
         mntr_tmr_sum(ipos)=ifsum
         mntr_tmr_num = mntr_tmr_num + 1
         if (mntr_tmr_mpos.lt.ipos) mntr_tmr_mpos = ipos
         call mntr_log(mntr_id,lp_inf,
     $       'Registered timer ['//trim(lname)//']: '//trim(ldscr))
      endif

      return
      end subroutine
!=======================================================================
!> @brief Check if timer name is registered and return its id.
!! @ingroup monitor
!! @param[out] mid      timer id
!! @param[in]  mname    timer name
      subroutine mntr_tmr_is_name_reg(mid,mname)
      implicit none

      include 'SIZE'
      include 'PARALLEL'        ! ISIZE
      include 'MNTRLOGD'
      include 'MNTRTMRD'
      include 'FRAMELP'

      ! argument list
      integer mid
      character*(*) mname

      ! local variables
      character*10  lname
      character*3 str
      integer slen,slena

      integer il, ipos
!-----------------------------------------------------------------------
      ! check name length
      slena = len_trim(adjustl(mname))
      ! remove trailing blanks
      slen = len_trim(mname) - slena + 1
      if (slena.gt.mntr_lstl_mnm) then
         call mntr_log(mntr_id,lp_deb,
     $          'too long timer name; shortenning')
         slena = min(slena,mntr_lstl_mnm)
      endif
      call blank(lname,mntr_lstl_mnm)
      lname= mname(slen:slen+slena- 1)
      call capit(lname,slena)

      ! find module
      ipos = 0

      ! to ensure consistency I do it on master and broadcast result
      if (nid.eq.mntr_pid0) then
         ! check if module is already registered
         do il=1,mntr_tmr_mpos
            if (mntr_tmr_id(mntr_tmr_mark,il).ge.0.and.
     $         mntr_tmr_name(il).eq.lname) then
               ipos = il
               exit
            endif
         enddo
      endif

      ! broadcast ipos
      call bcast(ipos,isize)

      if (ipos.eq.0) then
         mid = -1
         call mntr_log(mntr_id,lp_inf,
     $        'timer ['//trim(lname)//'] not registered')
      else
         mid = ipos
         write(str,'(I3)') ipos
         call mntr_log(mntr_id,lp_vrb,
     $        'timer ['//trim(lname)//'] registered with id='//str)
      endif

      return
      end subroutine
!=======================================================================
!> @brief Check if timer id is registered. This operation is performed locally
!! @ingroup monitor
!! @param[in] mid      timer id
!! @return mntr_tmr_is_id_reg
      logical function mntr_tmr_is_id_reg(mid)
      implicit none

      include 'SIZE'
      include 'PARALLEL'
      include 'MNTRLOGD'
      include 'MNTRTMRD'
      include 'FRAMELP'

      ! argument list
      integer mid
!-----------------------------------------------------------------------
      mntr_tmr_is_id_reg = mntr_tmr_id(mntr_tmr_mark,mid).ge.0

      return
      end function
!=======================================================================
!> @brief Check if timer id is registered. This operation is performed locally
!! @ingroup monitor
!! @param[in] mid       timer id
!! @param[in] icount    count increase
!! @param[in] time      time increase
      subroutine mntr_tmr_add(mid,icount,time)
      implicit none

      include 'SIZE'
      include 'MNTRLOGD'
      include 'MNTRTMRD'
      include 'FRAMELP'

      ! argument list
      integer mid, icount
      real time

      ! local variables
      character*3 str
!-----------------------------------------------------------------------
      if (mntr_tmr_id(mntr_tmr_mark,mid).ge.0) then
         mntr_tmrv_timer(mntr_tmr_count,mid) =
     $        mntr_tmrv_timer(mntr_tmr_count,mid) + icount

         mntr_tmrv_timer(mntr_tmr_time,mid) =
     $        mntr_tmrv_timer(mntr_tmr_time,mid) + time
      else
         write(str,'(I3)') mid
         call mntr_log(mntr_id,lp_inf,
     $       'timer id='//trim(str)//' in mntr_tmr_add not registered')
      endif

      return
      end subroutine
!=======================================================================
!> @brief Print registered timers showing tree structure
!! @ingroup monitor
      subroutine mntr_tmr_summary_print()
      implicit none

      include 'SIZE'
      include 'MNTRLOGD'
      include 'MNTRTMRD'
      include 'FRAMELP'

      ! local variables
      integer il, jl, maxlev, stride
      parameter (stride=2)
      integer olist(2,mntr_tmr_id_max), ierr, itmp
      real timmin(mntr_tmr_id_max),timmax(mntr_tmr_id_max)
      character*35 ftm
      character*3 str

      ! functions
      integer iglmax
      real glmax, glmin, dnekclock
!-----------------------------------------------------------------------
      call mntr_log(mntr_id,lp_prd,
     $         'Summary of registered timers')

      ! finalise framework timing
      mntr_frame_tmini = dnekclock() - mntr_frame_tmini
      call mntr_tmr_add(mntr_frame_tmr_id,1,mntr_frame_tmini)

      ! get ordered list
      call mntr_tmr_get_olist(olist, ierr)
      ierr = iglmax(ierr,1)
      if (ierr.gt.0) then
         call mntr_error(mntr_id,"Inconsistent timer tree.")
         return
      endif

      ! sum contributions from children if they are marked with mntr_tmr_sum
      ! find max level for this run
      maxlev = 1
      do il=1,mntr_tmr_num
         maxlev = max(maxlev,olist(2,il))
      enddo

      do il=maxlev,1,-1
         do jl=1,mntr_tmr_num
            if (olist(2,jl).eq.il.and.mntr_tmr_sum(olist(1,jl))) then
               itmp = mntr_tmr_id(mntr_tmr_mark,olist(1,jl))
               ! sum iteration count
               mntr_tmrv_timer(mntr_tmr_count,itmp) =
     $             mntr_tmrv_timer(mntr_tmr_count,itmp) +
     $             mntr_tmrv_timer(mntr_tmr_count,olist(1,jl))
               ! sum timer
               mntr_tmrv_timer(mntr_tmr_time,itmp) =
     $             mntr_tmrv_timer(mntr_tmr_time,itmp) +
     $             mntr_tmrv_timer(mntr_tmr_time,olist(1,jl))
            endif
         enddo
      enddo


      ! get max, min timers
      do il=1,mntr_tmr_mpos
         if (mntr_tmr_id(mntr_tmr_mark,il).ge.0) then
            timmin(il) = glmin(mntr_tmrv_timer(mntr_tmr_time,il),1)
            timmax(il) = glmax(mntr_tmrv_timer(mntr_tmr_time,il),1)
         endif
      enddo

      if (nid.eq.mntr_pid0) then

         if(ierr.eq.0.and.mntr_lp_def.le.lp_prd) then

            ! modify max level
            maxlev = maxlev + 1

            ! print description
            if (mntr_iftdsc) then
               write (*,*) ' '
               do il=1,mntr_tmr_num
                  write(str,'(I3)') stride*(olist(2,il))
                  ftm = '("[",A,"]",'//trim(str)//'X,A,'
                  write(str,'(I3)') stride*(maxlev-olist(2,il))
                  ftm = trim(ftm)//trim(str)//'X,": ",A)'
                  jl = olist(1,il)
                  write(*,ftm)
     $               mntr_mod_name(mntr_tmr_id(mntr_tmr_mod,jl)),
     $               mntr_tmr_name(jl), trim(mntr_tmr_dscr(jl))
               enddo
            endif

            ! print values
            write(*,*) ' '
            write(str,'(I3)') mntr_lstl_mnm +stride*maxlev-1
            ftm='(A11,1X,A'//trim(adjustl(str))//',1X,":",4A15)'
            write(*,ftm) 'Module name','Timer name','Count','Min time',
     $                   'Max time', 'Max/count'
            do il=1,mntr_tmr_num
               write(str,'(I3)') stride*(olist(2,il))
               ftm = '("[",A,"]",'//trim(str)//'X,A,'
               write(str,'(I3)') stride*(maxlev-olist(2,il))
               ftm = trim(ftm)//trim(str)//'X,":",4E15.8)'
               jl = olist(1,il)
               write(*,ftm) mntr_mod_name(mntr_tmr_id(mntr_tmr_mod,jl)),
     $           mntr_tmr_name(jl), mntr_tmrv_timer(mntr_tmr_count,jl),
     $           timmin(jl),timmax(jl),
     $           timmax(jl)/max(1.0,mntr_tmrv_timer(mntr_tmr_count,jl))
            enddo
            write(*,*) ' '
         endif
      endif

      return
      end subroutine
!=======================================================================
!> @brief Provide ordered list of registered timers for printing.
!! @ingroup monitor
!! @param[out]   olist    ordered list
!! @param[out]   ierr     error flag
      subroutine mntr_tmr_get_olist(olist,ierr)
      implicit none

      include 'SIZE'
      include 'FRAMELP'
      include 'MNTRLOGD'
      include 'MNTRTMRD'

      ! argument list
      integer olist(2,mntr_tmr_id_max), ierr

      ! local variables
      integer ind(mntr_tmr_id_max), level, parent, ipos
      integer slist(2,mntr_tmr_id_max), itmp1(2)
      integer npos, key
      integer il, jl
      integer istart, in, itest
!-----------------------------------------------------------------------
      ierr = 0

      ! sort timer index array
      ! copy data removing possible empty slots
      npos=0
      do il=1,mntr_tmr_mpos
         if (mntr_tmr_id(mntr_tmr_mark,il).ge.0) then
            npos = npos + 1
            slist(1,npos) = mntr_tmr_id(mntr_tmr_mark,il)
            slist(2,npos) = il
         endif
      enddo
      if(npos.ne.mntr_tmr_num) then
         ierr = 1
         call mntr_log(mntr_id,lp_inf,
     $         'Inconsistent timer number; return')
         return
      endif

      ! sort with respect to parent id
      key = 1
      call ituple_sort(slist,2,npos,key,1,ind,itmp1)

      ! sort within children of single parent with respect to child id
      istart = 1
      itest = slist(1,istart)
      do il=1,npos
         if(itest.ne.slist(1,il).or.il.eq.npos) then
           if (il.eq.npos.and.itest.eq.slist(1,il)) then
              jl = npos + 1
           else
              jl = il
           endif
           in = jl - istart
           if (itest.eq.0.and.in.ne.1) then
              call mntr_log(mntr_id,lp_inf,
     $         'Must be single root of the graph; return')
              ierr = 2
              return
           endif
           if (in.gt.1) then
              key = 2
              call ituple_sort(slist(1,istart),2,in,key,1,ind,itmp1)
           endif
           if (il.ne.npos) then
              itest = slist(1,il)
              istart = il
           endif
         endif
      enddo

      parent = 0
      level = 0
      ipos = 1
      call mntr_build_ord_list(olist,slist,npos,ipos,parent,level)

      return
      end subroutine
!=======================================================================

