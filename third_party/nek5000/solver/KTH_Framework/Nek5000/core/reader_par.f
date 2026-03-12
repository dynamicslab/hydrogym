c-----------------------------------------------------------------------
      subroutine readat_par
C
C     Read in run parameters from .par file
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'RESTART'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
c
      logical ifbswap

      call setDefaultParam

      if(nid.eq.0) call par_read(ierr)
      call bcast(ierr,isize)
      if(ierr .ne. 0) call exitt
      call bcastParam

      call usrdat0

      call read_re2_hdr(ifbswap, .true.)

      call chkParam

      call mapelpr  ! read .map file, est. gllnid, etc.

      call read_re2_data(ifbswap)

      call nekgsync()

      return
      end
c-----------------------------------------------------------------------
      subroutine setDefaultParam
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'RESTART'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'

      loglevel = 1
      optlevel = 1

      call rzero(param,200)
      call rzero(uparam,20)

      param(10) = 0    ! stop at numSteps
      param(14) = 0    ! iostep
      param(15) = 0    ! iotime 

      param(21) = 1e-5 ! pressure tolerance
      param(22) = 1e-7 ! velocity tolerance

      param(26) = 0.5  ! target Courant number
      param(27) = 2    ! 2nd order in time
      param(28) = 0    ! use same torder for mesh solver

      param(31) = 0    ! zero perturbations
      param(32) = 0    ! read all BC from .re2 
      param(33) = 0    ! use default field index for BCs to read from .re2 

      param(40) = 0    ! XXT 

      param(41) = 0    ! additive SEMG
      param(42) = 0    ! GMRES for iterative solver w/ nonsymmetric weighting
      param(43) = 0    ! additive multilevel scheme (requires param(42).eq.0)
      param(44) = 0    ! base top-level additive Schwarz on restrictions of E

      param(47) = 0.4  ! viscosity for mesh elasticity solver

      param(59) = 1    ! No fast operator eval

      param(65) = 1    ! just one i/o node
      param(66) = 6    ! write in binary
      param(67) = 6    ! read in binary
      param(93) = mxprev ! number of vectors for projection

      param(94) = 5    ! projection for helmholz solves (controled by ifprojfld) 
      param(95) = 0    ! projection for pressure solve
      param(99) = 4    ! enable dealising

      param(101) = 0   ! no additional modes
      param(103) = 0   ! filter weight
      filterType = 0   ! no filtering

      param(160) = 1   ! cvode use normal mode 
      param(161) = 2   ! cvode use stiff integration 
      param(162) = 0   ! cvode absolute tolerance
      param(163) = -1  ! cvode realtive tolerance
      param(164) = 100 ! cvode don't limit internal dt
      param(165) = 1   ! cvode increment factor DQJ
      param(166) = 0   ! cvode use default ratio linear/non-linear tolerances
      param(167) = 0   ! cvode use no preconditioner

      restol(0) = param(22)
      restol(1) = param(22)
      do i=1,ldimt
         restol(1+i) = param(22) 
      enddo

      iftmsh(0) = .false. 
      iftmsh(1) = .false. 
      do i=1,ldimt
         iftmsh(1+i) = .false.
      enddo

      ifxyo = .true.
      ifvo  = .false.
      ifpo  = .false.
      ifto  = .false.   
      do i=1,ldimt1
         ifpsco(i) = .false.
      enddo 

      ifadvc(1) = .true.  
      do i=1,ldimt
         ifadvc(i+1) = .true.  
      enddo 

      iffilter(1) = .false.  
      do i=1,ldimt
         iffilter(i+1) = .false.  
      enddo 

      ifdgfld(0) = .false. 
      ifdgfld(1) = .false. 
      do i=1,ldimt
         ifdgfld(i+1) = .false.  
      enddo 

      ifdiff(1) = .true.  
      do i=1,ldimt
         ifdiff(i+1) = .true.  
      enddo 

      ifdeal(1) = .true.  
      do i=1,ldimt
         ifdeal(i+1) = .true.  
      enddo 

      do i=1,ldimt
         idpss(i) = -1
      enddo 

      meshPartitioner=3 ! HYBRID (RSB+RCB)

      ifprojfld(0) = .false. 
      ifprojfld(1) = .false. 
      do i=1,ldimt
         ifprojfld(1+i) = .false.
      enddo

      ifflow    = .false.
      ifheat    = .false.  
      iftran    = .true.   
      ifaxis    = .false.  
      ifstrs    = .false. 
      iflomach  = .false. 
      ifmvbd    = .false.
      ifchar    = .false.  
      ifmhd     = .false. 
      ifuservp  = .false.  
      ifcyclic  = .false.
      ifusermv  = .false.
      ifmgrid   = .true.
      ifessr    = .false.
      ifreguo   = .false.
      ifbase    = .true.   
      ifpert    = .false. 
      ifaziv    = .false. 
      ifmoab    = .false.  
      ifcvode   = .false.

      ifdg      = .false.
      ifsync    = .false.  
      ifanls    = .false.  
      ifcoup    = .false.  
      ifvcoup   = .false.  
      ifexplvis = .false.
      ifcons    = .false.   

      ifmodel   = .false. 
      ifkeps    = .false.
      ifschclob = .false. 

      ifdp0dt   = .false.
      ifreguo   = .false.   ! dump on the GLL mesh

      fem_amg_param(1) = 0
      crs_param(1) = 0

      call izero(matype,16*ldimt1)
      call rzero(cpgrp ,48*ldimt1)

      call blank (initc,15*132)

      return
      end
c-----------------------------------------------------------------------
      subroutine par_read(ierr)
c
c     parse .par file and set run parameters
c
c     todo:
c     - check for invalid values for a given key
c     - print default value to screen
c     - separate settings for tol, proj, dealiasing for ps
c     - mhd support

      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'ADJOINT'
      INCLUDE 'RESTART'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
      INCLUDE 'TSTEP'

      character*132 c_out,txt, txt2

      call finiparser_load(parfle,ierr)
      if(ierr .ne. 0) return

      call par_verify(ierr)
      if(ierr .ne. 0) return

c set parameters
      call finiparser_getDbl(d_out,'general:loglevel',ifnd)
      if(ifnd .eq. 1) loglevel = d_out

      call finiparser_getDbl(d_out,'general:optlevel',ifnd)
      if(ifnd .eq. 1) optlevel = d_out

      call finiparser_getString(c_out,'general:stopAt',ifnd)
      if (ifnd .eq. 1) then
         call capit(c_out,132)
         if (index(c_out,'ENDTIME') .eq. 1) then
            call finiparser_getDbl(d_out,'general:endTime',ifnd)
            if (ifnd .eq. 1) then
               param(10) = d_out
            else
               write(6,*) 'general:endTime'
               write(6,*) 'is required for general:stopAt = endTime!'
               goto 999
            endif
         else if (index(c_out,'NUMSTEPS') .eq. 1) then
            call finiparser_getDbl(d_out,'general:numSteps',ifnd)
            if (ifnd .eq. 1) then
               param(11) = d_out 
            else
               write(6,*) 'general:numSteps'
               write(6,*) 'is required for general:stopAt = numSteps!'
               goto 999
            endif
         else
            write(6,*) 'value: ',trim(c_out)
            write(6,*) 'is invalid for general:stopAt!'
            goto 999
         endif
      else
         call finiparser_getDbl(d_out,'general:numSteps',ifnd)
         if (ifnd .eq. 1) then 
            param(11) = d_out 
         endif
      endif

      call finiparser_getDbl(d_out,'general:dt',ifnd)
      if (ifnd .eq. 1) then
         param(12) = d_out
      endif

      param(12) = -1*abs(param(12))
      call finiparser_getBool(i_out,'general:variableDt',ifnd)
      if (ifnd .eq. 1) then
         if (i_out .eq. 1) then
            param(12) = abs(param(12)) 
            call finiparser_getDbl(d_out,'general:targetCFL',ifnd)
            if (ifnd .eq. 1) then
               param(26) = d_out
            else
               write(6,*) 'general:targetCFL'
               write(6,*) 'is required for general:variableDt!'
               goto 999
            endif
         endif
      endif

      call finiparser_getDbl(d_out,'general:writeInterval',ifnd)
      if (ifnd .eq. 1) param(15) = d_out
      call finiparser_getString(c_out,'general:writeControl',ifnd)
      if (ifnd .eq. 1) then
         call capit(c_out,132)
         if (index(c_out,'RUNTIME') .eq. 1) then
            param(14) = d_out
            param(15) = 0
         else if (index(c_out,'TIMESTEP') .eq. 1) then
            param(14) = 0
            param(15) = d_out
         else if (index(c_out,'ELAPSEDTIME') .eq. 1) then
            param(14) = d_out
            param(15) = 0
            timeioe = 1
         else
            write(6,*) 'value: ',trim(c_out)
            write(6,*) 'is invalid for general:writeControl!'
            goto 999
         endif
      endif

      call finiparser_getDbl(d_out,'pressure:residualTol',ifnd)
      if(ifnd .eq. 1) param(21) = d_out 
      call finiparser_getDbl(d_out,'velocity:residualTol',ifnd)
      if(ifnd .eq. 1) then
        restol(1) = d_out 
        param(22) = d_out
      endif

      call finiparser_find(i_out,'temperature',ifnd)
      if(ifnd .eq. 1) then
        ifheat = .true.
        ifto   = .true.
        idpss(1) = 0 ! Helmholtz is default
      endif

      j = 0
      do i = 1,99
         write(txt,"('scalar',i2.2)") i
         call finiparser_find(i_out,txt,ifnd)
         if (ifnd .eq. 1) j = j + 1
      enddo
      if (j.gt.ldimt-1) then
         write(6,*) 'found more scalars than specified in SIZE!' 
         goto 999
      endif

      j = 0
      do i = 1,ldimt-1
         write(txt,"('scalar',i2.2)") i
         call finiparser_find(i_out,txt,ifnd)
         if (ifnd .eq. 1) then
            j = j + 1 
            ifpsco(i) = .true.
            idpss(i+1) = 0 ! Helmholtz is default
         endif
      enddo
      param(23) = j ! number of scalars 
      n = param(23)
 
      is = 2
      if (ifheat) is = 1 

      do i = is,n+1

      if (i.eq.1) then
        txt = 'temperature'
      else
        write(txt,"('scalar',i2.2)") i-1
      endif

      call finiparser_getString(c_out,trim(txt)//':solver',ifnd)
      call capit(c_out,132)
      if(ifnd .eq. 1) then 
        if (index(c_out,'CVODE') .eq. 1) then
          idpss(i) = 1
          call finiparser_getDbl(d_out,trim(txt)//':absoluteTol',ifnd)
          if (ifnd .eq. 1) then
             atol(i+1) = d_out 
          else
             write(6,*) trim(txt) // ':absoluteTol' 
             write(6,*) 'is required for ',trim(txt)//':solver = CVODE'
             goto 999
          endif
        else if (index(c_out,'HELM') .eq. 1) then
          continue
        else if (index(c_out,'NONE') .eq. 1) then
          idpss(i) = -1
        else
           write(6,*) 'value: ',trim(c_out)
           write(6,*) 'is invalid for ',trim(txt)//':solver!' 
           goto 999
        endif
      else
        call finiparser_getDbl(d_out,trim(txt)//':residualTol',ifnd)
        if (ifnd .eq. 1) then
           restol(i+1) = d_out 
        endif
      endif
      call finiparser_getBool(i_out,trim(txt)//':residualProj',ifnd)
      if (ifnd .eq. 1) then
         ifprojfld(i+1) = .false.
         if(i_out .eq. 1) ifprojfld(i+1) = .true.
      endif
 
      enddo

      call finiparser_getDbl(d_out,'cvode:absoluteTol',ifnd)
      if(ifnd .eq. 1) param(162) = d_out 
      call finiparser_getDbl(d_out,'cvode:relativeTol',ifnd)
      if(ifnd .eq. 1) param(163) = d_out 
      call finiparser_getDbl(d_out,'cvode:dtmax',ifnd)
      if(ifnd .eq. 1) param(164) = d_out 
      call finiparser_getDbl(d_out,'cvode:DQJincrementFactor',ifnd)
      if(ifnd .eq. 1) param(165) = d_out 
      call finiparser_getDbl(d_out,'cvode:ratioLNLtol',ifnd)
      if(ifnd .eq. 1) param(166) = d_out 

      call finiparser_getString(c_out,'cvode:preconditioner',ifnd)
      call capit(c_out,132)
      if(ifnd .eq. 1) then 
        if (index(c_out,'USER') .eq. 1) then
           param(167) = 1
        else if (index(c_out,'NONE') .eq. 1) then
           param(167) = 0
        else
           write(6,*) 'value: ',trim(c_out)
           write(6,*) 'is invalid for cvode:preconditioner!' 
           goto 999
        endif
      endif

      j = 0
      do i = 1,ldimt
         if (idpss(i).ge.0) j = j + 1
      enddo
      if (j .ge. 1) then ! we have to solve for temp and/or ps
         ifheat = .true.  
      else
         ifheat = .false.
      endif

      call finiparser_getDbl(d_out,'magnetic:viscosity',ifnd)
      if(ifnd .eq. 1) param(29) = d_out 
      if(param(29).lt.0.0) param(29) = -1.0/param(29)

      call finiparser_getDbl(d_out,'mesh:numberOfBCFields',ifnd)
      if(ifnd .eq. 1) param(32) = int(d_out)

      call finiparser_getDbl(d_out,'mesh:firstBCFieldIndex',ifnd)
      if(ifnd .eq. 1) param(33) = int(d_out)

      call finiparser_getString(c_out,'pressure:solver',ifnd)
      if (ifnd .eq. 1) then 
         call capit(c_out,132)
         if (index(c_out,'GMRES') .eq. 1) then
            param(42) = 0
         else if (index(c_out,'CGFLEX') .eq. 1) then
            param(42) = 2
         else
            write(6,*) 'value: ',trim(c_out)
            write(6,*) 'not supported for pressure:solver!'
            goto 999
         endif
      endif

      call finiparser_getString(c_out,'pressure:preconditioner',ifnd)
      if (ifnd .eq. 1) then 
         call capit(c_out,132)
         if (index(c_out,'SEMG_XXT') .eq. 1) then
            param(40) = 0
        else if (index(c_out,'SEMG_AMG_HYPRE') .eq. 1) then
            param(40) = 2
         else if (index(c_out,'SEMG_AMG') .eq. 1) then
            param(40) = 1
         else if (index(c_out,'FEM_AMG_HYPRE') .eq. 1) then
            param(40) = 3
         else
           write(6,*) 'value: ',trim(c_out)
           write(6,*) 'is invalid for pressure:preconditioner!'
           goto 999
         endif
      endif 

      call finiparser_getBool(i_out,'general:writeDoublePrecision',ifnd)
      if(ifnd .eq. 1 .and. i_out .eq. 1) param(63) = 1 

      call finiparser_getDbl(d_out,'general:writeNFiles',ifnd)
      if(ifnd .eq. 1) param(65) = int(d_out) 

      call finiparser_getBool(i_out,'velocity:residualProj',ifnd)
      if(ifnd .eq. 1) then
        ifprojfld(1) = .false.
        if(i_out .eq. 1) ifprojfld(1) = .true. 
      endif

      call finiparser_getBool(i_out,'pressure:residualProj',ifnd)
      if(ifnd .eq. 1) then
        param(95) = 0
        if(i_out .eq. 1) param(95) = 5 
      endif

      call finiparser_getBool(i_out,'general:dealiasing',ifnd)
      if(ifnd .eq. 1 .and. i_out .eq. 0) param(99) = -1 

c     filtering parameters
      call finiparser_getString(c_out,'general:filtering',ifnd)
      if (ifnd .eq. 1) then
c        stabilization type: none, explicit or hpfrt    
         call capit(c_out,132)
         if (index(c_out,'NONE') .eq. 1) then
            filterType = 0
            goto 101
         else if (index(c_out,'EXPLICIT') .eq. 1) then
            filterType = 1
            call ltrue(iffilter,size(iffilter))
         else if (index(c_out,'HPFRT') .eq. 1) then
            filterType = 2
            call ltrue(iffilter,size(iffilter))
         else
           write(6,*) 'value: ',c_out
           write(6,*) 'is invalid for general:filtering!'
           goto 999
         endif
         call finiparser_getDbl(d_out,'general:filterWeight',ifnd)
         if (ifnd .eq. 1) then
            param(103) = d_out 
         else
            write(6,*) 'general:filterWeight'
            write(6,*) 'is required for general:filtering!'
            goto 999
         endif
         call finiparser_getDbl(d_out,'general:filterModes',ifnd)
         if (ifnd .eq. 1) then
            param(101) = int(d_out) - 1
            if (int(param(101)).eq.0) filterType = 0
         else
            call finiparser_getDbl
     $           (d_out,'general:filterCutoffRatio',ifnd)
            if (ifnd .eq. 1) then
              dtmp = anint(lx1*(1.0 - d_out)) 
              param(101) = max(dtmp-1,0.0)
              if (abs(1.0 - d_out).lt.0.01) filterType = 0
            else 
              write(6,*) 'general:filterCutoffRatio or filterModes'
              write(6,*) 'is required for general:filtering!'
              goto 999
            endif
         endif
 101     continue 
      endif

      call finiparser_getString(c_out,'cvode:mode',ifnd)
      call capit(c_out,132)
      if (index(c_out,'NORMAL') .eq. 1) param(160) = 1
      if (index(c_out,'NORMAL_TSTOP' ) .eq. 1) param(160) = 3
 
      do i = 1,20
         call blank(txt,132)
         write(txt,"('general:userParam',i2.2)") i
         call finiparser_getDbl(d_out,txt,ifnd)
         if(ifnd .eq. 1) uparam(i) = d_out
      enddo

c set logical flags
      call finiparser_getString(c_out,'general:timeStepper',ifnd)
      if (ifnd .eq. 1) then
        call capit(c_out,132)

        if (index(c_out,'BDF1') .eq. 1) then
           param(27) = 1 
        else if (index(c_out,'BDF2') .eq. 1) then
           param(27) = 2 
        else if (index(c_out,'BDF3') .eq. 1) then
           param(27) = 3 
        else
           write(6,*) 'value: ',trim(c_out)
           write(6,*) 'is invalid for general:timeStepper!'
           goto 999
        endif
      endif

      call finiparser_getString(c_out,'general:extrapolation',ifnd)
      if (ifnd .eq. 1) then
        call capit(c_out,132)
        if (index(c_out,'OIFS') .eq. 1) then
           ifchar = .true.

           call finiparser_getDbl(d_out,'general:targetCFL',ifnd)
           if (ifnd .eq. 1) then
              param(26) = d_out
           else
              write(6,*) 'general:targetCFL'
              write(6,*) 'is required for general:extrapolation!'
              goto 999
           endif
        else if (index(c_out,'STANDARD') .eq. 1) then
           continue
        else
           write(6,*) 'value: ',trim(c_out)
           write(6,*) 'is invalid for general:extrapolation!'
           goto 999
        endif
      endif

      call finiparser_find(i_out,'velocity',ifnd)
      if(ifnd .eq. 1) then
        ifflow = .true.
        ifvo   = .true.
        ifpo   = .true.
      endif

      call finiparser_getString(c_out,'mesh:motion',ifnd)
      if (ifnd .eq. 1) then
       call capit(c_out,132)
       if (index(c_out,'ELASTICITY') .eq. 1) then
          ifmvbd = .true.
          call finiparser_getDbl(d_out,'mesh:viscosity',ifnd)
          if (ifnd .eq. 1) param(47) = d_out
          call finiparser_getBool(i_out,'mesh:residualProj',ifnd)
          if (ifnd .eq. 1) then
             ifprojfld(0) = .false.
             if(i_out .eq. 1) ifprojfld(0) = .true. 
          endif
        else if (index(c_out,'USER') .eq. 1) then
          ifmvbd = .true.
          ifusermv = .true.
        else if (index(c_out,'NONE') .eq. 1) then
          continue
        else
          write(6,*) 'value: ',trim(c_out)
          write(6,*) 'is invalid for mesh:motion!'
          goto 999
        endif
      endif
      call finiparser_getDbl(d_out,'mesh:residualTol',ifnd)
      if(ifnd .eq. 1) restol(0) = d_out 

      call finiparser_getBool(i_out,'problemType:axiSymmetry',ifnd)
      if(ifnd .eq. 1) then
        ifaxis = .false.
        if(i_out .eq. 1) ifaxis = .true.
      endif

      call finiparser_getBool(i_out,'problemType:swirl',ifnd)
      if(ifnd .eq. 1) then
        ifaziv = .false.
        if(i_out .eq. 1) ifaziv = .true.
      endif

      call finiparser_getBool(i_out,'problemType:cyclicBoundaries',ifnd)
      if(ifnd.eq.1) then
        ifcyclic = .false.
        if(i_out .eq. 1) ifcyclic = .true.
      endif

      call finiparser_getString(c_out,'problemType:equation',ifnd)
      call capit(c_out,132)
      if (index(c_out,'STEADYSTOKES').eq.1) then
         iftran = .false.
         ifadvc(1) = .false.
      else if (index(c_out,'STOKES').eq.1) then
         ifadvc(1) = .false.
      else if (index(c_out,'STEADYHEAT').eq.1) then
         iftran = .false.
         ifflow = .false. 
         ifheat = .true.
         ifadvc(2) = .false.
      else if (index(c_out,'LOWMACHNS').eq.1) then
         iflomach = .true.
      else if (index(c_out,'INCOMPLINNS').eq.1 .or.
     $         index(c_out,'INCOMPLINADJNS').eq.1) then
         ifpert = .true.
         if (index(c_out,'INCOMPLINADJNS').eq.1) ifadj  = .true.
         call finiparser_getDbl
     $        (d_out,'problemType:numberOfPerturbations',ifnd)
         if (ifnd .eq. 1) then
            param(31) = int(d_out) 
         else
            param(31) = 1 
         endif
         call finiparser_getBool
     $        (i_out,'problemType:solveBaseFlow',ifnd)
         if (ifnd .eq. 1) then
            ifbase = .false.
            if(i_out .eq. 1) ifbase = .true.
         else
            write(6,*) 'problemType:solveBaseFlow'
            write(6,*) 'is required for ', trim(c_out) 
            goto 999
         endif
      else if (index(c_out,'COMPNS') .eq. 1) then
#ifdef CMTNEK
         continue
#else
         write(6,*) 'value: ',trim(c_out)
         write(6,*) 'not supported for problemType:equation!'
         write(6,*) 'Recompile with CMTNEK ...'
         goto 999
#endif
      else if (index(c_out,'INCOMPMHD') .eq. 1) then
         write(6,*) 'value: ',trim(c_out)
         write(6,*) 'not yet supported for problemType:equation!'
         goto 999
      endif

      call finiparser_getBool(i_out,
     &                        'problemType:stressFormulation',ifnd)
      if(ifnd .eq. 1) then
        ifstrs = .false.
        if(i_out .eq. 1) ifstrs = .true.
      endif

      call finiparser_getBool(i_out,
     &                        'problemType:variableProperties',ifnd)
      if(ifnd .eq. 1) then
        ifuservp = .false.
        if(i_out .eq. 1) ifuservp = .true.
      endif

      call finiparser_getBool(i_out,'problemType:dp0dt',ifnd)
      if(ifnd .eq. 1) then
        if(i_out .eq. 1) ifdp0dt = .true.
      endif

      call finiparser_getBool(i_out,'cvode:stiff',ifnd)
      if(ifnd .eq. 1) then
        param(161) = 1 ! AM
        if(i_out .eq. 1) param(161) = 2 !BDF
      endif

c set advection
      call finiparser_getBool(i_out,'velocity:advection',ifnd)
      if(ifnd .eq. 1) then
        ifadvc(1) = .false.
        if(i_out .eq. 1) ifadvc(1) = .true.
      endif
 
      call finiparser_getBool(i_out,'temperature:advection',ifnd)
      if(ifnd .eq. 1) then
        ifadvc(2) = .false.
        if(i_out .eq. 1) ifadvc(2) = .true.
      endif

      do i = 1,ldimt-1
         write(txt,"('scalar',i2.2,a)") i,':advection'
         call finiparser_getBool(i_out,txt,ifnd)
         if(ifnd .eq. 1) then
           ifadvc(i+2) = .false.
           if(i_out .eq. 1) ifadvc(i+2) = .true.
         endif
      enddo

c set mesh-field mapping
      call finiparser_getBool(i_out,'temperature:conjugateHeatTransfer',
     &                        ifnd)
      if(ifnd .eq. 1) then
        if(i_out .eq. 1) then
          iftmsh(0) = .true.
          iftmsh(2) = .true.
        endif
      endif

      do i = 1,ldimt-1
         write(txt,"('scalar',i2.2,a)") i,':conjugateHeatTransfer'
         call finiparser_getBool(i_out,txt,ifnd)
         if(ifnd .eq. 1) then
           iftmsh(i+2) = .false.
           if(i_out .eq. 1) iftmsh(i+2) = .true.
         endif
      enddo

c set output flags
      call finiparser_getBool(i_out,'mesh:writeToFieldFile',ifnd) 
      if(ifnd .eq. 1) then
        ifxyo = .false.
        if(i_out .eq. 1) ifxyo = .true.
      endif

      call finiparser_getBool(i_out,'velocity:writeToFieldFile',ifnd)
      if(ifnd .eq. 1) then
        ifvo = .false.
        if(i_out .eq. 1) ifvo = .true.
      endif

      call finiparser_getBool(i_out,'pressure:writeToFieldFile',ifnd)
      if(ifnd .eq. 1) then
        ifpo = .false.
        if(i_out .eq. 1) ifpo = .true.
      endif

      call finiparser_getBool(i_out,'temperature:writeToFieldFile',ifnd)
      if(ifnd .eq. 1) then
        ifto = .false.
        if(i_out .eq. 1) ifto = .true.
      endif

      do i = 1,ldimt-1
         write(txt,"('scalar',i2.2,a)") i,':writeToFieldFile'
         call finiparser_getBool(i_out,txt,ifnd)
         if(ifnd .eq. 1) then
           ifpsco(i) = .false.
           if(i_out .eq. 1) ifpsco(i) = .true.
         endif
      enddo

c set properties
      call finiparser_getDbl(d_out,'velocity:viscosity',ifnd)
      if(ifnd .eq. 1) cpfld(1,1) = d_out 
      if (cpfld(1,1) .lt.0.0) cpfld(1,1)  = -1.0/cpfld(1,1)
      call finiparser_getDbl(d_out,'velocity:density',ifnd)
      if(ifnd .eq. 1) cpfld(1,2) = d_out 

      call finiparser_getDbl(d_out,'temperature:conductivity',ifnd)
      if(ifnd .eq. 1) cpfld(2,1) = d_out 
      if (cpfld(2,1) .lt.0.0) cpfld(2,1)  = -1.0/cpfld(2,1)
      call finiparser_getDbl(d_out,'temperature:rhoCp',ifnd)
      if(ifnd .eq. 1) cpfld(2,2) = d_out 

      do i = 1,ldimt-1
         write(txt,"('scalar',i2.2,a)") i,':diffusivity'
         call finiparser_getDbl(d_out,txt,ifnd)
         if(ifnd .eq. 1) cpfld(2+i,1) = d_out 
         if(cpfld(2+i,1) .lt.0.0) cpfld(2+i,1)  = -1.0/cpfld(2+i,1)
         write(txt,"('scalar',i2.2,a)") i,':density'
         call finiparser_getDbl(d_out,txt,ifnd)
         if(ifnd .eq. 1) cpfld(2+i,2) = d_out 
      enddo

c set restart options
      call finiparser_findTokens('general:startFrom', ',' , ifnd)
      do i = 1,min(ifnd,15)
         call finiparser_getToken(initc(i),i)
         if(index(initc(i),'0') .eq. 1) call blank(initc(i),132)
      enddo

c set partitioner options
      call finiparser_getString(c_out,'mesh:partitioner',ifnd)
      call capit(c_out,132)
      if(index(c_out,'RSB').eq.1) then
         meshPartitioner=1
      else if(index(c_out,'RCB').eq.1) then
         meshPartitioner=2
      else if (index(c_out,'RCBRSB').eq.1) then
         meshPartitioner=3
      else if (index(c_out,'METIS').eq.1) then
         meshPartitioner=4
      endif

100   if(ierr.eq.0) call finiparser_dump()
      return

c error handling
 999  continue
      ierr = 1
      goto 100

      end
c-----------------------------------------------------------------------
      subroutine bcastParam
C
C     Broadcast run parameters to all processors
C
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'TSTEP'
      INCLUDE 'RESTART'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
      INCLUDE 'ADJOINT'
      INCLUDE 'CVODE'

      call bcast(loglevel, isize)
      call bcast(optlevel, isize)

      call bcast(param , 200*wdsize)
      call bcast(uparam, 20*wdsize)

      call bcast(filterType, wdsize)

      call bcast(atol ,  (ldimt1+1)*wdsize)
      call bcast(restol, (ldimt1+1)*wdsize)

      call bcast(ifchar , lsize)
      call bcast(iftran  , lsize)
      call bcast(ifflow  , lsize)
      call bcast(ifheat  , lsize)     
      call bcast(iflomach, lsize)
      call bcast(ifstrs  , lsize)
      call bcast(ifmvbd  , lsize)
      call bcast(ifusermv, lsize)
      call bcast(ifdp0dt,  lsize)
      call bcast(ifaxis  , lsize)
      call bcast(ifcyclic, lsize)
      call bcast(ifmhd   , lsize)
      call bcast(ifuservp, lsize)
      call bcast(ifpert, lsize)
      call bcast(ifbase, lsize)
      call bcast(ifmoab, lsize)
      call bcast(ifaziv, lsize)
      call bcast(ifadj , lsize)

      call bcast(ifadvc ,  ldimt1*lsize)
      call bcast(ifdiff ,  ldimt1*lsize)
      call bcast(ifdeal ,  ldimt1*lsize)
      call bcast(iffilter, ldimt1*lsize)

      call bcast(idpss    ,  ldimt*isize)

      call bcast(meshPartitioner,isize)

      call bcast(iftmsh   , (ldimt1+1)*lsize)
      call bcast(ifprojfld, (ldimt1+1)*lsize)

      call bcast(cpfld, 3*ldimt1*wdsize)

      call bcast(ifxyo , lsize)
      call bcast(ifvo  , lsize)
      call bcast(ifpo  , lsize)
      call bcast(ifto  , lsize)
      call bcast(ifpsco, ldimt1*lsize)

      call bcast(initc, 15*132*csize) 

      call bcast(timeioe,sizeof(timeioe))

c set some internals 
      if (ldim.eq.3) if3d=.true.
      if (ldim.ne.3) if3d=.false.

      param(1) = cpfld(1,2)
      param(2) = cpfld(1,1)
      param(7) = cpfld(2,2)
      param(8) = cpfld(2,1)

      npscal=int(param(23)) ! number of scalar sections
      npscl1=npscal+1
      npscl2=npscal+2
 
      npert = abs(param(31)) 

      if (if3d) ifaxis = .false.
      if (ifaxis) param(99) = 3             ! For now, at least.

      ifldmhd = npscal + 3
      if (ifmhd) then
         ifessr = .true.
         ifchar = .false.
         npscl1 = npscl1 + 1
         cpfld(ifldmhd,1) = param(29)  ! magnetic viscosity
         cpfld(ifldmhd,2) = param( 1)  ! magnetic rho same as for fluid
      endif

      if (ifaxis.and..not.ifsplit) then ! Use standard Schwarz/PCG solver
         ifmgrid   = .false.
         param(42) = 1.00000  !  p042 0=gmres/1=pcg
         param(43) = 1.00000  !  p043 0=semg/1=schwarz
         param(44) = 1.00000  !  p044 0=E-based/1=A-based prec.
      endif

      if (.not.iftran) then
         if (ifflow.and.ifsplit) then
            iftran=.true. ! no steady Stokes support for Pn/Pn
         else
            param(11) = 1.0
            param(12) = 1.0
            param(19) = 0.0
         endif
      endif

      cv_nfld = 0
      do i = 1,ldimt
         if (idpss(i) .gt. 0) then
           cv_nfld = cv_nfld + 1
           ifcvfld(i+1) = .true.
         endif
      enddo
      if (cv_nfld.gt.0) ifcvode = .true.

      return
      END
c-----------------------------------------------------------------------
      subroutine chkParam
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'RESTART'
      INCLUDE 'PARALLEL'
      INCLUDE 'CTIMER'
c
      neltmx=np*lelt
      nelvmx=np*lelv

      neltmx=min(neltmx,lelg)
      nelvmx=min(nelvmx,lelg)

      nelgt = iglmax(nelgt,1)
      nelgv = iglmax(nelgv,1)

      if (npscal+1.gt.ldimt) then
         if(nid.eq.0) then
           write(6,21) ldimt,npscal+1
   21      format(//,2x,'Error: Nek has been compiled'
     $             /,2x,'       for max.',i4,' scalars. This run'
     $             /,2x,'       requires that ldimt be set to',i4,'.')
         endif
         call exitt
      endif

      if (nelgt.gt.neltmx.or.nelgv.gt.nelvmx) then
         if (nid.eq.0) then
          lelt_needed = nelgt/np
          if (mod(nelgt,np).ne.0) lelt_needed = lelt_needed + 1 
          write(6,82) lelt,lelg,lelt_needed,np,nelgt
   82         format(//,2X,'Error: Nek has has been compiled for'
     $         ,/,2X,      '       number of elements/proc  (lelt):',i12
     $         ,/,2X,      '       total number of elements (lelg):',i12
     $         ,/,2X
     $         ,/,2X,'This run requires:'
     $         ,/,2X,'   lelt >= ',i12,'  for np = ',i12
     $         ,/,2X,'   lelg >= ',i12,/)
         endif
         call exitt
      endif

      if (ifmvbd) then
         if (lx1.ne.lx1m.or.ly1.ne.ly1m.or.lz1.ne.lz1m) 
     $    call exitti
     $    ('Error: Mesh motion requires lx1m=lx1 etc. in SIZE . $',lx1m)
      endif

      IF(ldimr.NE.LDIM) THEN
         IF(NID.EQ.0) THEN
           WRITE(6,10) LDIM,ldimr
   10      FORMAT(//,2X,'Error: Nek has been compiled'
     $             /,2X,'       for spatial dimension equal to',I2,'.'
     $             /,2X,'       The mesh file has dimension',I2,'.')
         ENDIF
         call exitt
      ENDIF

      if (lpert.lt.npert) then
         if(nid.eq.0) write(6,*) 
     $   'ERROR: Increase lpert in SIZE to', npert
         call exitt
      endif

      IF (NPSCAL+1.GT.LDIMT .AND. IFMHD) THEN
         if(nid.eq.0) then
           WRITE(6,22) LDIMT,NPSCAL+1
   22      FORMAT(/s,2X,'Error: Nek has been compiled'
     $             /,2X,'       for',I4,' scalars.  A MHD run'
     $             /,2X,'       requires that LDIMT be set to',I4,'.')
         endif
         call exitt
      ENDIF

      if (if3d) then
         if (ly1.ne.lx1.or.lz1.ne.lx1) then
            if (nid.eq.0) write(6,13) lx1,ly1,lz1
   13       format('ERROR: lx1,ly1,lz1:',3i5,' must be equal for 3D')
            call exitt
         endif
         if (ly2.ne.lx2.or.lz2.ne.lx2) then
            if (nid.eq.0) write(6,14) lx2,ly2,lz2
   14       format('ERROR: lx2,ly2,lz2:',3i5,' must be equal for 3D')
            call exitt
         endif
      else
         if (ly1.ne.lx1.or.lz1.ne.1) then
            if (nid.eq.0) write(6,12) lx1,ly1,lz1
   12       format('ERROR: ',3i5,' must have lx1=ly1; lz1=1, for 2D')
            call exitt
         endif
         if (ly2.ne.lx2.or.lz2.ne.1) then
            if (nid.eq.0) write(6,11) lx2,ly2,lz2
   11       format('ERROR: ',3i5,' must have lx2=ly2; lz2=1, for 2D')
            call exitt
         endif
      endif

      if (lgmres.lt.5 .and. param(42).eq.0) then
         if(nid.eq.0) write(6,*)
     $   'WARNING: lgmres might be too low!'
      endif

      if (ifsplit) then
         if (lx1.ne.lx2) then
            if (nid.eq.0) write(6,43) lx1,lx2
   43    format('ERROR: lx1,lx2:',2i4,' must be equal for IFSPLIT=T')
            call exitt
         endif
      else
         if (lx2.ne.lx1-2) then
            if (nid.eq.0) write(6,44) lx1,lx2
   44    format('ERROR: lx1,lx2:',2i4,' lx2 must be lx-2 for IFSPLIT=F')
           call exitt
         endif
      endif

      if (param(40).eq.3 .and. .not.ifsplit) then
         call exitti
     $    ('ERROR: Selected preconditioner requires lx2=lx1$',lx2)
      endif

      if (ifsplit .and. ifuservp .and. .not.ifstrs) then
         if(nid.eq.0) write(6,*) 
     $   'Enable stress formulation to support PN/PN and IFUSERVP=T' 
         ifstrs = .true.
      endif

      if (ifcyclic .and. .not.ifstrs) then
         if(nid.eq.0) write(6,*) 
     $   'Enable stress formulation to support cyclic BC' 
         ifstrs = .true.
      endif

      ktest = (lx1-lx1m) + (ly1-ly1m) + (lz1-lz1m)
      if (ifstrs.and.ktest.ne.0) then
         if(nid.eq.0) write(6,*) 
     $   'ERROR: Stress formulation requires lx1m=lx1, etc. in SIZE'
         call exitt
      endif

      if (ifsplit .and. ifmhd) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: No MHD support for Pn-Pn'
         call exitt
      endif

      if (ifmhd .and. lbx1.ne.lx1) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For MHD, need lbx1=lx1, etc.; Change SIZE '
         call exitt
      endif

      if (ifpert .and. lpx1.ne.lx1) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For Lyapunov, need lpx1=lx1, etc.; Change SIZE '
      endif

      if (ifpert .and. ifsplit) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For Lyapunov, need lx2=lx1-2, etc. in SIZE'
      endif 

      if (iflomach .and. .not.ifsplit) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For lowMach, need lx2=lx1, etc.; Change SIZE '
         call exitt
      endif

      if (iflomach .and. .not.ifheat) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: For lowMach, need to solve for temperature too!'
         call exitt
      endif

      if (ifdp0dt .and. .not.iflomach) then
         if(nid.eq.0) write(6,*) 
     $   'ABORT: Varying p0 requires lowMach! '
         call exitt
      endif

      if (ifchar .and. param(99).lt.0) then
        if (nid.eq.0) write(6,*) 
     &     'ABORT: Characteristic scheme needs dealiasing!'
        call exitt
      endif

      if (.not.ifsplit .and. ifaxis .and. ifstrs) then
        if (nid.eq.0) write(6,*)
     $    'ABORT: Axisymetric and stress formulation not supported ' //
     $    'for PN/PN-2$'
        call exitt
      endif

      if (ifchar.and.(nelgv.ne.nelgt)) call exitti(
     $ 'ABORT: Characteristics not supported w/ conj. ht transfer$',1)

      if (param(99).gt.-1 .and. (lxd.lt.lx1 .or. lyd.lt.ly1 .or.
     &   lzd.lt.lz1)) then
         if(nid.eq.0) write(6,*)
     &   'ABORT: Dealiasing space too small; Check lxd,lyd,lzd in SIZE '
         call exitt
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine par_verify(ierr)

      INCLUDE 'PARDICT'

      
      character*132  key
      character*1024 val

      character*132 txt
      character*1   tx1(132)
      equivalence   (tx1,txt)

      ierr = 0

      call finiparser_getDictEntries(n)
      do i = 1,n
         call finiparser_getPair(key,val,i,ifnd)
         call capit(key,132)

         is = index(key,'_') ! ignore user keys
         if (is.eq.1) goto 10

         do j = 1,PARDICT_NKEYS ! do we find the key in the par-dictionary 
            if(index(pardictkey(j),key).eq.1) goto 10      

            is = index(key,'SCALAR')
            if(is .eq. 1) then
              call chcopy(txt,key,132)
              call chcopy(tx1(is+6),'%%',2) 
              if(index(pardictkey(j),txt).eq.1) goto 10
            endif
         enddo
         write(6,*) 'ERROR: Par file contains unknown key ', key
         ierr = ierr + 1
   10 enddo

      return
      end
