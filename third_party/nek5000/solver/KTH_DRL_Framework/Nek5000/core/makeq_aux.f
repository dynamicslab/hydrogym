      subroutine makeq_aux

      include 'SIZE'
      include 'TOTAL'

      logical  if_conv_std

      if_conv_std = .true.
      if (ifmhd.and.ifaxis) if_conv_std = .false. ! conv. treated in induct.f

      if (if_conv_std) call makeuq ! this will zero bq first

      if (ifdp0dt .and.  ifield.eq.2 .and. .not.ifcvfld(ifield)) then
            dd = dp0thdt * (gamma0 - 1.)/gamma0
            ntot = nx1*ny1*nz1*nelv
            call add2s2 (bq(1,1,1,1,ifield-1),bm1,dd,ntot) 
      endif

      if(filterType.eq.2) call make_hpf
    
      return
      end
