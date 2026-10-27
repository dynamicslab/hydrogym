c-----------------------------------------------------------------------
      subroutine makeq

C     Generate forcing function for the solution of a passive scalar.
C     !! NOTE: Do not change the content of the array BQ until the current

      include 'SIZE'
      include 'TOTAL'

      logical  if_conv_std
      common /SCRUZ/ w1(lx1,ly1,lz1,lelt)

      nxyz = nx1*ny1*nz1
      ntot = nxyz*nelv

      if_conv_std = .true.
      if (ifmhd.and.ifaxis) if_conv_std = .false. ! conv. treated in induct.f

      call makeq_aux ! nekuq, etc.

      if (ifadvc(ifield) .and. .not.ifchar .and. if_conv_std) then

         if (ifcvfld(ifield) .and. ifmvbd) then
            call sub2 (vx, wx, ntot)
            call sub2 (vy, wy, ntot)
            call sub2 (vz, wz, ntot)
         endif

         call convab

         if (ifcvfld(ifield) .and. ifmvbd) then
            call add2 (vx, wx, ntot)
            call add2 (vy, wy, ntot)
            call add2 (vz, wz, ntot)
         endif

      endif

      if (iftran) then

         if (ifcvfld(ifield)) then

           if (ifdiff(ifield)) then
              ntot = nx1*ny1*nz1*nelfld(ifield)
              call wlaplacian(w1,t(1,1,1,1,ifield-1),
     &                        vdiff(1,1,1,1,ifield),ifield)
              call add2(bq(1,1,1,1,ifield-1),w1,ntot)
           endif

         else

           if (ifmvbd.and..not.ifchar) call admesht

           call makeabq

           if (ifchar.and.ifadvc(ifield)) then
              call convch
           else
              call makebdq
           endif

         endif

      endif

      return
      end
