c-----------------------------------------------------------------------
      subroutine makeq

C     Generate forcing function for the solution of a passive scalar.
C     !! NOTE: Do not change the content of the array BQ until the current

      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'

      logical  if_conv_std
      common /SCRUZ/ w1(lx1,ly1,lz1,lelt)

      nxyz = lx1*ly1*lz1
      ntot = nxyz*nelv

      etime0 = dnekclock()      

      if (nio.eq.0.and.loglevel.gt.2)
     $   write(6,*) 'makeq', ifield

      if_conv_std = .true.
      if (ifmhd.and.ifaxis) if_conv_std = .false. ! conv. treated in induct.f

      call makeq_aux ! nekuq, etc.

      if (ifadvc(ifield) .and. if_conv_std) then

         if (ifcvfld(ifield)) then
            if (ifmvbd) then
               call sub2 (vx, wx, ntot)
               call sub2 (vy, wy, ntot)
               call sub2 (vz, wz, ntot)
             endif

             call convab

             if (ifmvbd) then
                call add2 (vx, wx, ntot)
                call add2 (vy, wy, ntot)
                call add2 (vz, wz, ntot)
             endif
         else
             if (.not.ifchar) call convab
         endif

      endif

      if (iftran) then

         if (ifcvfld(ifield)) then

           if (ifdiff(ifield)) then
              ntot = lx1*ly1*lz1*nelfld(ifield)
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

      tmakq=tmakq+(dnekclock()-etime0)

      return
      end
