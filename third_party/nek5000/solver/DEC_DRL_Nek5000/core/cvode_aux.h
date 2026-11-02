      subroutine cvunpack(w1,y)
c
c     copy the internal cvode vector y to nek array w1 
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      real w1(lx1,ly1,lz1,lelt,1)
      real y(1)

      nxyz = nx1*ny1*nz1

      j = 1
        do ifield = 2,nfield
           if (ifcvfld(ifield)) then
              ntot = nxyz*nelfld(ifield)
              call copy   (w1(1,1,1,1,ifield-1),y(j),ntot)
              call bcdirsc(w1(1,1,1,1,ifield-1)) ! restore dirichlet bcs
              j = j + ntot
           endif
        enddo

      if (ifdp0dt) then
         p0th = y(j)
         j = j + 1 
      endif

      return
      end
c----------------------------------------------------------------------
      subroutine cvpack(y,w1,ifrhs)
c
c     copy the nek array w1 to the internal cvode vector y
c     note: assumes temperature is stored in ifield=2 (only for ifdp0dt)
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      real y(1)
      real w1(lx1,ly1,lz1,lelt,1)
      logical ifrhs

      common /scrsf/ dtmp(lx1,ly1,lz1,lelt)


      nxyz = nx1*ny1*nz1

      if (ifrhs .and. ifdp0dt) then
         call qthermal(.false.) ! computes dp0thdt  
         dd = (gamma0 - 1.)/gamma0
         dd = dd * dp0thdt
         ntot = nxyz*nelfld(2)
         call invers2(dtmp,vtrans(1,1,1,1,2),ntot)
         call cmult(dtmp,dd,ntot)
      endif

      j = 1
      do ifield = 2,nfield
         if (ifcvfld(ifield)) then
            ntot = nxyz*nelfld(ifield)
            call copy (y(j),w1(1,1,1,1,ifield-1),ntot)
            if (ifrhs) then
               if (ifdp0dt .and. ifield.eq.2) call add2(y(j),dtmp,ntot) 
               call col2(y(j),tmask(1,1,1,1,ifield-1),ntot)
            endif
            j = j + ntot
         endif
      enddo

      if (ifdp0dt) then
         y(j) = p0th
         if (ifrhs) y(j) = dp0thdt
      endif

      return
      end

