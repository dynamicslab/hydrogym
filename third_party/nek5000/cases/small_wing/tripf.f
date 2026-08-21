!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!    Trip forcing
!    intermediate implementation
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!-----------------------------------------------------------------------
      subroutine tripf

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'TRIPF'

      integer k

      integer z,i
      real p,b
      real tamps, tampt, tdt
      integer num_modes

      tamps = wallpar(1)
      tampt = wallpar(2)
      tdt   = wallpar(3)
      num_modes = int(wallpar(4))
c
c     Generate the time independent part fzt(z,2)
c     at first iteration

      if (istep.eq.1) then
c
c     Get random distribution and rescale
c
         do k=1,nwalls
            call rand_func(fzt2(1,k),znek(1,k),kpts(k),seed,num_modes)
            do z=1,kpts(k)
               fzt2(z,k)=tamps*fzt2(z,k)
            end do
         enddo
         ntdt=-2
      end if
c
c     Generate new time dependent part if necessary
c     to be able to recreate the trip of restarted simulations,
c     loop from ntdt=-1 up to present trip count.
c
      do i=ntdt+1,int(time/tdt)
         do k=1,nwalls
            do z=1,kpts(k)
               fzt3(z,k)=fzt4(z,k)
            end do
         enddo
c
c     Get random distribution and rescale
c
         do k=1,nwalls
            call rand_func(fzt4(1,k),znek(1,k),kpts(k),seed,num_modes)
            do z=1,kpts(k)
               fzt4(z,k)=tampt*fzt4(z,k)
            enddo
         enddo
      enddo
c
c     Update trip count as actual time divided by time scale
c
      ntdt=int(time/tdt)
c
c     Generate the z-dependence of the trip
c     as a smooth transition between old and new trip vectors
c     p is varying from 0 to 1 for a given trip count.
c
      p=(time-real(ntdt)*tdt)/tdt
      b=p*p*(3.-2.*p)
      do k=1,nwalls
         do z=1,kpts(k)
            fzt1(z,k)=fzt2(z,k)+(1.-b)*fzt3(z,k)+b*fzt4(z,k)
         enddo
      enddo

      end subroutine tripf

!---------------------------------------------------------------------- 
      subroutine ARnumnodes

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'TRIPF'
cc MA:      include 'GEOM_DEF'
      include 'GEOM'

      integer ARatio
      real ARx1max, ARx1min, ARx2max, ARx2min
      integer n

      n = nelv*lx1*ly1*lz1

      ARx1max = glmax(xm1,n)
      ARx2max = glmax(ym1,n)
      ARx1min = glmin(xm1,n)
      ARx2min = glmin(xm1,n)
      ARatio = int((ARx1max-ARx1min)/(ARx2max-ARx2min))
      if (ARatio .lt. (ARx1max-ARx1min)/(ARx2max-ARx2min)) then
         ARatio = ARatio + 1
      endif
      wallpar(4) = wallpar(4)*ARatio
      
c      write(*,*) 'Wall parameter is:', wallpar(4) 

      return
      end
c-----------------------------------------------------------------------
      subroutine readwallfile

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'TRIPF'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
      integer len,ierr
      character*132 wallfname
      character*1 wallfnam1(132)
      equivalence (wallfnam1,wallfname)

      ierr = 0
      if (nid .eq. 0) then
         call blank(wallfname,132)
         len = ltrunc(SESSION,132)
         call chcopy(wallfnam1(1),SESSION,len)
         call chcopy(wallfnam1(len+1),'.wall',5)
         open(unit=75,file=wallfname,err=30,status='old')
         read(75,*,err=30)
         read(75,*,err=30) nwalls
         read(75,*,err=30) nwallpar
         read(75,*,err=30) npwallpar
         goto 31
 30      ierr=1
 31      continue
      endif
      call err_chk(ierr,'Error reading .wall file.$')
      call bcast(nwalls, ISIZE)
      call bcast(nwallpar, ISIZE)
      call bcast(npwallpar, ISIZE)

      call blank(direction,maxwalls)

      if(nwalls .gt. maxwalls .or. nwallpar .gt. maxwallpar
     $           .or. npwallpar .gt. maxpwallpar ) then
         if(nid .eq. 0) then
           write(6,*) 'Too many walls/parameters in ',wallfname
         endif
         call exitt
      endif
      if (nid .eq. 0) then
c   read global parameters
        read(75,*,err=32)
        do i=1,nwallpar
          read(75,*,err=32) wallpar(i)
        end do
c   read wall definitions and parameters
        read(75,*,err=32)
        read(75,*,err=32)
        do i=1,nwalls
          read(75,*,err=32) direction(i)
          read(75,*,err=32) tripx(i),tripy(i),tripz(i)
          do j=1,npwallpar
            read(75,*,err=32) (pwallpar(k,j,i), k=1,3)
          end do
        end do
        goto 33
 32     ierr=1
 33     continue
      endif
     
      call err_chk(ierr,'Not enough walls.$')
      call bcast(wallpar,nwallpar*WDSIZE)
      call bcast(direction,nwalls*CSIZE)
      call bcast(tripx,nwalls*WDSIZE)
      call bcast(tripy,nwalls*WDSIZE)
      call bcast(tripz,nwalls*WDSIZE)
      call bcast(pwallpar(1,1,1),3*npwallpar*nwalls*WDSIZE)

      return
      end
c----------------------------------------------------------------------
      subroutine znekgen(wall)

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'TOTAL'
      include 'TRIPF'

      real dx1(lx1,ly1,lz1,lelv), bouxm1(lx1,ly1,lz1,lelv)
      real dx2(lx1,ly1,lz1,lelv), bouxm2(lx1,ly1,lz1,lelv)
      real dr2(lx1,ly1,lz1,lelv), bouxm3(lx1,ly1,lz1,lelv)
      real tripx1, tripx2

      real vals(maxlxyz,nelv), valsSort(maxlxyz,nelv)
      real valsf(nelv), valsfSort(nelv)
      real gvalsf(np*lelv),gvalsfw(np*lelv),gvalsfSort(np*lelv)
      integer valsfw(nelv), gvalsfi(np*lelv), wall
      real gCloseGLL(2), lCloseGLL, realTemp
      real znekw(lelv*maxlxyz)
      integer lCloseGLLid, cGLLnid, intTemp
      integer cvals, cvals1, cvals2, myit, itx, ity, itz

      itx = lx1
      ity = ly1
      itz = lz1
c   compute the differences
      if (direction(wall) .eq. 'x') then
         bouxm1 = ym1
         bouxm2 = zm1
         bouxm3 = xm1
         myit = itx
         tripx1 = tripy(wall)
         tripx2 = tripz(wall)
         itx = 1
      elseif (direction(wall) .eq. 'y') then
         bouxm1 = xm1
         bouxm2 = zm1
         bouxm3 = ym1
         myit = ity
         tripx1 = tripx(wall)
         tripx2 = tripz(wall)
         ity = 1
      else
         bouxm1 = xm1
         bouxm2 = ym1
         bouxm3 = zm1
         tripx1 = tripx(wall)
         tripx2 = tripy(wall)
         myit = itz
         itz = 1
      endif

      dx1 = tripx1 - bouxm1
      dx2 = tripx2 - bouxm2
      dr2 = dx1*dx1 + dx2*dx2
      lCloseGLL = dr2(1,1,1,1)
      lCloseGLLid = 1

c   calculate the local minimum distance
      do j = 1, nelv
      do iz = 1,itz
        do iy = 1,ity
          do ix = 1,itx
            if (dr2(ix,iy,iz,j) .lt. lCloseGLL) then
              lCloseGLL = dr2(ix,iy,iz,j)
              lCloseGLLid = ix + lx1*(iy-1) + lx1*ly1*(iz-1)
     $                         + (j-1)*lx1*ly1*lz1
            end if
          end do
        end do
      end do
      end do
      gCloseGLL(1) = lCloseGLL

c      print *, 'ISM', nid, lCloseGLL
c   pick the global minimum distance
      call gop(gCloseGLL(1),realTemp,'m  ',1)

c   chose a proc who has this distance
      if (lCloseGLL .eq. gCloseGLL(1)) then
         cGLLnid = nid
      else
         cGLLnid = 0
      end if
      call igop(cGLLnid,intTemp,'M  ',1)

c   share its x,y value to everyone
      if (cGLLnid .eq. nid) then
         gCloseGLL(1) = bouxm1(lCloseGLLid,1,1,1)
         gCloseGLL(2) = bouxm2(lCloseGLLid,1,1,1)
      else
         gCloseGLL(1) = 0.
         gCloseGLL(2) = 0.
      end if
c      print *, 'ISM', nid, gCloseGLL(1), gCloseGLL(2),tripx1,tripx2
      call bcastn0(gCloseGLL(1),2*WDSIZE,cGLLnid)

c      print *, 'ISM', nid, gCloseGLL(1), gCloseGLL(2),tripx1,tripx2
c   sort the first z-value of each element containing the tripping points
      cvals = 0
      do j = 1,nelv
       do iz = 1, itz
         do iy = 1, ity
           do ix = 1, itx
             if (bouxm1(ix,iy,iz,j) .eq. gCloseGLL(1)
     $         .and. bouxm2(ix,iy,iz,j) .eq. gCloseGLL(2)) then
               cvals = cvals + 1
               if (direction(wall) .eq. 'x') then
                 vals(1:myit,cvals) = bouxm3(:,iy,iz,j)
               elseif (direction(wall) .eq. 'y') then
                 vals(1:myit,cvals) = bouxm3(ix,:,iz,j)
               else
                 vals(1:myit,cvals) = bouxm3(ix,iy,:,j)
               end if
               valsf(cvals) = bouxm3(ix,iy,iz,j)
               goto 100
             end if
           end do
         end do
       end do
 100   continue
      end do
      call sorts(valsfSort,valsf,valsfw,cvals)

c      print *, 'ISM', nid, cvals, valsfSort(1:cvals)
c   remove duplicate and share with everyone
      gvalsf = huge(1.0)
      if (cvals .gt. 0) then
         cvals1 = 1
         valsSort(:,1) = vals(:,valsfw(1))
         gvalsf(1 + nid*lelv) = valsfSort(1)
      else
         cvals1 = 0
      end if
      do i = 2, cvals
         if(valsfSort(i) .ne. valsfSort(i-1)) then
           cvals1 = cvals1 + 1
           valsSort(:,cvals1) = vals(:,valsfw(i))
           gvalsf(cvals1 + nid*lelv) = valsfSort(i)
         end if
      end do
      call gop(gvalsf, gvalsfw,'m  ', np*lelv)


c   define kpts (lx * number of direction elements), nnelx1x2 (nx1*nx2), znek
      call sorts(gvalsfSort,gvalsf,gvalsfi,np*lelv)
c      print *, 'ISM1', nid, cvals, gvalsfSort(1:10)
      cvals2 = 1
      do i = 1,np*lelv
        if (gvalsfSort(i) .ne. gvalsfSort(cvals2)) then
           cvals2 = cvals2 + 1
           gvalsfSort(cvals2) = gvalsfSort(i)
        endif
        if (i .ne. cvals2) gvalsfSort(i) = huge(1.0)
      end do
c      print *, 'ISM2', nid, cvals, gvalsfSort(1:10)
      znek(:,wall) = huge(1.0)
      cvals2 = 1
      do i = 1, lelv
        if (gvalsfSort(i) .eq. huge(1.0)) then
c          print *, 'ISM fini', i, lz1
          kpts(wall) = (i-1)*myit
          nnelx1x2(wall) = nelgv/(i-1)
          exit
        end if
        if (gvalsf(cvals2 + nid*lelv) .eq. gvalsfSort(i)) then
          do j = 1,myit !i*lz1,(i+1)*lz1
            znek((i-1)*myit+j,wall) = valsSort(j,cvals2)
          end do
          cvals2 = cvals2 + 1
        end if
      end do
      call gop(znek(1,wall), znekw,'m  ', kpts(wall))

      if (nid .eq. 0) then
      do i=1,kpts(wall)
         print *,'ISM', znek(i,wall)
      end do
      end if
c   Done!

ccccccccccccccc
c   print the values for script processing (no more needed)

c      if (nid .eq. 0) write (6,"('ISM2',x,i7)") nelgv
c      if (nid .eq. 0) write (6,"('ISM3',x,i7)") lz1
c      write(clz1,"(i2)") lz1
c      do k = 1,nelv
c        do i = 1, lx1*ly1
c          if (xm1(i,1,1,k) .eq. gCloseGLL(1)
c     $       .and. ym1(i,1,1,k) .eq. gCloseGLL(2)) then
c           write (6,"('ISM1'," // adjustr(clz1) // "(x,g25.16E4))")
c     $            (zm1(i,1,j,k),j=1,lz1)
c          end if
c       end do
c      end do
c      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine bcastn0(buf,len,proc)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      real*4 buf(1)

      call mpi_bcast (buf,len,mpi_byte,proc,nekcomm,ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine rand_func(rand_vec,zvec,zpts,seed,num_modes)

      implicit none

      integer seed,k
      integer zpts
      real zvec(1:zpts),bb
      real rand_vec(zpts)

      real pi
      parameter (pi = 3.1415926535897932385)
      integer num_modes
c      parameter (num_modes = 10)
c
c     Local variables
c
      integer z,m
      real zlength
      real phase
      real theta
c
c     External function
c
      real ran2
c
c     Compute length of z-interval
c
      zlength = zvec(zpts) - zvec(1)
      if (zlength .eq. 0.) zlength = 1.
      do z=1,zpts
         rand_vec(z) = 0.0
      enddo
c
c     Compute m sinus modes
c
      do m=1,num_modes
         bb = ran2(seed)
         phase = 2.*pi*bb
         do z=1,zpts
            theta = 2.*pi*m*zvec(z)/zlength
            rand_vec(z) = rand_vec(z) + sin(theta + phase)
         enddo
      enddo

      end subroutine rand_func
!---------------------------------------------------------------------- 

      real function ran2(idum)
c
c     A simple portable random number generator
c
c     Requires 32-bit integer arithmetic
c     Taken from Numerical Recipes, William Press et al.
c     gives correlation free random numbers but does not have a very large
c     dynamic range, i.e only generates 714025 different numbers
c     for other use consult the above
c     Set idum negative for initialization
c
      implicit none

      integer idum,ir(97),m,ia,ic,iff,iy,j
      real rm
      parameter (m=714025,ia=1366,ic=150889,rm=1./m)
      save iff,ir,iy
      data iff /0/

      if (idum.lt.0.or.iff.eq.0) then
c
c     Initialize
c
         iff=1
         idum=mod(ic-idum,m)
         do j=1,97
            idum=mod(ia*idum+ic,m)
            ir(j)=idum
         end do
         idum=mod(ia*idum+ic,m)
         iy=idum
      end if
c
c     Generate random number
c
      j=1+(97*iy)/m
      iy=ir(j)
      ran2=iy*rm
      idum=mod(ia*idum+ic,m)
      ir(j)=idum

      end function ran2
c-----------------------------------------------------------------------
      subroutine readtrip_par

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'TRIPF'

      real tdt, tampt
c    read TRIP FORCING parameters from forparam.i

      if (istep.eq.0.) then
         open(unit=1011,status='old',file='forparam.i')
         if(nid.eq.0)  write(6,*)'Tripping parameters '
         read(1011,*) xup
         read(1011,*) yup
         if(nid.eq.0)  write(6,*) 'x_up = ', xup, 'y_up = ', yup
         read(1011,*) xlo
         read(1011,*) ylo
         if(nid.eq.0)  write(6,*) 'x_lo = ', xlo, 'y_lo = ', ylo
         read(1011,*) radiusx
         if(nid.eq.0)  write(6,*) 'radius =  ', radiusx
         read(1011,*) radiusy
         if(nid.eq.0)  write(6,*) 'radius =  ', radiusy
         read(1011,*) tdt
         if(nid.eq.0)  write(6,*) 'tdt =  ', tdt
         read(1011,*) tampt
         if(nid.eq.0)  write(6,*) 'tampt =  ', tampt
         read(1011,*) alpha_elipse
         if(nid.eq.0)  write(6,*) 'alpha =  ', alpha_elipse
      
         close(unit=1011)
      end if

      return
      end subroutine readtrip_par

C=======================================================================


 
