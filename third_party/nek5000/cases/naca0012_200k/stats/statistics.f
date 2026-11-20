!=======================================================================
! Name        : time_series
! Author      : modified by Adam Peplinski
! Version     : last modification 2015.02.20
! Copyright   : GPL
! Description : This is a set of routines to generate statistics for wing simulations.
!=======================================================================
!     Trbulent DUCT flow, run_time statistics
!     PARAMETERS to define
!     .rea file: iastep (p68), nv (p69) and n2ptc (p70)
!     SIZE file: nstat,ldist
C=======================================================================

      subroutine avg_stat_all

      include 'SIZE'
      include 'TOTAL'
      include 'ZPER'  
      include 'mpif.h'
 

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
      common /avgcmnr/ atime,timel
      common /c_p0/ p0(lx1,ly1,lz1,lelt)
      common /cvflow_r/ flow_rate,base_flow,domain_length,xsec

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
C ---------------------------------------------------------------------- C
C -------------- Define the various quantities on a 3D array------------ C
C ---------------------------------------------------------------------- C
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

      real stat(lx1*ly1*lz1*lelt)
      real stat_d(lx1*ly1*lz1*lelt)

      real pm1(lx1*ly1*lz1*lelt)     
      real wk1(lx1*ly1*lz1)
      real wk2(lx1*ly1*lz1)

      real duidxj(lx1*ly1*lz1,lelt,3*ldim)
      real ur(lx1*ly1*lz1),us(lx1*ly1*lz1),ut(lx1*ly1*lz1)
      real vr(lx1*ly1*lz1),vs(lx1*ly1*lz1),vt(lx1*ly1*lz1)
      real wr(lx1*ly1*lz1),ws(lx1*ly1*lz1),wt(lx1*ly1*lz1)

      real u_avg, uu_avg
      real xlmin, xlmax, domain_x
      real ylmin, ylmax, domain_y
      real zlmin, zlmax, domain_z
      
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
C ---------------------------------------------------------------------- C
C -------------- Define the various quantities on a 2D array ----------- C
C ---------------------------------------------------------------------- C
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

      real stat_xy_t(lx1*ly1*lelx*lely)

      real stat_xy_dist_t(ldist)
      real stat_xy_dist  (ldist,nstat)

      real w1(lx1,ly1,lelx,lely), w2(lx1,ly1,lelx,lely)

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
c      logical ifverbose
      integer icalld
      save    icalld
      data    icalld  /0/

      integer icall2
      save    icall2
      data    icall2 /-9/

      real atime,timel,times
      real my_surf_mean

      integer indts, nrec, ss
      save    indts, nrec, ss
      save    times
      save    domain_x, domain_y, domain_z

      character*80 pippo
      character*80 val1, val2, val3, val4, val5, val6
      character*80 val7, val8, val9, val10, val11
      character*80 inputname1, inputname2, inputname3

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal

!     ADAM; for test
!     simple timing
      real ltim_init, ltim_min, ltim_max
      save ltim_init, ltim_min, ltim_max
      real ltm1, ltm2, ltm3, ltm4

!     functions
      real dnekclock, glmax, glmin

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
!     ADAM; for test
      ltm1 = dnekclock()

!     Interval to save stat files
      iastep = param(68)
      if  (iastep.eq.0) iastep=param(15) ! same as iostep

!     Interval of time record
!     Remainders nsteps/nv and iastep/nv should be 0
      nv=param(69)     
                       
!     Interval to dump files for 2-point-correlations
      n2ptc=param(70)  

!     These definitions are required for z_average
      nelx = lelx
      nely = lely
      nelz = lelz

!     Total number of grid-points per core (3D arrays)
      ntot = nx1*ny1*nz1*nelv

!     Total number of grid-points (2D arrays - undistributed)
      ntot_2d = nx1*ny1*lelx*lely

!     2D grid-points per core ntot_2d/np is nsend, must be integer
      if (mod(ntot_2d,np).eq.0) then
         nsend = ntot_2d/np
      else  
         nsend = ntot_2d/np+1
      end if

!     Check that allocated memory with ldist is enough for nsend
      if (nsend.gt.ldist) then
         write(*,*) 'increase ldist to at least ',nsend
         call exitt()
      end if

!     Initialize variables in first time-step
      if (icalld.eq.0) then

!     ADAM; for test
         ltim_min = 0.0

!     ADAM; for test
         ltm2 = dnekclock()

         icalld = icalld + 1 ! Initialization flag
         atime  = 0. ! Initialize averaging time of the stat file
         timel  = time ! Initialize starting time of the nv interval

         ! Extract domain limits
         xlmin = glmin(xm1,ntot)
         xlmax = glmax(xm1,ntot)
         domain_x = xlmax - xlmin

         ylmin = glmin(ym1,ntot)          
         ylmax = glmax(ym1,ntot)
         domain_y = ylmax - ylmin
         
         zlmin = glmin(zm1,ntot)          
         zlmax = glmax(zm1,ntot)
         domain_z = zlmax - zlmin
         
         ! Initialize 3D arrays
         call rzero(stat,ntot)
         call rzero(stat_d,ntot)
         
         ! Initialize undistributed 2D array
         call rzero(stat_xy_t,ntot_2d)
         
!     ADAM; possibly not necessary
         ! Initialize distributed 2D arrays
         call rzero(stat_xy_dist,ldist*nstat)
         call rzero(stat_xy_dist_t,ldist)

!     ADAM; for test
         ltim_init = dnekclock() - ltm2

      end if

!     Find inverse (jacmi) of the Jacobian array (jacm1)
      if (istep.ne.icall2) then
         call invers2(jacmi,jacm1,ntot)
         icall2=istep
      endif

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

!     Time span of current nv interval
      dtime = time  - timel

!     For ifverbose calls
c      ifverbose=.false.
c      if (istep.le.10) ifverbose=.false.
c      if  (mod(istep,iastep).eq.0) ifverbose=.false. ! get rid of output for now

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

!     Initialize number of records and starting flag of stat file
      if (istep.le.1) then
         nrec  = 0
         ss    = 1
      endif

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

!     Two-point correlations
c      if (mod(istep,n2ptc).eq.0.and.istep.gt.1) then
c         param(66)=6.
c         call outpost(vx,vy,vz,p0,p0,'2pt')
c         param(66)=6.
c      endif                                                                                   


C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C  

!     Every nv time-steps we average in time
      if (mod(istep,nv).eq.0.and.istep.gt.1) then

!     If ss=1 it means that it is the beginning of a stat file.
!     We store the time as times to write it out on the stat file header
         if (ss.eq.1) then
            times = time
            ss    = 2
         endif

!     We increase the count of number of records.
!     Our time average is composed by a number of individual values taken every
!     nv time-steps, which each constitute a record
      nrec  = nrec + 1

!     Update total time over which the current stat file is averaged
      atime = atime + dtime
      
!     ADAM; just for test
      if(0.eq.1) then

!     Map pressure to velocity mesh
      call mappr(p0,pr,wk1,wk2) 

!     Calculate reference pressure
      pmean = -my_surf_mean(p0,1,'W  ',ierr)

!     Add p0 and pmean, which subtracts the reference pressure
      call cadd(p0,pmean,ntot)

!     Compute derivative tensor
      call comp_derivat(duidxj,vx,vy,vz,ur,us,ut,vr,vs,vt,wr,ws,wt)

!     adding time series here
!     notice I use p0, so the mean pressure is subtracted from pressure
      call stat_pts(duidxj, p0)

!     ADAM; just for test
      endif

!     Start statistics
      if (atime.ne.0..and.dtime.ne.0.) then
         ! Time average is accum=alpha*accum+beta*new
         beta  = dtime/atime
         alpha = 1.-beta
         
         ! Statistics with z-averaging before time averaging
         ! 2D statistics stored in distributed arrays stast_xy_dist

c-----------------------------------------------------------------------

         call z_average(stat_xy_t,vx,w1,w2)                               ! <u>z
!         call z_average(stat_xy_t,xm1,w1,w2)
!         call z_average(stat_xy_t,ym1,w1,w2)
         call avgt(stat_xy_dist(1,1),stat_xy_dist_t,stat_xy_t,            ! <u>zt
     &        alpha,beta,nsend)                                           

!     ADAM; just for test
         if(0.eq.1) then

         call z_average(stat_xy_t,vy,w1,w2)                               ! <v>z
         call avgt(stat_xy_dist(1,2),stat_xy_dist_t,stat_xy_t,            ! <v>zt
     &        alpha,beta,nsend)

         call z_average(stat_xy_t,vz,w1,w2)                               ! <w>z
         call avgt(stat_xy_dist(1,3),stat_xy_dist_t,stat_xy_t,
     &        alpha,beta,nsend)                                           ! <w>zt
         
         call z_average(stat_xy_t,p0,w1,w2)                               ! <p>z
         call avgt(stat_xy_dist(1,4),stat_xy_dist_t,stat_xy_t,            ! <p>zt
     &        alpha,beta,nsend)                                           

c-----------------------------------------------------------------------

         call prod2(stat,vx,vx,ntot)                                      !  uu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uu>z
         call avgt(stat_xy_dist(1,5),stat_xy_dist_t,stat_xy_t,            ! <uu>zt
     &        alpha,beta,nsend)

         call prod2(stat,vy,vy,ntot)                                      !  vv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vv>z
         call avgt(stat_xy_dist(1,6),stat_xy_dist_t,stat_xy_t,            ! <vv>zt
     &        alpha,beta,nsend)

         call prod2(stat,vz,vz,ntot)                                      !  ww
         call z_average(stat_xy_t,stat,w1,w2)                             ! <ww>z
         call avgt(stat_xy_dist(1,7),stat_xy_dist_t,stat_xy_t,            ! <ww>zt
     &        alpha,beta,nsend)

         call prod2(stat,p0,p0,ntot)                                      !  pp
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pp>z
         call avgt(stat_xy_dist(1,8),stat_xy_dist_t,stat_xy_t,            ! <pp>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod2(stat,vx,vy,ntot)                                      !  uv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uv>z
         call avgt(stat_xy_dist(1,9),stat_xy_dist_t,stat_xy_t,            ! <uv>zt
     &        alpha,beta,nsend)

         call prod2(stat,vy,vz,ntot)                                      !  vw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vw>z
         call avgt(stat_xy_dist(1,10),stat_xy_dist_t,stat_xy_t,           ! <vw>zt
     &        alpha,beta,nsend)

         call prod2(stat,vx,vz,ntot)                                      !  uw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uw>z
         call avgt(stat_xy_dist(1,11),stat_xy_dist_t,stat_xy_t,           ! <uw>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod2(stat,p0,vx,ntot)                                      !  pu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pu>z
         call avgt(stat_xy_dist(1,12),stat_xy_dist_t,stat_xy_t,           ! <pu>zt
     &        alpha,beta,nsend)

         call prod2(stat,p0,vy,ntot)                                      !  pv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pv>z
         call avgt(stat_xy_dist(1,13),stat_xy_dist_t,stat_xy_t,           ! <pv>zt
     &        alpha,beta,nsend)

         call prod2(stat,p0,vz,ntot)                                      !  pw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pw>z
         call avgt(stat_xy_dist(1,14),stat_xy_dist_t,stat_xy_t,           ! <pw>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'pux')                            !   dudx
         call prod2(stat,p0,stat_d,ntot)                                  !  pdudx
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdudx>z
         call avgt(stat_xy_dist(1,15),stat_xy_dist_t,stat_xy_t,           ! <pdudx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'puy')                            !   dudy
         call prod2(stat,p0,stat_d,ntot)                                  !  pdudy
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdudy>z
         call avgt(stat_xy_dist(1,16),stat_xy_dist_t,stat_xy_t,           ! <pdudy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'puz')                            !   dudz
         call prod2(stat,p0,stat_d,ntot)                                  !  pdudz
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdudz>z
         call avgt(stat_xy_dist(1,17),stat_xy_dist_t,stat_xy_t,           ! <pdudz>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'pvx')                            !   dvdx
         call prod2(stat,p0,stat_d,ntot)                                  !  pdvdx
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdvdx>z
         call avgt(stat_xy_dist(1,18),stat_xy_dist_t,stat_xy_t,           ! <pdvdx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'pvy')                            !   dvdy
         call prod2(stat,p0,stat_d,ntot)                                  !  pdvdy
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdvdy>z
         call avgt(stat_xy_dist(1,19),stat_xy_dist_t,stat_xy_t,           ! <pdvdy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'pvz')                            !   dvdz
         call prod2(stat,p0,stat_d,ntot)                                  !  pdvdz
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdvdz>z
         call avgt(stat_xy_dist(1,20),stat_xy_dist_t,stat_xy_t,           ! <pdvdz>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'pwx')                            !   dwdx
         call prod2(stat,p0,stat_d,ntot)                                  !  pdwdx
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdwdx>z
         call avgt(stat_xy_dist(1,21),stat_xy_dist_t,stat_xy_t,           ! <pdwdx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'pwy')                            !   dwdy
         call prod2(stat,p0,stat_d,ntot)                                  !  pdwdy
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdwdy>z
         call avgt(stat_xy_dist(1,22),stat_xy_dist_t,stat_xy_t,           ! <pdwdy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'pwz')                            !   dwdz
         call prod2(stat,p0,stat_d,ntot)                                  !  pdwdz
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pdwdz>z
         call avgt(stat_xy_dist(1,23),stat_xy_dist_t,stat_xy_t,           ! <pdwdz>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod3(stat,vx,vx,vx,ntot)                                   !  uuu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuu>z
         call avgt(stat_xy_dist(1,24),stat_xy_dist_t,stat_xy_t,           ! <uuu>zt
     &        alpha,beta,nsend)

         call prod3(stat,vy,vy,vy,ntot)                                   !  vvv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vvv>z
         call avgt(stat_xy_dist(1,25),stat_xy_dist_t,stat_xy_t,           ! <vvv>zt
     &        alpha,beta,nsend)

         call prod3(stat,vz,vz,vz,ntot)                                   !  www
         call z_average(stat_xy_t,stat,w1,w2)                             ! <www>z
         call avgt(stat_xy_dist(1,26),stat_xy_dist_t,stat_xy_t,           ! <www>zt
     &        alpha,beta,nsend)

         call prod3(stat,p0,p0,p0,ntot)                                   !  ppp
         call z_average(stat_xy_t,stat,w1,w2)                             ! <ppp>z
         call avgt(stat_xy_dist(1,27),stat_xy_dist_t,stat_xy_t,           ! <ppp>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod3(stat,vx,vx,vy,ntot)                                   !  uuv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuv>z
         call avgt(stat_xy_dist(1,28),stat_xy_dist_t,stat_xy_t,           ! <uuv>zt
     &        alpha,beta,nsend)

         call prod3(stat,vx,vx,vz,ntot)                                   !  uuw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuw>z
         call avgt(stat_xy_dist(1,29),stat_xy_dist_t,stat_xy_t,           ! <uuw>zt
     &        alpha,beta,nsend)

         call prod3(stat,vy,vy,vx,ntot)                                   !  vvu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vvu>z
         call avgt(stat_xy_dist(1,30),stat_xy_dist_t,stat_xy_t,           ! <vvu>zt
     &        alpha,beta,nsend)

         call prod3(stat,vy,vy,vz,ntot)                                   !  vvw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vvw>z
         call avgt(stat_xy_dist(1,31),stat_xy_dist_t,stat_xy_t,           ! <vvw>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod3(stat,vz,vz,vx,ntot)                                   !  wwu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <wwu>z
         call avgt(stat_xy_dist(1,32),stat_xy_dist_t,stat_xy_t,           ! <wwu>zt
     &        alpha,beta,nsend)

         call prod3(stat,vz,vz,vy,ntot)                                   !  wwv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <wwv>z
         call avgt(stat_xy_dist(1,33),stat_xy_dist_t,stat_xy_t,           ! <wwv>zt
     &        alpha,beta,nsend)

         call prod3(stat,vx,vy,vz,ntot)                                   !  uvw
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uvw>z
         call avgt(stat_xy_dist(1,34),stat_xy_dist_t,stat_xy_t,           ! <uvw>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod4(stat,vx,vx,vx,vx,ntot)                                !  uuuu
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuuu>z
         call avgt(stat_xy_dist(1,35),stat_xy_dist_t,stat_xy_t,           ! <uuuu>zt
     &        alpha,beta,nsend)

         call prod4(stat,vy,vy,vy,vy,ntot)                                !  vvvv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <vvvv>z
         call avgt(stat_xy_dist(1,36),stat_xy_dist_t,stat_xy_t,           ! <vvvv>zt
     &        alpha,beta,nsend)

         call prod4(stat,vz,vz,vz,vz,ntot)                                !  wwww
         call z_average(stat_xy_t,stat,w1,w2)                             ! <wwww>z
         call avgt(stat_xy_dist(1,37),stat_xy_dist_t,stat_xy_t,           ! <wwww>zt
     &        alpha,beta,nsend)

         call prod4(stat,p0,p0,p0,p0,ntot)                                !  pppp
         call z_average(stat_xy_t,stat,w1,w2)                             ! <pppp>z
         call avgt(stat_xy_dist(1,38),stat_xy_dist_t,stat_xy_t,           ! <pppp>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call prod4(stat,vx,vx,vx,vy,ntot)                                !  uuuv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuuv>z
         call avgt(stat_xy_dist(1,39),stat_xy_dist_t,stat_xy_t,           ! <uuuv>zt
     &        alpha,beta,nsend)

         call prod4(stat,vx,vx,vy,vy,ntot)                                !  uuvv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uuvv>z
         call avgt(stat_xy_dist(1,40),stat_xy_dist_t,stat_xy_t,           ! <uuvv>zt
     &        alpha,beta,nsend)

         call prod4(stat,vx,vy,vy,vy,ntot)                                !  uvvv
         call z_average(stat_xy_t,stat,w1,w2)                             ! <uvvv>z
         call avgt(stat_xy_dist(1,41),stat_xy_dist_t,stat_xy_t,           ! <uvvv>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv2(stat_d,duidxj,ntot,'e11')                            !  e11: (du/dx)^2+(du/dy)^2+(du/dz)^2
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e11>z
         call avgt(stat_xy_dist(1,42),stat_xy_dist_t,stat_xy_t,           ! <e11>zt
     &        alpha,beta,nsend)

         call deriv2(stat_d,duidxj,ntot,'e22')                            !  e22: (dv/dx)^2+(dv/dy)^2+(dv/dz)^2
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e22>z
         call avgt(stat_xy_dist(1,43),stat_xy_dist_t,stat_xy_t,           ! <e22>zt
     &        alpha,beta,nsend)

         call deriv2(stat_d,duidxj,ntot,'e33')                            !  e33: (dw/dx)^2+(dw/dy)^2+(dw/dz)^2
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e33>z
         call avgt(stat_xy_dist(1,44),stat_xy_dist_t,stat_xy_t,           ! <e33>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv2(stat_d,duidxj,ntot,'e12')                            !  e12: du/dx*dv/dx+du/dy*dv/dy+du/dz*dv/dz
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e12>z
         call avgt(stat_xy_dist(1,45),stat_xy_dist_t,stat_xy_t,           ! <e12>zt
     &        alpha,beta,nsend)

         call deriv2(stat_d,duidxj,ntot,'e13')                            !  e13: du/dx*dw/dx+du/dy*dw/dy+du/dz*dw/dz
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e13>z
         call avgt(stat_xy_dist(1,46),stat_xy_dist_t,stat_xy_t,           ! <e13>zt
     &        alpha,beta,nsend)

         call deriv2(stat_d,duidxj,ntot,'e23')                            !  e23: dv/dx*dw/dx+dv/dy*dw/dy+dv/dz*dw/dz>
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <e23>z
         call avgt(stat_xy_dist(1,47),stat_xy_dist_t,stat_xy_t,           ! <e23>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'aaa')                            !  dw/dx*dw/dx
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dw/dx*dw/dx>z
         call avgt(stat_xy_dist(1,48),stat_xy_dist_t,stat_xy_t,           ! <dw/dx*dw/dx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'bbb')                            !  dw/dy*dw/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dw/dy*dw/dy>z
         call avgt(stat_xy_dist(1,49),stat_xy_dist_t,stat_xy_t,           ! <dw/dy*dw/dy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'ccc')                            !  dw/dx*dw/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dw/dx*dw/dy>z
         call avgt(stat_xy_dist(1,50),stat_xy_dist_t,stat_xy_t,           ! <dw/dx*dw/dy>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'ddd')                            !  du/dx*du/dx
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dx*du/dx>z
         call avgt(stat_xy_dist(1,51),stat_xy_dist_t,stat_xy_t,           ! <du/dx*du/dx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'eee')                            !  du/dy*du/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dy*du/dy>z
         call avgt(stat_xy_dist(1,52),stat_xy_dist_t,stat_xy_t,           ! <du/dy*du/dy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'fff')                            !  du/dx*du/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dx*du/dy>z
         call avgt(stat_xy_dist(1,53),stat_xy_dist_t,stat_xy_t,           ! <du/dx*du/dy>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'ggg')                            !  dv/dx*dv/dx
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dv/dx*dv/dx>z
         call avgt(stat_xy_dist(1,54),stat_xy_dist_t,stat_xy_t,           ! <dv/dx*dv/dx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'hhh')                            !  dv/dy*dv/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dv/dy*dv/dy>z
         call avgt(stat_xy_dist(1,55),stat_xy_dist_t,stat_xy_t,           ! <dv/dy*dv/dy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'iii')                            !  dv/dx*dv/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <dv/dx*dv/dy>z
         call avgt(stat_xy_dist(1,56),stat_xy_dist_t,stat_xy_t,           ! <dv/dx*dv/dy>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

         call deriv1(stat_d,duidxj,ntot,'jjj')                            !  du/dx*dv/dx
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dx*dv/dx>z
         call avgt(stat_xy_dist(1,57),stat_xy_dist_t,stat_xy_t,           ! <du/dx*dv/dx>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'kkk')                            !  du/dy*dv/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dy*dv/dy>z
         call avgt(stat_xy_dist(1,58),stat_xy_dist_t,stat_xy_t,           ! <du/dy*dv/dy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'lll')                            !  du/dx*dv/dy
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dx*dv/dy>z
         call avgt(stat_xy_dist(1,59),stat_xy_dist_t,stat_xy_t,           ! <du/dx*dv/dy>zt
     &        alpha,beta,nsend)

         call deriv1(stat_d,duidxj,ntot,'mmm')                            !  du/dy*dv/dx
         call z_average(stat_xy_t,stat_d,w1,w2)                           ! <du/dy*dv/dx>z
         call avgt(stat_xy_dist(1,60),stat_xy_dist_t,stat_xy_t,           ! <du/dy*dv/dx>zt
     &        alpha,beta,nsend)

c-----------------------------------------------------------------------

!     ADAM; just for test
      endif

      endif
      endif

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C 
C ------------------------------------------------------------------------------- C
C ------- average the statistical quantities in the homogeneous directions ------ C
C   planar_average_z; average in the homogeneous streamwise (axial) z-direction   C
C ------------------------------------------------------------------------------- C
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C 

!     When a stat file is saved we restart the averaging time
      if (mod(istep,iastep).eq.0.and.istep.gt.1) then

         atime = 0.

      endif

!     Assign starting time of new nv interal
!     It does not depend  on whether we stored a stat file or not
      timel = time

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C 
C ------------------------------------------------------------------------------- C
C ------------ Write statistics to file ----------------------------------------- C
C ------------------------------------------------------------------------------- C
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C 

!     Initialize counter of stat files
      if(nid.eq.0.and.istep.eq.1)  indts = 0

!     Write stat files every iastep time-steps
      if(istep.gt.0 .and. 
     &   mod(istep,iastep).eq.0) then


!     ADAM; for test
         ltm2 = dnekclock()

         ! Generate name of stat file
         if (nid.eq.0) then
            indts = indts + 1 ! Update stat file counter
            write(pippo,'(i4.4)') indts ! Write counter
            inputname1 = 'stat'//trim(pippo) ! First stat file has format stat0001

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C
C ------------------------------------------------------------------------------- C
C ----- Inputname1 -------------------------------------------------------------- C
C ------------------------------------------------------------------------------- C         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C

            ! Create stat file with name inputname1     
            open(unit=33,form='unformatted',file=inputname1)
            
            ! Number of grid-points to be written: 2D slices
            m=nx1*ny1*lelx*lely

            ! Values defining header of stat file
            write(val1,'(1p15e17.9)') 1/param(2) ! Reynolds number        
            write(val2,'(1p15e17.9)') domain_x,domain_y,domain_z ! domain size
            write(val3,'(9i9)') lelx,lely,lelz ! number of elements 
            write(val4,'(9i9)') nx1-1,ny1-1,nz1-1 ! polynomial order
            write(val5,'(9i9)')       nstat ! number of saved statistics 
            write(val6,'(1p15e17.9)') times ! start time
            write(val7,'(1p15e17.9)') time + nv*DT ! end time
            write(val8,'(1p15e17.9)') nrec*nv*DT ! time interval
            write(val9,'(1p15e17.9)') nrec*DT ! effective average time
            write(val10,'(1p15e17.9)') DT ! time step
            write(val11,'(9i9)')      nrec ! number of time records

            ! Write header
            write(33) '(Re ='//trim(val1)
     &           //') (Lx, Ly, Lz ='//trim(val2)
     &           //') (nelx, nely, nelz ='//trim(val3)
     &           //') (Polynomial order ='//trim(val4)
     &           //') (Nstat ='//trim(val5)
     &           //') (start time ='//trim(val6)
     &           //') (end time ='//trim(val7)
     &           //') (time interval ='//trim(val8)
     &           //') (effective average time ='//trim(val9)
     &           //') (time step ='//trim(val10)
     &           //') (nrec ='//trim(val11)
     &           //')'

            ! Write values in the header
            write(33) 1/param(2),
     &           domain_x, domain_y, domain_z,
     &           lelx    , lely    , lelz,
     &           nx1-1   , ny1-1   , nz1-1,
     &           nstat,
     &           times,
     &           time + nv*DT,
     &           nrec*nv*DT,
     &           nrec*DT,
     &           DT,
     &           nrec
         end if

         ! Write statistics for each of the nstsat fields
         do i=1,nstat
            ! Gather distributed 2D fields in stat_xy_dist
            ! into the undistributed field stat_xy_t for writing
            call mpi_gather(stat_xy_dist(1,i),nsend,nekreal,
     &           stat_xy_t,nsend,nekreal,0,nekcomm,ierror)

            ! Write content of undistributed 2D field stat_xy_t
            if (nid.eq.0) then
               write(33) (stat_xy_t(j),j=1,m)
            end if
         end do
         
         ! Close file
         if (nid.eq.0) then
            
             close(33)
             
             nrec = 0 ! Restart number of records to 0
             times = time + nv*DT ! Restart starting time for next file
          end if

!     ADAM; for test
          ltim_max = ltim_max + dnekclock() - ltm2

       endif


!     ADAM; for test
       ltim_min = ltim_min + dnekclock() - ltm1
       if (NID.eq.0) then
          write(*,*) 'Simple timing ',ltim_init, ltim_min, ltim_max
          write(*,*) 'Simple counting ', indts, nrec
       endif

       return
       end

c-----------------------------------------------------------------------
c-----------------------------------------------------------------------

      function my_surf_mean(u,ifld,bc_in,ierr)

      include 'SIZE'
      include 'TOTAL'

      real u(1)

      integer e,f,fmid(6)
      character*3 bc_in
      real my_surf_mean,xmin,xmax,xf,yf,zf

!     Face midpoints on each direction
      im = (1+nx1)/2 
      jm = (1+ny1)/2
      km = (1+nz1)/2

!     Grid-point number within element corresponding to each face     
      fmid(4) =   1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x = -1
      fmid(2) = nx1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x =  1
      fmid(1) =  im + nx1*(  1-1) + nx1*ny1*( km-1) !  y = -1
      fmid(3) =  im + nx1*(ny1-1) + nx1*ny1*( km-1) !  y =  1
      fmid(5) =  im + nx1*( jm-1) + nx1*ny1*(  1-1) !  z = -1
      fmid(6) =  im + nx1*( jm-1) + nx1*ny1*(nz1-1) !  z =  1
      
!     Minimum and maximum x to obtain reference pressure
      xmin = 0.3
      xmax = 0.5

!     Initialize sum of pressure*area and area
      usum = 0
      asum = 0

!     Go through all faces on all elements
      nface = 2*ndim
      do e=1,nelv
      do f=1,nface

!     Obtain coordinates of the face center
         xf    = xm1(fmid(f),1,1,e)
         yf    = ym1(fmid(f),1,1,e)
         zf    = zm1(fmid(f),1,1,e)

!     First condition is that BC on that face is Wall. 
!     Then x coordinate of face center must be between xmin and xmax
         if (cbc(f,e,ifld).eq.bc_in
     &        .and. xf.gt.xmin .and. xf.le.xmax) then

            ! Compute the weighted sum of pressure over face f of element e
            call fcsum2(usum_f,asum_f,u,e,f) 

            ! Add up to obtain total usum (pressure*area) and total asum (area)
            usum = usum + usum_f
            asum = asum + asum_f

         endif
      enddo
      enddo

!     Sum across processors
      usum = glsum(usum,1)  
      asum = glsum(asum,1)
      ierr      = 1

!     Return variable is pressure*area divided by area
      if (asum.gt.0) my_surf_mean = usum/asum
      if (asum.gt.0) ierr      = 0

      return
      end

c-----------------------------------------------------------------------

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
         duidxj(k,e,1) = jacmi(k,e)*(ur(k)*rxm1(k,1,1,e)+
     $        us(k)*sxm1(k,1,1,e)+
     $        ut(k)*txm1(k,1,1,e))
         duidxj(k,e,2) = jacmi(k,e)*(vr(k)*rym1(k,1,1,e)+
     $        vs(k)*sym1(k,1,1,e)+
     $        vt(k)*tym1(k,1,1,e))
         duidxj(k,e,3) = jacmi(k,e)*(wr(k)*rzm1(k,1,1,e)+
     $        ws(k)*szm1(k,1,1,e)+
     $        wt(k)*tzm1(k,1,1,e))
         duidxj(k,e,4) = jacmi(k,e)*(ur(k)*rym1(k,1,1,e)+
     $        us(k)*sym1(k,1,1,e)+
     $        ut(k)*tym1(k,1,1,e))
         duidxj(k,e,5) = jacmi(k,e)*(vr(k)*rzm1(k,1,1,e)+
     $        vs(k)*szm1(k,1,1,e)+
     $        vt(k)*tzm1(k,1,1,e))
         duidxj(k,e,6) = jacmi(k,e)*(wr(k)*rxm1(k,1,1,e)+
     $        ws(k)*sxm1(k,1,1,e)+
     $        wt(k)*txm1(k,1,1,e))
         duidxj(k,e,7) = jacmi(k,e)*(ur(k)*rzm1(k,1,1,e)+
     $        us(k)*szm1(k,1,1,e)+
     $        ut(k)*tzm1(k,1,1,e))
         duidxj(k,e,8) = jacmi(k,e)*(vr(k)*rxm1(k,1,1,e)+
     $        vs(k)*sxm1(k,1,1,e)+
     $        vt(k)*txm1(k,1,1,e))
         duidxj(k,e,9) = jacmi(k,e)*(wr(k)*rym1(k,1,1,e)+
     $        ws(k)*sym1(k,1,1,e)+
     $        wt(k)*tym1(k,1,1,e))
      enddo
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine avgt(avg,f,full,alpha,beta,n)
      include 'SIZE'
      include 'mpif.h'

      integer n,k
      real avg(n),f(n)
      real full(lx1*ly1*lelx*lely)
      real alpha, beta

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal

!     Scatter undistributed 2D array to distributed one
      call mpi_scatter(full,n,nekreal,
     &     f,n,nekreal,0,nekcomm,ierror)

!     Time average is computed on distributed array
      do k=1,n
         avg(k) = alpha*avg(k)+beta*f(k)
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine prod2(avg,f,g,n)
      implicit none
      integer n,k
      real avg(n),f(n),g(n)

      do k=1,n
         avg(k) = f(k)*g(k)
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine prod3(avg,f,g,h,n)
      implicit none
      integer n,k
      real avg(n),f(n),g(n),h(n)

      do k=1,n
         avg(k) = f(k)*g(k)*h(k)
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine prod4(avg,f,g,h,s,n)
      implicit none
      integer n,k
      real avg(n),f(n),g(n),h(n),s(n)

      do k=1,n
         avg(k) = f(k)*g(k)*h(k)*s(k)
      enddo

      return
      end

c-----------------------------------------------------------------------

      subroutine deriv1(avg,duidxj,n,name)
      include 'SIZE'

      integer n,k
      real duidxj(lx1*ly1*lz1,lelt,3*ldim)
      real avg(n)
      character*3 name

      if (name .eq. 'pux') then
         do k=1,n
            avg(k) = duidxj(k,1,1)
         enddo
      elseif (name .eq. 'puy') then
         do k=1,n
            avg(k) = duidxj(k,1,4)
         enddo
      elseif (name .eq. 'puz') then
         do k=1,n
            avg(k) = duidxj(k,1,7)
         enddo
      elseif (name .eq. 'pvx') then
         do k=1,n
            avg(k) = duidxj(k,1,8)
         enddo
      elseif (name .eq. 'pvy') then
         do k=1,n
            avg(k) = duidxj(k,1,2)
         enddo
      elseif (name .eq. 'pvz') then
         do k=1,n
            avg(k) = duidxj(k,1,5)
         enddo
      elseif (name .eq. 'pwx') then
         do k=1,n
            avg(k) = duidxj(k,1,6)
         enddo
      elseif (name .eq. 'pwy') then
         do k=1,n
            avg(k) = duidxj(k,1,9)
         enddo
      elseif (name .eq. 'pwz') then
         do k=1,n
            avg(k) = duidxj(k,1,3)
         enddo
      elseif (name .eq. 'aaa') then
         do k=1,n
            avg(k) = duidxj(k,1,6)*duidxj(k,1,6)
         enddo
      elseif (name .eq. 'bbb') then
         do k=1,n
            avg(k) = duidxj(k,1,9)*duidxj(k,1,9)
         enddo
      elseif (name .eq. 'ccc') then
         do k=1,n
            avg(k) = duidxj(k,1,6)*duidxj(k,1,9)
         enddo
      elseif (name .eq. 'ddd') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,1)
         enddo
      elseif (name .eq. 'eee') then
         do k=1,n
            avg(k) = duidxj(k,1,4)*duidxj(k,1,4)
         enddo
      elseif (name .eq. 'fff') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,4)
         enddo
      elseif (name .eq. 'ggg') then
         do k=1,n
            avg(k) = duidxj(k,1,8)*duidxj(k,1,8)
         enddo
      elseif (name .eq. 'hhh') then
         do k=1,n
            avg(k) = duidxj(k,1,2)*duidxj(k,1,2)
         enddo
      elseif (name .eq. 'iii') then
         do k=1,n
            avg(k) = duidxj(k,1,8)*duidxj(k,1,2)
         enddo
      elseif (name .eq. 'jjj') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,8)
         enddo
      elseif (name .eq. 'kkk') then
         do k=1,n
            avg(k) = duidxj(k,1,4)*duidxj(k,1,2)
         enddo
      elseif (name .eq. 'lll') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,2)
         enddo
      elseif (name .eq. 'mmm') then
         do k=1,n
            avg(k) = duidxj(k,1,4)*duidxj(k,1,8)
         enddo
      endif

      return
      end

c-----------------------------------------------------------------------

      subroutine deriv2(avg,duidxj,n,name)
      include 'SIZE'

      integer n,k
      real duidxj(lx1*ly1*lz1,lelt,3*ldim)
      real avg(n)
      character*3 name

      if (name .eq. 'e11') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,1)+
     $           duidxj(k,1,4)*duidxj(k,1,4)+duidxj(k,1,7)*duidxj(k,1,7)
         enddo
      elseif (name .eq. 'e22') then
         do k=1,n
            avg(k) = duidxj(k,1,8)*duidxj(k,1,8)+
     $           duidxj(k,1,2)*duidxj(k,1,2)+duidxj(k,1,5)*duidxj(k,1,5)
         enddo
      elseif (name .eq. 'e33') then
         do k=1,n
            avg(k) = duidxj(k,1,6)*duidxj(k,1,6)+
     $           duidxj(k,1,9)*duidxj(k,1,9)+duidxj(k,1,3)*duidxj(k,1,3)
         enddo
      elseif (name .eq. 'e12') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,8)+
     $           duidxj(k,1,4)*duidxj(k,1,2)+duidxj(k,1,7)*duidxj(k,1,5)
         enddo
      elseif (name .eq. 'e13') then
         do k=1,n
            avg(k) = duidxj(k,1,1)*duidxj(k,1,6)+
     $           duidxj(k,1,4)*duidxj(k,1,9)+duidxj(k,1,7)*duidxj(k,1,3)
         enddo
      elseif (name .eq. 'e23') then
         do k=1,n
            avg(k) = duidxj(k,1,8)*duidxj(k,1,6)+
     $           duidxj(k,1,2)*duidxj(k,1,9)+duidxj(k,1,5)*duidxj(k,1,3)
         enddo
      endif

      return
      end

c-----------------------------------------------------------------------
