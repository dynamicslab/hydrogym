      subroutine particle_paths
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cpathr/ time0,time1,time2
      common /cpatha/ v0x(maxpts),v1x(maxpts),v2x(maxpts)
     $              , v0y(maxpts),v1y(maxpts),v2y(maxpts)
     $              , v0z(maxpts),v1z(maxpts),v2z(maxpts)
c
      parameter (maxpart=500)
      common /cparta/ xpart  (3,maxpart)
      common /cparti/ iclr_part(maxpart)
c
      real xnew(3)
c
      ntot  = nx*ny*nz*nel
c
      call get_part(ierr)
      if (ierr.eq.0) return
c
      call prs  ('Input start,stop, skip fld numbers:$')
      call reiii (istart_fld,iend_fld,iskip)
c
c
      iframe = 0
      do ifld=istart_fld,iend_fld,iskip
c
         iframe = iframe+1
c
         call copy(v0x,v1x,ntot)
         call copy(v0y,v1y,ntot)
         call copy(v0z,v1z,ntot)
         time0 = time1
c
         call copy(v1x,v2x,ntot)
         call copy(v1y,v2y,ntot)
         call copy(v1z,v2z,ntot)
         time1 = time2
c
         call copy(v2x,u ,ntot)
         call copy(v2y,v ,ntot)
         call copy(v2z,w ,ntot)
         time2 = time
c
         ndumps = ifld
         call getfld(ndumps,ierr,.true.)
c
         if (iframe.ge.4) then
c
c           Advance particles from time1 to time2
c
            nstep = 20
            dt = (time2-time1)/nstep
            do ipart=1,npart
c
c              color and start particle path
c
               call color    (iclr_part(ipart))
c
c
c              Integrate path from time1 to time2
c
               call out_xpos (xpart(1,ipart),0)
               do istep=1,nstep
                  timep = time1 + dt*(istep-1)
                  call rk4pp     (xnew,xpart(1,ipart),timep,dt,ndim)
                  call out_xpos  (xnew,istep)
                  call copy      (xpart(1,ipart),xnew,ndim)
               enddo
            enddo
         endif
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_xpos(xpos,i)
      real xpos(3)
c
      xxis=xisom(xpos(1),xpos(2),xpos(3))
      yyis=yisom(xpos(1),xpos(2),xpos(3))
      if (i.eq.0) then
         call movec(xxis,yyis)
      else
         call drawc(xxis,yyis)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rk4pp(wnew,w,t,dt,n)
      real wnew(1),w(1),t,dt
      real wh(4),f1(4),f2(4),f3(4),f4(4)
c
c     RK4:
c
      dt2 = dt/2.
      dt3 = dt/3.
      dt6 = dt/6.
c
      t2 = t+dt2
      tt = t+dt
c
      call compute_fp(f1,w ,t )
      call add3s2    (wh,w,f1,dt2,n)
c
      call compute_fp(f2,wh,t2)
      call add3s2    (wh,w,f2,dt2,n)
c
      call compute_fp(f3,wh,t2)
      call add3s2    (wh,w,f3,dt ,n)
c
      call compute_fp(f4,wh,tt)
c
      call copy      (wnew,w,n)
      call add2s2    (wnew,f1,dt6,n)
      call add2s2    (wnew,f2,dt3,n)
      call add2s2    (wnew,f3,dt3,n)
      call add2s2    (wnew,f4,dt6,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_fp(vel,xpos,t_str)
c
c     Compute RHS of ODE for particle path
c
c        INPUT:   xpos  -- particle position
c                 t_str -- time of interest
c
c        OUTPUT:  vel   -- velocity at point
c
c
c     The velocity field is known at time0,time1,time2, and time, with
c
c         time0  <  time1  =<  t_str  =<  time2  <  time
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cpathr/ time0,time1,time2
      common /cpatha/ v0x(maxpts),v1x(maxpts),v2x(maxpts)
     $              , v0y(maxpts),v1y(maxpts),v2y(maxpts)
     $              , v0z(maxpts),v1z(maxpts),v2z(maxpts)
c
      real vel(3),xpos(3),r(3)
      real vtmp(3,0:3),coef(0:3)
c
c
      call get_coef(coef,t_str,time0,time1,time2,time)
c
      call findre(ie,r,xpos,rminmax,ierr)
      nxyz  = nx*ny*nz
      ieoff = nxyz*(ie-1) + 1
c
      if (if3d) then
c
         call evalsc( vtmp(1,0) , v0x(ieoff) , r , 1 )
         call evalsc( vtmp(2,0) , v0y(ieoff) , r , 1 )
         call evalsc( vtmp(3,0) , v0z(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,1) , v1x(ieoff) , r , 1 )
         call evalsc( vtmp(2,1) , v1y(ieoff) , r , 1 )
         call evalsc( vtmp(3,1) , v1z(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,2) , v2x(ieoff) , r , 1 )
         call evalsc( vtmp(2,2) , v2y(ieoff) , r , 1 )
         call evalsc( vtmp(3,2) , v2z(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,3) , u  (ieoff) , r , 1 )
         call evalsc( vtmp(2,3) , v  (ieoff) , r , 1 )
         call evalsc( vtmp(3,3) , w  (ieoff) , r , 1 )
         do j=1,3
            vel(j) = 0.
            do i=0,3
               vel(j) = vel(j) + coef(i)*vtmp(j,i)
            enddo
         enddo
c
      else
c
         call evalsc( vtmp(1,0) , v0x(ieoff) , r , 1 )
         call evalsc( vtmp(2,0) , v0y(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,1) , v1x(ieoff) , r , 1 )
         call evalsc( vtmp(2,1) , v1y(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,2) , v2x(ieoff) , r , 1 )
         call evalsc( vtmp(2,2) , v2y(ieoff) , r , 1 )
c
         call evalsc( vtmp(1,3) , u  (ieoff) , r , 1 )
         call evalsc( vtmp(2,3) , v  (ieoff) , r , 1 )
c
         do j=1,2
            vel(j) = 0.
            do i=0,3
               vel(j) = vel(j) + coef(i)*vtmp(j,i)
            enddo
         enddo
c
c
      endif
c
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_coef(coef,t,t0,t1,t2,t3)
c
c     Compute coefficients for lagrangian interpolat at time t
c     with knots at t0--t3.
c
      real coef(0:3),tk(0:3)
c
      tk(0) = t0
      tk(1) = t1
      tk(2) = t2
      tk(3) = t3
c
      do j=0,3
         coef(j) = 1.
         do i=0,3
            if (i.ne.j) coef(j) = coef(j)*(t-tk(i))/(tk(j)-tk(i))
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine get_part(isel)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (maxpart=500)
      common /cparta/ xpart  (3,maxpart)
      common /cparti/ iclr_part(maxpart)
c
c     if (if3d) then
c        call prs('Input format: (0) cancel (1) Keyboard (2) File?$')
c     else
c        call prs
c    $   ('Input format: (0) cancel (1) Keyboard (2) File (3) Mouse?$')
c     endif
c     call rei(isel)
c
      isel = 2
      if (isel.eq.2) then
         open (unit=68,file='part.path',err=991)
         npart = 0
         do i=1,maxpart
c           Note:  Colors should be between 2 and 15, (0=white,1=black)
            read(68,*,end=100) (xpart(j,i),j=1,ndim),iclr_part(i)
            iclr_part(i) = min(15,iclr_part(i))
            iclr_part(i) = max( 0,iclr_part(i))
            npart = npart+1
         enddo
  100    return
      endif
c
  991 continue
      call prs('Could not open file, "part.path"')
      isel=0
      return
      end
c-----------------------------------------------------------------------
