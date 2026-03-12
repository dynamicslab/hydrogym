c-----------------------------------------------------------------------
c
c    Stability limits:
c
c    AB3:    .7236                     w/safety (1.2):   .603
c
c    RK3:    1.73   (sqrt 3)           w/safety (1.2):   1.44
c
c    RK4:    2.828                     w/safety (1.2):   2.36
c
c    SEM Safety factor:  1.52 for N=3
c                     <  1.20 for N=16
c                     ~  1.16 for N=256
c
c-----------------------------------------------------------------------
      subroutine setup_convect(igeom)
      include 'SIZE'
      include 'TOTAL'
      logical ifnew

      common /cchar/ ct_vx(0:lorder+1) ! time for each slice in c_vx()

      common /scruz/ cx  (lx1*ly1*lz1*lelt)
     $ ,             cy  (lx1*ly1*lz1*lelt)
     $ ,             cz  (lx1*ly1*lz1*lelt)
     $ ,             hmsk(lx1*ly1*lz1*lelt)


      if (igeom.eq.1) return
      if (param(99).lt.0) return ! no dealiasing

      if (ifchar) then

         nelc = nelv
         if (ifmhd) nelc = max(nelv,nelfld(ifldmhd))
         if (ifmhd) call exitti('no characteristics for mhd yet$',istep)

         ifnew = .true.
         if (igeom.gt.2) ifnew = .false.

         if (ifgeom) then ! Moving mesh
            call opsub3(cx,cy,cz,vx,vy,vz,wx,wy,wz)
            call set_conv_char(ct_vx,c_vx,cx,cy,cz,nelc,time,ifnew)
c            call set_char_mask(hmsk,cx,cy,cz) ! mask for hyperbolic system 
         else
            call set_conv_char(ct_vx,c_vx,vx,vy,vz,nelc,time,ifnew)
c            call set_char_mask(hmsk,vx,vy,vz) ! mask for hyperbolic system 
         endif

         n=lx1*ly1*lz1*nelv
         call rone(hmsk,n)            ! TURN OFF MASK
         call set_binv (bmnv, hmsk,n) ! Store binvm1*(hyperbolic mask)
         if(ifmvbd) then
            call set_bdivw(bdivw,hmsk,n) ! Store Bdivw *(hyperbolic mask)
         else
            call rzero(bdivw,n*lorder)
         endif
         call set_bmass(bmass,hmsk,n) ! Store binvm1*(hyperbolic mask)

      else

         if (.not.ifpert) then
            if (ifcons) then
               call set_convect_cons (vxd,vyd,vzd,vx,vy,vz)
            else
               call set_convect_new  (vxd,vyd,vzd,vx,vy,vz)
            endif
         endif

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine set_char_mask(mask,u,v,w) ! mask for hyperbolic system 

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'TSTEP'
      integer msk(0:1)
      character      cb*3
      parameter (lxyz1=lx1*ly1*lz1)
      common /ctmp1/ work(lxyz1,lelt)

      real mask(lxyz1,1),u(lxyz1,1),v(lxyz1,1),w(lxyz1,1)

      integer e,f

      nfaces= 2*ldim
      ntot1 = lx1*ly1*lz1*nelv
      call rzero (work,ntot1)
      call rone  (mask,NTOT1)

      ifldv = 1
      do 100 e=1,nelv
      do 100 f=1,nfaces
         cb=cbc(f,e,ifldv)
         if (cb(1:1).eq.'v' .or. cb(1:1).eq.'V' .or.
     $       cb.eq.'mv ' .or. cb.eq.'MV ') then

           call faccl3 (work(1,e),u(1,e),unx(1,1,f,e),f)
           call faddcl3(work(1,e),v(1,e),uny(1,1,f,e),f)
           if (if3d) 
     $     call faddcl3(work(1,e),w(1,e),unz(1,1,f,e),f)

           call fcaver (vaver,work,e,f)

           if (vaver.lt.0) call facev (mask,e,f,0.0,lx1,ly1,lz1)
         endif
         if (cb(1:2).eq.'ws' .or. cb(1:2).eq.'WS') 
     $   call facev (mask,e,f,0.0,lx1,ly1,lz1)
 100  continue
      call dsop(mask,'MUL',lx1,ly1,lz1)

      return
      end
c-----------------------------------------------------------------------
      subroutine set_dealias_rx
C
C     Eulerian scheme, add convection term to forcing function
C     at current time step.
C
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'TSTEP' ! for istep

      common /dealias1/ zd(lxd),wd(lxd)
      integer e

      integer ilstep
      save    ilstep
      data    ilstep /-1/

      if (.not.ifgeom.and.ilstep.gt.1) return  ! already computed
      if (ifgeom.and.ilstep.eq.istep)  return  ! already computed
      ilstep = istep

      nxyz1 = lx1*ly1*lz1
      nxyzd = lxd*lyd*lzd

      call zwgl (zd,wd,lxd)  ! zwgl -- NOT zwgll!

      if (if3d) then
c
         do e=1,nelv

c           Interpolate z+ and z- into fine mesh, translate to r-s-t coords

            call intp_rstd(rx(1,1,e),rxm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,2,e),rym1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,3,e),rzm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,4,e),sxm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,5,e),sym1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,6,e),szm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,7,e),txm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,8,e),tym1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,9,e),tzm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd

            l = 0
            do k=1,lzd
            do j=1,lyd
            do i=1,lxd
               l = l+1
               w = wd(i)*wd(j)*wd(k)
               do ii=1,9
                  rx(l,ii,e) = w*rx(l,ii,e)
               enddo
            enddo
            enddo
            enddo
         enddo

      else ! 2D
c
         do e=1,nelv

c           Interpolate z+ and z- into fine mesh, translate to r-s-t coords

            call intp_rstd(rx(1,1,e),rxm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,2,e),rym1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,3,e),sxm1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd
            call intp_rstd(rx(1,4,e),sym1(1,1,1,e),lx1,lxd,if3d,0) ! 0 --> fwd

            l = 0
            do j=1,lyd
            do i=1,lxd
               l = l+1
               w = wd(i)*wd(j)
               do ii=1,4
                  rx(l,ii,e) = w*rx(l,ii,e)
               enddo
            enddo
            enddo
         enddo

      endif

      return
      end
c-----------------------------------------------------------------------

