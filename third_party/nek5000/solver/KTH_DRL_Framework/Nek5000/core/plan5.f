c-----------------------------------------------------------------------
      subroutine plan5(igeom)

c     Two-step Richardson Extrapolation.
c     Operator splitting technique.

      include 'SIZE'
      include 'TOTAL'

      common /scrns/  resv  (lx1*ly1*lz1*lelv,3)

      n   = lx1*ly1*lz1*nelv
      n2  = lx2*ly2*lz2*nelv
      dt2 = dt/2
      dti = 1/dt

      if (igeom.eq.2) then

      if (ifmvbd) call opcopy
     $  (wxlag(1,1,1,1,2),wylag(1,1,1,1,2),wzlag(1,1,1,1,2),xm1,ym1,zm1)

      do i=1,n
         s = bm1(i,1,1,1)*vtrans(i,1,1,1,1)*dti  ! Add  density*mass/dt,
         vxlag(i,1,1,1,2)=s*vx(i,1,1,1)          ! equivalent to using
         vylag(i,1,1,1,2)=s*vy(i,1,1,1)          ! density*mass/(dt/2)
         vzlag(i,1,1,1,2)=s*vz(i,1,1,1)          ! in the first place...
      enddo

      call midstep(vxlag,vylag,vzlag,prlag,0,dt)  ! One step of Pn-Pn-2

      do i=1,n                                          ! Add  density*mass/dt,
         bfx(i,1,1,1)=bfx(i,1,1,1)+vxlag(i,1,1,1,2)     ! equivalent to using
         bfy(i,1,1,1)=bfy(i,1,1,1)+vylag(i,1,1,1,2)     ! density*mass/(dt/2)
         bfz(i,1,1,1)=bfz(i,1,1,1)+vzlag(i,1,1,1,2)     ! in the first place...
      enddo

      if (ifmvbd) then
        call opcopy
     $  (xm1,ym1,zm1,wxlag(1,1,1,1,2),wylag(1,1,1,1,2),wzlag(1,1,1,1,2))
        call geom_reset(0)
      endif

      time = time-dt2
      call midstep(vx,vy,vz,pr,1,dt2)      ! One step of Pn-Pn-2, dt/2

      time = time+dt2
      call setup_convect(2)  ! Map vx --> vxd
      call setprop

      call midstep(vx,vy,vz,pr,0,dt2)      ! One step of Pn-Pn-2, dt/2

      do i=1,n
         vx(i,1,1,1)=2*vx(i,1,1,1)-vxlag(i,1,1,1,1)
         vy(i,1,1,1)=2*vy(i,1,1,1)-vylag(i,1,1,1,1)
         vz(i,1,1,1)=2*vz(i,1,1,1)-vzlag(i,1,1,1,1)
      enddo

      do i=1,n2
         pr(i,1,1,1)=2*pr(i,1,1,1)-prlag(i,1,1,1,1)
      enddo

      call ortho(pr)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine midstep(ux,uy,uz,pu,iresv,dtl)
      include 'SIZE'
      include 'TOTAL'

      parameter (lv=lx1*ly1*lz1*lelt)
      real ux(1),uy(1),uz(1),pu(1)

      common /p5var/ rhs2   (lx1*ly1*lz1*lelv,3)

      common /scrns/  resv  (lx1*ly1*lz1*lelv,3)
     $ ,              dv1   (lx1*ly1*lz1*lelv)
     $ ,              dv2   (lx1*ly1*lz1*lelv)
     $ ,              dv3   (lx1*ly1*lz1*lelv)
      common /scrvh/  h1    (lx1*ly1*lz1*lelv)
     $ ,              h2    (lx1*ly1*lz1*lelv)


      if (lx1.eq.lx2) 
     $   call exitti('midstep requires lx2=lx1-2 in SIZE$',lx2)

      ifield = 1                ! Set field for velocity
      n   = lx1*ly1*lz1*nelv
      n2  = lx2*ly2*lz2*nelv

      dti = 1/dtl
      call copy    (h1,vdiff ,n)
      call cmult2  (h2,vtrans,dti,n)

      if (iresv.eq.0) then ! bfx etc is preserved if iresv=1

                    call makeuf  ! Initialize bfx, bfy, bfz
        if (ifmvbd) call admeshv ! Add div(W.u_i)

        call convop(resv(1,1),vx)
        call convop(resv(1,2),vy)
        call convop(resv(1,3),vz)

        do i=1,n
           b=vtrans(i,1,1,1,1)*bm1(i,1,1,1)
           s=1./dtl
           bfx(i,1,1,1)=bfx(i,1,1,1)+b*(s*vx(i,1,1,1)-resv(i,1))
           bfy(i,1,1,1)=bfy(i,1,1,1)+b*(s*vy(i,1,1,1)-resv(i,2))
           bfz(i,1,1,1)=bfz(i,1,1,1)+b*(s*vz(i,1,1,1)-resv(i,3))
        enddo

      endif

      call adv_geom(dtl) ! Advance the geometry

      call opcopy  (ux,uy,uz,vx,vy,vz)

      call bcdirvc (ux,uy,uz,v1mask,v2mask,v3mask) ! Don't forget bcneutr !
      call ophx    (resv(1,1),resv(1,2),resv(1,3),ux,uy,uz,h1,h2)

      call copy(rhs2,resv,lx1*ly1*lz1*lelv*3)

      do i=1,n
         resv(i,1)=bfx(i,1,1,1)-resv(i,1) ! rhs = rhs - H*u
         resv(i,2)=bfy(i,1,1,1)-resv(i,2)
         resv(i,3)=bfz(i,1,1,1)-resv(i,3)
      enddo

      tolhv = abs(param(22))
      call ophinv(dv1,dv2,dv3
     $   ,resv(1,1),resv(1,2),resv(1,3),h1,h2,tolhv,nmxv)

      call opadd2(ux,uy,uz,dv1,dv2,dv3)

      bd(1) = 1.0
      call rzero(pu,n2)

      dt_old = dt
      dt = dtl
      call incomprn(ux,uy,uz,pu)
      dt = dt_old

      return
      end
c-----------------------------------------------------------------------
      subroutine adv_geom(dtl) ! Advance the geometry
      include 'SIZE'
      include 'TOTAL'

      param(28) = 1   !  This forces Euler Forward for Mesh Update
                      !  Note: p28 must be set prior to call settime

      dt_tmp = dt     ! Save "full" dt value
      dt     = dtl

      ifield = 1
      call gengeom(2)
      ifield = 1

      dt     = dt_tmp ! Restore dt

      return
      end
c-----------------------------------------------------------------------
