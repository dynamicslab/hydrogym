c-----------------------------------------------------------------------
c
c     To do:
c
c        Differing BC's imposed for ophinv, incomprn, etc.
c
c        1-shot Fast solver for Helmholtz and pressure
c
c
c-----------------------------------------------------------------------
      subroutine induct (igeom)
c
c     Solve the convection-diffusion equation for the B-field, with
c     projection onto a div-free space.
c
c
      include 'SIZE'
      include 'INPUT'
      include 'EIGEN'
      include 'SOLN'
      include 'TSTEP'
      include 'MASS'

      common /scrns/  resv1 (lx1,ly1,lz1,lelv)
     $ ,              resv2 (lx1,ly1,lz1,lelv)
     $ ,              resv3 (lx1,ly1,lz1,lelv)
     $ ,              dv1   (lx1,ly1,lz1,lelv)
     $ ,              dv2   (lx1,ly1,lz1,lelv)
     $ ,              dv3   (lx1,ly1,lz1,lelv)
      common /scrvh/  h1    (lx1,ly1,lz1,lelv)
     $ ,              h2    (lx1,ly1,lz1,lelv)

      ifield = ifldmhd

      if (igeom.eq.1) then  ! old geometry, old velocity

         call makebsource_mhd

      else

         call lagbfield
         call lagvel

         call elsasserh(igeom)

         call vol_flow        ! check for fixed flow rate


      endif
c
      return
      end
c--------------------------------------------------------------------
      subroutine lagbfield
c
c     Keep old B-field(s)
c
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
c
      do ilag=nbdinp-1,2,-1
         call opcopy
     $    ( bxlag(1,ilag  ),bylag(1,ilag  ),bzlag(1,ilag  )
     $    , bxlag(1,ilag-1),bylag(1,ilag-1),bzlag(1,ilag-1) )
      enddo
      call opcopy (bxlag,bylag,bzlag,bx,by,bz)
c
      return
      end
c--------------------------------------------------------------------
      subroutine makebsource_mhd
c
c     Make rhs for induction equation
c
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'INPUT'
      include 'TSTEP'
      include 'CTIMER'

      if (icalld.eq.0) tbmhd=0.0
      icalld = icalld+1
      nbmhd  = icalld
      etime1 = dnekclock()
c
      ifield = 1
                                      call makeuf
c
      ifield = ifldmhd
                                      call makeufb
      if (ifaxis) then
c        do ifield = 2,3
         do ifield = 2,npscal+1
                                      call makeuq  ! nonlinear terms
         enddo
         ifield = ifldmhd
      endif

      if (ifnav.and.(.not.ifchar)) then
         call advab_elsasser_fast
      endif
      if (ifchar) then
         write(6,*) 'No IFCHAR for MHD, yet.'
         call exitt
      endif

      ifield = 1
      if (iftran)                     call makeabf
                                      call makebdf
c
      ifield = ifldmhd
      if (iftran)                     call makextb
                                      call makebdfb
c
      tbmhd=tbmhd+(dnekclock()-etime1)
      return
      end
c--------------------------------------------------------------------
      subroutine makeufb
c
c     Compute and add: (1) user specified forcing function (FX,FY,FZ)
c
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      time = time-dt
      call nekuf   (bmx,bmy,bmz)
      call opcolv2 (bmx,bmy,bmz,vtrans(1,1,1,1,ifield),bm1)
      time = time+dt
c
      return
      end
c--------------------------------------------------------------------
      subroutine makextb
c
c     Add extrapolation terms to magnetic source terms
c
c     (nek5 equivalent for velocity is "makeabf")
c
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
C
      common /scrns/ ta1 (lx1,ly1,lz1,lelv)
     $ ,             ta2 (lx1,ly1,lz1,lelv)
     $ ,             ta3 (lx1,ly1,lz1,lelv)
c
      ntot1 = lx1*ly1*lz1*nelv
c
      ab0 = ab(1)
      ab1 = ab(2)
      ab2 = ab(3)
      call add3s2 (ta1,bbx1,bbx2,ab1,ab2,ntot1)
      call add3s2 (ta2,bby1,bby2,ab1,ab2,ntot1)
      call copy   (bbx2,bbx1,ntot1)
      call copy   (bby2,bby1,ntot1)
      call copy   (bbx1,bmx,ntot1)
      call copy   (bby1,bmy,ntot1)
      call add2s1 (bmx,ta1,ab0,ntot1)
      call add2s1 (bmy,ta2,ab0,ntot1)
      if (ldim.eq.3) then
         call add3s2 (ta3,bbz1,bbz2,ab1,ab2,ntot1)
         call copy   (bbz2,bbz1,ntot1)
         call copy   (bbz1,bmz,ntot1)
         call add2s1 (bmz,ta3,ab0,ntot1)
      endif
c
      return
      end
c--------------------------------------------------------------------
      subroutine makebdfb
C
C     Add contributions to magnetic source from lagged BD terms.
C
      include 'SIZE'
      include 'SOLN'
      include 'MASS'
      include 'GEOM'
      include 'INPUT'
      include 'TSTEP'
C
      COMMON /SCRNS/ TA1(LX1,LY1,LZ1,LELV)
     $ ,             TA2(LX1,LY1,LZ1,LELV)
     $ ,             TA3(LX1,LY1,LZ1,LELV)
     $ ,             TB1(LX1,LY1,LZ1,LELV)
     $ ,             TB2(LX1,LY1,LZ1,LELV)
     $ ,             TB3(LX1,LY1,LZ1,LELV)
     $ ,             H2 (LX1,LY1,LZ1,LELV)
C
      NTOT1 = lx1*ly1*lz1*NELV
      CONST = 1./DT
      CALL CMULT2(H2,vtrans(1,1,1,1,ifield),CONST,NTOT1)
      CALL OPCOLV3c (TB1,TB2,TB3,BX,BY,BZ,BM1,bd(2))
C
      DO 100 ILAG=2,NBD
         IF (IFGEOM) THEN
            CALL OPCOLV3c(TA1,TA2,TA3,BXLAG (1,ILAG-1),
     $                                BYLAG (1,ILAG-1),
     $                                BZLAG (1,ILAG-1),
     $                                BM1LAG(1,1,1,1,ILAG-1),bd(ilag+1))
         ELSE
            CALL OPCOLV3c(TA1,TA2,TA3,BXLAG (1,ILAG-1),
     $                                BYLAG (1,ILAG-1),
     $                                BZLAG (1,ILAG-1),
     $                                BM1                   ,bd(ilag+1))
         ENDIF
         CALL OPADD2  (TB1,TB2,TB3,TA1,TA2,TA3)
 100  CONTINUE
      CALL OPADD2col (BMX,BMY,BMZ,TB1,TB2,TB3,h2)
c
      return
      end
c--------------------------------------------------------------------
      subroutine cresvib(resv1,resv2,resv3,h1,h2)
c
c     Account for inhomogeneous Dirichlet boundary contributions 
c     in rhs of induction eqn.
c                                               n
c     Also, subtract off best estimate of grad p
c
      include 'SIZE'
      include 'TOTAL'
      real           resv1 (lx1,ly1,lz1,1)
      real           resv2 (lx1,ly1,lz1,1)
      real           resv3 (lx1,ly1,lz1,1)
      real           h1    (lx1,ly1,lz1,1)
      real           h2    (lx1,ly1,lz1,1)
      common /scruz/ w1    (lx1,ly1,lz1,lelv)
     $ ,             w2    (lx1,ly1,lz1,lelv)
     $ ,             w3    (lx1,ly1,lz1,lelv)
c
c
      call bcdirvc (bx,by,bz,b1mask,b2mask,b3mask)
      call extrapp (pm,pmlag)
      call opgradt (resv1,resv2,resv3,pm)
      call opadd2  (resv1,resv2,resv3,bmx,bmy,bmz)
c     call opcopy  (resv1,resv2,resv3,bmx,bmy,bmz)
      call ophx    (w1,w2,w3,bx,by,bz,h1,h2)
      call opsub2  (resv1,resv2,resv3,w1,w2,w3)
c
      return
      end
c--------------------------------------------------------------------
      subroutine cresvibp(resv1,resv2,resv3,h1,h2)
c
c     Account for inhomogeneous Dirichlet boundary contributions 
c     in rhs of momentum eqn.
c                                               n
c     Also, subtract off best estimate of grad p
c
      include 'SIZE'
      include 'TOTAL'
      real           resv1 (lx1,ly1,lz1,1)
      real           resv2 (lx1,ly1,lz1,1)
      real           resv3 (lx1,ly1,lz1,1)
      real           h1    (lx1,ly1,lz1,1)
      real           h2    (lx1,ly1,lz1,1)
      common /scruz/ w1    (lx1,ly1,lz1,lelv)
     $ ,             w2    (lx1,ly1,lz1,lelv)
     $ ,             w3    (lx1,ly1,lz1,lelv)
c
      call bcdirvc (vx,vy,vz,v1mask,v2mask,v3mask)
      if (ifstrs)  call bcneutr
      call extrapp (pr,prlag)
      call opgradt (resv1,resv2,resv3,pr)
      call opadd2  (resv1,resv2,resv3,bfx,bfy,bfz)
c     call opcopy  (resv1,resv2,resv3,bfx,bfy,bfz)
      call ophx    (w1,w2,w3,vx,vy,vz,h1,h2)
      call opsub2  (resv1,resv2,resv3,w1,w2,w3)
c
      return
      end
c--------------------------------------------------------------------
      subroutine incomprn (ux,uy,uz,up)
c
c     Project U onto the closest incompressible field
c
c     Input:  U     := (ux,uy,uz)
c
c     Output: updated values of U, iproj, proj; and
c             up    := pressure currection req'd to impose div U = 0
c
c
c     Dependencies: ifield ==> which "density" (vtrans) is used.
c
c     Notes  1.  up is _not_ scaled by bd(1)/dt.  This should be done
c                external to incompr().
c
c            2.  up accounts _only_ for the perturbation pressure,
c                not the current pressure derived from extrapolation.
c
c
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'
c
      common /scrns/ w1    (lx1,ly1,lz1,lelv)
     $ ,             w2    (lx1,ly1,lz1,lelv)
     $ ,             w3    (lx1,ly1,lz1,lelv)
     $ ,             dv1   (lx1,ly1,lz1,lelv)
     $ ,             dv2   (lx1,ly1,lz1,lelv)
     $ ,             dv3   (lx1,ly1,lz1,lelv)
     $ ,             dp    (lx2,ly2,lz2,lelv)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $ ,             h2    (lx1,ly1,lz1,lelv)
      common /scrhi/ h2inv (lx1,ly1,lz1,lelv)

      parameter(nset = 1 + lbelv/lelv)
      common /orthov/ pset(lx2*ly2*lz2*lelv*mxprev,nset)
      common /orthbi/ nprv(2)
      logical ifprjp

      ifprjp=.false.    ! Project out previous pressure solutions?
      istart=param(95)  
      if (istep.ge.istart.and.istart.ne.0) ifprjp=.true.

      if (icalld.eq.0) tpres=0.0
      icalld = icalld+1
      npres  = icalld
      etime1 = dnekclock()

      ntot1  = lx1*ly1*lz1*nelv
      ntot2  = lx2*ly2*lz2*nelv
      intype = 1

      call rzero   (h1,ntot1)
      call copy    (h2,vtrans(1,1,1,1,ifield),ntot1)
      call invers2 (h2inv,h2,ntot1)

      call opdiv   (dp,ux,uy,uz)

      bdti = -bd(1)/dt
      call cmult   (dp,bdti,ntot2)

      call add2col2(dp,bm2,usrdiv,ntot2) ! User-defined divergence.

      call ortho   (dp)

      i = 1 + ifield/ifldmhd
      if (ifprjp)   call setrhsp  (dp,h1,h2,h2inv,pset(1,i),nprv(i))
                    scaledt = dt/bd(1)
                    scaledi = 1./scaledt
                    call cmult(dp,scaledt,ntot2)        ! scale for tol
                    call esolver  (dp,h1,h2,h2inv,intype)
                    call cmult(dp,scaledi,ntot2)
      if (ifprjp)   call gensolnp (dp,h1,h2,h2inv,pset(1,i),nprv(i))

      call add2(up,dp,ntot2)

      call opgradt  (w1 ,w2 ,w3 ,dp)
      call opbinv   (dv1,dv2,dv3,w1 ,w2 ,w3 ,h2inv)
      dtb  = dt/bd(1)
      call opadd2cm (ux ,uy ,uz ,dv1,dv2,dv3, dtb )

      if (ifmhd)  call chkptol	! to avoid repetition

      tpres=tpres+(dnekclock()-etime1)

      return
      end
c-----------------------------------------------------------------------
      subroutine extrapp(p,plag)
C
C     Pressure extrapolation
C
      INCLUDE 'SIZE'
      INCLUDE 'SOLN'
      INCLUDE 'TSTEP'

      real  p    (lx2,ly2,lz2,1)
     $     ,plag (lx2,ly2,lz2,1)

      common /cgeom/ igeom

      ntot2 = lx2*ly2*lz2*nelv

      if (nbd.eq.2.and.nbdinp.gt.2.and.igeom.le.2) then
         call copy(plag,p,ntot2)
      elseif (nbd.eq.3.and.igeom.le.2) then

         const = dtlag(1)/dtlag(2)

         do i=1,ntot2
            pnm1          = p   (i,1,1,1)
            pnm2          = plag(i,1,1,1)
            p   (i,1,1,1) = pnm1 + const*(pnm1-pnm2)
            plag(i,1,1,1) = pnm1
         enddo

      elseif (nbd.gt.3) then
         WRITE (6,*) 'Pressure extrapolation cannot be completed'
         WRITE (6,*) 'Try a lower-order temporal scheme'
         call exitt
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine opzero(ux,uy,uz)
      include 'SIZE'
      include 'TOTAL'
      real ux(1),uy(1),uz(1)
c
      n = lx1*ly1*lz1*nelfld(ifield)
      call rzero(ux,n)
      call rzero(uy,n)
      if (if3d) call rzero(uz,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine opnorm(unorm,ux,uy,uz,type3)
      include 'SIZE'
      include 'TOTAL'
      character*3 type3
      real ux(1),uy(1),uz(1)
      real un(3),wn(3)
c
      n = lx1*ly1*lz1*nelfld(ifield)
      if (type3.eq.'L2 ') then
         if (if3d) then
            un(1) = vlsc3(ux,ux,bm1,n)
            un(2) = vlsc3(uy,uy,bm1,n)
            un(3) = vlsc3(uz,uz,bm1,n)
            un(1) = un(1) + un(2) + un(3)
            unorm = glsum(un(1),1)
            if (unorm.gt.0) unorm = sqrt(unorm/volvm1)
         else
            un(1) = vlsc3(ux,ux,bm1,n)
            un(2) = vlsc3(uy,uy,bm1,n)
            un(1) = un(1) + un(2)
            unorm = glsum(un(1),1)
            if (unorm.gt.0) unorm = sqrt(unorm/volvm1)
         endif
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lorentz_force (lf,b1,b2,b3,w1,w2)
c
c     Compute Lorentz force
c
c     Input:  B     := (b1,b2,b3)
c
c     Output: lf(1,ldim)
c
c     Work arrays: w1(ltot) and w2(ltot)
c
c     The output will not be continuous.  However, it will be in
c     the form appropriate for incorporation as a body force term 
c     in the variational formulation of the Navier-Stokes equations.
c
c     (i.e.,   rhs(NS) = rhs(NS) + B*lf,  where B is the mass matrix)
c
      include 'SIZE'
c
      real lf(lx1*ly1*lz1*lelv,ldim)
      real b1(lx1*ly1*lz1*lelv)
      real b2(lx1*ly1*lz1*lelv)
      real b3(lx1*ly1*lz1*lelv)
c
      call curl(lf,b1,b2,b3,.false.,w1,w2)
c
      ntot = lx1*ly1*lz1*nelv
c
      do i=1,ntot
         c1 = lf(i,2)*b3(i) - lf(i,3)*b2(i)
         c2 = lf(i,3)*b1(i) - lf(i,1)*b3(i)
         c3 = lf(i,1)*b2(i) - lf(i,2)*b1(i)
         lf(i,1) = c1
         lf(i,2) = c2
         lf(i,3) = c3
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine curl(vort,u,v,w,ifavg,work1,work2)
c
      include 'SIZE'
      include 'TOTAL'
c
      logical ifavg
c
      parameter(lt=lx1*ly1*lz1*lelv)
      real vort(lt,3),work1(1),work2(1),u(1),v(1),w(1)
c
      ntot  = lx1*ly1*lz1*nelv
      if (if3d) then
c        work1=dw/dy ; work2=dv/dz
           call dudxyz(work1,w,rym1,sym1,tym1,jacm1,1,2)
           call dudxyz(work2,v,rzm1,szm1,tzm1,jacm1,1,3)
           call sub3(vort(1,1),work1,work2,ntot)
c        work1=du/dz ; work2=dw/dx
           call dudxyz(work1,u,rzm1,szm1,tzm1,jacm1,1,3)
           call dudxyz(work2,w,rxm1,sxm1,txm1,jacm1,1,1)
           call sub3(vort(1,2),work1,work2,ntot)
c        work1=dv/dx ; work2=du/dy
           call dudxyz(work1,v,rxm1,sxm1,txm1,jacm1,1,1)
           call dudxyz(work2,u,rym1,sym1,tym1,jacm1,1,2)
           call sub3(vort(1,3),work1,work2,ntot)
      else
c        work1=dv/dx ; work2=du/dy
           call dudxyz(work1,v,rxm1,sxm1,txm1,jacm1,1,1)
           call dudxyz(work2,u,rym1,sym1,tym1,jacm1,1,2)
           call sub3(vort(1,3),work1,work2,ntot)
      endif
c
c    Avg at bndry
c
      if (ifavg) then
         ifielt = ifield
         ifield = 1
         if (if3d) then
            do idim=1,ldim
               call col2  (vort(1,idim),bm1,ntot)
               call dssum (vort(1,idim),lx1,ly1,lz1)
               call col2  (vort(1,idim),binvm1,ntot)
            enddo
         else
            call col2  (vort(1,3),bm1,ntot)    ! NOTE:  This differs from
            call dssum (vort(1,3),lx1,ly1,lz1) ! "comp_vort", which returns
            call col2  (vort(1,3),binvm1,ntot) ! vorticity as 1st entry in vort
         endif
         ifield = ifielt
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lorentz_force2(lf,b1,b2,b3)
c
c     Compute Lorentz force
c
c     Input:  B     := (b1,b2,b3)
c
c     Output: lf(1,ldim)
c
c     Work arrays: w1(ltot) and w2(ltot)
c
c     The output will not be continuous.  However, it will be in
c     the form appropriate for incorporation as a body force term 
c     in the variational formulation of the Navier-Stokes equations.
c
c     (i.e.,   rhs(NS) = rhs(NS) + B*lf,  where B is the mass matrix)
c
      include 'SIZE'
c
      real lf(lx1*ly1*lz1*ldim,lelt)
      real b1(lx1*ly1*lz1,lelt)
      real b2(lx1*ly1*lz1,lelt)
      real b3(lx1*ly1*lz1,lelt)
c
      integer e
c
      do e = 1,nelt   ! NOTE:  the order is different from v. 1
         call lorentz_force_e(lf(1,e),b1(1,e),b2(1,e),b3(1,e),e)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine lorentz_force_e(lf,b1,b2,b3,e)
c
c     Compute Lorentz force field for a single element
c
c     Input:  B     := (b1,b2,b3)
c
c     Output: lf(1,ldim)
c
c     Work arrays: cb(lxyzd) and w2(lxyzd)
c
c     The output will not be continuous.  However, it will be in
c     the form appropriate for incorporation as a body force term 
c     in the variational formulation of the Navier-Stokes equations.
c     (i.e.,   rhs(NS) = rhs(NS) + B*lf,  where B is the mass matrix)
c
c     Dealiasing strategy:
c
c       e      e  -1   ~ T  ~e   ~ ~      
c     lf  = ( B  )     I    B  ( I j x I b )
c       i              ~                    i
c
c            ~
c     Here,  I is the interpolant from N to M, where M = 3/2 N.
c
c            ~                                  -1
c            j is a special curl  (sans Jacobian   )
c
c            b is the B-field on the N-points
c
c            ~e
c            B  is the local mass matrix on the M-points (sans Jacabian)
c
c             e
c            B  is the local mass matrix on the N-points (with Jacabian)
c
c     The last B is req'd to compensate for the subsequent multiplication 
c     by B that takes place once the rhs is formed.
c
c
c
      include 'SIZE'
      include 'DEALIAS'
      include 'GEOM'
      include 'WZ'
c
      real lf(lx1*ly1*lz1,3)
      real b1(lx1*ly1*lz1)
      real b2(lx1*ly1*lz1)
      real b3(lx1*ly1*lz1)
      integer d,e
c
      integer icalld
      save    icalld
      data    icalld /0/
c
      common /ctmp1x/ lfd(lxd*lyd*lzd,3)
     $             ,  bd (lxd*lyd*lzd,3)
     $             ,  cb (lx1*ly1*lz1,3)
     $             ,  cbd(lxd*lyd*lzd,3)
      real lfd

      if (icalld .eq. 0) then
          write(6,*) 'CALL SET PROJ',lx1,lxd
          call setmap (lx1,lxd)   ! Set up interpolation operators
          call setproj(lx1,lxd)   ! Set up interpolation operators
          icalld = icalld + 1
      endif
c
      call spec_curl_e (cb,b1,b2,b3                  !local curl, w/o Jacobian
     $                 , rxm1(1,1,1,e),rym1(1,1,1,e),rzm1(1,1,1,e)
     $                 , sxm1(1,1,1,e),sym1(1,1,1,e),szm1(1,1,1,e)
     $                 , txm1(1,1,1,e),tym1(1,1,1,e),tzm1(1,1,1,e) )
c
      do d=1,ldim                                   ! interpolate to M points
         call specmp(cbd(1,d),lxd,cb(1,d),lx1,im1d,im1dt,bd)
      enddo
      call specmp(bd(1,1),lxd,b1,lx1,im1d,im1dt,lfd)
      call specmp(bd(1,2),lxd,b2,lx1,im1d,im1dt,lfd)
      call specmp(bd(1,3),lxd,b3,lx1,im1d,im1dt,lfd)
c
      nxyzd = lxd*lyd*lzd
      do i=1,nxyzd
         lfd(i,1) = cbd(i,2)*bd(i,3) - cbd(i,3)*bd(i,2) ! Curl B x B
         lfd(i,2) = cbd(i,3)*bd(i,1) - cbd(i,1)*bd(i,3)
         lfd(i,3) = cbd(i,1)*bd(i,2) - cbd(i,2)*bd(i,1)
      enddo
c
c     Project back and simultaneous collocate with local quadrature weights
c
c                                 ~        ~        ~
c        P := Pz x Py x Px  =  Iz*B  x  Iy*B  x  Ix*B
c
c
c           ~            M
c     where B = diag (rho  )
c                        i
c
      call specmp(lf(1,1),lx1,lfd(1,1),lxd,pmd1,pmd1t,cbd)
      call specmp(lf(1,2),lx1,lfd(1,2),lxd,pmd1,pmd1t,cbd)
      call specmp(lf(1,3),lx1,lfd(1,3),lxd,pmd1,pmd1t,cbd)
c
c     Finally, divide by local mass matrix in anticipation of subsequent
c     multiply by BM1.
c
      nxyz = lx1*ly1*lz1
      do i=1,nxyz
c        scale = 1./(w3m1(i,1,1)*jacm1(i,1,1,e))
         scale = 1./jacm1(i,1,1,e)
         lf(i,1) = scale*lf(i,1)
         lf(i,2) = scale*lf(i,2)
         lf(i,3) = scale*lf(i,3)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine spec_curl_e (cb,b1,b2,b3,rx,ry,rz,sx,sy,sz,tx,ty,tz) 
c
c     local curl, multiplied by Jacobian
c
      include 'SIZE'
      include 'DXYZ'
c
      real cb(lx1*ly1*lz1,3)  ! Output J*curl B  (J:=Jacobian)
      real b1(1),b2(1),b3(1)  ! Input B-field
c
      real rx(1),ry(1),rz(1)  ! Metrics
      real sx(1),sy(1),sz(1)
      real tx(1),ty(1),tz(1)
c
      common /ctmp0x/ br(lx1*ly1*lz1),bs(lx1*ly1*lz1),bt(lx1*ly1*lz1)
c
c              / db3     db2 \
c     cb1 = J  | ---  -  --- |    ! Keep J ( J:=Jacobian)
c              \ dy      dz  /
c
c
c              / db1     db3 \
c     cb2 = J  | ---  -  --- |    ! Keep J ( J:=Jacobian)
c              \ dz      dx  /
c
c
c              / db2     db1 \
c     cb3 = J  | ---  -  --- |    ! Keep J ( J:=Jacobian)
c              \ dx      dy  /
c
c
c     Note:
c
c      db2      db2   dr     db2   ds     db2   dt       
c    J --- =    --- J --  +  --- J --  +  --- J --     
c      dz       dr    dz     ds    dz     dt    dz      
c
c              etc.
c
c
c
      nxyz = lx1*ly1*lz1 
c
      N=lx1-1
      call local_grad3(br,bs,bt,b1,N,1,dxm1,dxtm1)
      do i=1,nxyz
         cb(i,2) =  (br(i)*rz(i)+bs(i)*sz(i)+bt(i)*tz(i))
         cb(i,3) = -(br(i)*ry(i)+bs(i)*sy(i)+bt(i)*ty(i))
      enddo
c
      call local_grad3(br,bs,bt,b2,N,1,dxm1,dxtm1)
      do i=1,nxyz
         cb(i,1) = -(br(i)*rz(i)+bs(i)*sz(i)+bt(i)*tz(i))
         cb(i,3) =  (br(i)*rx(i)+bs(i)*sx(i)+bt(i)*tx(i))
     $           +  cb(i,3)
      enddo
c
      call local_grad3(br,bs,bt,b3,N,1,dxm1,dxtm1)
      do i=1,nxyz
         cb(i,1) =  (br(i)*ry(i)+bs(i)*sy(i)+bt(i)*ty(i))
     $           +  cb(i,1)
         cb(i,2) = -(br(i)*rx(i)+bs(i)*sx(i)+bt(i)*tx(i))
     $           +  cb(i,2)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine specx(b,nb,a,na,ba,ab,w)
c
      include 'SIZE'
      include 'INPUT'
      real b(1),a(1)
      real w(1)
c
      n=na*na*na
      do i=1,n
         b(i) = a(i)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine phys_to_elsasser(u1,u2,u3,b1,b2,b3,n)
c
      real u1(1),u2(1),u3(1),b1(1),b2(1),b3(1)
c
      do i=1,n
c
         zpx = u1(i) + b1(i)
         zpy = u2(i) + b2(i)
         zpz = u3(i) + b3(i)
c
         zmx = u1(i) - b1(i)
         zmy = u2(i) - b2(i)
         zmz = u3(i) - b3(i)
c
         u1(i) = zpx
         u2(i) = zpy
         u3(i) = zpz
c
         b1(i) = zmx
         b2(i) = zmy
         b3(i) = zmz
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine elsasser_to_phys(u1,u2,u3,b1,b2,b3,n)
c
      real u1(1),u2(1),u3(1),b1(1),b2(1),b3(1)
c
      do i=1,n
c
         zpx = 0.5*( u1(i) + b1(i) )
         zpy = 0.5*( u2(i) + b2(i) )
         zpz = 0.5*( u3(i) + b3(i) )
c
         zmx = 0.5*( u1(i) - b1(i) )
         zmy = 0.5*( u2(i) - b2(i) )
         zmz = 0.5*( u3(i) - b3(i) )
c
         u1(i) = zpx
         u2(i) = zpy
         u3(i) = zpz
c
         b1(i) = zmx
         b2(i) = zmy
         b3(i) = zmz
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine phys_to_elsasser2(u1,b1,n)
c
      real u1(1),b1(1)
c
      do i=1,n
         zpx = u1(i) + b1(i)
         zmx = u1(i) - b1(i)
         u1(i) = zpx
         b1(i) = zmx
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine elsasser_to_phys2(u1,b1,n)
c
      real u1(1),b1(1)
c
      do i=1,n
         zpx = 0.5*( u1(i) + b1(i) )
         zmx = 0.5*( u1(i) - b1(i) )
         u1(i) = zpx
         b1(i) = zmx
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine elsasserh(igeom)
c
c
c     Solve MHD in Elsasser variables
c
c
      include 'SIZE'
      include 'INPUT'
      include 'EIGEN'
      include 'SOLN'
      include 'TSTEP'
      include 'MASS'
      include 'GEOM'
C
      common /scrnt/  besv1 (lbx1,lby1,lbz1,lbelv)
     $ ,              besv2 (lbx1,lby1,lbz1,lbelv)
     $ ,              besv3 (lbx1,lby1,lbz1,lbelv)
      COMMON /SCRNS/  RESV1 (LX1,LY1,LZ1,LELV)
     $ ,              RESV2 (LX1,LY1,LZ1,LELV)
     $ ,              RESV3 (LX1,LY1,LZ1,LELV)
     $ ,              DV1   (LX1,LY1,LZ1,LELV)
     $ ,              DV2   (LX1,LY1,LZ1,LELV)
     $ ,              DV3   (LX1,LY1,LZ1,LELV)
      COMMON /SCRVH/  H1    (LX1,LY1,LZ1,LELV)
     $ ,              H2    (LX1,LY1,LZ1,LELV)
c
      n  = lx1*ly1*lz1*nelv
c
c     New geometry, new velocity
c
      intype = -1
c
      ifield = 1
      call sethlm   (h1,h2,intype)
      call cresvibp (resv1,resv2,resv3,h1,h2)
c
      ifield = ifldmhd
      call sethlm   (h1,h2,intype)
      call cresvib  (besv1,besv2,besv3,h1,h2)
c
c
      ifield = 1
      call sethlm   (h1,h2,intype)

      call ophinv   (dv1,dv2,dv3,resv1,resv2,resv3,h1,h2,tolhv,nmxv)

      call opadd2   (vx,vy,vz,dv1,dv2,dv3)

      if (filterType.eq.1) then
         alpha_filt=param(103)                        ! Optional Filtering
         call q_filter(alpha_filt)
      endif

      call incomprn (vx,vy,vz,pr) ! project U onto div-free space

      ifield = ifldmhd
      call sethlm   (h1,h2,intype)

      call ophinv   (dv1,dv2,dv3,besv1,besv2,besv3,h1,h2,tolhv,nmxv)
      call opadd2   (bx,by,bz,dv1,dv2,dv3)

      call incomprn (bx,by,bz,pm) ! project B onto div-free space

      return
      end
c--------------------------------------------------------------------
      subroutine compute_cfl(cfl,u,v,w,dt)
c
c     Given velocity field (u,v,w) and dt, compute current CFL number.
c
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
      include 'WZ'
      include 'SOLN'
c
      real u(lx1,ly1,lz1,nelv),v(lx1,ly1,lz1,nelv),w(lx1,ly1,lz1,nelv)
c
c     Store the inverse jacobian to speed up this operation
c
      common /cfldx/ dri(lx1),dsi(ly1),dti(lz1)
c
      integer e
c
      integer icalld
      save    icalld
      data    icalld /0/
c
      if (icalld.eq.0) then
         icalld=1
         call getdr(dri,zgm1(1,1),lx1)
         call getdr(dsi,zgm1(1,2),ly1)
         if (if3d) call getdr(dti,zgm1(1,3),lz1)
      endif

      cfl = 0.
      l   = 0

      if (if3d) then
         nxyz = lx1*ly1*lz1
         do e=1,nelv
            do k=1,lz1
            do j=1,ly1
            do i=1,lx1
               l = l+1
               ur = ( u(i,j,k,e)*rxm1(i,j,k,e)
     $            +   v(i,j,k,e)*rym1(i,j,k,e)
     $            +   w(i,j,k,e)*rzm1(i,j,k,e) ) * jacmi(l,1)
               us = ( u(i,j,k,e)*sxm1(i,j,k,e)
     $            +   v(i,j,k,e)*sym1(i,j,k,e)
     $            +   w(i,j,k,e)*szm1(i,j,k,e) ) * jacmi(l,1)
               ut = ( u(i,j,k,e)*txm1(i,j,k,e)
     $            +   v(i,j,k,e)*tym1(i,j,k,e)
     $            +   w(i,j,k,e)*tzm1(i,j,k,e) ) * jacmi(l,1)
 
               cflr = abs(dt*ur*dri(i))
               cfls = abs(dt*us*dsi(j))
               cflt = abs(dt*ut*dti(k))
 
               cflm = cflr + cfls + cflt
               cfl  = max(cfl,cflm)

               cflf(i,j,k,e) = cflm
 
            enddo
            enddo
            enddo
         enddo
      else
         nxyz = lx1*ly1
         do e=1,nelv
            do j=1,ly1
            do i=1,lx1
               l = l+1
               ur = ( u(i,j,1,e)*rxm1(i,j,1,e)
     $            +   v(i,j,1,e)*rym1(i,j,1,e) ) * jacmi(l,1)
               us = ( u(i,j,1,e)*sxm1(i,j,1,e)
     $            +   v(i,j,1,e)*sym1(i,j,1,e) ) * jacmi(l,1)

               cflr = abs(dt*ur*dri(i))
               cfls = abs(dt*us*dsi(j))

               cflm = cflr + cfls
               cfl  = max(cfl,cflm)

               cflf(i,j,1,e) = cflm

            enddo
            enddo
         enddo
      endif
c
      cfl = glmax(cfl,1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine getdr(dri,zgm1,lx1)
      real dri(lx1),zgm1(lx1)
c
      dri(1) = zgm1(2) - zgm1(1)   !  Compute 1/Dx
      do i=2,lx1-1
         dri(i) = 0.5*( zgm1(i+1) - zgm1(i-1) )
      enddo
      dri(lx1) = zgm1(lx1) - zgm1(lx1-1)
c
      call invcol1(dri,lx1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine ophinv(o1,o2,o3,i1,i2,i3,h1,h2,tolh,nmxhi)
C
C     Ok = (H1*A+H2*B)-1 * Ik  (implicit)
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'VPROJ'
      include 'TSTEP'
c      logical ifproj
 
      real o1 (lx1,ly1,lz1,1) , o2 (lx1,ly1,lz1,1) , o3 (lx1,ly1,lz1,1)
      real i1 (lx1,ly1,lz1,1) , i2 (lx1,ly1,lz1,1) , i3 (lx1,ly1,lz1,1)
      real h1 (lx1,ly1,lz1,1) , h2 (lx1,ly1,lz1,1)
 
c      ifproj = .false.
c      if (param(94).gt.0)    ifproj = .true.
c      if (ifprojfld(ifield)) ifproj = .true.
c 
c      if (.not.ifproj) then
c         if (ifield.eq.1) call ophinv
c     $      (o1,o2,o3,i1,i2,i3,h1,h2,tolh,nmxhi)
c         if (ifield.eq.ifldmhd) call ophinv
c     $      (o1,o2,o3,i1,i2,i3,h1,h2,tolh,nmxhi)
c         return
c      endif
 
      mtmp = param(93)
      do i=1,2*ldim
         ivproj(1,i) = min(mxprev,mtmp) - 1
      enddo
 
      imesh = 1
 
      if (ifstrs) then
         matmod = 0
         call hmhzsf  ('NOMG',o1,o2,o3,i1,i2,i3,h1,h2,
     $                  v1mask,v2mask,v3mask,vmult,
     $                  tolh,nmxhi,matmod)
      else
         if (ifield.eq.1) then
            call hsolve ('VELX',o1,i1,h1,h2,v1mask,vmult
     $                         ,imesh,tolh,nmxhi,1
     $                         ,vproj(1,1),ivproj(1,1),binvm1)
            call hsolve ('VELY',o2,i2,h1,h2,v2mask,vmult
     $                         ,imesh,tolh,nmxhi,2
     $                         ,vproj(1,2),ivproj(1,2),binvm1)
            if (if3d)
     $      call hsolve ('VELZ',o3,i3,h1,h2,v3mask,vmult
     $                         ,imesh,tolh,nmxhi,3
     $                         ,vproj(1,3),ivproj(1,3),binvm1)
         elseif (ifield.eq.ifldmhd) then  ! B-field
            call hsolve (' BX ',o1,i1,h1,h2,b1mask,vmult
     $                         ,imesh,tolh,nmxhi,1
     $                         ,vproj(1,4),ivproj(1,4),binvm1)
            call hsolve (' BY ',o2,i2,h1,h2,b2mask,vmult
     $                         ,imesh,tolh,nmxhi,2
     $                         ,vproj(1,5),ivproj(1,5),binvm1)
            if (if3d)
     $      call hsolve (' BZ ',o3,i3,h1,h2,b3mask,vmult
     $                         ,imesh,tolh,nmxhi,3
     $                         ,vproj(1,6),ivproj(1,6),binvm1)
         endif
      endif
C
      return
      end
c--------------------------------------------------------------------
      subroutine set_ifbcor
      include 'SIZE'
      include 'GEOM'
      include 'INPUT'
c     include 'TSTEP'   ! ifield?

      common  /nekcb/ cb
      character cb*3

      ifbcor = .true.
      ifield = ifldmhd

      nface  = 2*ldim
      do iel=1,nelv
      do ifc=1,nface
         cb = cbc(ifc,iel,ifield)
         if  (cb.eq.'ndd' .or. cb.eq.'dnd' .or. cb.eq.'ddn')
     $           ifbcor = .false.
      enddo
      enddo

      call gllog(ifbcor , .false.)

      if (nio.eq.0)  write (6,*) 'IFBCOR   =',ifbcor

      return
      end
c--------------------------------------------------------------------
c      subroutine set_ifbcor_old(ifbcor)
cc
cc     This is a quick hack for the rings problem - it is not general,
cc     but will also work fine for the periodic in Z problem
cc
c      include 'SIZE'
c      include 'TOTAL'
c
c      logical ifbcor
c
c      integer e,f
c      character*1 cb1(3)
c
c      itest = 0
c
c      do e=1,nelv
c      do f=1,2*ldim
c         call chcopy(cb1,cbc(f,e,ifldmhd),3)
c         if (cb1(3).eq.'n'.or.cb1(3).eq.'N') itest = 1
c      enddo
c      enddo
c
c      itest = iglmax(itest,1)
c
c      ifbcor = .true.  ! adjust mean pressure to rmv hydrostatic mode
c
c
c      if (itest.eq.1) ifbcor = .false.  ! do not adjust mean pressure
c
c      return
c      end
c--------------------------------------------------------------------
      subroutine setrhsp(p,h1,h2,h2inv,pset,niprev)
C
C     Project soln onto best fit in the "E" norm.
C
      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'SOLN'
      include 'TSTEP'

      real p    (lx2,ly2,lz2,lelv)
      real h1   (lx1,ly1,lz1,lelv)
      real h2   (lx1,ly1,lz1,lelv)
      real h2inv(lx1,ly1,lz1,lelv)
      real pset (lx2*ly2*lz2*lelv,mxprev)

      parameter (ltot2=lx2*ly2*lz2*lelv)
      common /orthox/ pbar(ltot2),pnew(ltot2)
      common /orthos/ alpha(mxprev),work(mxprev)
      common /orthoi/ nprev,mprev

      nprev = niprev
      if (nprev.eq.0) return

c     Diag to see how much reduction in the residual is attained.
      ntot2  = lx2*ly2*lz2*nelv
      alpha1 = glsc3(p,p,bm2inv,ntot2)
      if (alpha1.gt.0) then
         alpha1 = sqrt(alpha1/volvm2)
      else
         return
      endif

      call updrhse(p,h1,h2,h2inv,ierr) ! Update rhs's if E-matrix has changed
c     if (ierr.eq.1) Nprev=0           ! Doesn't happen w/ new formulation

      do i=1,nprev  ! Perform Gram-Schmidt for previous soln's.
         alpha(i) = vlsc2(p,pset(1,i),ntot2)
      enddo
      call gop(alpha,work,'+  ',nprev)

      call rzero(pbar,ntot2)
      do i=1,nprev
         call add2s2(pbar,pset(1,i),alpha(i),ntot2)
      enddo
C
      intetype = 1
      call cdabdtp(pnew,pbar,h1,h2,h2inv,intetype)
      call sub2   (p,pnew,ntot2)

c    ................................................................
      alpha2 = glsc3(p,p,bm2inv,ntot2) ! Diagnostics
      if (alpha2.gt.0) then
         alpha2 = sqrt(alpha2/volvm2)
         ratio  = alpha1/alpha2
         n10=min(10,nprev)
c         if (nio.eq.0) write(6,11) istep,nprev,(alpha(i),i=1,n10)
c         if (nio.eq.0) write(6,12) istep,nprev,alpha1,alpha2,ratio
   11    format(2i5,' alpha:',1p10e12.4)
   12    format(i6,i4,1p3e12.4,' alph12')

         if (nio.eq.0) write(6,13) istep,'  Project PRES  ',
     &                         alpha2,alpha1,ratio,nprev,mxprev
   13    format(i11,a,6x,1p3e13.4,i4,i4)
      endif
c    ................................................................

      return
      end
c-----------------------------------------------------------------------
      subroutine gensolnp(p,h1,h2,h2inv,pset,nprev)
C
C     Reconstruct the solution to the original problem by adding back
C     the previous solutions
C
      include 'SIZE'
      include 'INPUT'

      real p    (lx2,ly2,lz2,lelv)
      real h1   (lx1,ly1,lz1,lelv)
      real h2   (lx1,ly1,lz1,lelv)
      real h2inv(lx1,ly1,lz1,lelv)
      real pset (lx2*ly2*lz2*lelv,mxprev)

      parameter (ltot2=lx2*ly2*lz2*lelv)
      common /orthox/ pbar(ltot2),pnew(ltot2)
      common /orthos/ alpha(mxprev),work(mxprev)

      mprev=param(93)
      mprev=min(mprev,mxprev)

      ntot2=lx2*ly2*lz2*nelv

      if (nprev.lt.mprev) then
         nprev = nprev+1
         call copy  (pset(1,nprev),p,ntot2)        ! Save current solution
         call add2  (p,pbar,ntot2)                 ! Reconstruct solution.
         call econjp(pset,nprev,h1,h2,h2inv,ierr)  ! Orthonormalize set

         if (ierr.eq.1) then
          nprev = 1
          call copy  (pset(1,nprev),p,ntot2)       ! Save current solution
          call econjp(pset,nprev,h1,h2,h2inv,ierr) !   and orthonormalize.
        endif

      else                                         !          (uses pnew).
         nprev = 1
         call add2  (p,pbar,ntot2)                 ! Reconstruct solution.
         call copy  (pset(1,nprev),p,ntot2)        ! Save current solution
         call econjp(pset,nprev,h1,h2,h2inv,ierr)  !   and orthonormalize.
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine econjp(pset,nprev,h1,h2,h2inv,ierr)

c     Orthogonalize the soln wrt previous soln's for which we already
c     know the soln.

      include 'SIZE'
      include 'INPUT'

      real p    (lx2,ly2,lz2,lelv)
      real h1   (lx1,ly1,lz1,lelv)
      real h2   (lx1,ly1,lz1,lelv)
      real h2inv(lx1,ly1,lz1,lelv)
      real pset (lx2*ly2*lz2*lelv,mxprev)

      parameter (ltot2=lx2*ly2*lz2*lelv)
      common /orthox/ pbar(ltot2),pnew(ltot2)
      common /orthos/ alpha(mxprev),work(mxprev)

      ierr  = 0

      ntot2=lx2*ly2*lz2*nelv

C
C     Gram Schmidt, w re-orthogonalization
C
      npass=1
c     if (abs(param(105)).eq.2) npass=2
      do ipass=1,npass

         intetype=1
         call cdabdtp(pnew,pset(1,nprev),h1,h2,h2inv,intetype)
         alphad = glsc2(pnew,pset(1,nprev),ntot2) ! compute part of the norm

         nprev1 = nprev-1
         do i=1,nprev1   !   Gram-Schmidt
            alpha(i) = vlsc2(pnew,pset(1,i),ntot2)
         enddo
         if (nprev1.gt.0) call gop(alpha,work,'+  ',nprev1)

         do i=1,nprev1
            alpham = -alpha(i)
            call add2s2(pset(1,nprev),pset(1,i),alpham,ntot2)
            alphad = alphad - alpha(i)**2
         enddo
      enddo
C
C    .Normalize new element in P~
C
      if (alphad.le.0) then
         write(6,*) 'ERROR:  alphad .le. 0 in econjp',alphad,nprev
         ierr = 1
         return
      endif
      alphad = 1./sqrt(alphad)
      call cmult(pset(1,nprev),alphad,ntot2)

      return
      end
c-----------------------------------------------------------------------
      subroutine advab_elsasser_fast
C
C     Eulerian scheme, add convection term to forcing function
C     at current time step.
C
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
c
      parameter (lxy=lx1*ly1*lz1,ltd=lxd*lyd*lzd)
      common /scrns/ wk(2*ltd)
     $             , fx(lxy),fy(lxy),fz(lxy)
     $             , gx(lxy),gy(lxy),gz(lxy)
     $             , zr(ltd),zs(ltd),zt(ltd)
     $             , tr(ltd,3),zp(ltd,3),zm(ltd,3)

      integer e
      integer icalld
      save    icalld
      data    icalld /0/

      if (icalld.eq.0) call set_dealias_rx
      icalld = icalld + 1

      nxyz1 = lx1*ly1*lz1
      nxyzd = lxd*lyd*lzd


      if (if3d) then
c
         do e=1,nelv

c           Interpolate z+ and z- into fine mesh, translate to r-s-t coords

c           write(6,*) nid,' inside fast',e,nxyz1

            call add3(wk,vx(1,1,1,e),bx(1,1,1,e),nxyz1)
            call intp_rstd(zp(1,1),wk,lx1,lxd,if3d,0) ! 0 --> forward

            call add3(wk,vy(1,1,1,e),by(1,1,1,e),nxyz1)
            call intp_rstd(zp(1,2),wk,lx1,lxd,if3d,0)

            call add3(wk,vz(1,1,1,e),bz(1,1,1,e),nxyz1)
            call intp_rstd(zp(1,3),wk,lx1,lxd,if3d,0)

            call sub3(wk,vx(1,1,1,e),bx(1,1,1,e),nxyz1)
            call intp_rstd(zm(1,1),wk,lx1,lxd,if3d,0)

            call sub3(wk,vy(1,1,1,e),by(1,1,1,e),nxyz1)
            call intp_rstd(zm(1,2),wk,lx1,lxd,if3d,0)

            call sub3(wk,vz(1,1,1,e),bz(1,1,1,e),nxyz1)
            call intp_rstd(zm(1,3),wk,lx1,lxd,if3d,0)

            do i=1,nxyzd  ! Convert convector (zm) to r-s-t coordinates
               tr(i,1)=
     $            rx(i,1,e)*zm(i,1)+rx(i,2,e)*zm(i,2)+rx(i,3,e)*zm(i,3)
               tr(i,2)=
     $            rx(i,4,e)*zm(i,1)+rx(i,5,e)*zm(i,2)+rx(i,6,e)*zm(i,3)
               tr(i,3)=
     $            rx(i,7,e)*zm(i,1)+rx(i,8,e)*zm(i,2)+rx(i,9,e)*zm(i,3)
            enddo


            call grad_rst(zr,zs,zt,zp(1,1),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(fx,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zp(1,2),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(fy,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zp(1,3),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(fz,wk,lx1,lxd,if3d,1) ! Project back to coarse


            do i=1,nxyzd  ! Convert convector (zp) to r-s-t coordinates
               tr(i,1)=
     $            rx(i,1,e)*zp(i,1)+rx(i,2,e)*zp(i,2)+rx(i,3,e)*zp(i,3)
               tr(i,2)=
     $            rx(i,4,e)*zp(i,1)+rx(i,5,e)*zp(i,2)+rx(i,6,e)*zp(i,3)
               tr(i,3)=
     $            rx(i,7,e)*zp(i,1)+rx(i,8,e)*zp(i,2)+rx(i,9,e)*zp(i,3)
            enddo


            call grad_rst(zr,zs,zt,zm(1,1),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(gx,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zm(1,2),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(gy,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zm(1,3),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)+tr(i,3)*zt(i)
            enddo
            call intp_rstd(gz,wk,lx1,lxd,if3d,1) ! Project back to coarse

            tmp = -0.5 ! vtrans() assumed to be 1.0 !
            do i=1,nxyz1
               bfx(i,1,1,e) = bfx(i,1,1,e) + tmp*( fx(i) + gx(i) )
               bmx(i,1,1,e) = bmx(i,1,1,e) + tmp*( fx(i) - gx(i) )

               bfy(i,1,1,e) = bfy(i,1,1,e) + tmp*( fy(i) + gy(i) )
               bmy(i,1,1,e) = bmy(i,1,1,e) + tmp*( fy(i) - gy(i) )

               bfz(i,1,1,e) = bfz(i,1,1,e) + tmp*( fz(i) + gz(i) )
               bmz(i,1,1,e) = bmz(i,1,1,e) + tmp*( fz(i) - gz(i) )
            enddo

         enddo

      else ! 2D
c
         do e=1,nelv

c           Interpolate z+ and z- into fine mesh, translate to r-s-t coords

            call add3(wk,vx(1,1,1,e),bx(1,1,1,e),nxyz1)
            call intp_rstd(zp(1,1),wk,lx1,lxd,if3d,0) ! 0 --> forward

            call add3(wk,vy(1,1,1,e),by(1,1,1,e),nxyz1)
            call intp_rstd(zp(1,2),wk,lx1,lxd,if3d,0)

            call sub3(wk,vx(1,1,1,e),bx(1,1,1,e),nxyz1)
            call intp_rstd(zm(1,1),wk,lx1,lxd,if3d,0)

            call sub3(wk,vy(1,1,1,e),by(1,1,1,e),nxyz1)
            call intp_rstd(zm(1,2),wk,lx1,lxd,if3d,0)

            do i=1,nxyzd  ! Convert convector (zm) to r-s-t coordinates
               tr(i,1)=
     $            rx(i,1,e)*zm(i,1)+rx(i,2,e)*zm(i,2)
               tr(i,2)=
     $            rx(i,3,e)*zm(i,1)+rx(i,4,e)*zm(i,2)
            enddo

            call grad_rst(zr,zs,zt,zp(1,1),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)
            enddo
            call intp_rstd(fx,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zp(1,2),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)
            enddo
            call intp_rstd(fy,wk,lx1,lxd,if3d,1) ! Project back to coarse

            do i=1,nxyzd  ! Convert convector (zp) to r-s-t coordinates
               tr(i,1)=
     $            rx(i,1,e)*zp(i,1)+rx(i,2,e)*zp(i,2)
               tr(i,2)=
     $            rx(i,3,e)*zp(i,1)+rx(i,4,e)*zp(i,2)
            enddo

            call grad_rst(zr,zs,zt,zm(1,1),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)
            enddo
            call intp_rstd(gx,wk,lx1,lxd,if3d,1) ! Project back to coarse

            call grad_rst(zr,zs,zt,zm(1,2),lxd,if3d)
            do i=1,nxyzd ! mass matrix included, per DFM (4.8.5)
               wk(i) = tr(i,1)*zr(i)+tr(i,2)*zs(i)
            enddo
            call intp_rstd(gy,wk,lx1,lxd,if3d,1) ! Project back to coarse

            tmp = -0.5 ! vtrans() assumed to be 1.0 !
            do i=1,nxyz1
               bfx(i,1,1,e) = bfx(i,1,1,e) + tmp*( fx(i) + gx(i) )
               bmx(i,1,1,e) = bmx(i,1,1,e) + tmp*( fx(i) - gx(i) )

               bfy(i,1,1,e) = bfy(i,1,1,e) + tmp*( fy(i) + gy(i) )
               bmy(i,1,1,e) = bmy(i,1,1,e) + tmp*( fy(i) - gy(i) )
            enddo
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine cfl_check
c
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'MASS'
      include 'TSTEP'
c
      common /scrns/ ta1 (lx1,ly1,lz1,lelv)
     $ ,             ta2 (lx1,ly1,lz1,lelv)
     $ ,             ta3 (lx1,ly1,lz1,lelv)
     $ ,             tb1 (lx1,ly1,lz1,lelv)
     $ ,             tb2 (lx1,ly1,lz1,lelv)
     $ ,             tb3 (lx1,ly1,lz1,lelv)


      call opcopy(ta1,ta2,ta3,vx,vy,vz)
      call opcopy(tb1,tb2,tb3,bx,by,bz)

      ntot1 = lx1*ly1*lz1*nelv
      call phys_to_elsasser(ta1,ta2,ta3,tb1,tb2,tb3,ntot1) ! crude, but effective

      call compute_cfl(cflp,ta1,ta2,ta3,dt)   !  vx = U+B
      call compute_cfl(cflm,tb1,tb2,tb3,dt)   !  bx = U-B

      courno = max(cflp,cflm)

      if (nio.eq.0) write(6,1) istep,time,dt,cflp,cflm
    1 format(i9,1p4e15.7,' CFL')

      return
      end
c--------------------------------------------------------------------
