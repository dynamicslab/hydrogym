c===============================================================================
c     pff@cfm.brown.edu   3/19/96
c
c
c     This  is a suite of routines for solving overlapping subdomain
c     problems with finite elements on distributed memory machines.
c
c     The overall hierarchy of data structures is as follows:
c
c         System        -  index set denoted by       _glob
c
c            Processor  -  index set denoted by       _pglob
c
c              .Domain  -  index set denoted by       _dloc  (1,2,3,...,n_dloc)
c
c              .Sp.Elem -  index set denoted by       _sloc  (1,2,3,...,n_sloc)
c
c
c     A critical component of the parallel DD solver is the gather-scatter
c     operation.   As there may be more than one domain on a processor, and
c     communication must be minimized, it is critical that communication be
c     processor oriented, not domain oriented.  Hence domains will access data
c     via the dloc_to_pglob interface, and the pglob indexed variables will
c     be accessed via a gather-scatter which interacts via the pglob_glob
c     interface.   Noticed that, in a uni-processor application, the pglob_glob
c     map will be simply the identity.
c
c===============================================================================
c
      subroutine set_overlap
c
c     Set up arrays for overlapping Schwartz algorithm *for pressure solver*
c
      include 'SIZE'
      include 'DOMAIN'
      include 'ESOLV'
      include 'INPUT'
      include 'TSTEP'
c
      REAL*8 dnekclock,t0
c
      parameter (          n_tri = 7*ltotd )
      common /scrns/  tri (n_tri)
      integer         tri,elem
c
      common /screv/ x(2*ltotd)
      common /scrvh/ y(2*ltotd)
      common /scrch/ z(2*ltotd)
c
      common /ctmp0/ nv_to_t(2*ltotd)
c
      parameter (lia = ltotd - 2 - 2*lelt)
      common /scrcg/ ntri(lelt+1),nmask(lelt+1)
     $             , ia(lia)
c
      common /scruz/ color   (4*ltotd)
      common /scrmg/ ddmask  (4*ltotd)
      common /ctmp1/ mask    (4*ltotd)

      parameter(lxx=lx1*lx1, levb=lelv+lbelv)
      common /fastd/  df(lx1*ly1*lz1,levb)
     $             ,  sr(lxx*2,levb),ss(lxx*2,levb),st(lxx*2,levb)


      integer e

      if (lx1.eq.2) param(43)=1.
      if (lx1.eq.2.and.nid.eq.0) write(6,*) 'No mgrid for lx1=2!'

      if (ifaxis) ifmgrid = .false.
      if (param(43).ne.0) ifmgrid = .false.

      npass = 1
      if (ifmhd) npass = 2
      do ipass=1,npass
         ifield = 1

         if (ifsplit.and.ifmgrid) then

            if (ipass.gt.1) ifield = ifldmhd

            call swap_lengths
            call gen_fast_spacing(x,y,z)
 
            call hsmg_setup
            call h1mg_setup

         elseif (.not.ifsplit) then ! Pn-Pn-2

            if (ipass.gt.1) ifield = ifldmhd

            if (param(44).eq.1) then !  Set up local overlapping solves 
               call set_fem_data_l2(nel_proc,ndom,n_o,x,y,z,tri)
            else
               call swap_lengths
            endif
 
            e = 1
            if (ifield.gt.1) e = nelv+1

            call gen_fast_spacing(x,y,z)
            call gen_fast(df(1,e),sr(1,e),ss(1,e),st(1,e),x,y,z)

            call init_weight_op
            if (param(43).eq.0) call hsmg_setup
         endif

         call set_up_h1_crs

      enddo
 
      return
      end
c-----------------------------------------------------------------------
      subroutine overflow_ck(n_req,n_avail,signal)
c
c     Check for buffer overflow
c
      character*11 signal
c
      nid = mynode()
c
      if (n_req.gt.n_avail) then
         write(6,9) nid,n_req,n_avail,nid,signal
    9    format(i7,' ERROR: requested array space (',i12
     $            ,') exceeds allocated amount (',i12,').'
     $            ,/,i12,' ABORTING.',3x,a11
     $            ,/,i12,' ABORTING.',3x,'from overflow_ck call.')
         call exitt
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine iunswap(b,ind,n,temp)
      integer b(1),ind(1),temp(1)
c
c     sort associated elements by putting item(jj)
c     into item(i), where jj=ind(i).
c
      do 20 i=1,n
         jj=ind(i)
         temp(jj)=b(i)
   20 continue
      do 30 i=1,n
   30 b(i)=temp(i)
      return
      end
c-----------------------------------------------------------------------
      subroutine set_fem_data_l2(nep,nd,no,x,y,z,p)
      include 'SIZE'
      include 'DOMAIN'
      include 'NONCON'
      include 'TOTAL'
c
      real x(lx1,ly1,lz1,1),y(lx1,ly1,lz1,1),z(lx1,ly1,lz1,1)
      real p(lx1,ly1,lz1,1)
      integer ce,cf
c
      common /ctmp0/ w1(lx1,ly1),w2(lx1,ly1)
c
c     First, copy local geometry to temporary, expanded, arrays
c
      nxy1  = nx1*ny1
      nxy2  = nx2*ny2
      nxyz1 = nx1*ny1*nz1
      nxyz2 = nx2*ny2*nz2
      ntot1 = nelv*nxyz1
      ntot2 = nelv*nxyz2
c
      call rzero(x,ntot1)
      call rzero(y,ntot1)
      call rzero(z,ntot1)
c
c     Take care of surface geometry
c
      call map_face12(x,xm1,w1,w2)
      call map_face12(y,ym1,w1,w2)
      call map_face12(z,zm1,w1,w2)
c
c     Take care of interior geometry
c
      iz1 = 0
      if (if3d) iz1=1
      do ie=1,nelv
         do iz=1,nz2
         do iy=1,ny2
         do ix=1,nx2
            x(ix+1,iy+1,iz+iz1,ie) = xm2(ix,iy,iz,ie)
            y(ix+1,iy+1,iz+iz1,ie) = ym2(ix,iy,iz,ie)
            z(ix+1,iy+1,iz+iz1,ie) = zm2(ix,iy,iz,ie)
         enddo
         enddo
         enddo
      enddo
c
c
c     Compute absolute value of distance between pressure and vel. planes
c
      call dface_add1sa(x)
      call dface_add1sa(y)
      call dface_add1sa(z)
c
c     Sum this distance
c
      ifielt = ifield
      ifield = 1
c     call dssum(x,nx1,ny1,nz1)
c     call dssum(y,nx1,ny1,nz1)
c     call dssum(z,nx1,ny1,nz1)
c
c     Scale children values by 0.5.  This is a hack, based on assumption
c     that number of children sharing a parent edge is 2
c
      scale = 0.5
      if (if3d) scale = 0.25
      do im   = 1 , mort_m
         cf = noncon_f(im)
         ce = noncon_e(im)
         call faces(x,scale,ce,cf,nx1,ny1,nz1)
         call faces(y,scale,ce,cf,nx1,ny1,nz1)
         if (if3d) call faces(z,scale,ce,cf,nx1,ny1,nz1)
      enddo
      call gs_op_many(gsh_fld(ifield), x,y,z,x,x,x,ndim, 1,1,0)
c
      ifield = ifielt
c
c     Dirichlet (pmask=0) faces  all set...
c
      nel_proc = nelv
      ndom     = nelv
c     ndom     = 1
c
      n_o =  1
c
c     These are the return values, since the calling routine doesn't
c     know DOMAIN
c
      nep     = nel_proc
      nd      = ndom
      no      = n_o
      ifield  = ifielt
      return
      end
c-----------------------------------------------------------------------
      subroutine map_face12(x2,x1,w1,w2)
      include 'SIZE'
      include 'INPUT'
      include 'IXYZ'
c
c     Interpolate iface of x1 (GLL pts) onto x2 (GL pts).
c     Work arrays should be of size nx1*nx1 each.
c
      real x2(nx1*ny1*nz1,1)
      real x1(nx1*ny1*nz1,1)
      real w1(1),w2(1)
c
c
      do ie=1,nelv
       if (if3d) then
        call map_one_face12(x2(1,ie),x1(1,ie),1,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),2,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),3,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),4,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),5,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),6,ixm12,ixtm12,w1,w2)
       else
        call map_one_face12(x2(1,ie),x1(1,ie),1,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),2,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),3,ixm12,ixtm12,w1,w2)
        call map_one_face12(x2(1,ie),x1(1,ie),4,ixm12,ixtm12,w1,w2)
       endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine map_one_face12(x2,x1,iface,i12,i12t,w1,w2)
      include 'SIZE'
      include 'INPUT'
c
c     Interpolate iface of x1 (GLL pts) onto interior of face of x2 (GL pts).
c     Work arrays should be of size nx1*nx1 each.
c
      real x2(nx1,ny1,nz1)
      real x1(nx1,ny1,nz1)
      real w1(1),w2(1)
      real i12(1),i12t(1)
c
c
c     Extract surface data from x1
c
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,IFACE)
      i=0
      do iz=kz1,kz2
      do iy=ky1,ky2
      do ix=kx1,kx2
         i     = i+1
         w1(i) = x1(ix,iy,iz)
      enddo
      enddo
      enddo
c
c     Interpolate from mesh 1 to 2
c
      if (if3d) then
         call mxm(i12 ,nx2,w1,nx1,w2,nx1)
         call mxm(w2,nx2,i12t,nx1,w1,nx2)
      else
         call mxm(i12 ,nx2,w1,nx1,w2,  1)
         call copy(w1,w2,nx2)
      endif
c
c     Write to interior points on face of x2
c
      kx1=min(kx1+1,nx1,kx2)
      kx2=max(kx2-1,  1,kx1)
      ky1=min(ky1+1,ny1,ky2)
      ky2=max(ky2-1,  1,ky1)
      kz1=min(kz1+1,nz1,kz2)
      kz2=max(kz2-1,  1,kz1)
c
      i=0
      do iz=kz1,kz2
      do iy=ky1,ky2
      do ix=kx1,kx2
         i     = i+1
         x2(ix,iy,iz) = w1(i)
      enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dface_add1sa(x)
c     Compute |face-interior|
c
      include 'SIZE'
      include 'INPUT'
      real x(nx1,ny1,nz1,1)
c
      do ie=1,nelv
c
        if (if3d) then
c
          do iz=2,nz1-1
          do ix=2,nx1-1
            x(ix,1  ,iz,ie)=abs(x(ix,1  ,iz,ie) - x(ix,2    ,iz,ie))
            x(ix,ny1,iz,ie)=abs(x(ix,ny1,iz,ie) - x(ix,ny1-1,iz,ie))
          enddo
          enddo
c
          do iz=2,nz1-1
          do iy=2,ny1-1
            x(1  ,iy,iz,ie)=abs(x(1  ,iy,iz,ie) - x(2    ,iy,iz,ie))
            x(nx1,iy,iz,ie)=abs(x(nx1,iy,iz,ie) - x(nx1-1,iy,iz,ie))
          enddo
          enddo
c
          do iy=2,ny1-1
          do ix=2,nx1-1
            x(ix,iy,1  ,ie)=abs(x(ix,iy,1  ,ie) - x(ix,iy,2    ,ie))
            x(ix,iy,nz1,ie)=abs(x(ix,iy,nz1,ie) - x(ix,iy,nz1-1,ie))
          enddo
          enddo
c
        else
c
c         2D
c
          do ix=2,nx1-1
            x(ix,1  ,1,ie)=abs(x(ix,1  ,1,ie) - x(ix,2    ,1,ie))
            x(ix,ny1,1,ie)=abs(x(ix,ny1,1,ie) - x(ix,ny1-1,1,ie))
          enddo
          do iy=2,ny1-1
            x(1  ,iy,1,ie)=abs(x(1  ,iy,1,ie) - x(2    ,iy,1,ie))
            x(nx1,iy,1,ie)=abs(x(nx1,iy,1,ie) - x(nx1-1,iy,1,ie))
          enddo
c
        endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine faces(a,s,ie,iface,nx,ny,nz)
C
C     Scale face(IFACE,IE) of array A by s.
C     IFACE is the input in the pre-processor ordering scheme.
C
      INCLUDE 'SIZE'
      DIMENSION A(NX,NY,NZ,LELT)
      CALL FACIND (KX1,KX2,KY1,KY2,KZ1,KZ2,NX,NY,NZ,IFACE)
      DO 100 IZ=KZ1,KZ2
      DO 100 IY=KY1,KY2
      DO 100 IX=KX1,KX2
         A(IX,IY,IZ,IE)=S*A(IX,IY,IZ,IE)
  100 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
