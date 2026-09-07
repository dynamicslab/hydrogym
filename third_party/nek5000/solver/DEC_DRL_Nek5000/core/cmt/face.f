c-----------------------------------------------------------------------
      subroutine iface_vert_int8cmt(nx,ny,nz,fa,va,jz0,jz1,nel)
      include 'SIZE'
      integer*8 fa(nx*nz,2*ndim,nel),va(0:nx+1,0:ny+1,jz0:jz1,nel)
      integer e,f

      n = nx*nz*2*ndim*nel
      call izero8(fa,n)

      mx1 = nx+2
      my1 = ny+2
      mz1 = nz+2
      if (ndim.eq.2) mz1=1

      nface = 2*ndim
      do e=1,nel
      do f=1,nface
         call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,f)

         if     (f.eq.1) then ! EB notation
            ky1=ky1-1
            ky2=ky1
         elseif (f.eq.2) then
            kx1=kx1+1
            kx2=kx1
         elseif (f.eq.3) then
            ky1=ky1+1
            ky2=ky1
         elseif (f.eq.4) then
            kx1=kx1-1
            kx2=kx1
         elseif (f.eq.5) then
            kz1=kz1-1
            kz2=kz1
         elseif (f.eq.6) then
            kz1=kz1+1
            kz2=kz1
         endif

         i = 0
         do iz=kz1,kz2
         do iy=ky1,ky2
         do ix=kx1,kx2
            i = i+1
            fa(i,f,e)=va(ix,iy,iz,e)
         enddo
         enddo
         enddo
      enddo
      enddo

      return
      end 

!-----------------------------------------------------------------------

      subroutine setup_cmt_gs(dg_hndl,nx,ny,nz,nel,melg,vertex,gnv,gnf)

!     Global-to-local mapping for gs

      include 'SIZE'
      include 'TOTAL'

      integer   dg_hndl
      integer   vertex(*)

      integer*8 gnv(*),gnf(*),ngv

      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      mx = nx+2
      call set_vert(gnv,ngv,mx,nel,vertex,.false.) ! lives in navier8.f

      mz0 = 1
      mz1 = 1
      if (if3d) mz0 = 0
      if (if3d) mz1 = nz+1
      call iface_vert_int8cmt(nx,ny,nz,gnf,gnv,mz0,mz1,nelt) 

      nf = nx*nz*2*ndim*nelt !total number of points on faces BETTA BE 4-byte!
      call gs_setup(dg_hndl,gnf,nf,nekcomm,np)

      return
      end

!-----------------------------------------------------------------------

      subroutine setup_cmt_commo
      include 'SIZE'
      include 'TOTAL'
      include 'DG'

      parameter(lf=lx1*lz1*2*ldim*lelt)
      common /c_is1/ glo_num_face(lf)
     $             , glo_num_vol((lx1+2)*(ly1+2)*(lz1+2)*lelt)
      integer*8 glo_num_face,glo_num_vol,ngv

      call setup_cmt_gs(dg_hndl,nx1,ny1,nz1,nelt,nelgt,vertex,
     >                  glo_num_vol,glo_num_face)
      call cmt_set_fc_ptr(nelt,nx1,ny1,nz1,ndg_face,iface_flux)

      return
      end

!-----------------------------------------------------------------------

      subroutine cmt_set_fc_ptr(nel,nx,ny,nz,nface,iface)

!     Set up pointer to restrict u to faces ! NOTE: compact
! JH062314 Now 2D so we can strip faces by element and not necessarily
!          from the whole field

      include 'SIZE'
      include 'TOTAL'

      integer nx, ny, nz, nel
      integer nface,iface(nx*nz*2*ldim,*)
      integer e,f,ef

      call dsset(nx,ny,nz) ! set skpdat. lives in connect1.f

      nxyz = nx*ny*nz
      nxz  = nx*nz
      nfpe = 2*ndim
      nxzf = nx*nz*nfpe ! red'd mod to area, unx, etc.

      do e=1,nel
      do f=1,nfpe

         ef     = eface(f)
         js1    = skpdat(1,f)
         jf1    = skpdat(2,f)
         jskip1 = skpdat(3,f)
         js2    = skpdat(4,f)
         jf2    = skpdat(5,f)
         jskip2 = skpdat(6,f)

         i = 0
         do j2=js2,jf2,jskip2
         do j1=js1,jf1,jskip1

            i = i+1
            k = i+nxz*(ef-1)           ! face numbering
            iface(k,e) = j1+nx*(j2-1)  ! cell numbering

         enddo
         enddo

      enddo
      enddo
      nface = nxzf*nel

      return
      end

!-----------------------------------------------------------------------

      subroutine full2face_cmt(nel,nx,ny,nz,iface,faces,vols)

! JH062314 Store face data from nel full elements (volume data). Merely
!          selection for the time being (GLL grid), but if we need to
!          extrapolate to faces (say, from Gauss points), this is where
!          we'd do it.

      include 'SIZE'
      include 'TOTAL'

      integer  nel,nx,ny,nz
      integer  iface(nx*nz*2*ldim,1)
      real     faces(nx*nz   ,2*ldim,nel)
      real     vols (nx,ny,nz       ,1  )
      integer  e,i,j

      n= nx*nz*2*ndim
      do e=1,nel
      do j=1,n
         i=iface(j,e)
         faces(j,1,e) = vols(i,1,1,e)
      enddo
      enddo

      return
      end

!-----------------------------------------------------------------------

      subroutine add_face2full_cmt(nel,nx,ny,nz,iface,vols,faces)

      include 'SIZE'
      include 'TOTAL'

      integer   nel,nx,ny,nz
      integer   iface(nx*nz*2*ldim,*)
      real     faces(nx*nz   ,2*ldim,*  )
      real     vols (nx,ny,nz       ,nel)
      integer  ie,i,j

      n= nx*nz*2*ndim
      do ie=1,nel
      do j=1,n
         i=iface(j,ie)
         vols(i,1,1,ie) = vols(i,1,1,ie)+faces(j,1,ie)
      enddo
      enddo

      return
      end
