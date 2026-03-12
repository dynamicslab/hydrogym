#define INTP_HMAX 20

c-----------------------------------------------------------------------
      subroutine interp_setup(ih,tolin,nmsh,nelm)
c
c input:
c tolin ... tolerance newton solve (use 0 for default)
c nmsh  ... polynomial order for mesh (use 0 to fallback to lx1-1)
c nelm  ... number of local mesh elements (typically nelt) 
c 
c output:
c ih    ... handle
c

      include 'SIZE'
      include 'INPUT'
      include 'GEOM'

      common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal

      common /intp_h/ ih_intp(2,INTP_HMAX)
      common /intp/   tol

      data ihcounter /0/
      save ihcounter

      real xmi, ymi, zmi
      common /CBXMI/ xmi(lx1*ly1*lz1*lelt),
     $               ymi(lx1*ly1*lz1*lelt),
     $               zmi(lx1*ly1*lz1*lelt)
 
      real w(2*lx1**3)

      tol = max(5e-13,tolin)
      npt_max = 128
      bb_t    = 0.01

      if (nio.eq.0) 
     $   write(6,*) 'call intp_setup ','tol=', tol

      if (nmsh.ge.1 .and. nmsh.lt.lx1-1) then
         if (nio.eq.0) write(6,*) 'Nmsh for findpts:',nmsh
         nxi = nmsh+1
         nyi = nxi
         nzi = 1
         if (if3d) nzi = nxi
         do ie = 1,nelm
           call map_m_to_n(xmi((ie-1)*nxi*nyi*nzi + 1),nxi,xm1(1,1,1,ie)
     $                     ,lx1,if3d,w,size(w))
           call map_m_to_n(ymi((ie-1)*nxi*nyi*nzi + 1),nyi,ym1(1,1,1,ie)
     $                     ,ly1,if3d,w,size(w))
           if (if3d) 
     $     call map_m_to_n(zmi((ie-1)*nxi*nyi*nzi + 1),nzi,zm1(1,1,1,ie)
     $                     ,lz1,if3d,w,size(w))
         enddo

         n = nelm*nxi*nyi*nzi 
         call fgslib_findpts_setup(ih_intp1,nekcomm,npp,ldim,
     &                             xm1,ym1,zm1,nx1,ny1,nz1,
     &                             nelm,nx1,ny1,nz1,bb_t,nelm+2,nelm+2,
     &                             npt_max,tol)

         call fgslib_findpts_setup(ih_intp2,nekcomm,npp,ldim,
     $                             xmi,ymi,zmi,nxi,nyi,nzi,
     $                             nelm,2*nxi,2*nyi,2*nzi,bb_t,n,n,
     $                             npt_max,tol)
      else
         n = nelm*nx1*ny1*nz1
         call fgslib_findpts_setup(ih_intp1,nekcomm,npp,ldim,
     &                             xm1,ym1,zm1,nx1,ny1,nz1,
     &                             nelm,2*nx1,2*ny1,2*nz1,bb_t,n,n,
     &                             npt_max,tol)
         ih_intp2 = ih_intp1
      endif

      ihcounter = ihcounter + 1 
      ih = ihcounter
      if (ih .gt. INTP_HMAX)
     $   call exitti('Maximum number of handles exceeded!$',INTP_HMAX)
      ih_intp(1,ih) = ih_intp1
      ih_intp(2,ih) = ih_intp2

      return
      end
c-----------------------------------------------------------------------
      subroutine interp_nfld(out,fld,nfld,xp,yp,zp,n,iwk,rwk,nmax,
     $                       iflp,ih)
c
c output:
c out       ... interpolation value(s) dim (n,nfld)
c
c input:
c fld       ... source field(s)
c nfld      ... number of fields
c xp,yp,zp  ... interpolation points
c n         ... number of points dim(xp,yp,zp)
c iwk       ... integer working array dim(nmax,3)
c rwk       ... real working array dim(nmax,ldim+1)
c nmax      ... leading dimension of iwk and rwk
c iflp      ... locate interpolation points (proc,el,r,s,t)
c ih        ... handle
c
      include 'SIZE'

      common /intp/   tol
      common /intp_h/ ih_intp(2,INTP_HMAX)

      real    fld(*),out(*)
      real    xp(*),yp(*),zp(*)

      logical iflp

      real    rwk(nmax,*)
      integer iwk(nmax,*) 

      integer nn(2)
      logical ifot

      if (ih.lt.1 .or. ih.gt.INTP_HMAX) 
     $   call exitti('invalid intp handle!$',ih)

      ih_intp1 = ih_intp(1,ih)
      ih_intp2 = ih_intp(2,ih)

      ifot = .false. ! transpose output field

      if (nio.eq.0 .and. loglevel.gt.2) 
     $   write(6,*) 'call intp_nfld', ih, ih_intp1, ih_intp2

      if (n.gt.nmax) then
         write(6,*)
     &    'ABORT: n>nmax in intp_nfld', n, nmax
         call exitt
      endif

      ! locate points (iel,iproc,r,s,t)
      nfail = 0
      if(iflp) then
        if(nio.eq.0 .and. loglevel.gt.2) write(6,*) 'call findpts'
        call fgslib_findpts(ih_intp2,
     &                      iwk(1,1),1,
     &                      iwk(1,3),1,
     &                      iwk(1,2),1,
     &                      rwk(1,2),ldim,
     &                      rwk(1,1),1,
     &                      xp,1,
     &                      yp,1,
     &                      zp,1,n)
        do in=1,n
           ! check return code
           if(iwk(in,1).eq.1) then
             if(rwk(in,1).gt.10*tol) then
               nfail = nfail + 1
               if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     xp(in),yp(in),zp(in),rwk(in,1)
             endif
           elseif(iwk(in,1).eq.2) then
             nfail = nfail + 1
             if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        xp(in),yp(in),zp(in)
           endif
        enddo
      endif

      ! evaluate inut field at given points
      ltot = lelt*lx1*ly1*lz1
      do ifld = 1,nfld
         iin    = (ifld-1)*ltot + 1
         iout   = (ifld-1)*n + 1
         is_out = 1
         if(ifot) then ! transpose output
           iout   = ifld
           is_out = nfld
         endif
         call fgslib_findpts_eval(ih_intp1,out(iout),is_out,
     &                            iwk(1,1),1,
     &                            iwk(1,3),1,
     &                            iwk(1,2),1,
     &                            rwk(1,2),ldim,n,
     &                            fld(iin))
      enddo

      nn(1) = iglsum(n,1)
      nn(2) = iglsum(nfail,1)
      if(nio.eq.0) then
        if(nn(2).gt.0 .or. loglevel.gt.2) write(6,1) nn(1),nn(2)
  1     format('   total number of points = ',i12,/,'   failed = '
     &         ,i12,/,' done :: intp_nfld')
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine interp_free(ih)

      common /intp_h/ ih_intp(2,INTP_HMAX)

      ih_intp1 = ih_intp(1,ih)
      ih_intp2 = ih_intp(2,ih)

      call fgslib_findpts_free(ih_intp1)
      call fgslib_findpts_free(ih_intp2)

      return
      end
c-----------------------------------------------------------------------
      subroutine intp_setup(tolin)

      call exitti('intp_setup is deprecated, see release notes!!$',1)

      return
      end
c-----------------------------------------------------------------------
      subroutine intp_do(fldout,fldin,nfld,xp,yp,zp,n,iwk,rwk,nmax,iflp)

      call exitti('intp_do is deprecated, see release notes!!$',1)

      return
      end
c-----------------------------------------------------------------------
      subroutine intp_free()

      call exitti('intp_free is deprecated, see release notes!!$',1)

      return
      end
