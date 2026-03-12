c-----------------------------------------------------------------------
      subroutine get_session_info(intracomm)
      include 'mpif.h'
      include 'SIZE'
      include 'GLOBALCOM' 
      include 'TSTEP' 
      include 'INPUT' 
      
      common /happycallflag/ icall
      common /nekmpi/ nid_,np,nekcomm,nekgroup,nekreal
      integer nid_global_root(0:nsessmax-1)
      character*132 session_mult(0:nsessmax-1), path_mult(0:nsessmax-1)

      logical ifhigh

C     nsessmax = upper limit for number of sessions
C     npmax = upper limit for number of processors in each session 


C     Read from a file: number of sessions (nsessions), 
C     session name (SESSION_MULT(n-1)),
C     number of pocessors in each session (npsess(n-1)) 
C     and path (PATH_MULT(n-1))


      call mpi_initialized(mpi_is_initialized, ierr) !  Initialize MPI
      if ( mpi_is_initialized .eq. 0 ) call mpi_init (ierr)

      call mpi_comm_size(mpi_comm_world,np_global,ierr)
      call mpi_comm_rank(mpi_comm_world,nid_global,ierr)

      nid=nid_global
      nekcomm=mpi_comm_world

      ierr = 0
      if (nid_global.eq.0) then
         open (unit=8,file='SESSION.NAME',status='old',err=24)
         read(8,*) nsessions
         do n=0,nsessions-1
            call blank(session_mult(n),132)
            call blank(path_mult(n)   ,132)
            read(8,10) session_mult(n)
            read(8,10) path_mult(n)
            read(8,*)  npsess(n)
         enddo
 10      format(a132)
         close(unit=8)
         goto 23
 24      ierr = 1
      endif
 23   continue
      
      call err_chk(ierr,' Cannot open SESSION.NAME!$')

      call bcast(nsessions,4)

      if (nsessions.gt.2) 
     &  call exitti('More than 2 sessions are currently 
     &  not supported!$',1)

      do n=0,nsessions-1
         call bcast(npsess(n),4)
         call bcast(session_mult(n),132)
         call bcast(path_mult(n),132)
      enddo

      npall=0
      do n=0,nsessions-1
         npall=npall+npsess(n)
      enddo
     
C     Check if number of processors in each session is consistent 
C     with the total number of processors

      if (npall.ne.np_global) 
     &  call exitti('Wrong number of processors!$',1)

C     Assign key for splitting into multiple groups

      nid_global_root_next=0
      do n=0,nsessions-1
         nid_global_root(n)=nid_global_root_next
         nid_global_root_next=nid_global_root(n)+npsess(n)
         if (nid_global.ge.nid_global_root(n).and.
     &   nid_global.lt.nid_global_root_next) 
     &    idsess=n
      enddo
         
      call mpi_comm_split(mpi_comm_world,idsess,nid,intracomm,ierr)
 
      session = session_mult(idsess)
      path    = path_mult   (idsess)

C     Intercommunications set up only for 2 sessions 

      if (nsessions.gt.1) then

         if (idsess.eq.0) idsess_neighbor=1
         if (idsess.eq.1) idsess_neighbor=0
 
         call mpi_intercomm_create(intracomm,0,mpi_comm_world, 
     &     nid_global_root(idsess_neighbor), 10,intercomm,ierr)

         np_neighbor=npsess(idsess_neighbor)
      
         call iniproc(intracomm)

         ifhigh=.true.
         call mpi_intercomm_merge(intercomm, ifhigh, iglobalcomm, ierr)
      
         ifneknek   = .true.
         ifneknekm  = .false.

         ninter = 1 ! Initialize NEKNEK interface extrapolation order to 1.

         icall = 0  ! Emergency exit call flag

      endif 

      return
      end
C-----------------------------------------------------------------------
      subroutine multimesh_create

      include 'SIZE'
      include 'TOTAL'
      integer iList_all(ldim+2,nmaxcom)
      real    pts(ldim,nmaxcom)

      integer icalld
      save    icalld
      data    icalld  /0/

C     Set interpolation flag: points with bc = 'int' get intflag=1. 
C     Boundary conditions are changed back to 'v' or 't'.

      if (icalld.eq.0) then
         call set_intflag
         call neknekmv()
         icalld = icalld + 1
      endif 

c     Every processor sends all the points with intflag=1 to the processors 
c     of remote session
c     (send_points)

C    Root processor receives all the boundary points from remote session
C    and redistributes it among its processors for efficient 
C    localization of points (recv_points)

      call exchange_points(pts,iList_all,npoints_all)
      
C     Find which boundary points of remote session are
C     within the local mesh (intpts_locate).  
C     Communicate the point info(iList) to the remote processors
C     Points which are inside the mesh are masked and
C     their identities are stored in array 

c     write(6,*) nelgv,nid,npoints_all,' npoints_all'
      call intpts_locate (pts,iList_all,npoints_all)

      return
      end
C-------------------------------------------------------------
      subroutine set_intflag 
      include 'SIZE'
      include 'TOTAL'
      include 'NEKNEK'
      character*3 cb
      character*2 cb2
      equivalence (cb2,cb)
      integer e,f

C     Set interpolation flag: points with boundary condition = 'int' 
c     get intflag=1. 
c
C     Boundary conditions are changed back to 'v' or 't'.

      if (ifflow) then
         ifield = 1
      elseif (ifheat) then
         ifield   = 2
      endif   

      nfaces = 2*ndim
      nel    = nelfld(ifield)
      
      nflag=nel*nfaces
      call izero(intflag,nflag)

      do e=1,nel
      do f=1,nfaces
         cb=cbc(f,e,ifield)
         if (cb2.eq.'in') then
            intflag(f,e)=1
            if (ifield.eq.2) cbc(f,e,ifield)='t  '
            if (ifield.eq.1) cbc(f,e,ifield)='v  '
            if (cb.eq.'inp') cbc(f,e,ifield)='o  ' ! Pressure
         endif
      enddo
      enddo

      return
      end
c-------------------------------------------------------------
      subroutine exchange_points(pts,iList_all,npoints_all)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'NEKNEK'
      include 'mpif.h'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
      real    pts(ldim,nmaxcom)   
      integer jsend((ldim+2)*nmaxl),jrecv((ldim+2)*nmaxl)
      common /exchr/ rsend(ldim*nmaxl), rrecv(ldim*nmaxl) 
      integer iList_all(ldim+2,nmaxcom)
    
C     Look for boundary points with Diriclet b.c. (candidates for
C     interpolation)

      
      if (ifflow) then
         ifield = 1
      elseif (ifheat) then
         ifield   = 2
      endif   

      nfaces = 2*ndim
      nel    = nelfld(ifield)

      ip = 0
      do 2010 iel=1,nel
      do 2010 iface=1,nfaces
         if (intflag(iface,iel).eq.1) then
            call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx1,ny1,nz1,iface)
            do 100 iz=kz1,kz2
            do 100 iy=ky1,ky2
            do 100 ix=kx1,kx2
               call nekasgn (ix,iy,iz,iel)
               ip=ip+1

               if (if3d) then
                 jsend((ldim+2)*(ip-1)+1)=ix
                 jsend((ldim+2)*(ip-1)+2)=iy
                 jsend((ldim+2)*(ip-1)+3)=iz
                 jsend((ldim+2)*(ip-1)+4)=iel
                 jsend((ldim+2)*(ip-1)+5)=nid

                 rsend(ldim*(ip-1)+1)=x
                 rsend(ldim*(ip-1)+2)=y
                 rsend(ldim*(ip-1)+3)=z
               else
                 jsend((ldim+2)*(ip-1)+1)=ix
                 jsend((ldim+2)*(ip-1)+2)=iy
                 jsend((ldim+2)*(ip-1)+3)=iel
                 jsend((ldim+2)*(ip-1)+4)=nid

                 rsend(ldim*(ip-1)+1)=x
                 rsend(ldim*(ip-1)+2)=y
               endif

               if (ip.gt.nmaxl) then
                  write(6,*) nid,
     &            ' ABORT: nbp (current ip) too large',ip,nmaxl
                  call exitt
               endif

  100       continue
         endif
 2010 continue

      nbp = ip

      call izero(iList_all,(ldim+2)*nmaxcom)
      call rzero(pts,  ldim*nmaxcom)

C     Send amount of points boundary points, its coordinates and 
c     its local identities 
C     to the the processors of remote session (for load balancing) 

C     Determine amount of points to send to each processor

      ndistrib = nbp/np_neighbor
      iremainder = nbp - ndistrib*np_neighbor

      ip=0
      iaddress=1
      iraddress=1

      do id=0,np_neighbor-1
        if(id.le.iremainder-1) then
            nsend = ndistrib + 1
            else 
            nsend = ndistrib
        endif

         len=isize
         call mpi_irecv (nrecv,len,mpi_byte, id, id, 
     &                                    intercomm, msg,ierr)
         call mpi_send  (nsend,len,mpi_byte, id, nid, 
     &                                    intercomm, ierr)

         call mpi_wait (msg,status,ierr)

         len=(ldim+2)*nrecv*isize
         call mpi_irecv (jrecv, len ,mpi_byte, id, 100+id, 
     &                                     intercomm, msg, ierr)
         len=(ldim+2)*nsend*isize
         call mpi_send (jsend(iaddress),len,mpi_byte, id, 100+nid, 
     &                                           intercomm, ierr)
         iaddress=iaddress+(ldim+2)*nsend

         call mpi_wait (msg, status, ierr)

         len=ldim*nrecv*wdsize
         call mpi_irecv (rrecv, len ,mpi_byte, id, 200+id, 
     &                                  intercomm, msg, ierr)
         len=ldim*nsend*wdsize
         call mpi_send (rsend(iraddress),len,mpi_byte, id, 200+nid, 
     &                                        intercomm, ierr)
         iraddress=iraddress+ldim*nsend

         call mpi_wait (msg, status, ierr)

         do n=1,nrecv
            ip = ip + 1

            if (ip.gt.nmaxcom) then
               write(6,*) nid,'ABORT: increase nmaxcom',ip,nmaxcom
               call exitt
            endif

            
           do j=1,ldim+2
              iList_all(j,ip)=jrecv((ldim+2)*(n-1)+j)
           enddo
           do j=1,ldim
               pts(j,ip)=rrecv(ldim*(n-1)+j)
           enddo

         enddo ! n

      enddo    ! id

      call neknekgsync()

      npoints_all = ip    

      return
      end
C-------------------------------------------------------------------------
      subroutine intpts_locate (pts,iList_all,npoints_all)

      include 'SIZE'
      include 'TOTAL'
      include 'NEKNEK'
      include 'mpif.h'  
      integer jsend((ldim+1)*nmaxl),jrecv((ldim+1)*nmaxl)
      integer rcode_all(nmaxcom),elid_all(nmaxcom),proc_all(nmaxcom)
      real    dist(nmaxcom)
      real    pts(ldim,nmaxcom)
      real    dist_all(nmaxcom)
      real    rst_all(nmaxcom*ldim)
      integer iList_all(ldim+2,nmaxcom)
      character*3 CB
      integer status(mpi_status_size)
      logical ifcomm

      integer icalld
      save    icalld
      data    icalld /0/

      if (icalld.le.1) then
         icalld=icalld+1
      else
         call intpts_done(inth_multi)
      endif

      call intpts_setup(-1.0,inth_multi) ! use default tolerance
        

      call findpts(inth_multi,rcode_all,1,
     &             proc_all,1,
     &             elid_all,1,
     &             rst_all,ndim,
     &             dist_all,1,
     &             pts(1,1),ndim,
     &             pts(2,1),ndim,
     &             pts(3,1),ndim,npoints_all)

      ierror=0

c     Rearrange arrays so that only the points which are found within 
c     the mesh are marked for interpolation: those are truly internal 
c     points (rcode_all=0) plus some of the "boundary points" 
c     (rcode_all=1) which fall on the boundaries of the elements 
c     of another domain (essentially also internal points) or 
c     wall or periodic boundaries 
c
c     Points with rcode_all(i)=1 which fall on the "interpolation" 
c     boundaries of another domain are treated with boundary 
c     conditions and not with interpolation routines, and are 
c     therefore excluded.
c
c     rcode_all, proc_all ... npoints_all represent all boundary points,
c     rcode, proc ... npoints - only interpolation points 
c     (used by interpolation routines)
c     Contained in a common block /multipts_r/, /multipts_i/ (in NEKNEK)

C

C     Exclude "true" boundary points 

      if (ifflow) then
         ifield = 1
      elseif (ifheat) then
         ifield   = 2
      endif 

      nel   = nelfld(ifield)
      nxyz  = nx1*ny1*nz1
      ntot  = nxyz*nel

      call izero(imask,ntot) 

      ip=0
      icount=0
      ierror=0

      do 100 i=1,npoints_all
         
      if (rcode_all(i).lt.2) then

        iel  = elid_all(i) + 1

        if (rcode_all(i).eq.1.and.dist_all(i).gt.1e-02) then
           if (ndim.eq.2) write(6,'(A,3E15.7)') 
     &     'WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     (pts(k,i),k=1,ndim),dist_all(i)  
           if (ndim.eq.3) write(6,'(A,4E15.7)') 
     &     'WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     (pts(k,i),k=1,ndim),dist_all(i)  
           goto 100
         endif
         ip=ip+1
         rcode(ip) = rcode_all(i)
         elid(ip)  = elid_all(i)
         proc(ip)  = proc_all(i)
         do j=1,ndim
           rst(ndim*(ip-1)+j)   = rst_all(ndim*(i-1)+j)
         enddo
         do n=1,ndim+1
           iList(n,ip) = iList_all(n,i)
         enddo

c        Check receiving remote processor id information 
         n=ndim+2
c        First point 
         if (ip.eq.1) then
            icount=1
            infosend(icount,1)=iList_all(n,i)
            infosend(icount,2)=ip
            iprocp=iList_all(n,i)
         else
            iproc  = iList_all(n,i)
            if (iproc.ne.iprocp) then
             if (iproc.gt.iprocp) then
               icount=icount+1
               infosend(icount,1)=iproc
               infosend(icount,2)=ip
               iprocp=iproc
             else
               write(6,*) 'Attention: intrinsic sorting is not achieved'
               write(6,'(a7,3i7,1x,a10)') 'switch', ip, 
     &         iproc,iprocp, session
               ierror = 1
               goto 100
             endif
            endif
         endif
      
      endif  !  rcode_all

 100  continue

      call iglmax(ierror,1)
      if (ierror.eq.1) call exitt

      npoints=ip
      npsend=icount

c     Store point counts (infosend(:,2)) instead of pointers 

      do n=2,npsend
         ip=infosend(n-1,2)
         ic=infosend(n,2)
         infosend(n-1,2)=ic-ip
      enddo
      infosend(npsend,2)=npoints-infosend(npsend,2)+1

      if (istep.eq.0) write(6,'(a7,i12,1x,a10)') 'found', npoints, 
     &                     session

c     Sending all processors information about the number 
c     of receiving points, including zero points
c     ifcomm=.false. means there are no more processors
c     with non-zero receiving points (either if npsend=0 
c     or if we already notified all the relevant processors)

      if (npsend.eq.0) then
         ifcomm=.false.
      else
         ifcomm=.true.
         icount=1
      endif   

      il=0
      do id=0,np_neighbor-1

         if (ifcomm) then

            idsend=infosend(icount,1)

            if (id.lt.idsend) then
               nsend=0
            elseif (id.eq.idsend) then
               nsend=infosend(icount,2)
               icount =icount+1
            endif

         else   !  not(ifcomm)
            nsend=0
         endif

         if (ifcomm.and.(icount.gt.npsend)) ifcomm=.false.

         len=isize   
         call mpi_send (nsend,len,mpi_byte, id, nid, intercomm, ierr)

         if (nsend.ne.0) then
c           Send the points identity to communicating processors
            len=(ldim+1)*nsend*isize
               do i=1,nsend
               il=il+1
               do j=1,ldim+1
                  jsend((ldim+1)*(i-1)+j)=iList(j,il)
               enddo
            enddo   

            call mpi_send(jsend,len,mpi_byte,id,100+nid, intercomm,ierr)

         endif

      enddo   


c     Receive information about sending processors
 6    irecvcnt=0
      il=0
      do id=0,np_neighbor-1
         len=isize

         call mpi_recv(nrecv,len,mpi_byte,id,id,intercomm,status,ierr)

         if (nrecv.ne.0) then
            irecvcnt=irecvcnt+1
            inforecv(irecvcnt,1)=id
            inforecv(irecvcnt,2)=nrecv

c     Receive information about point identities
            len=(ldim+1)*nrecv*isize
            call mpi_recv
     $        (jrecv,len,mpi_byte,id,100+id,intercomm,status,ierr)
         
c     Receiving processor nid masks which points he receives from 
c     remote processor id as interpolation points using the identity 
c     information contained in array 'jrecv'. Those points are masked 
c     with imask=1 (imask=0 for all other points). 

      do ii=1,nrecv
         ix=jrecv((ldim+1)*(ii-1)+1)
         iy=jrecv((ldim+1)*(ii-1)+2)
         iz = 1
         if (if3d) iz=jrecv((ldim+1)*(ii-1)+3)
         ie=jrecv((ldim+1)*(ii-1)+ldim+1)
         il=il+1
            iden(1,il)=ix
            iden(2,il)=iy
            if (if3d) iden(3,il)=iz
            iden(ldim+1,il)=ie
            imask(ix,iy,iz,ie)=1
         enddo

      endif  !  jrecv.ne.0
      
      enddo  !  all remote processors

      nprecv=irecvcnt

      call neknekgsync()
      return
      end
C----------------------------------------------------------------
      subroutine get_values(which_field)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKNEK'
      include 'mpif.h' 

      parameter (lt=lx1*ly1*lz1*lelt,lxyz=lx1*ly1*lz1)
      common /scrcg/ pm1(lt),wk1(lxyz),wk2(lxyz)

      character*3 which_field(nfld_neknek)
      real field(lx1*ly1*lz1*lelt,nfldmax)
      real rsend(nfldmax*nmaxl), rrecv(nfldmax*nmaxl)
      real fieldout(nfldmax,nmaxl)

      integer status(mpi_status_size)

c     Information about communication of points is contained in common 
c     block /proclist/.
 
C     iden(1,:) = ix
C     iden(2,:) = iy
C     iden(3,:) = iz
C     iden(4,:) = iel


c     Put field values for the field ifld=1,nfld_neknek to the working array 
c     field (:,:,:,:,nfld_neknek) used byt the findpts_value routine according 
c     to the field identificator which_field(ifld) 

      if (nfld_neknek.eq.0) 
     $ call exitti('Error: set nfld_neknek in usrchk. Session:$',idsess)
      

      call mappr(pm1,pr,wk1,wk2)  ! Map pressure to pm1 

      nv = nx1*ny1*nz1*nelv
      nt = nx1*ny1*nz1*nelt

      do ifld=1,nfld_neknek
        if (which_field(ifld).eq.'t' ) call copy(field(1,ifld),t  ,nt)
        if (which_field(ifld).eq.'vx') call copy(field(1,ifld),vx ,nt)
        if (which_field(ifld).eq.'vy') call copy(field(1,ifld),vy ,nt)
        if (which_field(ifld).eq.'vz') call copy(field(1,ifld),vz ,nt)
        if (which_field(ifld).eq.'pr') call copy(field(1,ifld),pm1,nt)

C       Find interpolation values      
         call findpts_eval(inth_multi,fieldout(ifld,1),nfldmax,
     &                     rcode,1,
     &                     proc,1,
     &                     elid,1,
     &                     rst,ndim,npoints,
     &                     field(1,ifld))

       enddo  

C     Send interpolation values to the corresponding processors 
C     of remote session

      call neknekgsync()

      il=0
      do n=1,npsend   
         id       = infosend(n,1)
         nsend    = infosend(n,2)
         do i=1,nsend
            il=il+1
            do ifld=1,nfld_neknek
               rsend(nfld_neknek*(i-1)+ifld)=fieldout(ifld,il)
           enddo
         enddo
         len=nfld_neknek*nsend*wdsize
         call mpi_send (rsend,len,mpi_byte, id, nid, intercomm, ierr)
      enddo

      il=0 
      do n=1,nprecv
         id    = inforecv(n,1)
         nrecv = inforecv(n,2)
         len=nfld_neknek*nrecv*wdsize
         call mpi_recv (rrecv,len,mpi_byte,id,id,intercomm,status,ierr)
         do i=1,nrecv ! Extract point identity
            il=il+1
            ix = iden(1,il)
            iy = iden(2,il)
            iz = 1
            if (if3d) iz=iden(3,il)
            ie=iden(ldim+1,il)      
            do ifld=1,nfld_neknek
               valint(ix,iy,iz,ie,ifld)=rrecv(nfld_neknek*(i-1)+ifld)
            enddo
         enddo
      enddo 

      call neknekgsync()

      return
      end
C--------------------------------------------------------------------------
      subroutine userchk_set_xfer
      include 'SIZE'
      include 'TOTAL'
      include 'NEKNEK'
      real l2,linf
      character*3 which_field(nfldmax+1)

c     nfld_neknek is the number of fields to interpolate.
c     nfld_neknek = 3 for just veliocities, nfld_neknek = 4 for velocities + temperature

      which_field(1)='vx'
      which_field(2)='vy'
      which_field(3)='vz'
      which_field(ndim+1)='pr'
      if (nfld_neknek.gt.ndim+1) which_field(ndim+2)='t'

      if (nsessions.gt.1) call get_values(which_field)

      return
      end
c------------------------------------------------------------------------
      subroutine bcopy
      include 'SIZE'
      include 'TOTAL'
      include 'NEKNEK'

      n    = nx1*ny1*nz1*nelt

      do k=1,nfld_neknek
         call copy(bdrylg(1,k,2),bdrylg(1,k,1),n)
         call copy(bdrylg(1,k,1),bdrylg(1,k,0),n)
         call copy(bdrylg(1,k,0),valint(1,1,1,1,k),n)
      enddo

c     Order of extrpolation is contolled by the parameter NINTER contained 
c     in NEKNEK. First order interface extrapolation, NINTER=1 (time lagging) 
c     is activated. It is unconditionally stable.  If you want to use 
c     higher-order interface extrapolation schemes, you need to increase 
c     ngeom to ngeom=3-5 for scheme to be stable.

      if (NINTER.eq.1.or.istep.eq.0) then
       c0=1
       c1=0
       c2=0
       else if (NINTER.eq.2.or.istep.eq.1) then
         c0=2
         c1=-1
         c2=0
       else 
         c0=3
         c1=-3
         c2=1
      endif
     
      do k=1,nfld_neknek
      do i=1,n
         ubc(i,1,1,1,k) = 
     $      c0*bdrylg(i,k,0)+c1*bdrylg(i,k,1)+c2*bdrylg(i,k,2)
      enddo
      enddo

      return
      end
C---------------------------------------------------------------------
      subroutine setintercomm(nekcommtrue,nptrue) 
      include 'SIZE' 
      include 'GLOBALCOM'
      common /nekmpi/ nid_,np,nekcomm,nekgroup,nekreal 
      
      nekcommtrue=nekcomm
      nekcomm=iglobalcomm

      nptrue=np
      np=np_global

      return
      end
c-----------------------------------------------------------------------
      subroutine unsetintercomm(nekcommtrue,nptrue)
      include 'SIZE' 
      include 'GLOBALCOM'
      common /nekmpi/ nid_,np,nekcomm,nekgroup,nekreal 

      nekcomm=nekcommtrue
      np=nptrue

      return
      end
c-----------------------------------------------------------------------
      function uglmin(a,n)
      real a(1)

      call happy_check(1)
      call setintercomm(nekcommtrue,nptrue)    ! nekcomm=iglobalcomml
      uglmin=glmin(a,n)
      call unsetintercomm(nekcommtrue,nptrue)  ! nekcomm=nekcomm_original

      return
      end
c-----------------------------------------------------------------------
      function uglamax(a,n)
      real a(1)

      call happy_check(1)
      call setintercomm(nekcommtrue,nptrue)    ! nekcomm=iglobalcomml
      uglamax=glamax(a,n)
      call unsetintercomm(nekcommtrue,nptrue)  ! nekcomm=nekcomm_original

      return
      end
c------------------------------------------------------------------------
      subroutine neknekgsync()
      include 'SIZE' 
      include 'GLOBALCOM'

      call happy_check(1)
      call mpi_barrier(intercomm,ierr)
      return
      end
c------------------------------------------------------------------------
      subroutine happy_check(ihappy)
      include 'SIZE'
      include 'TOTAL'
      common /happycallflag/ icall

c     Happy check
      call setintercomm(nekcommtrue,nptrue)    ! nekcomm=iglobalcomml
      iglhappy=iglmin(ihappy,1)
      call unsetintercomm(nekcommtrue,nptrue)  ! nekcomm=nekcomm_original
      if (ihappy.eq.1.and.iglhappy.eq.0) then
         if (nid.eq.0) then
         write (6,*) '       '
         write (6,'(A,1i7,A,1e13.5)') 
     $   ' Emergency exit due to the other session:',
     $     ISTEP,'   time =',TIME
         write (6,*)   
         endif
         icall=1
       call exitt
      endif 

      return
      end
c------------------------------------------------------------------------
      subroutine chk_outflow_short ! Assign neighbor velocity to outflow

c
c     This is just an experimental routine for PnPn only...
c


      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'NEKNEK'
      integer e,eg,f

      n = nx1*ny1*nz1*nelt
      ipfld=ndim
c     if (ifsplit) ipfld=ndim+1
      ipfld=ndim+1                ! Now for both split and nonsplit (12/14/15)
      itfld=ipfld+1

      do i=1,n ! The below has not been checked for ifheat=.true., pff 6/27/15
         if (imask(i,1,1,1).eq.1) then
            vx(i,1,1,1) = ubc(i,1,1,1,1)
            vy(i,1,1,1) = ubc(i,1,1,1,2)
            if (if3d)    vz(i,1,1,1)  = ubc(i,1,1,1,3)
            if (ifsplit) pr(i,1,1,1)  = ubc(i,1,1,1,ipfld)
            if (ifheat)  t(i,1,1,1,1) = ubc(i,1,1,1,itfld)
         endif
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine chk_outflow ! Assign neighbor velocity to outflow if
                             ! characteristics are going the wrong way.

      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'NEKNEK'
      integer e,eg,f

      nface = 2*ndim
      do e=1,nelv
      do f=1,nface
         if (cbc(f,e,1).eq.'o  ') then
           eg = lglel(e)
           call facind (i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)
           l=0
           do k=k0,k1
           do j=j0,j1
           do i=i0,i1
              l=l+1
              vo=vx(i,j,k,e)*unx(l,1,f,e)
     $          +vy(i,j,k,e)*uny(l,1,f,e)
     $          +vz(i,j,k,e)*unz(l,1,f,e)
              if (vo.lt.0) then            ! We have inflow
                 cbu = cbc(f,e,1)
                 call userbc(i,j,k,f,eg)
                 vx(i,j,k,e) = ux
                 vy(i,j,k,e) = uy
                 vz(i,j,k,e) = uz
              endif
           enddo
           enddo
           enddo
         endif
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine neknekmv()
      include 'SIZE'
      include 'TOTAL'

      integer imove

      imove=1
      if (ifmvbd) imove=0
      call neknekgsync()

      call setintercomm(nekcommtrue,nptrue)    ! nekcomm=iglobalcomml
      iglmove=iglmin(imove,1)
      call unsetintercomm(nekcommtrue,nptrue)  ! nekcomm=nekcomm_original

      if (iglmove.eq.0) then
         ifneknekm=.true.
      endif

      return
      end
c-----------------------------------------------------------------------

