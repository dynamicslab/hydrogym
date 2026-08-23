c Various support routines for perturbation related calculuations.
c------------------------------------------------------------------------
      subroutine flushBuffer(k)
      integer k

c     call flush(k)  ! this works most places (including the pgi compiler)
      call flush_(k) ! this works on nersc seaborg

      return
      end
c------------------------------------------------------------------------

      function opnormOld(v1,v2,v3,weight)
      include 'SIZE'
      include 'INPUT'
c
      real v1(1),v2(1),v3(1),weight(1)
      real normsq1,normsq2,normsq3,opnormOld,opnorm
c
      ntotv=nx1*ny1*nz1*nelv
      normsq1=glsc3(v1,weight,v1,ntotv)
      normsq2=glsc3(v2,weight,v2,ntotv)
      if(if3d) then
         normsq3=glsc3(v3,weight,v3,ntotv)
      else
         normsq3=0
      endif

      opnorm=normsq1+normsq2+normsq3
      if (opnorm.gt.0) opnormOld=sqrt(opnorm)
      return
      end
c-----------------------------------------------------------------------

      function opnorm2(v1,v2,v3)
      include 'SIZE'
      include 'TOTAL'
c
      real v1(1) , v2(1), v3(1)
      real normsq1,normsq2,normsq3,opnorm
c
      ntotv=nx1*ny1*nz1*nelv
      normsq1=glsc3(v1,bm1,v1,ntotv)
      normsq2=glsc3(v2,bm1,v2,ntotv)
      if(if3d) then
         normsq3=glsc3(v3,bm1,v3,ntotv)
      else
         normsq3=0
      endif

      opnorm2=normsq1+normsq2+normsq3
      if (opnorm2.gt.0) opnorm2=sqrt(opnorm2/volvm1)
      return
      end
c-----------------------------------------------------------------------

      function Tnorm(temp)
      include 'SIZE'
      include 'TOTAL'

      real temp(*)
c
      ntotv = nx1*ny1*nz1*nelv
      Tnorm = sqrt( glsc3(temp,BM1,temp,ntotv) /voltm1)
c
      return
      end
c--------------------------------------------
      function dmnorm(v1,v2,v3,temp)
c     Norm weighted by mass matrix
      include 'SIZE'
      include 'TOTAL'

      real v1(1),v2(1),v3(1),temp(1)
      real normsq1,normsq2,normsq3,normsqT,dMnorm
      common/normset/pran, ray, rayc

      ntotv=nx1*ny1*nz1*nelv
      normsq1=(rayc)*glsc3(v1,BM1,v1,ntotv)
      normsq2=(rayc)*glsc3(v2,BM1,v2,ntotv)
      if(if3d) then
         normsq3=(rayc)*glsc3(v3,BM1,v3,ntotv)
      else
         normsq3=0
      endif

      if(ifheat) then
         normsqT = (pran*ray*ray)*glsc3(temp,BM1,temp,ntotv)
      else
         normsqT = 0
      endif

      dmnorm=sqrt((normsq1+normsq2+normsq3+normsqT)/volvm1)

      return
      end

c---------------------------------------------------------------
      subroutine opscale(v1,v2,v3,temp,alpha)
c     v =  alpha*v
      include 'SIZE'
      include 'INPUT'

      real alpha
      real v1(1),v2(1),v3(1),temp(1)

      ltotv=lx1*ly1*lz1*lelv
      ltott=lx1*ly1*lz1*lelt

      call cmult(v1,alpha,ltotv)
      call cmult(v2,alpha,ltotv)
      if (if3d)   call cmult(v3,alpha,ltotv)
      if (ifheat) call cmult(temp,alpha,ltott*ldimt)

      return
      end

c---------------------------------------------------------------
      subroutine opscaleV(v1,v2,v3,alpha)
c     v =  alpha*v
      include 'SIZE'
      include 'INPUT'
      real alpha
      real v1(*),v2(*),v3(*)

      ntotv=nx1*ny1*nz1*nelv

      call cmult(v1,alpha,ntotv)
      call cmult(v2,alpha,ntotv)

      if (if3d)   call cmult(v3,alpha,ntotv)
c
      return
      end

c-----------------------------------------------------------------------
      subroutine computelyap
      include 'SIZE'
      include 'TOTAL'

      do jpp=1,npert         ! Loop through each Lyapunov eigenvalue
         call computelyap1
     $     (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),jpp)
      enddo

      return
      end
c-----------------------------------------------------------------------

      subroutine computelyap1(vxq,vyq,vzq,tq,jpp)
      include 'SIZE'
      include 'TOTAL'

      real vxq(1),vyq(1),vzq(1),tq(1)
      real lyapsum,twt
      common /pertsave/ timestart,tinitial

      integer icount
      save    icount
      data    icount /0/

      logical       if_restart,if_ortho_lyap
      common/restar/if_restart,if_ortho_lyap

      character*132 lyprestart
      common/restflename/lyprestart  !file for restart data

      twt = param(126) !time to wait to start computing exponents

      if (nio.eq.0) 
     $  write(6,8) istep,icount,time,twt,(lyap(k,jpp),k=1,3),jpp
    8 format(i9,i4,2f8.4,1p3e12.4,i3,'clyap')

      if(time.lt.twt) then
c
c        For a  fresh simulation, then we wait 5 vertical diffusion 
c        times before we start measuring, so in this case just rescale
c
         pertnorm = dmnorm(vxq,vyq,vzq,tq)
         pertinvnorm = 1.0/pertnorm
         call rescalepert(pertnorm,pertinvnorm,jpp)
         lyap(3,jpp) = pertnorm !store latest norm
         timestart   = time
         tinitial    = time
         icount      = 0
         return
      else
         if (jpp.eq.1) icount = icount + 1
      endif

      irescale = param(128)
      if (icount.eq.irescale) then

         lyapsum     = lyap(2,jpp)
         oldpertnorm = lyap(3,jpp)
         pertnorm=dmnorm(vxq,vyq,vzq,tq)
c
         lyap(1,jpp) = log(pertnorm/oldpertnorm)/(time-timestart)
         lyapsum     = lyapSum + log(pertnorm/oldpertnorm)
         lyap(2,jpp) = lyapSum

         if(nid.eq.0) then        ! write out results to the .lyp file
 
           write(6 ,1) istep,time,lyap(1,jpp),lyapsum,pertnorm,jpp
           write(79,2) time,lyap(1,jpp),lyapsum,pertporm,oldpertnorm,jpp
 1         format(i9,1p4e17.8,i4,'lyap')
 2         format(1p5e17.8,i4,'lyap')
           call flushbuffer(79)
 
           if (jpp.eq.1) open(unit=96,file=lyprestart)
           write(96,9) lyapsum,timestart,timeinit,jpp
 9         format(1p3e19.11,i9)
           if (jpp.eq.npert) close(96)
         endif

         pertinvnorm = 1.0/pertnorm
         call rescalepert(pertnorm,pertinvnorm,jpp)
         lyap(3,jpp) = pertnorm  !save current pertnorm as old pertnorm

         if (jpp.eq.npert) then
            icount    = 0
            timestart = time
         endif

         if_ortho_lyap = .false.
         if (param(125).gt.0) if_ortho_lyap = .true.
         if (jpp.eq.npert .and. if_ortho_lyap) call pert_ortho_norm

      endif

      return
      end
c-----------------------------------------------------------------------

      subroutine rescalepert(pertnorm,pertinvnorm,jpp)
      include 'SIZE'
      include 'TOTAL'

      ntotp = nx2*ny2*nz2*nelv

      call opscale                     !normalize vectors to unit norm
     $      (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),pertinvnorm)
      call cmult(prp(1,jpp),pertinvnorm,ntotp)

      call opscale(exx1p(1,jpp),exy1p(1,jpp),exz1p(1,jpp)
     $                           ,vgradt1p(1,1,jpp),pertinvnorm)
      call opscale(exx2p(1,jpp),exy2p(1,jpp),exz2p(1,jpp)
     $                           ,vgradt2p(1,1,jpp),pertinvnorm)

      ltotv = lx1*ly1*lz1*lelv
      ltotp = lx2*ly2*lz2*lelv

      call cmult( tlagp(1,1,1,jpp),pertinvnorm,ltotv*(lorder-1)*ldimt)
      call cmult(vxlagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(vylagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(vzlagp(1,1,jpp),pertinvnorm,ltotv*(lorder-1))
      call cmult(prlagp(1,1,jpp),pertinvnorm,ltotp*(Lorder-2))

      if (nio.eq.0) write(6,1) istep,pertnorm,pertinvnorm,jpp,'PNORM'
  1   format(i4,1p2e12.4,i4,a5)
      pertnorm = pertnorm*pertinvnorm

      return
      end
c-----------------------------------------------------------------------

      subroutine writehist(v1,v2,v3,temp,jpp)
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'GEOM'
      INCLUDE 'CTIMER'
      INCLUDE 'IXYZ'
      INCLUDE 'INPUT'
      INCLUDE 'MASS'
      INCLUDE 'PARALLEL'
      INCLUDE 'SOLN'
      real hdump(25)
      real  v1(nx1,ny1,nz1,nelv), v2(nx1,ny1,nz1,nelv)
     $    , v3(nx1,ny1,nz1,nelv), temp(nx1,ny1,nz1,nelv)

      IF (NHIS.GT.0) THEN
         IPART=0
         DO I=1,NHIS
            NVAR=0
            IX =LOCHIS(1,I)
            IY =LOCHIS(2,I)
            IZ =LOCHIS(3,I)
            IEG=LOCHIS(4,I)
            JNID=GLLNID(IEG)
            IE  =GLLEL(IEG)
            
            IF(HCODE(1,I).EQ.'U')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=V1(IX,IY,IZ,IE)
            ELSEIF(HCODE(1,I).EQ.'X')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=XM1(IX,IY,IZ,IE)
            ENDIF
            IF(HCODE(2,I).EQ.'V')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=V2(IX,IY,IZ,IE)
            ELSEIF(HCODE(2,I).EQ.'Y')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=YM1(IX,IY,IZ,IE)
            ENDIF
            IF(HCODE(3,I).EQ.'W')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=V3(IX,IY,IZ,IE)
            ELSEIF(HCODE(3,I).EQ.'Z')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=ZM1(IX,IY,IZ,IE)
            ENDIF
            IF(HCODE(5,I).EQ.'T')THEN
               NVAR=NVAR+1
               HDUMP(NVAR)=temp (IX,IY,IZ,IE)
            ENDIF

C--------------------------------------------------------------
C     Dump out the NVAR values for this history point
C--------------------------------------------------------------
            MTYPE=7000+I
            LEN=WDSIZE*NVAR
C     
C     If point on this processor, send data to node 0
            IF (NVAR.GT.0.AND.NID.NE.0.AND.JNID.EQ.NID) 
     $           CALL CSEND(MTYPE,HDUMP,LEN,NODE0,NULLPID)
C     
C     If processor 0, recv data (unless point resides on node0).
            IF (NVAR.GT.0.AND.NID.EQ.0.AND.JNID.NE.NID)
     $           CALL CRECV(MTYPE,HDUMP,LEN)
C     
            IF (NVAR.GT.0.AND.NID.EQ.0) then
               WRITE(146,'(1p8e16.8)')TIME,(HDUMP(II),II=1,NVAR)
               call flushBuffer(146)
            endif
C     
         enddo
      endif
      return
      end

c----------------------------------------------------------------------
      subroutine initialize
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      do jpp = 1,npert
         call initialize1(jpp)
      enddo

      return
      end
c----------------------------------------------------------------------
      subroutine initialize1(jpp)
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      real tmp(4),lyapSum,timeStart
      common/pertsave/timeStart,tinitial
      common/normset/pran,ray,rayc
      character*1 lyp(4),sch(5)
      character*4 lyp4
      character*5 sch5
      character*7 lypres,lypold
      CHARACTER*1 SESS1(132),PATH1(132),NAM1(132)
      EQUIVALENCE (SESSION,SESS1)
      EQUIVALENCE (PATH,PATH1)
      EQUIVALENCE (NAME,NAM1)
      equivalence (lyp,lyp4)
      equivalence (sch,sch5)
      data lyp4 /'.lyp'/
      data sch5 /'.schp'/
      data lypold /'.lypold'/
      data lypres /'.lypsum'/
      character*132 lypfile,lypfileold,lyprestart,lypsch
      common/restflename/lyprestart      !file where restart data is written
      logical if_restart
      common/restar/if_restart
      logical ifdebug
      common/debug/ ifdebug

      ntotv     = nx1*ny1*nz1*nelv
      ntotp     = nx2*ny2*nz2*nelv
      nconv_max = nbdinp
      nconv     = nconv_max 
      nconv     = 1
      nprex     = 0
      nprex_max = max(0,nbdinp-2)
     
      nr = nx1-1
      ns = ny1-1
      nt = nz1-1

      if(param(31).eq.0) return !param(31) is number of exponents
      
      tinitial = time

      call opzero(bfxp(1,jpp),bfyp(1,jpp),bfzp(1,jpp))

      itmp = lx1*ly1*lz1*lelv*(lorder-1)

      call rzero(vxlagp(1,1,jpp),itmp)
      call rzero(vylagp(1,1,jpp),itmp)
      if(if3d)   call rzero(vzlagp(1,1,jpp),itmp)
      if(ifheat) call rzero(tlagp (1,1,1,jpp),itmp*LdimT)

      call opzero(exx1p(1,jpp),exy1p(1,jpp),exz1p(1,jpp))
      call opzero(exx2p(1,jpp),exy2p(1,jpp),exz2p(1,jpp))

      pertvnorm = sqrt(rayc)*opnorm2(vxp(1,jpp),vyp(1,jpp),vzp(1,jpp))
      perttnorm = sqrt(pran*ray*ray)*tnorm(tp(1,1,jpp))
      pertnorm  = sqrt(pertvnorm**2+perttnorm**2)

      if(param(64).ne.1) then !fresh start, param(64) is restart flag
         call opzero(vxp(1,jpp),vyp(1,jpp),vzp(1,jpp))
         call rzero(tp(1,1,jpp),ntotv)
         call get_useric
         if_restart = .false.
         call rzero(lyap(1,jpp),3)
      else
         if(nio.eq.0) write(6,*)jpp,'norm of pert IC=',pertnorm
         if_restart  = .true.
         pertInvNorm = 1.0/pertNorm
         call rescalepert(pertnorm,pertinvnorm,jpp)
         if(nio.eq.0) write(6,*) jpp,'Rescaled the perturbation'
         call rzero(lyap(1,jpp),3)
         lyap(3,jpp) = pertnorm
      endif
c
      ifdebug=.false.

c     opening the lyp file in unit=79

      call blank(lypfile,132)
      call blank(lypfileold,132)
      call blank(lyprestart,132)
      call blank(lypsch,132)

      ls  = ltrunc(session,132)
      lpp = ltrunc(path,132)

      call chcopy(nam1(    1),path1,lpp)
      call chcopy(nam1(lpp+1),sess1,ls )
      l1 = lpp+ls+1
      ln = lpp+ls+4
         
      call chcopy(nam1  (l1),lyp , 4)
      call chcopy(lypfile    ,nam1,ln)

      lns = lpp+ls+5
      call chcopy(nam1 (l1),sch , 5)
      call chcopy(lypsch    ,nam1,lns)

      ln2 = lpp+ls+7
      call chcopy(nam1  (l1),lypold , 7)
      call chcopy(lypfileold    ,nam1,ln2)

      call chcopy(nam1  (l1),lypres , 7)
      call chcopy(lyprestart    ,nam1,ln2)

      if(nid.eq.0.and.jpp.eq.1) then
         write(6,*) 'opening file ',lypfile
         open(unit=79,file=lypfile,status='UNKNOWN')
         write(79,5757) ! put a header on the lyp file
5757     format(1x,"time",5x,"lyap",5x,"lyapsum",5x,"pertnorm",
     &          5x,"oldpertnorm")
 
         write(6,*) 'opening file ',lypsch
         open(unit=146,file=lypsch)
      endif
         
      if(if_restart) then
         if(nid.eq.0) then
            if (jpp.eq.1) then
               print*,'opening file ',lypfileold
               open(unit=86,file=lypfileold,status='OLD')
            endif
            read(86,*) tmp(1)    !lyapsum
            read(86,*) tmp(2)    !timeStart
            if (jpp.eq.npert) close(86)
         endif
         len = 4*wdsize
         call bcast(tmp,len)  !broadcast to all cpus
         lyapSum  =  tmp(1)
         timeStart = tmp(2)
         lyap(2,jpp) = lyapsum
         print*,'nid,lyapsum,timestart=',nid,lyapSum,timestart,jpp
         if(timeStart.gt.time) then
            write(6,*) nid,' reporting! timestart.gt.time! time=',time
            call exitt
         endif
      endif

      return
      end
c------------------------------------------------------------------------

      subroutine get_useric
c
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
c
      integer e,eg
c
      do jp=1,npert
         l = 0
         do iel=1,nelv
            ielg = lglel(iel)     ! Global element number
            do k=1,nz1
            do j=1,ny1
            do i=1,nx1
               l = l+1
c              call set_nekuse (l)
               call nekasgnp (i,j,k,iel,l)
               call useric   (i,j,k,ielg)
               vxP(l,jp)   = ux
               vyP(l,jp)   = uy
               vzP(l,jp)   = uz
               tP (l,1,jp) = temp
            enddo
            enddo
            enddo
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine out_pert  ! dump perturbation .fld files
c
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
c
      character*3 s3
c
      do jpp=1,npert
         write(s3,3) jpp
 3       format('p',i2.2)
         call outpost2
     $     (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),prp(1,jpp),tp(1,1,jpp),
     $     1,s3)
         call writehist
     $     (vxp(1,jpp),vyp(1,jpp),vzp(1,jpp),tp(1,1,jpp),jpp)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine pert_add2s2(i,j,scale)   ! xi = xi + scale * xj
      include 'SIZE'
      include 'TOTAL'

      ntotp = nx2*ny2*nz2*nelv
      ntotv = nx1*ny1*nz1*nelv
      ntott = nx1*ny1*nz1*nelt

      call add2s2(vxp(1,i),vxp(1,j),scale,ntotv)
      call add2s2(vyp(1,i),vyp(1,j),scale,ntotv)
      if (if3d) call add2s2(vzp(1,i),vzp(1,j),scale,ntotv)
      call add2s2(prp(1,i),prp(1,j),scale,ntotp)

      do l=1,lorder-1
         call add2s2(vxlagp(1,l,i),vxlagp(1,l,j),scale,ntotv)
         call add2s2(vylagp(1,l,i),vylagp(1,l,j),scale,ntotv)
         if (if3d) call add2s2(vzlagp(1,l,i),vzlagp(1,l,j),scale,ntotv)
         call add2s2(prlagp(1,l,i),prlagp(1,l,j),scale,ntotp)
      enddo

      call add2s2(exx1p(1,i),exx1p(1,j),scale,ntotv)
      call add2s2(exy1p(1,i),exy1p(1,j),scale,ntotv)
      if (if3d) call add2s2(exz1p(1,i),exz1p(1,j),scale,ntotv)

      call add2s2(exx2p(1,i),exx2p(1,j),scale,ntotv)
      call add2s2(exy2p(1,i),exy2p(1,j),scale,ntotv)
      if (if3d) call add2s2(exz2p(1,i),exz2p(1,j),scale,ntotv)

      if (ifheat) then
        do k=0,npscal
          k1=k+1
          ntotk = nx1*ny1*nz1*nelfld(k+2)
          call add2s2(tp(1,k1,i),tp(1,k1,j),scale,ntotk)
          do l=1,lorder-1
            call add2s2(tlagp(1,l,k1,i),tlagp(1,l,k1,j),scale,ntotk)
          enddo
          call add2s2(vgradt1p(1,k1,i),vgradt1p(1,k1,j),scale,ntotk)
          call add2s2(vgradt2p(1,k1,i),vgradt2p(1,k1,j),scale,ntotk)
        enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      function pert_inner_prod(i,j) ! weighted inner product vi^T vj
      include 'SIZE'
      include 'TOTAL'

      common/normset/pran, ray, rayc

      ntotv=nx1*ny1*nz1*nelv
      ntott=nx1*ny1*nz1*nelt

      s1 = rayc*glsc3(vxp(1,i),bm1,vxp(1,j),ntotv)
      s2 = rayc*glsc3(vyp(1,i),bm1,vyp(1,j),ntotv)
      s3 = 0
      if (if3d) s3 = rayc*glsc3(vzp(1,i),bm1,vzp(1,j),ntotv)

      t1 = 0
      if (ifheat) t1=pran*ray*ray*glsc3(tp(1,1,i),bm1,tp(1,1,j),ntott)

      pert_inner_prod = (s1+s2+s3+t1)/volvm1

      return
      end
c-----------------------------------------------------------------------
      subroutine pert_ortho_norm ! orthogonalize and rescale pert. arrays
      include 'SIZE'
      include 'TOTAL'

      do k=1,npert
         call pert_ortho_norm1(k)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine pert_ortho_norm1 (k) ! orthogonalize k against 1...k-1
      include 'SIZE'
      include 'TOTAL'

      do j=1,k-1
         scale = -pert_inner_prod(j,k)
         call pert_add2s2(k,j,scale)   ! xi = xi + scale * xj
      enddo
      scale = pert_inner_prod(k,k)
      if (scale.gt.0) scale = 1./scale
      call rescalepert(pertnorm,scale,k)

      return
      end
c-----------------------------------------------------------------------
