c-----------------------------------------------------------------------
      subroutine arow3nc(x,y,z,dxbar,dybar,dzbar,sarrowi,jclr)
c
c     New arrow routine, but no planar projection.    2/25/00   pff
c
      COMMON/SCALE/XFAC,YFAC,XZERO,YZERO
c
      real uu(3),d(3),nhat(3,3)
      save nhat
      data nhat / 1,0,0  ,  0,1,0  ,  0,0,1  /
c
C     Now draw shadow on plane at Zbase (without +DZ)
C     Sort of pale orange in heat map
C     FIX COLOR(WHITE USUALLY, YELLOOW FOR AXIS.  <FIX 3-D ARROWHEAD)!!??
C
      sarrow = 10.*sarrowi
      IF (DXBAR.EQ.0.0.AND.DYBAR.EQ.0.0.AND.DZBAR.EQ.0.0) return
      IF (sarrow.eq.0) return
c
c     write(6,1) 'move3',x,y,z
    1 format(a5,2x,3f12.5)
c
      CALL MOVE3(X,Y,Z)
C
C
C     Draw 3-dimensional arrow
      CALL PENW(3)
      CALL COLOR(jclr)
      DX = DXBAR*sarrow
      DY = DYBAR*sarrow
      DZ = DZBAR*sarrow
c
      xh = x+dx
      yh = y+dy
      zh = z+dz
      CALL DRAW3(xh,yh,zh)
c     write(6,1) 'dmove',dx,dy,dz
c     write(6,1) 'draw3',xh,yh,zh
c
c     New arrowheads  (pff 11/24/99)
c
      dn = dx*dx + dy*dy + dz*dz
      if (dn.gt.0) dn = sqrt(dn)
      dn = 0.04*dn
c
      dxn = -.14*dx
      dyn = -.14*dy
      dzn = -.14*dz
c
c     Origin of plane intersecting arrow
c
      o1 = xh+dxn
      o2 = yh+dyn
      o3 = zh+dzn
c
c     Find projection of arrow onto x y and z planes
c
      d(1) = dx
      d(2) = dy
      d(3) = dz
c
      imax = 1
      vmax = abs(dx)
      if (abs(dy).gt.vmax) imax = 2
      if (abs(dy).gt.vmax) vmax = abs(dy)
      if (abs(dz).gt.vmax) imax = 3
      if (abs(dz).gt.vmax) vmax = abs(dz)
c
c
      do i=1,3
         if (i.ne.imax) then
            call cross (uu,nhat(1,i),d)
            call norm3d(uu)
            call cmult (uu,dn,3)
c
            v1 = o1+uu(1)
            v2 = o2+uu(2)
            v3 = o3+uu(3)
c
            w1 = o1-uu(1)
            w2 = o2-uu(2)
            w3 = o3-uu(3)
c
            call move3(v1,v2,v3)
            call draw3(xh,yh,zh)
            call draw3(w1,w2,w3)
c
         endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine selite(nelpf,se_to_face,nfcs,jclr)
c
      include 'basics.inc'
      include 'basicsp.inc'
      integer nelpf(1),se_to_face(nfcs,1)
c
      call mespos
      call gencen
      call coef
c
      do je=1,nel
      do jf=1,nfcs
         if (nelpf(se_to_face(jf,je)).eq.1) then
            call hilitef(jf,je,jclr)
         endif
      enddo
      enddo
c
      write(6,*) 'selite:',jclr
      call prsi('selite?$',jclr)
      call res(ans,1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine hilitef(iface,ie,icolr)
C
C      Draw ISOMETRIC view of general 3D element number IE.
C
      INCLUDE 'basics.inc'
      INCLUDE 'basicsp.inc'
      CHARACTER*1 CB1
      INTEGER IFIRST,NCSOLD,NCSEG1,NCSGM1
      INTEGER ICEDG(3,16),IEDGFC(4,6)
      SAVE    ICEDG,IEDGFC
      SAVE    IFIRST,NCSOLD,NCSEG1,NCSGM1
C
      COMMON /HLINE/ HIK(NXM,50),XXIS(50),YYIS(50),ZZIS(50),
     $               XK(50),YK(50),ZK(50)
      COMMON /ILINE/ IC(8),NSKIPX(3)
      COMMON /LLINE/ IFEDG(12,2,NELM)
      LOGICAL IFEDG,IFCURV
C
      DATA    IFIRST,NCSOLD,NCSEG1,NCSGM1 /4*0/
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
c      -2D-  (NOTE:  Edges default to faces in 2D! Not consistent, but...)
     $                1,3,2,   2,4,2,   1,2,1,   3,4,1 /
C
C
C
      IF(IFIRST.EQ.0)THEN
         NFACES=2*NDIM
         IFIRST=1
C
C        Set up logicals determining whether edge is to be
C        displayed or not.
C
C        Assign EB's numbering scheme to PF's scheme.
C
         EFACE(1)=4
         EFACE(2)=2
         EFACE(3)=1
         EFACE(4)=3
         EFACE(5)=5
         EFACE(6)=6
C
C        Assign inverse of EB's numbering scheme to PF's scheme.
C
         EFACE1(1)=3
         EFACE1(2)=2
         EFACE1(3)=4
         EFACE1(4)=1
         EFACE1(5)=5
         EFACE1(6)=6
C
      ENDIF
      CALL COLOR(icolr)
      CALL PENW(1)
      NXYZ = NX*NY*NZ
      NSKIPX(1) = 1
      NSKIPX(2) = NX
      NSKIPX(3) = NX*NY
C
C     Fill corners - 1 through 8.
C
      IC(1) =                                 0
      IC(2) =                             (NX-1)
      IC(3) =                 NX*(NY-1)
      IC(4) =                 NX*(NY-1) + (NX-1)
      IC(5) =  NX*NY*(NZ-1)
      IC(6) =  NX*NY*(NZ-1)             + (NX-1)
      IC(7) =  NX*NY*(NZ-1) + NX*(NY-1)
      IC(8) =  NX*NY*(NZ-1) + NX*(NY-1) + (NX-1)
C
      IF (NCSOLD.NE.NCSEGS) THEN
C
C        Set up curve line segment evaluation array (exclude endpoints) :
C
         NCSOLD = NCSEGS
         NCSEG1 = NCSEGS + 1
         NCSGM1 = NCSEGS - 1
         DR1 = 2.0 / FLOAT(NCSEGS)
         DO 400 K=1,NCSGM1
            RR1 = -1.0 + DR1*FLOAT(K)
            CALL HLEGN(HIK(1,K),RR1,ZPTS,NX)
  400    CONTINUE
      ENDIF
C
C     Draw element edges (for now, we do the 4X redundant work)
C
         IEOFF=NXYZ*(IE-1)+1
C        Check if we can take a short cut in plotting the element bdry.
         IFCURV=.FALSE.
         DO 500 IEDG = 1,8
            IF (CCURVE(IEDG,IE).NE.' ') IFCURV=.TRUE.
  500    CONTINUE
C
C        Edges
C
         do 800 jedg = 1,4
            iedg = iedgfc(jedg,iface)
            CALL COLOR(icolr)
            IEDG0 = IEDG + 12
            IF (IF3D) IEDG0 = IEDG
            ICE1 = ICEDG(1,IEDG0)
            ICE2 = ICEDG(2,IEDG0)
            NSKIP = NSKIPX(ICEDG(3,IEDG0))
            IC0 = IEOFF+IC(ICE1)
            IC1 = IEOFF+IC(ICE2)
            XK(     1) = XP(IC0)
            YK(     1) = YP(IC0)
            ZK(     1) = ZP(IC0)
            XK(NCSEG1) = XP(IC1)
            YK(NCSEG1) = YP(IC1)
            ZK(NCSEG1) = ZP(IC1)
            IF (IFCURV) THEN
               CALL EVLINE(XK(2),XP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(YK(2),YP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL EVLINE(ZK(2),ZP(IC0),HIK,NX,NXM,NCSGM1,NSKIP)
               CALL VDRAW3A(XK,YK,ZK,NCSEG1)
            ELSE
               XK(2) = XP(IC1)
               YK(2) = YP(IC1)
               ZK(2) = ZP(IC1)
               CALL VDRAW3A(XK,YK,ZK,2)
            ENDIF
  800    CONTINUE
      return
      end
c-----------------------------------------------------------------------
      subroutine hilitev(iface,ie,icolr)
      call prsiii('Hilite:$',iface,ie,icolr)
      call hilitef(iface,ie,icolr)
      return
      end
c-----------------------------------------------------------------------
      subroutine outline(input,leni,io)
c
c     output line of length len =< leni
c
      character*1 input(1)
c
      len = ltrunc(input,leni)
      write(io,1) (input(i),i=1,len)
    1 format(132a1)
      return
      end
c-----------------------------------------------------------------------
      subroutine scanout(string,input,len,infile,outfile)
c
c     scan through infile until input is found and output
c     all lines save that containing input to "outfile"
c
      character*80 string
      character*1 input(1)
      integer infile,outfile
c
      character*1 string1(80)
c
      do line=1,100000
         call blank(string,80)
         read (infile ,80,end=100,err=100) string
         if (indx1(string,input,len).ne.0) return
         call chcopy(string1,string,80)
         lout = ltrunc(string1,80)
         write (outfile,81) (string1(j),j=1,lout)
      enddo
   80 format(a80)
   81 format(80a1)
c
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine out_rea
      include 'basics.inc'
c
c     Make a copy of new .rea file
c
      character*80 string
      character*80 oldname
      character*1  oldnam1(80)
      equivalence (oldnam1,oldname)
c
      character*1  apt(52)
      character*52 apt52
      equivalence (apt52,apt)
      data apt52/'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'/
c
      real x1(8),y1(8),z1(8)
c
c     First, reopen .rea file and scan down to geometry
c
      write(6,80)  session_name
   80 format(a80)
      idot = indx1(session_name,'.',1)
      if (idot.eq.0) idot = indx1(session_name,' ',1)
c
      call blank (oldname,80)
      call chcopy(oldnam1,session_name,idot-1)
      call chcopy(oldnam1(idot),'.rea',4)
      write(6,80) oldname
      open(unit=8,file=oldname,status='old',err=199)
      open(unit=9,file='new.rea',status='unknown',err=299)
c
c     Read up to geometry
      write(6,*) 'call scan out, nel'
      call scanout(string,'NEL,NDIM,NELV',13,8,9)
      write(6,*) 'done scan out, nel'
      call outline(string,80,9)
c
c     Print out new element coordinates
c
      zl=0.
      ic=0
      il=1
      do ie=1,nel
         ic = ic+1
         read(8,80) string
c
c        write(6,80) string
c        read(string,15) kgroup
c  15    format(43x,i5)
         kgroup = 0
c
         read(8,*) (x1(k),k=1,4)
         read(8,*) (y1(k),k=1,4)
         if (if3d) then
            read(8,*) (z1(k),k=1,4)
            read(8,*) (x1(k),k=5,8)
            read(8,*) (y1(k),k=5,8)
            read(8,*) (z1(k),k=5,8)
            zmin = vlmin(zl,8)
            zmax = vlmax(zl,8)
            dz   = zmax-zmin
            dz   = .01*dz
            if (abs(zl-zmax).gt.dz) then
c              New Z-level   (thanks Ed)
               il = il+1
               zl = zmax
               ic = 0
            endif
         endif
c
         ic     = mod(ic,52)
         write(9,20) ie,il,apt(ic),kgroup
   20    format(12x,'ELEMENT',i5,' [',i5,a1,']    GROUP',i6)
         write(9,22) (x(ie,k),k=1,4)
         write(9,22) (y(ie,k),k=1,4)
         if (if3d) then
            write(9,22) (z(ie,k),k=1,4)
            write(9,22) (x(ie,k),k=5,8)
            write(9,22) (y(ie,k),k=5,8)
            write(9,22) (z(ie,k),k=5,8)
         endif
   22    format(1p4e15.6)
      enddo
c
c     Skip to Boundary conditions (skipping curve sides for now)
c
      write(6,*) 'call scan out, bdry'
      call scanout(string,'BOUNDARY',8,8,9)
      write(6,*) 'done scan out, bdry',nflds
      call outline(string,80,9)
c

      nfaces = 2*ndim
      do jf=1,nflds
c
         jf2 = jf-2
         if (jf.eq.1) write(9,41)
         if (jf.eq.2) write(9,42)
         if (jf.eq.3) write(9,43) jf2
   41    format('  ***** FLUID   BOUNDARY CONDITIONS *****')
   42    format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   43    format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')
c
         do ie=1,nel
         do k =1,nfaces
            read(8,80) string
            if (nel.lt.1000) then
               write(9,60) (cbc(k,ie,jf),ie,k,bc(j,k,ie,jf),j=1,5)
            elseif (nel.lt.100 000) then
               write(9,61) (cbc(k,ie,jf),ie,k,bc(j,k,ie,jf),j=1,5)
            elseif (nel.lt.1 000 000) then
               write(9,62) (cbc(k,ie,jf),ie,bc(j,k,ie,jf),j=1,5)
            else
               write(9,63) (cbc(k,ie,jf),ie,bc(j,k,ie,jf),j=1,5)
            endif
   60       format(1x,a3,2i3,5g14.7)
   61       format(1x,a3,i5,i1,5g14.7)
   62       format(1x,a3,i6,5g14.7)
   63       format(1x,a3,i12,5g18.11)
         enddo
         enddo
      enddo
c
c     Scan through and output .rea file until end of file
c
      write(6,*) 'call scan out, last'
      call scanout(string,'xxxx',4,8,9)
      write(6,*) 'done scan out, last'
c
      close(unit=8) 
      close(unit=9) 
      return
c
  199 continue
      call prs('File not found$')
      write(6,*) oldname
      return
c
  299 continue
      call prs('File exists: new.rea$')
      close(unit=8) 
      return
c
      end
c-----------------------------------------------------------------------
      subroutine get_vert (vertex,nlv)
      include 'basics.inc'
      integer vertex(nlv,nel)
c
      character*80 mapname
      character*1  mapnam1(80)
      equivalence (mapnam1,mapname)
c
      write(6,80)  session_name
   80 format(a80)
      idot = indx1(session_name,'.',1)
      if (idot.eq.0) idot = indx1(session_name,' ',1)
c
      call blank (mapname,80)
      call chcopy(mapnam1,session_name,idot-1)
      call chcopy(mapnam1(idot),'.map',4)
      write(6,80)  mapname
c
      open(unit=80,file=mapname,status='old',err=99)
c
      read(80,*) neli,nnzi
c
c     Get vertices from rsb routine... hypercubic ordering
c
      do ie=1,neli
         read(80,*,err=99,end=99) iproc,(vertex(k,ie),k=1,nlv)
c        write(6,*) 'vtx ',(vertex(k,ie),k=1,nlv)
      enddo
c     write(6,*) 'continue?'
c     read (5,*) ans
c
      ivmin = ilmin(vertex,nlv*nel)
      ivmax = ilmax(vertex,nlv*nel)
      write(6,*)nel,' elements, yielding vertice in range:',ivmin,ivmax
      close(unit=80)
c
      if (neli.eq.nel) return
      write(6,*) 'PROBLEM in get_vert',neli,nel
      write(6,*) mapname
c
   99 continue
      write(6,*) 'PROBLEM in get_vert',neli,nel
      write(6,*) 'Cannot find map file:'
      write(6,*) mapname
      return
      end
c-----------------------------------------------------------------------
      subroutine face_id
     $             (se_to_face,nfctot,vertex,nlv,wrk,tmp_face,nfcs)
c
c     Assign a numbering to all element faces
c
      include 'basics.inc'
      integer se_to_face(nfcs,1)
      integer vertex(nlv,1),wrk(1)
      integer tmp_face(ndim,nfcs,1)
      integer key(3),ivtx(4)
c
      INTEGER icface(4,6)
      SAVE    icface
      DATA    icface/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8/
c
      nfc    = 2**(ndim-1)
      do ie=1,nel
      do iface = 1,nfcs
         do ic=1,nfc
            ivtx(ic) = vertex(icface(ic,iface),ie)
         enddo
c        write(6,*) ie,iface,' ivtx1',ivtx
c        call isortit(ivtx,wrk,nfc)  ! isort actually sorts
         call isort  (ivtx,wrk,nfc)  ! isort actually sorts
c        write(6,*) ie,iface,' IVTX2',ivtx
         call icopy(tmp_face(1,iface,ie),ivtx,ndim)
      enddo
      enddo
c     write(6,*) 'continue?'
c     read (5,*) ans
c
c     Note -- tmp_face is blown away by following call..
c
      nlfc = nel*nfcs
      key(1) = 1
      key(2) = 2
      key(3) = 3
c
      call irank_vec(se_to_face,nfctot,tmp_face,ndim,nlfc,key,ndim,wrk)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine surface_id(nelpf,nfctot,se_to_face,nfcs,vertex,nlv
     $                      ,srfef,srfvt,srfvx,nfc,vtxpt)
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer nelpf(nfctot),se_to_face(nfcs,1)
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(ndim,nfc,1)
      integer vertex(nlv,1)
      real    vtxpt(ndim,1)
c
      INTEGER IC(8)
      INTEGER icface(4,6)
      SAVE    icface
      DATA    icface/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8/
c
      integer v_permute(4)
      save    v_permute
      data    v_permute / 2  , 1 , 4 , 3 /
c
      integer iced(8)
      save    iced
      data    iced   / 1,2,4,3 , 5,6,8,7 /
c
c     Identify all element faces on the surface
c
      call izero(nelpf,nfctot)
      do je=1,nel
      do jf=1,nfcs
         nelpf(se_to_face(jf,je)) = nelpf(se_to_face(jf,je)) + 1
      enddo
      enddo
C
c
c     Build collection of surface faces:
c
      n=0
      do je=1,nel
      do jf=1,nfcs
         if (nelpf(se_to_face(jf,je)).eq.1) then
            n=n+1
c           call setarea(je)
c           write(6,*)je,jf,' nelpf:'
c    $       ,se_to_face(jf,je),nelpf(se_to_face(jf,je)),if3d
            call hilitef(jf,je,9)
            srfef(1,n) = jf
            srfef(2,n) = je
c
c           nfc = 4 = number of face corners
            do jc=1,nfc
c
               jd = jc
               if (jf.eq.1.or.jf.eq.4.or.jf.eq.5) jd=v_permute(jc)
               ice = iced(icface(jc,jf))
c
               ivtx         = vertex(icface(jc,jf),je)
               srfvt(jd,n)  = ivtx
c
               srfvx(1,jd,n) = x(je,ice)
               vtxpt(1,ivtx) = x(je,ice)
c
               srfvx(2,jd,n) = y(je,ice)
               vtxpt(2,ivtx) = y(je,ice)
c
               if (if3d) srfvx(3,jd,n) = z(je,ice)
               if (if3d) vtxpt(3,ivtx) = z(je,ice)
            enddo
         endif
      enddo
      enddo
      srfef(1,0) = n
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vec_vtx_min(vs,v,vt,m,n,nm,work,nd)
      integer vt(m,n)
      real    vs(nd,m,n),v(nd,m,n),work(nd,nm)
c
      big = 1.e20
      call cfill(work,big,nd*nm)
      do k=1,n
      do j=1,m
         iv = vt(j,k)
         do i=1,nd
            work(i,iv) = min(work(i,iv) , v(i,j,k))
         enddo
      enddo
      enddo
c
      do k=1,n
      do j=1,m
         iv = vt(j,k)
         do i=1,nd
            vs(i,j,k) = work(i,iv)
         enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vec_vtx_sum(vs,v,vt,m,n,nm,work,nd)
      integer vt(m,n)
      real    vs(nd,m,n),v(nd,m,n),work(nd,nm)
c
      call rzero(work,nd*nm)
      do k=1,n
      do j=1,m
         iv = vt(j,k)
         do i=1,nd
            work(i,iv) = work(i,iv) + v(i,j,k)
         enddo
      enddo
      enddo
c
      do k=1,n
      do j=1,m
         iv = vt(j,k)
         do i=1,nd
            vs(i,j,k) = work(i,iv)
         enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gen_srf_nrm(vn,vx,nd,m)
c
      real vn(nd,0:1,0:1),vx(nd,0:1,0:1)
      real v1(3),v2(3)
c
      do i=1,nd
         v1(i) = vx(i,1,0)-vx(i,0,0)
         v2(i) = vx(i,0,1)-vx(i,0,0)
      enddo
      call cross (vn(1,0,0),v1,v2)
      call norm3d(vn(1,0,0))
c
      do i=1,nd
         v1(i) = vx(i,1,1)-vx(i,1,0)
         v2(i) = vx(i,0,0)-vx(i,1,0)
      enddo
      call cross (vn(1,1,0),v1,v2)
      call norm3d(vn(1,1,0))
c
      do i=1,nd
         v1(i) = vx(i,0,0)-vx(i,0,1)
         v2(i) = vx(i,1,1)-vx(i,0,1)
      enddo
      call cross (vn(1,0,1),v1,v2)
      call norm3d(vn(1,0,1))
c
      do i=1,nd
         v1(i) = vx(i,0,1)-vx(i,1,1)
         v2(i) = vx(i,1,0)-vx(i,1,1)
      enddo
      call cross (vn(1,1,1),v1,v2)
      call norm3d(vn(1,1,1))
c
      return
      end
c-----------------------------------------------------------------------
      subroutine surfchk
c
c     Suite of routines to clean up mesh imperfections
c
      include 'basics.inc'
c
      integer se_to_face(6*nelm)
      integer vertex(8*nelm)
      integer tmp_face(3,6,nelm)
      integer nelpf(6,nelm)
c
      integer srfef(2,0:6*nelm),srfvt(4,6*nelm)
      real    srfvx(3,4,6*nelm),srfvn(3,4,6*nelm),srfnb(3,4,6*nelm)
      real    phil(8*nelm),phihat(8*nelm)
      real    vtxpt(24*nelm)
c
      common /srfci/ se_to_face,vertex,tmp_face,nelpf,srfef,srfvt
      common /srfcr/ srfvx,srfvn,srfnb,phil,phihat,vtxpt
c
      common /ctmp1/ wrk(3*8*nelm)
c
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      real    phib(1000)
      integer jk  (1000)
c
      ncrnr = 2**ndim
      nfcs  = 2*ndim
      nfc   = 2**(ndim-1)
      if (icalld.ge.0) then
         icalld = icalld + 1
         call get_vert(vertex,ncrnr)
      endif
      nvert = ilmax(vertex,ncrnr*nel)
c
      write(6,*) 'call face_id',nvert
      call face_id
     $          (se_to_face,nfctot,vertex,ncrnr,wrk,tmp_face,nfcs)
c
c
      write(6,*) 'call surface_id',nfctot
      call surface_id(nelpf,nfctot,se_to_face,nfcs,vertex,ncrnr
     $                   ,srfef,srfvt,srfvx,nfc,vtxpt)
c
      write(6,*) 'call defect_id',nfctot
      call selite(nelpf,se_to_face,nfcs,2)
      call defect_id(phib,jk,ndfct
     $              ,srfef,srfvt,srfvx,srfvn,nfc,ndim,nvert
     $              ,srfnb,phil,phihat,wrk)
c
c
c
      call prs('Defect correct choice? (1,2)$')
      call rei(idfct_choice)
      call selite(nelpf,se_to_face,nfcs,4)
c
      if (idfct_choice.eq.1) then
         call selite(nelpf,se_to_face,nfcs,4)
         call defect_crct1(srfef,srfvt,srfvx,srfvn,nfc,ndim,nvert
     $                   ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,ndfct
     $              ,nelpf,se_to_face,nfcs)
      elseif (idfct_choice.eq.2) then
         write(6,*) 'call defect_crct2',nfctot
         call defect_crct2(srfef,srfvt,srfvx,srfvn,nfc,ndim,nvert
     $                   ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,ndfct
     $              ,nelpf,se_to_face,nfcs)
      elseif (idfct_choice.eq.3) then
         write(6,*) 'call defect_crct3',nfctot
         call defect_crct3(srfef,srfvt,srfvx,srfvn,nfc,ndim,nvert
     $                   ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,ndfct
     $              ,nelpf,se_to_face,nfcs)
      endif
c
      call prs('Remap data??$')
      call res(ans,1)
      if (ans.eq.'y' .or. ans.eq.'Y') then
         write(6,*) 'call remap_xyz',nfctot
         call selite(nelpf,se_to_face,nfcs,8)
         call remap_xyz (nelpf,nfctot,se_to_face,nfcs,vertex,ncrnr
     $                               ,srfef,srfvt,srfvx,nfc,vtxpt)
         call selite(nelpf,se_to_face,nfcs,10)
      endif
c
      write(6,*) 'call out_rea'
      call out_rea
c
      return
      end
c-----------------------------------------------------------------------
      subroutine remap_xyz (nelpf,nfctot,se_to_face,nfcs,vertex,nlv
     $                      ,srfef,srfvt,srfvx,nfc,vtxpt)
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer nelpf(nfctot),se_to_face(nfcs,1)
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(ndim,nfc,1)
      integer vertex(nlv,1)
      real    vtxpt(ndim,1)
c
      INTEGER IC(8)
      INTEGER icface(4,6)
      SAVE    icface
      DATA    icface/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8/
c
      integer v_permute(4)
      save    v_permute
      data    v_permute / 2  , 1 , 4 , 3 /
c
      integer iced(8)
      save    iced
      data    iced   / 1,2,4,3 , 5,6,8,7 /
c
C
c     Update nekton vertex points
c
      do je=1,nel
      do jf=1,nfcs
         if (nelpf(se_to_face(jf,je)).eq.1) then
            call hilitef(jf,je,3)
            do jc=1,nfc
               ivtx      = vertex(icface(jc,jf),je)
               ice       = iced(icface(jc,jf))
               x(je,ice) = vtxpt(1,ivtx)
               y(je,ice) = vtxpt(2,ivtx)
               if (if3d) z(je,ice) = vtxpt(3,ivtx)
            enddo
         endif
      enddo
      enddo
c
      call mespos
      call gencen
      call coef
c
      do je=1,nel
      do jf=1,nfcs
         if (nelpf(se_to_face(jf,je)).eq.1) then
            call hilitef(jf,je,15)
         endif
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine remap_xyz1
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c
      call mespos
      call gencen
      call coef
c
c     do je=1,nel
c     do jf=1,nfcs
c        if (nelpf(se_to_face(jf,je)).eq.1) then
c           call hilitef(jf,je,15)
c        endif
c     enddo
c     enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine defect_id(phib,jk,n
     $              ,srfef,srfvt,srfvx,srfvn,nfc,nd,nvrt
     $              ,srfnb,phil,phihat,work)
c
      include 'basics.inc'
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(nd,nfc,1),srfvn(nd,nfc,1),srfnb(nd,nfc,1)
      real    phil(nfc,1),phihat(nfc,1)
      real    w1(1),w2(1),w3(1)
      real    work(nd,nvrt)
      real    eli(3),elj(3)
c
      real    phib(1000)
      integer jk  (1000)
      logical ifvtsum
c
c     Compute curvature, normal derivatives, etc. for each face
c
      nsrf = srfef(1,0)
c
      do k=1,nsrf
         call gen_srf_nrm(srfvn(1,1,k),srfvx(1,1,k),nd,nfc)
         do j=1,nfc
c           call arow3nc(srfvx(1,j,k),srfvx(2,j,k),srfvx(3,j,k)
c    $                  ,srfvn(1,j,k),srfvn(2,j,k),srfvn(3,j,k)
c    $                  ,sarrow,15)
         enddo
      enddo
c
c
c     Compute mean normals
c
      call vec_vtx_sum(srfnb,srfvn,srfvt,nfc,nsrf,nvrt,work,nd)
      do k=1,nsrf
      do j=1,nfc
         call norm3d(srfnb(1,j,k))
      enddo
      enddo
c
c     Compute:
c
c     ~       ~     ~ 
c     e_ij := x_j - x_i
c
c     _,      ~        ~ 
c     e_ij := e_ij / | e_ij |
c
c                _     _,
c     phi_ij := -n_i . e_ij
c
c
c
c                          ^
c     First pass, compute phi_i
c
      do k=1,nsrf
         l=0
         do j=0,1
         do i=0,1
            l=l+1
            ii = mod(i+1,2)
            jj = mod(j+1,2)
            do id=1,nd
               eli(id) = srfvx(id,1+ii+2*j,k)-srfvx(id,1+i+2*j,k)
               elj(id) = srfvx(id,1+i+2*jj,k)-srfvx(id,1+i+2*j,k)
            enddo
            call norm3d(eli)
            call norm3d(elj)
c           phil(l,k) = -0.5 *
c    $        (dot(eli,srfnb(1,l,k),nd)+dot(elj,srfnb(1,l,k),nd))
            phil(l,k) = -max
     $        (dot(eli,srfnb(1,l,k),nd),dot(elj,srfnb(1,l,k),nd))
         enddo
         enddo
      enddo
c
      m = nfc*nsrf+1
      call rone(work,nfc*nsrf)
      call vec_vtx_sum(work,work,srfvt,nfc,nsrf,nvrt,work(m,1),1)
c
c     call vec_vtx_sum(phil,phil,srfvt,nfc,nsrf,nvrt,work(m,1),1)
c     call invcol2    (phil,work,nfc*nsrf)
c
      call vec_vtx_min(phil,phil,srfvt,nfc,nsrf,nvrt,work(m,1),1)
c
c     Now, compare phi_i to it's neighbors
c
      ifvtsum = .true.
      do k=1,nsrf
         l=0
         do j=0,1
         do i=0,1
            l=l+1
            ii = mod(i+1,2)
            jj = mod(j+1,2)
            if (ifvtsum) then
               phihat(l,k) = 0.5*( phil(1+i+2*jj,k)+phil(1+ii+2*j,k) )
            else
               phihat(l,k) = min( phil(1+i+2*jj,k),phil(1+ii+2*j,k) )
            endif
         enddo
         enddo
      enddo
      write(6,*) 'this is nsrf:',nsrf,ifvtsum
      if (ifvtsum) then
        call vec_vtx_sum(phihat,phihat,srfvt,nfc,nsrf,nvrt,work(m,1),1)
        call invcol2    (phihat,work,nfc*nsrf)
      else
        call vec_vtx_min(phihat,phihat,srfvt,nfc,nsrf,nvrt,work(m,1),1)
      endif
c
c
c     Now, compare phi_i to it's neighbors, count isolated cases
c
      n=0
      do k=1,nsrf
      do j=1,nfc
c        if (phil(j,k).lt.phihat(j,k)) then
         if (phil(j,k).lt.0.and.phihat(j,k).ge.0) then
            ss = abs(phil(j,k))*sarrow
            if (n.lt.1000) then
               n=n+1
               phib(n) = phil(j,k)
               jk  (n) = j + nfc*(k-1)
            endif
         endif
      enddo
      enddo
c
      call sortit(phib,work(m,1),n)
      call iswap (jk  ,work(m+n+1,1),work(m,1),n)
c
c     Identify 40 worst offenders
c
      n40 = min(40,n)
      do k=1,n40
         ss   = abs(phil(jk(k),1))*sarrow
         jsrf = 1+(jk(k)-1)/nfc
         jc   = mod1(jk(k),nfc)
         iv   = srfvt(jc,jsrf)
         call arow3nc(srfvx(1,jk(k),1),srfvx(2,jk(k),1),srfvx(3,jk(k),1)
     $               ,srfnb(1,jk(k),1),srfnb(2,jk(k),1),srfnb(3,jk(k),1)
     $               ,ss,15)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine defect_crct1(srfef,srfvt,srfvx,srfvn,nfc,nd,nvrt
     $              ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,n
     $              ,nelpf,se_to_face,nfcs)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(nd,nfc,1),srfvn(nd,nfc,1),srfnb(nd,nfc,1)
      real    phil(nfc,1),phihat(nfc,1)
      real    w1(1),w2(1),w3(1)
      real    wrk (nd,nvrt)
      real    vtxpt(nd,nvrt)
      real    eli(3),elj(3)
c
      real    phib(1000)
      integer jk  (1000)
      logical ifvtsum,ifupdate
c
c     Compute vertex multiplicity
c
      nsrf = srfef(1,0)
      m = nfc*nsrf+1
      call rone(wrk,nfc*nsrf)
      call vec_vtx_sum(wrk,wrk,srfvt,nfc,nsrf,nvrt,wrk(m,1),1)
c
c     Correct vertices with most significant difficulties
c
      eps = 1.0
      n10 = min(10,n)
      k = 0
      do ko=1,n10
        call refresh2
        call selite(nelpf,se_to_face,nfcs,3)
        do ki=1,4
         k=k+1
c
c
         jsrf = (jk(k)-1)/nfc + 1
         jcrn = mod1(jk(k),nfc)
         iv   = srfvt(jcrn,jsrf)
         i    = mod(jcrn-1,2)
         j    = (jcrn-1)/2
         ii   = mod(i+1,2)
         jj   = mod(j+1,2)
c
         enormi = 0.
         enormj = 0.
         do id=1,nd
            eli(id)  = srfvx(id,1+ii+2*j,jsrf)-srfvx(id,1+i+2*j,jsrf)
            elj(id)  = srfvx(id,1+i+2*jj,jsrf)-srfvx(id,1+i+2*j,jsrf)
            enormi = enormi+eli(id)*eli(id)
            enormj = enormj+elj(id)*elj(id)
         enddo
         if (enormi.gt.0) enormi = sqrt(enormi)
         if (enormj.gt.0) enormj = sqrt(enormj)
c
         phili = phil(1+ii+2*j,jsrf) - phil(1+i+2*j,jsrf)
         philj = phil(1+i+2*jj,jsrf) - phil(1+i+2*j,jsrf)
         phili = max(phili,0.)
         philj = max(philj,0.)
c
         rmult = wrk(jk(k),1)
c
         dnv = eps*(enormi*phili + enormj*philj)/2.
         dnv = dnv*0.333
         dn  = dnv/rmult
         dx  = dn*srfnb(1,jk(k),1)
         dy  = dn*srfnb(2,jk(k),1)
         dz  = dn*srfnb(3,jk(k),1)
         if (ndim.eq.2) dz  = 0.
c
         call arow3nc(srfvx(1,jk(k),1),srfvx(2,jk(k),1),srfvx(3,jk(k),1)
     $               ,dx,dy,dz,1.,1)
         if (ki.eq.1) then
            call prsi('update? (y/n) $',999)
            call res(ans,1)
            ifupdate = .false.
            if (ans.eq.'y' .or. ans.eq.'Y') ifupdate = .true.
         endif
         if (ifupdate) then
            vtxpt(1,iv) = vtxpt(1,iv) + dx
            vtxpt(2,iv) = vtxpt(2,iv) + dy
            vtxpt(3,iv) = vtxpt(3,iv) + dz
         endif
       enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine defect_crct3(srfef,srfvt,srfvx,srfvn,nfc,nd,nvrt
     $              ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,n
     $              ,nelpf,se_to_face,nfcs)
c
c     Correct defect using transfinite interpolation
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(nd,nfc,1),srfvn(nd,nfc,1),srfnb(nd,nfc,1)
      real    phil(nfc,1),phihat(nfc,1)
      real    w1(1),w2(1),w3(1)
      real    wrk (nd,nvrt)
      real    vtxpt(nd,nvrt)
      real    eli(3),elj(3)
c
      real    phib(1000)
      integer jk  (1000)
      integer ibox (3,3)
      integer jbox (2,2,4)
      real    pbox (2,2,4)
      real    diff(3)
c
c     Compute vertex multiplicity
c
      nsrf = srfef(1,0)
      m = nfc*nsrf+1
      call rone(wrk,nfc*nsrf)
      call vec_vtx_sum(wrk,wrk,srfvt,nfc,nsrf,nvrt,wrk(m,1),1)
c
c     Correct vertices with most significant difficulties
c
      k = 0
      n10 = min(n,4)
      do kv=1,10
        do kk=1,4
         k    = k+1
         rmlt = wrk(jk(k),1)
         imlt = wrk(jk(k),1)
         write(6,*) k,'phib:',phib(k),jk(k),je,jf,imlt
         if (rmlt.gt.3.0.and.rmlt.lt.5.0) then
           jsrf = (jk(k)-1)/nfc + 1
           jcrn = mod1(jk(k),nfc)
           iv   = srfvt(jcrn,jsrf)
           i    = mod(jcrn-1,2)
           j    = (jcrn-1)/2
           ii   = mod(i+1,2)
           jj   = mod(j+1,2)
c
           jf = srfef(1,jsrf)
           je = srfef(2,jsrf)
c
c          Fill in 3x3 square
c
           if (kk.eq.1) then
             ibox(1,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,1) = srfvt(1+ii+2*j ,jsrf)
             ibox(1,2) = srfvt(1+i +2*jj,jsrf)
             ibox(2,2) = srfvt(1+i +2*j ,jsrf)
             call hilitev(jf,je,1)
c
             jbox(1,1,1) = srfvt(1+ii+2*jj,jsrf)
             jbox(2,1,1) = srfvt(1+ii+2*j ,jsrf)
             jbox(1,2,1) = srfvt(1+i +2*jj,jsrf)
             jbox(2,2,1) = srfvt(1+i +2*j ,jsrf)
c
             pbox(1,1,1) = phil(1+ii+2*jj,jsrf)
             pbox(2,1,1) = phil(1+ii+2*j ,jsrf)
             pbox(1,2,1) = phil(1+i +2*jj,jsrf)
             pbox(2,2,1) = phil(1+i +2*j ,jsrf)
c
           elseif (srfvt(1+ii+2*j ,jsrf).eq.ibox(2,1)) then
             ibox(3,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(3,2) = srfvt(1+i +2*jj,jsrf)
             call hilitev(jf,je,7)
c
             jbox(1,1,2) = srfvt(1+ii+2*j ,jsrf)
             jbox(1,2,2) = srfvt(1+i +2*j ,jsrf)
             jbox(2,1,2) = srfvt(1+ii+2*jj,jsrf)
             jbox(2,2,2) = srfvt(1+i +2*jj,jsrf)
c
             pbox(1,1,2) = phil(1+ii+2*j ,jsrf)
             pbox(1,2,2) = phil(1+i +2*j ,jsrf)
             pbox(2,1,2) = phil(1+ii+2*jj,jsrf)
             pbox(2,2,2) = phil(1+i +2*jj,jsrf)
c
           elseif (srfvt(1+i +2*jj,jsrf).eq.ibox(2,1)) then
             ibox(3,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(3,2) = srfvt(1+ii+2*j ,jsrf)
             call hilitev(jf,je,7)
c
             jbox(1,1,2) = srfvt(1+i +2*jj,jsrf)
             jbox(1,2,2) = srfvt(1+i +2*j ,jsrf)
             jbox(2,1,2) = srfvt(1+ii+2*jj,jsrf)
             jbox(2,2,2) = srfvt(1+ii+2*j ,jsrf)
c
             pbox(1,1,2) = phil(1+i +2*jj,jsrf)
             pbox(1,2,2) = phil(1+i +2*j ,jsrf)
             pbox(2,1,2) = phil(1+ii+2*jj,jsrf)
             pbox(2,2,2) = phil(1+ii+2*j ,jsrf)
c
           elseif (srfvt(1+ii+2*j ,jsrf).eq.ibox(1,2)) then
             ibox(1,3) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,3) = srfvt(1+i +2*jj,jsrf)
             call hilitev(jf,je,10)
c
             jbox(1,1,3) = srfvt(1+ii+2*j ,jsrf)
             jbox(1,2,3) = srfvt(1+ii+2*jj,jsrf)
             jbox(2,1,3) = srfvt(1+i +2*j ,jsrf)
             jbox(2,2,3) = srfvt(1+i +2*jj,jsrf)
c
             pbox(1,1,3) = phil(1+ii+2*j ,jsrf)
             pbox(1,2,3) = phil(1+ii+2*jj,jsrf)
             pbox(2,1,3) = phil(1+i +2*j ,jsrf)
             pbox(2,2,3) = phil(1+i +2*jj,jsrf)
c
           elseif (srfvt(1+i +2*jj,jsrf).eq.ibox(1,2)) then
             ibox(1,3) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,3) = srfvt(1+ii+2*j ,jsrf)
             call hilitev(jf,je,10)
c
             jbox(1,1,3) = srfvt(1+i +2*jj,jsrf)
             jbox(1,2,3) = srfvt(1+ii+2*jj,jsrf)
             jbox(2,1,3) = srfvt(1+i +2*j ,jsrf)
             jbox(2,2,3) = srfvt(1+ii+2*j ,jsrf)
c
             pbox(1,1,3) = phil(1+i +2*jj,jsrf)
             pbox(1,2,3) = phil(1+ii+2*jj,jsrf)
             pbox(2,1,3) = phil(1+i +2*j ,jsrf)
             pbox(2,2,3) = phil(1+ii+2*j ,jsrf)
c
           else 
             ibox(3,3) = srfvt(1+ii+2*jj,jsrf)
             call hilitev(jf,je,15)
c
             jbox(1,1,4) = srfvt(1+i +2*j ,jsrf)
             jbox(1,2,4) = srfvt(1+ii+2*j ,jsrf)
             jbox(2,1,4) = srfvt(1+i +2*jj,jsrf)
             jbox(2,2,4) = srfvt(1+ii+2*jj,jsrf)
c
             pbox(1,1,4) = phil(1+i +2*j ,jsrf)
             pbox(1,2,4) = phil(1+ii+2*j ,jsrf)
             pbox(2,1,4) = phil(1+i +2*jj,jsrf)
             pbox(2,2,4) = phil(1+ii+2*jj,jsrf)
c
           endif
         endif
        enddo
c
        write(6,*) 
        write(6,6) jbox(1,2,3),jbox(2,2,3),jbox(1,2,4),jbox(2,2,4)
        write(6,6) jbox(1,1,3),jbox(2,1,3),jbox(1,1,4),jbox(2,1,4)
        write(6,*) 
        write(6,6) jbox(1,2,1),jbox(2,2,1),jbox(1,2,2),jbox(2,2,2)
        write(6,6) jbox(1,1,1),jbox(2,1,1),jbox(1,1,2),jbox(2,1,2)
        write(6,*) 
   6    format(2(2i6,3x))
c
        write(6,*) 
        write(6,8) pbox(1,2,3),pbox(2,2,3),pbox(1,2,4),pbox(2,2,4)
        write(6,8) pbox(1,1,3),pbox(2,1,3),pbox(1,1,4),pbox(2,1,4)
        write(6,*) 
        write(6,8) pbox(1,2,1),pbox(2,2,1),pbox(1,2,2),pbox(2,2,2)
        write(6,8) pbox(1,1,1),pbox(2,1,1),pbox(1,1,2),pbox(2,1,2)
        write(6,*) 
   8    format(2(1p2e12.4,3x))
c
        call prsi('continue?$',999)
        call res(ans,1)
c
ccc           i1 = ibox(1,1)
ccc           i2 = ibox(1,3)
ccc           i3 = ibox(3,1)
ccc           i4 = ibox(3,3)
ccc   c
ccc           j1 = ibox(1,2)
ccc           j2 = ibox(3,2)
ccc           j3 = ibox(2,1)
ccc           j4 = ibox(2,3)
ccc   c
ccc           do l=1,ndim
ccc              xnew = 
ccc        $       0.5*(vtxpt(l,j1)+vtxpt(l,j2)+vtxpt(l,j3)+vtxpt(l,j4))
ccc        $     - .25*(vtxpt(l,i1)+vtxpt(l,i2)+vtxpt(l,i3)+vtxpt(l,i4))
ccc              diff(l)     = vtxpt(l,iv)-xnew
ccc              vtxpt(l,iv) = xnew
ccc           enddo
ccc           write(6,6) iv,(diff(l),l=1,ndim)
ccc       6   format(i9,' diffx:',3f12.5)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine defect_crct2(srfef,srfvt,srfvx,srfvn,nfc,nd,nvrt
     $              ,srfnb,vtxpt,phil,phihat,phib,jk,wrk,n
     $              ,nelpf,se_to_face,nfcs)
c
c     Correct defect using transfinite interpolation
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(nd,nfc,1),srfvn(nd,nfc,1),srfnb(nd,nfc,1)
      real    phil(nfc,1),phihat(nfc,1)
      real    w1(1),w2(1),w3(1)
      real    wrk (nd,nvrt)
      real    vtxpt(nd,nvrt)
      real    eli(3),elj(3)
c
      real    phib(1000)
      integer jk  (1000)
      integer ibox (3,3)
      real    diff(3),vtmpx(3),dn(3)
c
c     Compute vertex multiplicity
c
      nsrf = srfef(1,0)
      m = nfc*nsrf+1
      call rone(wrk,nfc*nsrf)
      call vec_vtx_sum(wrk,wrk,srfvt,nfc,nsrf,nvrt,wrk(m,1),1)
c
c     Correct vertices with most significant difficulties
c
      k = 0
      n10 = min(n,4)
      do kv=1,10
        call refresh2
        call selite(nelpf,se_to_face,nfcs,3)
        do kk=1,4
         k    = k+1
         rmlt = wrk(jk(k),1)
         imlt = wrk(jk(k),1)
         write(6,*) k,'phib:',phib(k),jk(k),je,jf,imlt
         if (rmlt.gt.3.0.and.rmlt.lt.5.0) then
           jsrf = (jk(k)-1)/nfc + 1
           jcrn = mod1(jk(k),nfc)
           iv   = srfvt(jcrn,jsrf)
           i    = mod(jcrn-1,2)
           j    = (jcrn-1)/2
           ii   = mod(i+1,2)
           jj   = mod(j+1,2)
c
           jf = srfef(1,jsrf)
           je = srfef(2,jsrf)
           call copy(dn,srfnb(1,jk(k),1),nd)
c
c          Fill in 3x3 square
c
           if (kk.eq.1) then
             ibox(1,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,1) = srfvt(1+ii+2*j ,jsrf)
             ibox(1,2) = srfvt(1+i +2*jj,jsrf)
             ibox(2,2) = srfvt(1+i +2*j ,jsrf)
             call hilitev(jf,je,1)
           elseif (srfvt(1+ii+2*j ,jsrf).eq.ibox(2,1)) then
             ibox(3,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(3,2) = srfvt(1+i +2*jj,jsrf)
             call hilitev(jf,je,7)
           elseif (srfvt(1+i +2*jj,jsrf).eq.ibox(2,1)) then
             ibox(3,1) = srfvt(1+ii+2*jj,jsrf)
             ibox(3,2) = srfvt(1+ii+2*j ,jsrf)
             call hilitev(jf,je,7)
           elseif (srfvt(1+ii+2*j ,jsrf).eq.ibox(1,2)) then
             ibox(1,3) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,3) = srfvt(1+i +2*jj,jsrf)
             call hilitev(jf,je,10)
           elseif (srfvt(1+i +2*jj,jsrf).eq.ibox(1,2)) then
             ibox(1,3) = srfvt(1+ii+2*jj,jsrf)
             ibox(2,3) = srfvt(1+ii+2*j ,jsrf)
             call hilitev(jf,je,10)
           else 
             ibox(3,3) = srfvt(1+ii+2*jj,jsrf)
             call hilitev(jf,je,15)
           endif
         endif
        enddo
c       call prsi('continue?$',999)
c       call res(ans,1)
c
        i1 = ibox(1,1)
        i2 = ibox(1,3)
        i3 = ibox(3,1)
        i4 = ibox(3,3)
c
        j1 = ibox(1,2)
        j2 = ibox(3,2)
        j3 = ibox(2,1)
        j4 = ibox(2,3)
c
        do l=1,ndim
           xnew = 
     $       0.5*(vtxpt(l,j1)+vtxpt(l,j2)+vtxpt(l,j3)+vtxpt(l,j4))
     $     - .25*(vtxpt(l,i1)+vtxpt(l,i2)+vtxpt(l,i3)+vtxpt(l,i4))
           diff(l)     = xnew-vtxpt(l,iv)
           vtmpx(l) = xnew
        enddo
c
        dvdn = dot(diff,dn,nd)
        if (dvdn.gt.0.) then
c        Move is in direction of outward pointing normal, keep it
c        call copy(vtxpt(1,iv),vtmpx,nd)
c
         write(6,4) 'updat:',dvdn,(dn(l),l=1,nd)
4        format(a6,1p4e12.4)
c
         write(6,6) iv,(diff(l),l=1,ndim)
    6    format(i9,' diffx:',3f12.5)
         call arow3nc(srfvx(1,jk(k),1),srfvx(2,jk(k),1),srfvx(3,jk(k),1)
     $               ,diff(1),diff(2),diff(3),1.,15)
         call arow3nc(vtxpt(1,iv),vtxpt(2,iv),vtxpt(3,iv)
     $               ,dn(1),dn(2),dn(3),dvdn,1)
         call prsi('update? (y/n) $',999)
         call res(ans,1)
         if (ans.eq.'y' .or. ans.eq.'Y') 
     $      call add2s2(vtxpt(1,iv),dn,dvdn,nd)
c
        else
         call prsr('Negative move:$',dvdn)
         call arow3nc(srfvx(1,jk(k),1),srfvx(2,jk(k),1),srfvx(3,jk(k),1)
     $               ,diff(1),diff(2),diff(3),1.,1)
        endif
      enddo
c
      return
      end
c-----------------------------------------------------------------------
cqq  c
cqq          do l=1,ndim
cqq             xnew = 
cqq       $       0.5*(vtxpt(l,j1)+vtxpt(l,j2)+vtxpt(l,j3)+vtxpt(l,j4))
cqq       $     - .25*(vtxpt(l,i1)+vtxpt(l,i2)+vtxpt(l,i3)+vtxpt(l,i4))
cqq             diff(l)     = vtxpt(l,iv)-xnew
cqq             vtxpt(l,iv) = xnew
cqq          enddo
cqq          write(6,6) iv,(diff(l),l=1,ndim)
cqq      6   format(i9,' diffx:',3f12.5)
cqq        enddo
cqq  c
cqq        return
cqq        end
c
c-----------------------------------------------------------------------
      subroutine mesh_smooth
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      common /smeshr/ xyzo(3,8,nelm),xyzl(3,8,nelm)
      common /smeshi/ ioset
c
      if (icalld.eq.0) then
         icalld = icalld + 1
         call set_mesh(xyzo)
      endif
c
      x0 = -9.e19
      y0 = -9.e19
      z0 = -9.e19
      x1 =  9.e19
      y1 =  9.e19
      z1 =  9.e19
      nvc = 2**ndim
C
    1 CONTINUE
c
      call set_mesh(xyzl)
c
      nchoic = 0
      nchoic = nchoic+1
      ITEM(nchoic)       =             'UP MENU'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Redraw mesh'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Set smoothing box'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Set surface layer'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Angle Smooth'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Laplacian Smooth'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Laplacian Angle'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'undo'
      nchoic = nchoic+1
      ITEM(nchoic)       =             'Reset mesh'
c
c     Menu's all set, prompt for user input:
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'NOCOVER')
c
      IF (CHOICE.EQ.'UP MENU') return
      IF (CHOICE.EQ.'Redraw mesh') then
         call refresh2
      ELSEIF (CHOICE.EQ.'Set smoothing box') THEN
         ntot= nx*ny*nz*nel
         xmn = vlmin(xp,ntot)
         xmx = vlmax(xp,ntot)
         CALL PRSrr(
     $   'Input X-locations defining smoothing box.$',xmn,xmx)
         ymn = vlmin(yp,ntot)
         ymx = vlmax(yp,ntot)
         CALL RERR(x0,x1)
         CALL PRSrr(
     $   'Input Y-locations defining smoothing box.$',ymn,ymx)
         zmn = vlmin(zp,ntot)
         zmx = vlmax(zp,ntot)
         CALL RERR(y0,y1)
         if (if3d) CALL PRSrr(
     $   'Input Z-locations defining smoothing box.$',zmn,zmx)
         if (if3d) CALL RERR(z0,z1)
c
      ELSEIF (CHOICE.EQ.'Set surface layer') THEN
         call set_surf_layer(x0,x1,y0,y1,z0,z1)
c
      ELSEIF (CHOICE.EQ.'Laplacian Angle') THEN
c
       call prs('Input update par1 & 2 (typ. 0.9,  .01):$')
       call rerr(ss,eps)
       call prs('Input update graph depth and nrep (1-9, 1-9):$')
       call reii(idpth,nrep)
c
       do i=1,nrep
          call smoother(wkv1,nvc,wkv2,wkv3,work,wrk2,jacm1
     $                                ,x0,x1,x0,x1,z0,z1,ss,eps,idpth)
          call angle_smoother
     $      (wkv1,nvc,wkv2,wkv3,work,wrk2,jacm1,x0,x1,x0,x1,z0,z1,eps)
          call remap_xyz1
          call show_plot
       enddo
c
      ELSEIF (CHOICE.EQ.'Laplacian Smooth') THEN
c
       call prs('Input update par1 & 2 (typ. 0.9,  .01):$')
       call rerr(ss,eps)
       call prs('Input update graph depth and nrep (1-9, 1-9):$')
       call reii(idpth,nrep)
c
       do i=1,nrep
          call smoother(wkv1,nvc,wkv2,wkv3,work,wrk2,jacm1
     $                                 ,x0,x1,x0,x1,z0,z1,ss,eps,idpth)
          call remap_xyz1
          call show_plot
       enddo
c
      ELSEIF (CHOICE.EQ.'Angle Smooth') THEN
c
       call prs('Input update epsilon ( <<1 ), nrep:$')
       call rerr(eps,rep)
       nrep = rep
       do i=1,nrep
          call angle_smoother
     $      (wkv1,nvc,wkv2,wkv3,work,wrk2,jacm1,x0,x1,x0,x1,z0,z1,eps)
          call remap_xyz1
          call show_plot
       enddo
c
c
      ELSEIF (CHOICE.EQ.'undo') THEN
       call reset_mesh(xyzl)
       call remap_xyz1
       call show_plot
      ELSEIF (CHOICE.EQ.'Reset mesh') THEN
       call reset_mesh(xyzo)
       call remap_xyz1
       call show_plot
      ENDIF
c
c     write(6,*) 'call out_rea'
c     call out_rea
c
      goto 1
      end
c-----------------------------------------------------------------------
      subroutine find_bc_crv(b,nv,cell,nvc,ncell)
c
      include 'basics.inc'
      integer cell(nvc,ncell),b(nv)
c
      integer efc(4,6)
      save    efc
      data    efc  / 1,2,5,6
     $             , 2,3,6,7
     $             , 3,4,7,8
     $             , 4,1,8,5
     $             , 1,2,3,4
     $             , 5,6,7,8 /
c
c
c     Identify any elements on the boundary or having curved sides
c
      call izero(b,nv)
c
      nfaces = 2*ndim
      ncrnf  = 2**(ndim-1)
      ifld = 2
      if (ifflow) ifld = 1
c
      do ie=1,nel
      do is=1,nfaces
c
         if (cbc(is,ie,ifld).ne.'E  ') then
            do iv=1,ncrnf
               b(cell(efc(iv,is),ie)) =  1
            enddo
         endif
c
         if (cbc(is,ie,ifld).eq.'v  ') then
            do iv=1,ncrnf
               b(cell(efc(iv,is),ie)) =  2
            enddo
         endif
c
         if (cbc(is,ie,ifld).eq.'SYM') then
c           Later this will be fixed to handle each spatial dimension
            do iv=1,ncrnf
               b(cell(efc(iv,is),ie)) = -2
            enddo
         endif
c
c        Check for curve sides (spherical in particular)
         if (ccurve(is,ie).eq.'s') then
            do iv=1,ncrnf
               b(cell(efc(iv,is),ie)) =  1
            enddo
         endif
       enddo
       enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine build_adj(ja,ia,n,cell,nvc,ncell,edge,ne,w,wk,ind)
c
c     Build an adjacency graph from cell-based data, given
c     edge array
c
c
      integer ja(9),ia(9),cell(nvc,ncell),w(2,1),ind(1),wk(1)
      integer edge(2,ne)
      integer key (2)
      integer w1,w2
c
      k=0
      do icell=1,ncell
      do ie   =1,ne
         w1 = cell(edge(1,ie),icell)
         w2 = cell(edge(2,ie),icell)
         k=k+1
         w(1,k) = min(w1,w2)
         w(2,k) = max(w1,w2)
      enddo
      enddo
6     format(a3,8i4)
      nv = k
c
c
      key(1) = 1
      key(2) = 2
      call ituple_s_merge(nw,w,2,nv,key,2,ind,wk)
      write(6,*) 'this is nw:',nw,nv
c
c
c     Get symmetric other-half
c
      w1 = 0
      w2 = 0
      do i=1,nw
         w(1,i+nw) = w(2,i)
         w(2,i+nw) = w(1,i)
         w1 = max(w1,w(1,i))
         w2 = max(w2,w(2,i))
      enddo
      nv = 2*nw
      write(6,*) 'this is nw:',nw,nv,w1,w2
      call ituple_s_merge(nw,w,2,nv,key,2,ind,wk)
      write(6,*) 'this is nw3',nw,nv,w1,w2
c
c
c     Take merged list of vertex pairs and build csr format adjancy array
c
      n     = 1
      nnz   = 1
      ia(1) = 1
      ja(1) = w(1,1)
      ilast = w(1,1)
      nnz   = nnz+1
      ja(2) = w(2,1)
      do i=2,nv
         if (w(1,i).eq.ilast) then
            nnz = nnz+1
            ja(nnz) = w(2,i)
         else
            nnz   = nnz+1
            n     = n+1
            ia(n) = nnz
            ja(nnz) = w(1,i)
            ilast   = w(1,i)
            nnz = nnz+1
            ja(nnz) = w(2,i)
         endif
      enddo
      ia(n+1) = nnz+1
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_gxyz(nv,cell,nvc,ncell,ww,ind,ierr)
c
c     Read a file until "key" is found or eof is found.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /cfilold/ filold
      character filold*17
c
      integer cell(nvc,1),ind(1),ww(0:1)
c
      common /file_pref/ file_prefix
      character*80 file_prefix
c
      character*80 string
      character*1  string1(80)
      equivalence (string1,string)
c
c
      integer ecrnr(8),kcell(8)
      save    ecrnr
      data    ecrnr  / 1,2,4,3,5,6,8,7 /
c
   80 format(a80)
   81 format(80a1)
c
c
c     Get map file from old input
c
      length = indx1(filold,'.',1)-1
      if (length.le.0) length = indx1(filold,' ',1)-1
      call blank (string,80)
      call chcopy(string,filold,length)
      call chcopy(string1(length+1),'.map',4)
c
c     Get current vertex map info
c
      nvc = 2**ndim
      call get_vert (cell,nvc)
c
      ncell=nel
      do ie=1,ncell
c        HMT's data (.map) is in the good h-cube ordering, swap
         call icopy(kcell,cell(1,ie),nvc)
         do k=1,nvc
            j=ecrnr(k)
            cell(k,ie) = kcell(j)
         enddo
      enddo
c
c     Sort data and gridpoints by global vertex number
c
      lmax  = 0
      do ie = 1,ncell
         do k=1,nvc
c           j     = ecrnr(k)
            l     = cell(k,ie)
            xp(l) = x (ie,k)
            yp(l) = y (ie,k)
            zp(l) = z (ie,k)
            lmax  = max(l,lmax)
c           write(6,6) 'x:',ie,k,l,xp(l),yp(l)
         enddo
      enddo
6     format(a2,3i9,2f12.4)
      nv = lmax
c
      ierr = 0
      return
      end
c-----------------------------------------------------------------------
      subroutine find_sm_box(b,nv,x0,x1,y0,y1,z0,z1)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer b(1)
c
      call qb(b,nv,xp,x0,x1)
      call qb(b,nv,yp,y0,y1)
      if (if3d) call qb(b,nv,zp,z0,z1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine qb(b,n,x,x0,x1)
c
      integer b(1)
      real    x(1)
c
      xn=min(x0,x1)
      xx=max(x0,x1)
      do i=1,n
         if (x(i).lt.xn) b(i) = 10
         if (x(i).gt.xx) b(i) = 10
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine find_bd_jac(b,nv,cell,nvc,nel,iebad,nebad)
c
      integer b(nv),cell(nvc,nel),iebad(nebad)
c
      do je=1,nebad
         ie=iebad(je)
         do j=1,nvc
            i=cell(j,ie)
            b(i) = 1
         enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine chk_jacob(vol,ratio,nebad,iebad,b,xp,yp,zp,cell,nvc)
      include 'basics.inc'
c
      integer iebad(1),cell(nvc,1),b(1)
      real xp(1),yp(1),zp(1),vol(1),ratio(1)
c
      parameter (lx3=2)
      common  /cjac/ dxm3(lx3,lx3),dxm3t(lx3,lx3),z3(lx3),w3(lx3)
     $  , xm3 (lx3,lx3,lx3), ym3 (lx3,lx3,lx3), zm3 (lx3,lx3,lx3)
     $  , xrm3(lx3,lx3,lx3), yrm3(lx3,lx3,lx3), zrm3(lx3,lx3,lx3)
     $  , xsm3(lx3,lx3,lx3), ysm3(lx3,lx3,lx3), zsm3(lx3,lx3,lx3)
     $  , xtm3(lx3,lx3,lx3), ytm3(lx3,lx3,lx3), ztm3(lx3,lx3,lx3)
     $  , jacm3(lx3,lx3,lx3)
      real jacm3
c
      integer inv (8)
      save    inv
      data    inv /1,2,4,3,5,6,8,7/
c
      integer ci
C
      rmin = 1.e20
      nx3 = 2
      ny3 = 2
      nz3 = 2
      if (ndim.eq.2) nz3=1
c
      call legend(z3,w3,nx3)
      call dgll  (dxm3,dxm3t,z3,nx3,nx3)
c     call outmat(dxm3 ,nx3,nx3,'dx3')
c     call outmat(dxm3t,nx3,nx3,'dxt')
c     call prs('continue?$')
c     call res(ans,1)
c
      NXY3  = NX3*NY3
      NYZ3  = NY3*NZ3
      NXYZ3 = NX3*NY3*NZ3
      NTOT3 = NXYZ3
      nebad = 0
C
C
C     Compute isoparametric partials.
C
      IF (NDIM.EQ.2) THEN
C
C        Two-dimensional case
C
         do ie=1,nel
            do j=1,4
               i=inv(j)
               xm3(i,1,1) = x(ie,j)
               ym3(i,1,1) = y(ie,j)
            enddo
c           Overload with new mesh data
            do j=1,4
               if (b(cell(j,ie)).le.0) then
                  i=inv(j)
                  xm3(i,1,1) = xp(cell(j,ie))
                  ym3(i,1,1) = yp(cell(j,ie))
               endif
            enddo
c
c
C           Use the appropriate derivative- and interpolation operator in
C           the y-direction (= radial direction if axisymmetric).
            CALL MXM(DXM3,NX3,XM3,NX3,XRM3,NY3)
            CALL MXM(DXM3,NX3,YM3,NX3,YRM3,NY3)
            CALL MXM(XM3,NX3,DXM3t,NY3,XSM3,NY3)
            CALL MXM(YM3,NX3,DXM3t,NY3,YSM3,NY3)
            CALL RZERO   (JACM3,NTOT3)
            CALL ADDCOL3 (JACM3,XRM3,YSM3,NTOT3)
            CALL SUBCOL3 (JACM3,XSM3,YRM3,NTOT3)
            CALL CHKJAC  (ierr,JACM3,NXYZ3)
            vol(ie) = vlsum(jacm3,nxyz3)/nxyz3
            rjmin = vlmin(jacm3,nxyz3)
            rjmax = vlmax(jacm3,nxyz3)
            ratio(ie) = rjmin/rjmax
            if (ierr.gt.0) then
               nebad=nebad+1
               iebad(nebad) = ie
            endif
         enddo
      ELSE
C
C        Three-dimensional case
C
         do ie=1,nel
c
            do j=1,8
               i=inv(j)
               xm3(i,1,1) = x(ie,j)
               ym3(i,1,1) = y(ie,j)
               zm3(i,1,1) = z(ie,j)
            enddo
c           Overload with new mesh data
            do j=1,8
               i=inv(j)
               if (b(cell(j,ie)).le.0) then
                  xm3(i,1,1) = xp(cell(j,ie))
                  ym3(i,1,1) = yp(cell(j,ie))
                  zm3(i,1,1) = zp(cell(j,ie))
               endif
            enddo
c
c
            CALL MXM(DXM3,NX3,XM3,NX3,XRM3,NYZ3)
            CALL MXM(DXM3,NX3,YM3,NX3,YRM3,NYZ3)
            CALL MXM(DXM3,NX3,ZM3,NX3,ZRM3,NYZ3)
            do iz=1,nz3
               CALL MXM(XM3(1,1,IZ),NX3,DXM3t,NY3,XSM3(1,1,IZ),NY3)
               CALL MXM(YM3(1,1,IZ),NX3,DXM3t,NY3,YSM3(1,1,IZ),NY3)
               CALL MXM(ZM3(1,1,IZ),NX3,DXM3t,NY3,ZSM3(1,1,IZ),NY3)
            enddo
            CALL MXM(XM3,NXY3,dxm3t,NZ3,XTM3,NZ3)
            CALL MXM(YM3,NXY3,dxm3t,NZ3,YTM3,NZ3)
            CALL MXM(ZM3,NXY3,dxm3t,NZ3,ZTM3,NZ3)
c
            CALL RZERO   (JACM3,NTOT3)
            CALL ADDCOL4 (JACM3,XRM3,YSM3,ZTM3,NTOT3)
            CALL ADDCOL4 (JACM3,XTM3,YRM3,ZSM3,NTOT3)
            CALL ADDCOL4 (JACM3,XSM3,YTM3,ZRM3,NTOT3)
            CALL SUBCOL4 (JACM3,XRM3,YTM3,ZSM3,NTOT3)
            CALL SUBCOL4 (JACM3,XSM3,YRM3,ZTM3,NTOT3)
            CALL SUBCOL4 (JACM3,XTM3,YSM3,ZRM3,NTOT3)
            CALL CHKJAC  (ierr,JACM3,NXYZ3)
            vol(ie) = vlsum(jacm3,nxyz3)/nxyz3
            rjmin = vlmin(jacm3,nxyz3)
            rjmax = vlmax(jacm3,nxyz3)
            ratio(ie) = rjmin/rjmax
            if (abs(ratio(ie)).lt.rmin) then
               iemin = ie
               rmin  = ratio(ie)
            endif
c           write(6,6) ie,vol(ie),rjmin,rjmax,ratio(ie)
   6        format(i5,' jac:',1p4e12.4)
c
            if (ierr.gt.0) then
               nebad=nebad+1
               iebad(nebad) = ie
               call move3(xm3,ym3,zm3)
               do i=1,8
                  ii = inv(i)
                  ci = cell(ii,ie)
c                 write(6,*) 'cell jac:',ie,b(ci),ci,ii,i
c                 write(6,4) ie,xm3(i,1,1),xp(ci),jacm3(i,1,1)
c                 write(6,4) ii,ym3(i,1,1),yp(ci),jacm3(i,1,1)
c                 write(6,4) ci,zm3(i,1,1),zp(ci),jacm3(i,1,1)
                  call draw3(xm3(i,1,1),ym3(i,1,1),zm3(i,1,1))
               enddo
c  4           format(i5,1p4e14.6)
c              write(6,*) ie,nebad
c              call prs('continue?$')
c              call res(ans,1)
            endif
         enddo
      endif
c
      ie = iemin
      write(6,6) ie,vol(ie),rjmin,rjmax,ratio(ie)
      call hilight2(iemin,15)
C
      return
      end
c-----------------------------------------------------------------------
      subroutine CHKJAC(ierr,JAC,N)
C
C     Check the array JAC for a change in sign.
C
      REAL JAC(N)
      ierr=1
      SIGN = JAC(1)
      DO 100 I=2,N
         IF (SIGN*JAC(I).LE.0.0) return
  100 CONTINUE
      ierr=0
      return
      end
c-----------------------------------------------------------------------
      function gs_smooth(x,y,z,b,c,a,ja,ia,n,g,gg,s,eps,icall)
      real     c(1),a(1),x(1),y(1),z(1)
      integer  ja(1),ia(1),b(1),g(1)
c
c
c     Input parameters
c
c
c     b(i)  -- boundary (and box) information, b(i)>0 ==> point doesn't move
c     icall -- 1 ==> compute initial distance,  a(ij)
c     s     -- stretch.  s=.85 implies that a(ij)/d_final = 1./0.85.
c     eps   -- scaling factor for amount of movement on this pass
c     g(i)  -- graph distance from node i to boundary
c     gg    -- gain adjustment factor, such that s_ij = s*gg/(gi+gj+gg-1)
c              gg = 1--10 typical.  gg=1e9 ==> ~no boundary gain
c
c
      if (icall.eq.1) then
c
c        Build stiffness matrix based on original xj-xi distances
c
         do i=1,n
            j0 = ia(i)
            j1 = ia(i+1)-1
            do ij=j0,j1
               j = ja(ij)
               if (j.ne.i) then
                  dx  = (x(i)-x(j))
                  dy  = (y(i)-y(j))
                  dz  = (z(i)-z(j))
                  dx2 = dx*dx + dy*dy + dz*dz
                  ds  = sqrt(dx2)
                  a(ij) = ds
                  c(ij) = 1.
c                 Increase the stiffness for links connected to the boundary?
c                 if (b(i).eq.1.or.b(j).eq.1) c(ij) = 2.
                  if (gg.gt.1 .or. g(i).ne.0 .or. g(j).ne.0) then
                     c(ij) = gg/(g(i)+g(j)+gg-1.)
                  else
                     c(ij) = 0.
                  endif
               else
                  a(ij) = 0.
               endif
c              write(6,1) i,j,g(i),g(j),gg,c(ij),a(ij)
c  1           format(4i6,1p3e12.5)
            enddo
         enddo
      endif
c
      dmx = 0.
      do i=1,n
         if (b(i).le.0) then
            j0 = ia(i)
            j1 = ia(i+1)-1
            dxa=0.
            dya=0.
            dza=0.
            dsm=1.e22
            do ij=j0,j1
               j = ja(ij)
               if (j.ne.i) then
                  dx  = -(x(i)-x(j))
                  dy  = -(y(i)-y(j))
                  dz  = -(z(i)-z(j))
                  dx2 = dx*dx + dy*dy + dz*dz
                  ds  = sqrt(dx2)
                  dsm = min(ds,dsm)
c
c                 Strength of link is proportional to 
c
c                 c(ij) := 1./(g(i)+g(j)),  g(i) := dist | v(i) - boundary |
c
                  scale = c(ij)*(1.-s*a(ij)/ds)
                  dxa = dxa + scale*dx
                  dya = dya + scale*dy
                  dza = dza + scale*dz
               endif
            enddo
            xo = x(i)
            yo = y(i)
            zo = z(i)
c
c           This will ultimately allow planar constraints
c
            if (b(i).eq.-1) dxa = 0
            if (b(i).eq.-2) dya = 0
            if (b(i).eq.-3) dza = 0
            if (b(i).gt.0) then
               dxa = 0
               dya = 0
               dza = 0
            endif
c
            x(i) = x(i) + eps*dxa
            y(i) = y(i) + eps*dya
            z(i) = z(i) + eps*dza
            dxn = dxa*dxa + dya*dya + dza*dza
            dxn = eps*sqrt(dxn)
            dmx = max(dmx,dxn)
         endif
      enddo
      gs_smooth = dmx
      return
      end
c-----------------------------------------------------------------------
      subroutine smoother(cell,nvc,ia,ja,ww,b,g
     $                            ,x0,x1,y0,y1,z0,z1,ss,eps,idpth)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ww(0:1)
      integer ia(1),ja(1)
c
      integer cell(nvc,1),b(1),g(1)
c
c
      integer edge(2,12)
      save    edge
      data    edge  / 1,2 , 2,3 , 3,4 , 4,1
     $              , 5,6 , 6,7 , 7,8 , 8,5
     $              , 1,5 , 2,6 , 3,7 , 4,8 /
c
      integer iebad(nelm)
      real    vol(nelm),volo(nelm),ratio(nelm)
      common /csmooth/ vol,volo,ratio
c
c
      icount= 0
      nebad = 0
      call izero(iebad,nelm)
 1000 continue
      icount = icount+1
c
      gg = idpth
      if (icount.gt.1) then
         call prs
     $    ('input relaxation params (s,eps,gg) (typ 0.9,.01,1-999):$')
         call rerrr(ss,eps,gg)
         idpth = gg
      endif
c
c
c     use connectivity graph to smooth mesh
c
      n1 = nvc*nel
      call get_gxyz(nv,cell,nvc,ncell,ww(n1),ww,ierr)
c
c     Identify boundary and curve side vertices
      nvc=2**ndim
      nfaces=2*ndim
      call izero(b,nv)
      call find_bc_crv(b,nv,cell,nvc,ncell)
c
c     Identify dof's inside smoothing box
      call find_sm_box(b,nv,x0,x1,y0,y1,z0,z1)
c
c     Identify dof's connected to elements yielding bad Jacobians
      call find_bd_jac(b,nv,cell,nvc,nel,iebad,nebad)
c
c
c
c     Build adjaceny graph
c
      ne = 2*ndim*(ndim-1)
      n1 = 14*ne*ncell
      n2 = 14*ne*ncell + n1
      call build_adj  (ja,ia,n,cell,nvc,ncell,edge,ne,ww,ww(n1),ww(n2))
c
c     Construct a connectivity list telling the topological distance from
c     boundary points
c
      call boundary_graph(g,b,ja,ia,n)
c
c
c     Smooth the mesh
c
      nnz = ia(n+1)-ia(1)
      do iter=1,80
         dxi=gs_smooth (xp,yp,zp,b,ww,ww(nnz),ja,ia,n,g,gg,ss,eps,iter)
c        dxi=gs_smooth2(xp,yp,zp,b,ww,ww(nnz),ja,ia,n,g,gg,ss,eps,iter)
         write(6,*) iter,' smooth:',dxi,eps
      enddo
      call chk_jacob(vol,ratio,nebad,iebad,b,xp,yp,zp,cell,nvc)
      if (nebad.gt.0.and.icount.lt.3) then
         write(6,*) 'Bad jacobian',nebad,icount
         goto 1000
      elseif (nebad.gt.0) then
         write(6,*) 'Still Bad jacobian, no smoothing',nebad,icount
      endif
c
      do ie=1,nel
      do i=1,nvc
        xo=x(ie,i)
        if (b(cell(i,ie)).le.0) then
           x(ie,i) = xp(cell(i,ie))
           y(ie,i) = yp(cell(i,ie))
           z(ie,i) = zp(cell(i,ie))
        endif
        jc=cell(i,ie)
      enddo
      enddo
    1 format('xc:',2i4,i9,4f12.5)
    2 format(a3,2i4,i9,2f12.5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine dilate_el(dxyz,xyz,eps,vol,volo,volm)
c
      real dxyz(3,8),xyz(3,8)
      real cg(3),xc,yc,zc
      equivalence (xc,cg(1)),(yc,cg(2)),(zc,cg(3))
c
      real l,lo
c
      d = 3.
c
      call rzero(cg,3)
      do i=1,8
         xc=xc+xyz(1,i)
         yc=yc+xyz(2,i)
         zc=zc+xyz(3,i)
      enddo
      xc = xc/8.
      yc = yc/8.
      zc = zc/8.
c
      l     = vol**(1./d)
      lo    = volo**(1./d)
      a     = -volo/volm
      e     = exp(a)
      scale = 1+e*(lo-l)/lo
c
      do i=1,8
      do j=1,3
         xyz(j,i) = cg(j) + scale*(xyz(j,i)-cg(j))
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      function vec_length(v,n)
      real v(1)
c
      s = 0.
      do i=1,n
         s = s + v(i)*v(i)
      enddo
      if (s.gt.0) s=sqrt(s)
      vec_length = s
c
      return
      end
c-----------------------------------------------------------------------
      subroutine angle_fix(dxyz,xyz)
c
c     update positions of quadrilateral to try to make it rectangular
c
c     Input:    xyz
c     Output:  dxyz -- change
c
      real dxyz(3,0:3),xyz(3,0:3)
c
      real v1(3),v2(3),vn(3),v1p(3),v2p(3),v1n(3),v2n(3)
      real len1,len2
c
c     Data assumed stored in EB preprocessor format  --  counter-clockwise
c
      call rzero(dxyz,3*4)
c
      one = 1.
      pi2 = 2.*atan(one)
c
      do ic=0,3
         ip = mod(ic+1,4)
         im = mod(ic+3,4)
         call sub3(v1,xyz(1,ip),xyz(1,ic),3)
         call sub3(v2,xyz(1,im),xyz(1,ic),3)
         len1 = vec_length(v1,3)
         len2 = vec_length(v2,3)
         call cross (vn,v1,v2)
         call norm3d(vn)
c
c        Compute new v2
c
         call cross (v2p,vn,v1)
         call norm3d(v2p)
         s = len2
         call cmult (v2p,s,3)
         call sub2  (v2p,v2,3)
         s = len1/(len1+len2)
         call cmult (v2p,s,3)
         call add2  (dxyz(1,im),v2p,3)
c
c
c        Compute new v1
c
         call cross (v1p,v2,vn)          ! v1' = target, new v1 orth. to v2
         call norm3d(v1p)
         s = len1
         call cmult (v1p,s,3)
         call sub2  (v1p,v1,3)
         s = len2/(len1+len2)
         call cmult (v1p,s,3)
         call add2  (dxyz(1,ip),v1p,3)
c
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine angle_smooth_el(dxyz,xyz,eps)
c
c     update positions of element to try to make it rectangular
c
c     Input:    xyz
c     Output:  dxyz -- change
c
      real dxyz(3,8),xyz(3,8)
      real dxyzs(3,4),xyzs(3,4)
      real cg(3)
c
      integer icface(4,6)
      save    icface
      data    icface/ 1,5,6,2, 2,3,7,6,       ! ordered ctr clockwise,
     $                3,4,8,7, 1,5,8,4,       ! EB notation
     $                1,2,3,4, 5,6,7,8/
c
c
      call rzero(dxyz,24)
c
c     Begin with angle corrections
c
      do iface=1,6
         do ic=1,4
            call copy(xyzs(1,ic),xyz(1,icface(ic,iface)),3)
         enddo
         call angle_fix(dxyzs,xyzs)
         do ic=1,4
            call add2(dxyz(1,icface(ic,iface)),dxyzs(1,ic),3)
         enddo
      enddo
c
c     Scale by eps
c
      call cmult(dxyz,eps,24)
c
c     Shift to have zero mean, so that element is not (self) translated
c
c
      call rzero(cg,3)
      do i=1,8
      do j=1,3
         cg(j)=cg(j)+dxyz(j,i)
      enddo
      enddo
      call cmult(cg,.125,3)
c
      do i=1,8
      do j=1,3
         dxyz(j,i) = dxyz(j,i) - cg(j)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine show_plot
      include 'basics.inc'
      include 'basicsp.inc'
c
      SUATTR = 'SCALAR PLOT'
      SCALPT = 'COLOR FILL'
      PLFORM = 'SURFACE PLOT'
      QUANTY = 'PRESSURE'
      ntot   = nx*ny*nz*nel
      call copy(work,p,ntot)
      call plotit(0)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_mesh(xyz)
      include 'basics.inc'
c
      real xyz(3,8,1)
c
      do ie=1,nel
      do i =1,8
         xyz(1,i,ie) = x(ie,i)
         xyz(2,i,ie) = y(ie,i)
         xyz(3,i,ie) = z(ie,i)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine reset_mesh(xyz)
      include 'basics.inc'
c
      real xyz(3,8,1)
c
      do ie=1,nel
      do i =1,8
         x(ie,i) = xyz(1,i,ie)
         y(ie,i) = xyz(2,i,ie)
         z(ie,i) = xyz(3,i,ie)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine angle_smoother
     $              (cell,nvc,ia,ja,ww,b,g,x0,x1,y0,y1,z0,z1,ein)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ww(0:1)
      integer ia(1),ja(1)
c
      integer cell(nvc,1),b(1),g(1)
c
      real dxyz(3,8),xyze(3,8)
c
c
      integer edge(2,12)
      save    edge
      data    edge  / 1,2 , 2,3 , 3,4 , 4,1
     $              , 5,6 , 6,7 , 7,8 , 8,5
     $              , 1,5 , 2,6 , 3,7 , 4,8 /
c
      integer iebad(nelm)
      real vol(nelm),volo(nelm),ratio(nelm)
      common /csmooth/ vol,volo,ratio
c
c
      epsi = abs(ein)
      icount= 0
      nebad = 0
      call izero(iebad,nelm)
 1000 continue
      icount = icount+1
c
c     use connectivity graph to smooth mesh
c
      n1 = nvc*nel
      call get_gxyz(nv,cell,nvc,ncell,ww(n1),ww,ierr)
      call chk_jacob(volo,ratio,nebad,iebad,b,xp,yp,zp,cell,nvc)
      volm = glmax(volo,nel)
      voli = glmin(volo,nel)
      write(6,*) 'Vol:',voli,volm
c
c     Identify boundary and curve side vertices
      nvc=2**ndim
      nfaces=2*ndim
      call izero(b,nv)
      call find_bc_crv(b,nv,cell,nvc,ncell)
c
c     Identify dof's inside smoothing box
      call find_sm_box(b,nv,x0,x1,y0,y1,z0,z1)
c
c     Identify dof's connected to elements yielding bad Jacobians
      call find_bd_jac(b,nv,cell,nvc,nel,iebad,nebad)
c
c     Build adjaceny graph
      ne = 2*ndim*(ndim-1)
      n1 = 14*ne*ncell
      n2 = 14*ne*ncell + n1
      call build_adj  (ja,ia,n,cell,nvc,ncell,edge,ne,ww,ww(n1),ww(n2))
c
c     Construct a connectivity list telling the topological distance from
c     boundary points
c
c     call boundary_graph2(g,b,ja,ia,n,xp,yp,zp)
      call boundary_graph(g,b,ja,ia,n)
c
c
c     Smooth the mesh
c
c     March through each element, obtaining changes associated with each
c
c     call prs('Input update epsilon ( <<1 ):$')
c     call rer(eps)
c
      do ie=1,nel
         expon = 1./(2.-abs(ratio(ie)))   !! put more emphasis on bad Jacobian cases
         eps = epsi**expon
         do i=1,nvc
            xyze(1,i) = x(ie,i)
            xyze(2,i) = y(ie,i)
            xyze(3,i) = z(ie,i)
         enddo
         call angle_smooth_el(dxyz,xyze,eps)
         dmax = vlamax(dxyz,24)
c        write(6,*) ie,' dmax:',dmax,volo(ie)
         do i=1,nvc
            xp(cell(i,ie)) = xp(cell(i,ie)) + dxyz(1,i)
            yp(cell(i,ie)) = yp(cell(i,ie)) + dxyz(2,i)
            zp(cell(i,ie)) = zp(cell(i,ie)) + dxyz(3,i)
         enddo
      enddo
      call chk_jacob(vol,ratio,nebad,iebad,b,xp,yp,zp,cell,nvc)
c
c
c     Dilate, to preserve volume of small elements
c
      if (ein.lt.0) goto 777
      do ie=1,nel
         do i=1,nvc
            xyze(1,i) = xp(cell(i,ie))
            xyze(2,i) = yp(cell(i,ie))
            xyze(3,i) = zp(cell(i,ie))
         enddo
         call dilate_el(dxyz,xyze,eps,vol(ie),volo(ie),volm)
         do i=1,nvc
            xp(cell(i,ie)) = xp(cell(i,ie)) + dxyz(1,i)
            yp(cell(i,ie)) = yp(cell(i,ie)) + dxyz(2,i)
            zp(cell(i,ie)) = zp(cell(i,ie)) + dxyz(3,i)
         enddo
      enddo
  777 continue
c
c     Update mesh points
c
      do ie=1,nel
      do i=1,nvc
        if (b(cell(i,ie)).le.0) then
           x(ie,i) = xp(cell(i,ie))
           y(ie,i) = yp(cell(i,ie))
           z(ie,i) = zp(cell(i,ie))
        endif
        jc=cell(i,ie)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine boundary_graph(g,b,ja,ia,n)
      integer  g(1),ja(1),ia(1),b(1)
      logical  ifdone
c
c
c     Build stiffness matrix based on original xj-xi distances
c
      do i=1,n
         g(i) = n+9999999
         if (b(i).eq.1) g(i)=0
      enddo
      do ipass = 1,20
         ifdone=.true.
         do i=1,n
            j0 = ia(i)
            j1 = ia(i+1)-1
            do ij=j0,j1
               j = ja(ij)
c              a_ij is a link
               if (abs(g(i)-g(j)).gt.1) ifdone = .false.
               if (g(i).gt.g(j)) g(i)=g(j)+1
               if (g(j).gt.g(i)) g(j)=g(i)+1
            enddo
         enddo
         if (ifdone) goto 10
      enddo
   10 continue
      maxgraph = ilmax(g,n)
      call prsii('npasses, graph depth:$',ipass,maxgraph)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine remap_xyz2(cell,nvc,b)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer cell(nvc,1),b(1)
c
      do ie=1,nel
      do i=1,nvc
        if (b(cell(i,ie)).le.0) then
           x(ie,i) = xp(cell(i,ie))
           y(ie,i) = yp(cell(i,ie))
           z(ie,i) = zp(cell(i,ie))
        endif
      enddo
      enddo
c
      call mespos
      call gencen
      call coef
c
      return
      end
c-----------------------------------------------------------------------
      subroutine build_graph(cell,nvc,ia,ja,ww,b,g
     $                           ,x0,x1,y0,y1,z0,z1)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ww(0:1)
      integer ia(1),ja(1)
c
      integer cell(nvc,1),b(1),g(1)
c
c
      integer edge(2,12)
      save    edge
      data    edge  / 1,2 , 2,3 , 3,4 , 4,1
     $              , 5,6 , 6,7 , 7,8 , 8,5
     $              , 1,5 , 2,6 , 3,7 , 4,8 /
c
      integer iebad(nelm)
c
c     Build connectivity graph 
c
      n1 = nvc*nel
      call get_gxyz(nv,cell,nvc,ncell,ww(n1),ww,ierr)
c
c     Identify boundary and curve side vertices
      nvc=2**ndim
      nfaces=2*ndim
      call izero(b,nv)
      call find_bc_crv(b,nv,cell,nvc,ncell)
c
c     Identify dof's inside smoothing box
      call find_sm_box(b,nv,x0,x1,y0,y1,z0,z1)
c
c
c     Build adjaceny graph
c
      ne = 2*ndim*(ndim-1)
      n1 = 14*ne*ncell
      n2 = 14*ne*ncell + n1
      call build_adj  (ja,ia,n,cell,nvc,ncell,edge,ne,ww,ww(n1),ww(n2))
c
      return
      end
c-----------------------------------------------------------------------
      subroutine build_surface_pointers(cell,ncrnr)
c
c     Suite of routines to make elements near to wall orthogonal to wall
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer cell(ncrnr,1)
c
      common /srfci/ se_to_face,vertex,tmp_face,nelpf,srfef,srfvt
      common /srfcr/ srfvx,srfvn,srfnb,phil,phihat,vtxpt
      common /ctmp1/ wrk(3*8*nelm)
c
      integer se_to_face(6*nelm)
      integer vertex(8*nelm)
      integer tmp_face(3,6,nelm)
      integer nelpf(6,nelm)
c
      integer srfef(2,0:6*nelm),srfvt(4,6*nelm)
      real    srfvx(3,4,6*nelm),srfvn(3,4,6*nelm),srfnb(3,4,6*nelm)
      real    phil(8*nelm),phihat(8*nelm)
      real    vtxpt(24*nelm)
c
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      real    phib(1000)
      integer jk  (1000)
c
      nfcs  = 2*ndim
      nfc   = 2**(ndim-1)
      if (icalld.ge.0) then
         icalld = icalld + 1
         call get_vert(vertex,ncrnr)
      endif
      nvert = ilmax(vertex,ncrnr*nel)
c
      write(6,*) 'call face_id',nvert
      call face_id
     $          (se_to_face,nfctot,vertex,ncrnr,wrk,tmp_face,nfcs)
c
c
      write(6,*) 'call surface_id',nfctot
      call surface_id(nelpf,nfctot,se_to_face,nfcs,vertex,ncrnr
     $                   ,srfef,srfvt,srfvx,nfc,vtxpt)
c
      write(6,*) 'call defect_id',nfctot
c     call selite(nelpf,se_to_face,nfcs,2)
      call mean_normal(srfnb,srfvn,srfef,srfvt,srfvx,nfc,ndim,nvert,wrk)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mean_normal(srfnb,srfvn,srfef,srfvt,srfvx,nfc,nd,nvrt
     $              ,work)
c
      include 'basics.inc'
      integer srfef(2,0:1),srfvt(nfc,1)
      real    srfvx(nd,nfc,1),srfvn(nd,nfc,1),srfnb(nd,nfc,1)
      real    work(nd,nvrt)
c
c
c
      nsrf = srfef(1,0)
c
      do k=1,nsrf
         call gen_srf_nrm(srfvn(1,1,k),srfvx(1,1,k),nd,nfc)
         do j=1,nfc
c           call arow3nc(srfvx(1,j,k),srfvx(2,j,k),srfvx(3,j,k)
c    $                  ,srfvn(1,j,k),srfvn(2,j,k),srfvn(3,j,k)
c    $                  ,sarrow,15)
         enddo
      enddo
c
c
c     Compute mean normals
c
      call vec_vtx_sum(srfnb,srfvn,srfvt,nfc,nsrf,nvrt,work,nd)
      do k=1,nsrf
      do j=1,nfc
         call norm3d(srfnb(1,j,k))
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine set_surf_layer(x0,x1,y0,y1,z0,z1)
c
c     Suite of routines to make elements near to wall orthogonal to wall
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer cell(1),b(1),ja(1),ia(1),ww(1),g(1)
      equivalence (cell,wkv1)
      equivalence (ja  ,wkv2)
      equivalence (ia  ,wkv3)
      equivalence (ww  ,work)
      equivalence (b   ,wrk2)
      equivalence (g   ,jacm1)
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      common /smeshr/ xyzo(3,8,nelm),xyzl(3,8,nelm)
      common /smeshi/ ioset
c
c     if (icalld.eq.0) then
c        icalld = icalld + 1
c        call set_mesh(xyzo)
c     endif
c
      nvc = 2**ndim
c
      call build_surface_pointers(cell,nvc)
c
      call build_graph(cell,nvc,ia,ja,ww,b,g
     $                           ,x0,x1,y0,y1,z0,z1)
c
      call prs('Input spacing off of wall, delta:$')
      call rer(delta)
      call prs('Input averaging factor , alpha:$')
      call rer(alpha)
c
      call update_surface_layer(ia,ja,b,delta,alpha)
      call remap_xyz2(cell,nvc,b)
      call show_plot
c
      write(6,*) 'call out_rea'
      call out_rea
c
      return
      end
c-----------------------------------------------------------------------
      subroutine update_surface_layer(ia,ja,b,delta,alpha)
c
c     Suite of routines to clean up mesh imperfections
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer ia(1),ja(1),b(1)
c
c
      common /srfci/ se_to_face,vertex,tmp_face,nelpf,srfef,srfvt
      common /srfcr/ srfvx,srfvn,srfnb,phil,phihat,vtxpt
c
      integer se_to_face(6*nelm)
      integer vertex(8*nelm)
      integer tmp_face(3,6,nelm)
      integer nelpf(6,nelm)
c
      integer srfef(2,0:6*nelm),srfvt(4,6*nelm)
      real    srfvx(3,4,6*nelm),srfvn(3,4,6*nelm),srfnb(3,4,6*nelm)
      real    phil(8*nelm),phihat(8*nelm)
      real    vtxpt(24*nelm)
c
c
      common /ctmp1/ wrk(3*8*nelm)
c
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      real    phib(1000)
      integer jk  (1000)
c
      ncrnr = 2**ndim
      nfcs  = 2*ndim
      nfc   = 2**(ndim-1)
      nsrf  = srfef(1,0)
c
      do js=1,nsrf
      do jc=1,nfc
         i = srfvt(jc,js)
c        write(6,*) 'ibcs:',i,b(i),jc,js,ia(i),ia(i+1)
         if (b(i).eq.1) then                 ! i is a wall boundary point
            j0 = ia(i)
            j1 = ia(i+1)-1
            do ij=j0,j1
               j = ja(ij)
c              write(6,*) 'jbcs:',j,b(j),jc,js
               if (b(j).le.0) then
c                 a_ij is a link with v_j interior
c
                  dx = xp(j)-xp(i)
                  dy = yp(j)-yp(i)
                  dz = zp(j)-zp(i)
c
                  deltn = dx*dx+dy*dy+dz*dz
                  deltn = sqrt(deltn)
c
                  if (delta.lt.0) deltn = abs(delta)
c
                  xpjn  = xp(i) - deltn*srfnb(1,jc,js)
                  ypjn  = yp(i) - deltn*srfnb(2,jc,js)
                  zpjn  = zp(i) - deltn*srfnb(3,jc,js)
c
                  xp(j) = alpha*xp(j) + (1.-alpha)*xpjn
                  yp(j) = alpha*yp(j) + (1.-alpha)*ypjn
                  zp(j) = alpha*zp(j) + (1.-alpha)*zpjn
c
                  write(6,*) deltn
                  write(6,6) i,'zwall:',xp(i),yp(i),zp(i)
                  write(6,6) j,'znew :',xp(j),yp(j),zp(j)
                  dx = xp(j)-xp(i)
                  dy = yp(j)-yp(i)
                  dz = zp(j)-zp(i)
c                 call arow3nc(xp(i),yp(i),zp(i),dx,dy,dz,1.,14)
               endif
            enddo
         endif
      enddo
      enddo
    6 format(i9,1x,a6,3f13.6)
c
      return
      end
c-----------------------------------------------------------------------
      function gs_smooth2(x,y,z,b,c,xyzn,ja,ia,n,g,gg,s,lam,icall)
      real     c(1),xyzn(3,1),x(1),y(1),z(1),lam
      integer  ja(1),ia(1),b(1),g(1)
c
c
c     Input parameters
c
c
c     b(i)  -- boundary (and box) information, b(i)>0 ==> point doesn't move
c     icall -- 1 ==> compute initial distance,  a(ij)
c     s     -- stretch.  s=.85 implies that a(ij)/d_final = 1./0.85.
c     lam   -- scaling factor for amount of movement on this pass
c     g(i)  -- graph distance from node i to boundary
c     gg    -- gain adjustment factor, such that s_ij = s*gg/(gi+gj+gg-1)
c              gg = 1--10 typical.  gg=1e9 ==> ~no boundary gain
c
      do i=1,n
         j0 = ia(i)
         j1 = ia(i+1)-1
c
         if (b(i).le.0) then
c
c           Make denominator
            den = 0.
            do ij=j0,j1
               j = ja(ij)
               if (j.ne.i) then
                  dx  = x(i)-x(j)
                  dy  = y(i)-y(j)
                  dz  = z(i)-z(j)
                  dx2 = dx*dx + dy*dy + dz*dz
                  dsi = 1./sqrt(dx2)
                  den = den + dsi
               endif
            enddo
            scali = lam/den
c
c           Update
c
            do ij=j0,j1
               j = ja(ij)
               if (j.ne.i) then
                  dx  = x(i)-x(j)
                  dy  = y(i)-y(j)
                  dz  = z(i)-z(j)
                  dx2 = dx*dx + dy*dy + dz*dz
                  sij = scali/sqrt(dx2)
                  xyzn(1,i) = x(i) - sij*dx
                  xyzn(2,i) = y(i) - sij*dy
                  xyzn(3,i) = z(i) - sij*dz
               endif
            enddo
         endif
      enddo
c
c
c     Update, and check max-displacement
c
      dxm = 0.
      do i=1,n
         if (b(i).le.0) then
            dx   = xyzn(1,i)-x(i)
            dy   = xyzn(2,i)-y(i)
            dz   = xyzn(3,i)-z(i)
            dx2  = dx*dx + dy*dy + dz*dz
            dxm  = max(dx2,dxm)
            x(i) = xyzn(1,i)
            y(i) = xyzn(2,i)
            z(i) = xyzn(3,i)
         endif
      enddo
      dxm = sqrt(dxm)
c
c
      gs_smooth2 = dxm
      return
      end
c-----------------------------------------------------------------------
