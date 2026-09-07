c-----------------------------------------------------------------------
      subroutine userf
c     Driver for user-defined functions
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /quantc/ quanto
      character*20 quanto
c
      nchoic = 1
      item(nchoic) = 'UP MENU'
      nchoic   = nchoic+1
      item(nchoic) = 'Scale vector component'
      nchoic   = nchoic+1
      item(nchoic) = 'Shift vector component'
      nchoic   = nchoic+1
      item(nchoic) = 'Const. per elem.'
      nchoic   = nchoic+1
c     item(nchoic) = 'Wave Space Z'
c     nchoic   = nchoic+1
      item(nchoic) = 'Read fields from file'
      nchoic   = nchoic+1
      item(nchoic) = 'COORDINATE'
      nchoic   = nchoic+1
      item(nchoic) = 'Error Distribution'
      nchoic   = nchoic+1
      item(nchoic) = 'x-z average'
      nchoic   = nchoic+1
      item(nchoic) = 'BIG IMAGE'
      nchoic   = nchoic+1
      item(nchoic) = '6-way average'
      nchoic   = nchoic+1
      item(nchoic) = 'Rad. vel. (R,Th,Pnchoic)'
      nchoic   = nchoic+1
      item(nchoic) = 'Enstrophy output'
      nchoic   = nchoic+1
      item(nchoic) = 'Anil''s problem'
      nchoic   = nchoic+1
      item(nchoic) = 'Gary''s problem'
      nchoic   = nchoic+1
      item(nchoic) = 'Miles''s problem'
      nchoic   = nchoic+1
      item(nchoic) = 'Miles''s problem 2'
      nchoic   = nchoic+1
      item(nchoic) = 'Miles''s steady state'
      nchoic   = nchoic+1
      item(nchoic) = 'Miles''s problem 4'
      nchoic   = nchoic+1
      item(nchoic) = 'Miles''s problem 5'
      nchoic   = nchoic+1
      item(nchoic) = 'Cluster BC'
      nchoic   = nchoic+1
      item(nchoic) = 'add constant'
      if (ndim.eq.2) then
         nchoic   = nchoic+1
         item(nchoic) = 'view .map file'
      endif
c
      nchoic   = 9 
c
      call menu(xmouse,ymouse,button,'Define USERF')
c
      if (choice.eq.'UP MENU') then
         return
c
c
      elseif (choice.eq.'COORDINATE') then
         ntot = nx*ny*nz*nel
         call copy(u,xp,ntot)
         call copy(v,yp,ntot)
         call copy(w,zp,ntot)
         call copy(wkv1,xp,ntot)
         call copy(wkv2,yp,ntot)
         call copy(wkv3,zp,ntot)
         QUANTY = 'VELOCITY'  ! quick hack
c
      elseif (choice.eq.'BIG IMAGE') then
         call big_image
c
      elseif (choice.eq.'Const. per elem.') then
         call const_per_elem
c
      elseif (choice.eq.'Error Distribution') then
         call error_dist
c
      elseif (choice.eq.'6-way average') then
         call xz6_average
c
      elseif (choice.eq.'x-z average') then
         call xz_average
c
      elseif (choice.eq.'Wave Space Z') then
         call wave_space_z
c
c
      elseif (choice.eq.'Cluster BC') then
         call cluster_bc
c
c
      elseif (choice.eq.'Shift vector component') then
         call shift_comp
c
c
      elseif (choice.eq.'view .map file') then
         call view_map
c
c
      elseif (choice.eq.'Scale vector component') then
         call scale_comp
c
c
      elseif (choice.eq.'Rad. vel. (R,Th,Phi)') then
         call rad_vel
c
c
      elseif (choice.eq.'Read fields from file') then
         call read_fld
c
c
      elseif (choice.eq.'Enstrophy output') then
         call enstrophy_out
c
c
      elseif (choice.eq.'Anil''s problem') then
         call set_anil
c
c
      elseif (choice.eq.'Gary''s problem') then
         call gkl_userf
c
c
      elseif (choice.eq.'Miles''s problem') then
         call compute_miles
c
      elseif (choice.eq.'Miles''s problem 2') then
         call compute_miles2
c
      elseif (choice.eq.'Miles''s steady state') then
         call compute_miles3
c
      elseif (choice.eq.'Miles''s problem 4') then
         call compute_miles4
c
      elseif (choice.eq.'Miles''s problem 5') then
         call compute_miles5
c
      elseif (choice.eq.'SQRT') then
         ntot = nx*ny*nz*nel
         call vsqrt(work,ntot)
c
      elseif (choice.eq.'add constant') then
         call add_const
c
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine rad_vel
c     Driver for user-defined functions
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c
      ntot = nx*ny*nz*nel
      do i=1,ntot
         radi    = sqrt(xp(i)*xp(i)+yp(i)*yp(i)+zp(i)*zp(i))
         wkv1(i) = (u(i)*xp(i)+v(i)*yp(i)+w(i)*zp(i))/radi
         wkv2(i) = 0.
         wkv3(i) = 0.
         rad2    = xp(i)*xp(i)+yp(i)*yp(i)
         if (rad2.gt.0) then
            rad2    = sqrt(rad2)
            wkv2(i) = (v(i)*xp(i)-u(i)*yp(i))/rad2
            wkv3(i) = (u(i)*u(i)+v(i)*v(i)+w(i)*w(i))
         endif
      enddo
      do i=1,ntot
         if (wkv3(i).gt.0.)
     $   wkv3(i) = sqrt(wkv3(i)**2 - wkv1(i)**2 - wkv2(i)**2)
      enddo
c
c     load component of choice into WORK
      call get_component
c
      return
      end
c-----------------------------------------------------------------------
C  TABLE OF CONTENTS
C   1. Examples
C   2. vortex_tracking : Driver
C   3. id_cores  : Identify the vortex core set
C   4. cluster_cores : Identifies each vortex core
C   5. defjcc : Indexes the cores
C   6. core_position : Locates each vortex core position
C   7. vrticity : Calculates vorticity
C   8. core_enstrophy : Calcs the enstrophy for each vortex core
C   9. bottom_stress : Calcs average stress exerted by bottom face of
C                      each bottom element
C  10. particle_forces : Calc the force on each particle
C**************************************************
C   GLOSSARY :
c   nface : # of bottom elements ; in common
c   nhemi : # of hemispheres on bottom ; in common
c   bttm_elist(k,j) : k = 1,nface, j = 1,2
c                     ie = bttm_list(k,1) is element #
c                     kh = bttm_list(k,2) is hemi #
c   nve : # of elements in Vortex Core Set (VCS) ; in common
c   nccolor : # of disjoint sets in VCS( # distinct cores); in common
c   (xpc,ypc,zpc)(ic) , ic = 1,nccolor :Position of each Core;incom
c   (vrt1,vrt2,vrt3)(m) ,m = 1,ntot  ;Vorticity Vector;incom
c   enstrophy(ic), ic = 1,nccolor : Enstrophy for each core ; incom
c   (strss1,strss2,strss3)(k),k=1,nface : Avg stress on bottom face of
c                each bottom element. incom
c   (drag1,drag2,drag3)(kh) , kh = 1,nhemi : Drag on each hemi;incom
c
C**************************************************
c-----------------------------------------------------------------------
      subroutine gkl_userf
c
c    Use this routine to define custom functions for viewing in postnek.
c     
c     Output should be in array work() for scalar functions.
c     For vector functions, output should be in arrays 
c        wkv1(), wkv2(), and wkv3().
c     
c     Mesh geometry is stored at points (xp,yp,zp), i=1,...,ntot,
c     where ntot=nx*ny*nz*nel, nx=ny=nz=order (poly deg+1), and 
c     nel = number of spectral elements.
c     
c     
c     Velocity, pressure and temperature are stored in a similar
c     manner:  (u,v,w,p,t).
c     
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c     HERE IS AN EXAMPLE:   
c
c     This computes the normal component of velocity in a spherical
c     shell geometry:
c
c      ntot = nx*ny*nz*nel
c      do i=1,ntot
c         radi    = sqrt(xp(i)*xp(i)+yp(i)*yp(i)+zp(i)*zp(i))
c         work(i) = (u(i)*xp(i)+v(i)*yp(i)+w(i)*zp(i))/radi
c      enddo
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c     See postnek3.f for add'l examples, (e.g., VORTICITY)
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c
c
      call vortex_tracking
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vortex_tracking
c
c     01-08-99 Driver routine to track vortices
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      integer icalld
      REAL drag1(nhemimx),drag2(nhemimx),drag3(nhemimx)
      save    icalld
      data    icalld  /0/
c---------------------------------------
C  $$$$$$   Build bttm_elist(k,j) here
C The loading of bttm_elist(k,1) is at the moment in sub bottom_stress
c at ~ lines 674 - 686 . bttm_elist(k,2) has not yet been
c loaded. Can be done here or there.
c$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
c
c  Compute vortex function, if it hasn't happened already
      ntot = nx*ny*nz*nel

      CALL new_vortex
      CALL copy(work,vortex,ntot)
c
c  Identify elements containing negative lam2(Vortex Cores)

      CALL id_cores
c
c  Cluster the elements and give them a color.(Identify & label
c  individual cores.

      CALL cluster_cores(icalld)
      icalld = icalld+1

c    LOCATE POSITION OF EACH VORTEX CORE
c   (xpc,ypc,zpc)(ic) , ic = 1,nccolor :Position of each Core;incom

       CALL core_position

C    CALC VORTICITY VECTOR (vrt1,vrt2,vrt3)(m)
c   (vrt1,vrt2,vrt3)(m) ,m = 1,ntot  ;Vorticity Vector;incom

       CALL vrticity

C    Calc Enstrophy for each core
c   enstrophy(ic), ic = 1,nccolor : Enstrophy for each core ; incom

       CALL core_enstrophy

C     Calc Avg Bottom Stress on each Bottom Face
c   (strss1,strss2,strss3)(k),k=1,nface : Avg stress on bottom face of
c                each bottom element. incom

c      CALL set_surface_jacobians (unx,uny,unz,area)
c      CALL bottom_stress         (unx,uny,unz,area,bttm_farea)

C       CALC FORCE ON EACH HEMI
c   (drag1,drag2,drag3)(kh) , kh = 1,nhemi : Drag on each hemi;incom

c       CALL particle_forces(bttm_farea)
c
c$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
c        The principal output of "cluster_cores" is array "core_color()"
c     It identifies the group, or color, of each contiguous set of 
c     elements containing a negative lam2 value.  It is accessed by
c     element number.  Hence, the loop:
c
c     do i=1,nve
c         ie  = iev(i)
c         ic  = core_color(ie)
c         call hilight2(ie,ic)
c     enddo
c
c     would pass element number ie, core number ic, to the routine
c     hilight2(), which would in turn high-light element ie with color
c     number ic  (ic=0 white, ic=2 blue, ic=15-red).  (Note that it's
c     better to use ic = mod(ic,13)+2, so that the core numbers remain
c     within the postnek color-map range.
c
c        In addition, cluster_cores returns the number of core colors, 
c    nccolor (in basicsp.inc), as well as a means for accessing elements
c   containing negative lam2 values on a color by color (group by group)
c   basis.   For example, the following loop achieves the same as above:
c
c     do i=1,nccolor
c        do j=jcc(i),jcc(i+1)-1
c           ie = iev   (j)
c           ic = ccolor(j)
c           call hilight2(ie,ic)
c        enddo
c     enddo
c
c     This second form is convenient for computing volume integrals
c     over distinct core clusters, etc., since nccolor reflects the
c     number of clusters, and [jcc(i),jcc(i+1)-1] contains the index
c     range over iev() of elements within the cluster.
c
c$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
      return
      end
c-----------------------------------------------------------------------
      subroutine id_cores
c
c     01-07-99 Routine to indentify Vortex Cores Set
c     NVE is the # elements in this set.
c     The arrays are in common VRTXTRK in basicsp.inc
c     NVE,IEV(),IXV(),IYV(),IZV(),LAM2()
c
      include 'basics.inc'
      include 'basicsp.inc'
C  LOCAL
      INTEGER ie,ix,iy,iz,ntot,nc,m
      INTEGER ixe,iye,ize
      REAL lm2,lm2mn,lme
c
c     ind(i,j,k,iel)=i+nx*(j-1)+nx*ny*(k-1)+nx*ny*nz*(iel-1)
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c      Initialize LAM2() to +1.0 . 
c      Then if ie is in VCS,LAM2(ie) <= 0 .If ie is NOT in VCS,then
c      LAM2(ie) = 1.0 . This could be useful in cluster identification.
       DO ie = 1,nel
          LAM2(ie) = 1.0
       END DO
       ntot = nx*ny*nz*nel
       nve = 0
c
       eps = .02*glmin(vortex,ntot)
       if (eps.ge.0.) return
c
       DO ie = 1,nel

          lm2mn = 1.0
          DO iz = 1,nz
            DO iy = 1,ny
              DO ix = 1,nx
c                m = IND(ix,iy,iz,ie)
                 m = ix+nx*(iy-1)+nx*ny*(iz-1)+nx*ny*nz*(ie-1)
                 lm2 = vortex(m)
                 IF(lm2 .LE. eps) THEN
                   IF(lm2 .LT. lm2mn) THEN
                      lm2mn = lm2
                      ixe = ix
                      iye = iy
                      ize = iz
                   END IF
                 END IF
              END DO
            END DO
          END DO
          IF( lm2mn .LE. eps) THEN
            nve = 1 + nve
            IEV(nve) = ie
            IXV(nve) = ixe
            IYV(nve) = iye
            IZV(nve) = ize
            LAM2(ie) = lm2mn
          END IF

       END DO
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c   VCS = Vortex Cores Set
c   VCS is a set of NVE elements such that for 1 <= nc <= NVE, setting
c ie = IEV(nc) is an element which contains at least one collocation
c point for which lambda2 <= 0 .If m = IND(IXV(nc),IYV(nc),IZV(nc),ie) 
c then the most neg lambda2 in the ie element is at (xp(m),yp(m),zp(m))&
c its value is LAM2(ie). NOTE LAM(ie) = vortex(m)
c  NOTE!!! LAM2() is indexed with ie rather than nc. This was done 
c  with the expectation that it would be useful in cluster
c  identification . That is LAM2(ie) <=0 ==> ie in VCS . Whereas
c  LAM2(ie) = +1  ==> ie NOT in VCS .
c 
c    The set VCS is the union of "clusters" which need to be identified.
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c     See postnek3.f for add'l examples, (e.g., VORTICITY)
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - --
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cluster_cores(icalld)
c
c     01-08-99 
c
c     Take the set of elements identified to have a negative
c     lam2 values and cluster them by neighbor-neighbor associations.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c   Local
c
      common /ctmp1/ ww(2*nelm)
      common /ctmp2/ color(nelm),flag(nelm),items(nelm)
      integer color,flag,items
      integer cur_color
c
      real    l2,lmin
c
      integer ncco
      save    ncco
c
c
c
c
c     Algorithm:  March over all elements having lam2<0.
c                 For each flag .ne. 2, check neighbors.
c                 Set neighbor's flag to 1 if we add it to set.
c                 Set my flag to 2.
c
c
      nc = nve
      call izero(color,nel)
      call izero(flag ,nel)
c
      ntot = nx*ny*nz*nel
      eps = .02*glmin(vortex,ntot)
c
c   jfld specifies use of fluid (not temperature) bc's to find coupling
      jfld = 1
c
      cur_color = 0
      do iec = 1,nc
         ie = iev(iec)
         if (color(ie).eq.0) then
c
c           Begin greedy sweep for this color, bump color
            cur_color = cur_color + 1
c
c           New set -- apply greedy algorithm until n_in_list=0
c
            n_in_list = 1
            n_done    = 0
            items(1)   = ie
            do i=1,nel
               je = items(i)
               do jface = 1,2*ndim
                  if (cbc(jface,je,jfld).eq.'E  ' .or.
     $                cbc(jface,je,jfld).eq.'P  '     ) then
                     ke = bc(1,jface,je,jfld)
                     if (flag(ke).eq.0 .and. lam2(ke).lt.eps) then
c                       bump list for greedy algorithm
                        n_in_list = n_in_list+1
                        items(n_in_list) = ke
                        flag (ke)        = 1
                     endif
                  endif
               enddo
c
c              color myself, decrement list count
c
               color(je) = cur_color
               flag (je) = 2
               n_done = n_done+1
               if (n_done .eq. n_in_list) goto 100
            enddo
  100       continue
c
         endif
c
      enddo
c
c     Part II:  Reassign colors according to appropriate criteria
c
c    i)  First pass, ranked by most negative lam 2 values
c   ii)  Subsequently, ranked by previous coloring (to preserve history)
c
c        Key arrays:
c
c          ieminl2(ic) -- array of element numbers representing location
c                         of minimum lam2 values for *each* core
c
c     First, gather like colors together.
c
      do iec=1,nc 
         ie = iev(iec)
         ccolor(iec) = color(ie)
      enddo
c
      call irank(ccolor,items,nc)
      call iswap(ccolor,ww,items,nc)
      call iswap(iev   ,ww,items,nc)
      call iswap(ixv   ,ww,items,nc)
      call iswap(iyv   ,ww,items,nc)
      call iswap(izv   ,ww,items,nc)
c
c
c     Pass 1 -- temporary color assignments
c
c     count number of colors: jcc() is a pointer to 1st element of
c                                   a given color in the color list
c
      call defjcc(jcc,nccolor,ccolor,nc)
c
c     Pass 2 -- assign colors based upon previous color scheme
c
      if (icalld.ne.0) then
         do iold=1,ncco
            iem = ieminl2(iold)
            if (lam2(iem).le.eps) then
c
c              This core location still has a negative lam2, use
c              old color to color the current core.
c
               ico = core_color(iem)
               icn = color     (iem)
               do ic=jcc(icn),jcc(icn+1)-1
c                 Color everything in this group with the old color
                  ccolor(ic) = -ico
               enddo
            endif
         enddo
      endif
c
c     Color any remaining cores, sorted with most negative
c     lam2 values labeled highest (max |lam2| ==> "newest"
c
      icnext = ilmin(ccolor,nc)
      if (icnext.gt.0) then
         icnext = 0
      else
         icnext = abs(icnext)
      endif
c
c     Find min lam2 for each core
c
      do ic=1,nccolor
         lmin = 0.
         do j=jcc(ic),jcc(ic+1)-1
            ie = iev(j)
            l2 = lam2(ie)
            if (l2.lt.lmin) then
               ieminl2(ic) = ie
               lmin        = l2
            endif
         enddo
c        assign lam2min to entire core
         do j=jcc(ic),jcc(ic+1)-1
            ww(j) = abs(lmin)
         enddo
      enddo
c
c     Sort by |lmin|
c
c     write(6,*) 'num colors:',nccolor,nc,icnext
      call rank  (ww    ,items,nc)
      call iswap (ccolor,ww,items,nc)
      call iswap (iev   ,ww,items,nc)
      call iswap (ixv   ,ww,items,nc)
      call iswap (iyv   ,ww,items,nc)
      call iswap (izv   ,ww,items,nc)
      call defjcc(jcc,nccolor,ccolor,nc)
c
c     March over cores, assigning next color (icnext) to unassigned cores
c     (unassigned cores have color .ge. 0)
      do ic=1,nccolor
         if (ccolor(jcc(ic)).ge.0) then
            icnext = icnext+1
            do j=jcc(ic),jcc(ic+1)-1
               ccolor(j) = icnext
            enddo
         endif
      enddo
c
c     Reset color flags to positive
      do i=1,nc
         ccolor(i) = abs(ccolor(i))
      enddo
c
c
c     Finally, Sort by color
c
      call irank(ccolor,items,nc)
      call iswap(ccolor,ww,items,nc)
      call iswap(iev   ,ww,items,nc)
      call iswap(ixv   ,ww,items,nc)
      call iswap(iyv   ,ww,items,nc)
      call iswap(izv   ,ww,items,nc)
      call defjcc(jcc,nccolor,ccolor,nc)
c
c
c     Find min lam2 for each core (again, according to new ordering)
c
      do ic=1,nccolor
         lmin = 0.
         do j=jcc(ic),jcc(ic+1)-1
            ie = iev(j)
            l2 = lam2(ie)
            if (l2.lt.lmin) then
               ieminl2(ic) = ie
               lmin        = l2
            endif
         enddo
      enddo
c
c     Save color data for next analysis
      ncco = nccolor
      call izero(core_color,nel)
      do ic=1,nc
         core_color(iev(ic)) = ccolor(ic)
      enddo
c
c     To test, color each element by its vortex-core number
      do ic=1,nve
          icc = mod(2*ccolor(ic),13)+2
          call hilight2(iev(ic),icc)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine defjcc(jcc,nccolor,ccolor,nc)
      integer jcc(1),ccolor(1)
c
      nccolor  = 1
      jcc (nccolor) = 1
      do iec=2,nc
         if (ccolor(iec).ne.ccolor(iec-1)) then
            nccolor = nccolor+1
            jcc (nccolor) = iec
            write(6,*) 'num colors1'
     $     ,nccolor,nc,iec,ccolor(iec),ccolor(iec-1)
         endif
      enddo
      jcc (nccolor+1) = nc+1
c
      write(6,*) 'num colors2',nccolor,nc
c
      return
      end
c-----------------------------------------------------------------------
C    Synopsis : There are nve elements in the Vortex Core Set.
C
C             The VCS is the union of nccolor  disjoint clusters.
C
C             For each i = 1,nve ; ie = iev(i) is the element # (index)
C             and ic = core_color(ie) is the cluster #  (index).
C
C             If the members of VCS are indexed by j = 1,nve and
C             if the cluster index is ic = 1,nccolor ;then the
C             j interval j = jcc(i),jcc(i+1) - 1 are the indices for 
C             those members of VCS which belong to the ic-th cluster.
C             So ie = iev(j) for j = jcc(i) ,jcc(i+1)-1 are the
C             elements in cluster ic .
C
C             For each cluster ic = 1,nccolor , ie = ieminl2(ic) is
C             the element # of that element in that cluster where lam2
C             is MOST negative. The node indices in that element are
C             (ixv(ie),iyv(ie),ize(ie)). Thus the location is determined
C             from m = ind(ixv(ie),...,ie)& (xp(m),yp(m),zp(m)). These
C             coordinates will be taken as the "center" of cluster ic .
C        
C-----------------------------------------
      subroutine core_position
c
c     01-13-99 : For each cluster,calculates the position of the
c                "center" of the cluster.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     LOCAL
      integer ic,m,ie,ix,iy,iz
c---------------------------------
       DO ic = 1,nccolor
         ie = ieminl2(ic)
         ix = ixv(ie)
         iy = iyv(ie)
         iz = izv(ie)
         m = ix+nx*(iy-1)+nx*ny*(iz-1)+nx*ny*nz*(ie-1)
         xpc(ic) = xp(m)
         ypc(ic) = yp(m)
         zpc(ic) = zp(m)
       END DO
c--------------------------------
       return
       end
C-----------------------------------------
      subroutine vrticity
c
c     01-13-99 : Calculates the vorticity vector :
c                (vrt1,vrt2,vrt3)(m) 
c
      include 'basics.inc'
      include 'basicsp.inc'
      PARAMETER (LXYZ=NXM*NYM*NZM)
      COMMON /CTMP4/ WORK1(LXYZ),WORK2(LXYZ)
c
c     LOCAL
      integer ic,m,ie,ix,iy,iz
      INTEGER NXYZ,IEL,IEOFF
c--------------------------------
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR vrticity -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
           IF (if3d) THEN
              NXYZ  = NX*NY*NZ
              DO 3 IEL=1,NEL
                IEOFF = (IEL-1)*NXYZ+1
C             WORK1=DW/DY ; WORK2=DV/DZ
                CALL DUDXYZ(WORK1,W,RYM1,SYM1,TYM1,JACM1,IEL)
                CALL DUDXYZ(WORK2,V,RZM1,SZM1,TZM1,JACM1,IEL)
                CALL SUB3(vrt1(IEOFF),WORK1,WORK2,NXYZ)
C             WORK1=DU/DZ ; WORK2=DW/DX
                CALL DUDXYZ(WORK1,U,RZM1,SZM1,TZM1,JACM1,IEL)
                CALL DUDXYZ(WORK2,W,RXM1,SXM1,TXM1,JACM1,IEL)
                CALL SUB3(vrt2(IEOFF),WORK1,WORK2,NXYZ)
C             WORK1=DV/DX ; WORK2=DU/DY
                CALL DUDXYZ(WORK1,V,RXM1,SXM1,TXM1,JACM1,IEL)
                CALL DUDXYZ(WORK2,U,RYM1,SYM1,TYM1,JACM1,IEL)
                CALL SUB3(vrt3(IEOFF),WORK1,WORK2,NXYZ)
    3         CONTINUE
              if (ifdssum) then
                 call dsavg(vrt1,work)
                 call dsavg(vrt2,work)
                 call dsavg(vrt3,work)
              endif
           ELSE
C             WORK1=DV/DX ; WORK2=DU/DY
              NXYZ  = NX*NY*NZ
              DO 5 IEL=1,NEL
                IEOFF = (IEL-1)*NXYZ+1
                CALL DUDXYZ(WORK1,V,RXM1,SXM1,TXM1,JACM1,IEL)
                CALL DUDXYZ(WORK2,U,RYM1,SYM1,TYM1,JACM1,IEL)
                CALL SUB3(vrt3(IEOFF),WORK1,WORK2,NXYZ)
    5         CONTINUE
              if (ifdssum) call dsavg(vrt3,work)
           ENDIF
c          write(6,*) ' done computing vorticity',ifdssum
c--------------------------------
       return
       end
C-----------------------------------------
      subroutine core_enstrophy
c
c     01-13-99 : For each cluster,calculates the enstrophy of the
c                cluster. That is the integral of the square modulus
c                of the vorticity.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     LOCAL
      integer i,j,ic,m,ie,ix,iy,iz
      real summ,sume,gwt,fcj
c--------------------------------
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR vrticity -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
      DO ic = 1,nccolor
        summ = 0.0
        DO j = jcc(ic),jcc(ic+1) -1
            ie = iev(j)
            sume = 0.0
            DO iz = 1,nz
              DO iy = 1,ny
                DO ix = 1,nx
                  m = ix+nx*(iy-1)+nx*ny*(iz-1)+nx*ny*nz*(ie-1)
                  gwt = wght(ix)*wght(iy)*wght(iz)
                  fcj = jacm1(m)
                  trm = vrt1(m)**2 + vrt2(m)**2 + vrt3(m)**2
                  sume = sume + trm*fcj*gwt
                END DO
              END DO
            END DO
            summ = summ + sume
        END DO
        enstrophy(ic) = summ
      END DO
c--------------------------------
       return
       end
c-----------------------------------------------------------------------
      subroutine compute_stresses
c
c     Compute stresses on desired objects, over desired number 
c     of .fld files
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real tdrag(3),vdrag(3),pdrag(3),sarea
      common /draghemi/ tau
c
c
      call prs  ('Input start,stop, skip fld numbers:$')
      call reiii (istart_fld,iend_fld,iskip)
c
      open(unit=14,file='tdrag.out')
      open(unit=15,file='vdrag.out')
      open(unit=16,file='pdrag.out')
      call find_hemi
c
      iskip = max(iskip,1)
c
      call prs ('Input non-dimensional period of oscillation:$')
      call rer (tau)
c
c
      iframe = 0
      do ifld=istart_fld,iend_fld,iskip
c
         iframe = iframe+1
         ndumps = ifld
c
         quanty = 'VORTICITY'
         compon = 'X'
         call getfld(ndumps,ierr,.true.,.true.)
c
         quanty = 'PRESSURE'
         call setwrk(.true.)
c
c        Compute lift on hemispheres   (for now, just one hemi)
c
         call get_hemi_forces(vdrag,pdrag,sarea)
         do k=1,ndim
            tdrag(k) = vdrag(k) + pdrag(k)
         enddo
c
         write(14,14) ifld,time,(tdrag(k),k=1,3),sarea,' tdrag'
         write(6 ,14) ifld,time,(tdrag(k),k=1,3),sarea,' tdrag'
c
         write(15,14) ifld,time,(vdrag(k),k=1,3),sarea,' vdrag'
         write(16,14) ifld,time,(pdrag(k),k=1,3),sarea,' pdrag'
c
   14    format(i5,1p5e13.4,a6)
c
      enddo
      close(unit=14)
      close(unit=15)
      close(unit=16)
      call pris(iframe,' values of drag written to t,p,vdrag.out.$')
c
      return
      end
C-----------------------------------------------------------------------
      subroutine get_hemi_forces(vdrag,pdrag,sarea)
c
c     Compute stresses on hemisphere
c
      include 'basics.inc'
      include 'basicsp.inc'
      real vdrag(3),pdrag(3),sarea
      real mu
c
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR forces -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
c
      nxyz = nx*ny*nz
      nxzf = 2*ndim*nx*nz
c
c
c     Get ambient pressure
c
      p0=0
c     call user_line_obj_gkl(p0)    UNCOMMENT THIS FOR HEMI ON PLATE LIFT
c
      mu = param(2)
      if (mu.lt.0.) mu = abs(1./param(2))
c
      dragv1 = 0.
      dragv2 = 0.
      dragv3 = 0.
      dragp1 = 0.
      dragp2 = 0.
      dragp3 = 0.
      sarea  = 0.
c
      do j = 1,nsrf
         jfc = srf_list(j)
         ifc = mod1(jfc,6)
         ie  = 1  + (jfc-1)/6
c
         jv = 1 + nxyz*(ie-1)
         jf = 1 + nxzf*(ie-1)
         call fc_strs(s1,s2,s3,p1,p2,p3,sa
     $               ,unx(jf),uny(jf),unz(jf),area(jf)
     $               ,vrt1(jv),vrt2(jv),vrt3(jv),p(jv),mu,p0,ifc,ie,xp)
c
c        dragv -- viscous  forces
c        dragp -- pressure forces
c        sarea -- integrated area  (just as a check)
c
         dragv1 = dragv1 + s1
         dragv2 = dragv2 + s2
         dragv3 = dragv3 + s3
         dragp1 = dragp1 + p1
         dragp2 = dragp2 + p2
         dragp3 = dragp3 + p3
         sarea  = sarea  + sa
c
      enddo
      vdrag(1) = dragv1
      vdrag(2) = dragv2
      vdrag(3) = dragv3
      pdrag(1) = dragp1
      pdrag(2) = dragp2
      pdrag(3) = dragp3
c
      return
      end
c-----------------------------------------------------------------------
      subroutine fc_strs (s1,s2,s3,p1,p2,p3,sa
     $                   ,unx,uny,unz,area,v1,v2,v3,p,mu,pavg,ifc,ie,xp)
c
c     compute stresses
c
c     s1,s2,s3  -  viscous drag,lift,etc.
c     p1,p2,p3  -  pressure drag,lift,etc.
c     v1,v2,v3  -  vorticity components
c     p         -  pressure
c     mu        -  dynamic viscosity
c     pavg      -  ambient pressure
c
c     unx,uny,unz - components of unit normal
c     area        - surface jacobian
c
c     ifc         - face being computed (1-6, in preprocessor notation)
c
      include 'basics.inc'
      common /draghemi/ tau
c
      real s1,s2,s3
      real unx(nx,nz,6),uny(nx,nz,6),unz(nx,nz,6),area(nx,nz,6)
      real v1(nx,ny,nz),v2(nx,ny,nz),v3(nx,ny,nz),p(nx,ny,nz)
      real xp(nx,ny,nz,1)
      real mu,pavg
c
      integer efc
C
C     Set up counters for stride across arbitrary face, using dssum notation
C
      CALL DSSET(NX,NY,NZ)
      JS1    = SKPDAT(ifc,1)
      JF1    = SKPDAT(ifc,2)
      JSKIP1 = SKPDAT(ifc,3)
      JS2    = SKPDAT(ifc,4)
      JF2    = SKPDAT(ifc,5)
      JSKIP2 = SKPDAT(ifc,6)
c
      efc=eface(ifc)
c
c
      suma  = 0.0
      sum1  = 0.0
      sum2  = 0.0
      sum3  = 0.0
      sumn1 = 0.0
      sumn2 = 0.0
      sumn3 = 0.0
c
      one       = 1.
      twopi     = 8.*atan(one)
      twopi_tau = twopi/tau
      dpdx      = -twopi_tau*cos(time*twopi_tau)
c
      i = 0
      do 100 j2=js2,jf2,jskip2
      do 100 j1=js1,jf1,jskip1
         i = i+1
c
         fcj   = area(i,1,efc)
         suma  = suma + fcj
c
         trm1  = uny(i,1,efc)*v3(j1,j2,1) - unz(i,1,efc)*v2(j1,j2,1)
         trm2  = unz(i,1,efc)*v1(j1,j2,1) - unx(i,1,efc)*v3(j1,j2,1)
         trm3  = unx(i,1,efc)*v2(j1,j2,1) - uny(i,1,efc)*v1(j1,j2,1)
c
         sum1  = sum1 + trm1*fcj
         sum2  = sum2 + trm2*fcj
         sum3  = sum3 + trm3*fcj
c
c
c
c        Compute sumni -- normal forces
c
c        Notes:    (pff 3/31/99)
c
c        Nominally:   sumn1 = pressure force in x-direction
c                     sumn2 =    "       "   "  y-direction
c                     sumn3 =    "       "   "  z-direction
c
c        sumn1 = sumn1 + fcj*unx(i,1,efc)*p(j1,j2,1)
c        sumn2 = sumn2 + fcj*uny(i,1,efc)*p(j1,j2,1)
c        sumn3 = sumn3 + fcj*unz(i,1,efc)*p(j1,j2,1)
c
c        HOWEVER, for GKL's  problem, these have been modified
c        for the following reasons.   The x-direction pressure
c        requires addition of the mean pressure gradient, which
c        is imposed as a body force in the calculation.  The y-
c        direction force is 0 due to symmetry, so we use sumn2
c        to store the projected z-area, just as a parity check.
c        The z-normal force includes the mean pressure around
c        the base of the hemisphere (computed elsewhere) in order
c        to account for the indeterminate hydrostatic pressure.
c
c        Thuse, presently, we have:
c
c            sumn1 = pressure force in x-direction, acct'g for body force
c            sumn3 =    "       "   "  z-direction, acct'g for base press.
c            sumn2 = projected area in Z-direction
c
c
         sumn1 = sumn1 + fcj*unx(i,1,efc)*(p(j1,j2,1)+
     $                                            dpdx*xp(j1,j2,1,ie))
         sumn2 = sumn2 + fcj*uny(i,1,efc)* p(j1,j2,1)
         sumn3 = sumn3 + fcj*unz(i,1,efc)*(p(j1,j2,1)-pavg)
c        DIAGNOSTIC Hack -- sumn2 is projected area, should be negative
c        sumn2 = sumn2 + fcj*unz(i,1,efc)
c
  100 CONTINUE
c
c     trm1   = sum1 /suma
c     trm2   = sum2 /suma
c     trm3   = sum3 /suma
c     trmn1  = sumn1/suma
c     trmn2  = sumn2/suma
c     trmn3  = sumn3/suma
c
      s1     = mu*sum1
      s2     = mu*sum2
      s3     = mu*sum3
      p1     = sumn1
      p2     = sumn2
      p3     = sumn3
      sa     = suma
c
      if (ie.eq.10) write(6,*) 'dpdx',time,dpdx,p1,tau
c
      return
      end
c-----------------------------------------------------------------------
      subroutine user_line_obj_gkl(pavg)
c
c     routine to allow user defined line objects
c
c     Right now this is just hacked to compute the mean pressure around
c     a hemisphere.   pff 3/19/99
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     Line objects are defined as an ordered set of points in R^d, d=2 or 3.
c
c     More than one object may be defined.
c
c
c     Relevant arrays/scalars are:
c
c
      parameter (mopts=9) ! = 9000)
c
      common /rlobjs/    x_l_objs(mopts),y_l_objs(mopts),z_l_objs(mopts)
     $  ,wkl_objs(mopts),w1l_objs(mopts),w2l_objs(mopts),w3l_objs(mopts)
c
      integer  num_l_pts(0:100)
c
c
c
c     For the hemisphere problem, we seek the perimeter at the
c     base of the hemisphere, denoted by ccurve = 'C'
c
c     Two options:   a)  Choose GLL points and use GLL quadrature
c                    b)  Just define a circle and interpolate.
c
c     For now, we'll choose (b), such that we're well away from the
c     troublesome element boundaries.
c
      num_l_objs    = 1
      num_l_pts(0)  = 1
      num_l_pts(1)  = 101
      num_l_pts(2)  = 301
      num_l_pts(3)  = 701
      xcenter = 0.
      ycenter = 0.
      radius  = 1.01
      z0      = 0.01
      one     = 1.
      pi      = 4.*atan(one)
      do l=1,num_l_objs
c
         i0   = num_l_pts(l-1)
         npt  = num_l_pts(l)-num_l_pts(l-1)
c
         sum     = 0.
         dtheta  = pi/npt
         do i=num_l_pts(l-1),num_l_pts(l)
c
            j=i-num_l_pts(l-1)
            theta = j*dtheta
            x_l_objs(i) =  radius*cos(theta)
            y_l_objs(i) = -radius*sin(theta)
            z_l_objs(i) =  z0
c
c           Interpolate current fields onto object points
c
            call interp(wkl_objs(i),x_l_objs(i),y_l_objs(i),z_l_objs(i)
     $                ,idum,work,ierr)
            call interp(w1l_objs(i),x_l_objs(i),y_l_objs(i),z_l_objs(i)
     $                ,idum,wkv1,ierr)
            call interp(w2l_objs(i),x_l_objs(i),y_l_objs(i),z_l_objs(i)
     $                ,idum,wkv2,ierr)
            call interp(w3l_objs(i),x_l_objs(i),y_l_objs(i),z_l_objs(i)
     $                ,idum,wkv3,ierr)
c           Compute integral, specific to hemisphere app.
c
            if (ierr.eq.1) then
               call prsi('Coud not interpolate point in line_obj.$',i)
               call prrrr(x_l_objs(i),y_l_objs(i),z_l_objs(i))
            elseif (i.eq.num_l_pts(l-1).or.i.eq.num_l_pts(l)) then
               sum = sum + 0.5*wkl_objs(i)
            else
               sum = sum + wkl_objs(i)
            endif
         enddo
c
c        Mean pressure
c
         sum  = sum/npt
         pavg = sum
         call prsr('This is mean pressure:$',pavg)
c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine find_hemi
c
c     Find hemispheres, using currently available surface finding
c     routines.
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     Find hemispheres, using currently available surface finding
c     routines.
c
c
      x0=0.
      y0=0.
      z0=1.1
c
      call prs  ('Input top of sphere (xyz):$')
      call rerrr(x0,y0,z0)
      ivec = 3
      nlstp=0
      call setrstp_set(ivec,x0,y0,z0)
      nsrf = nlstp
c
c     Fill in srf_list
c
      do i=1,nlstp
         lista=abs(list(i))
         call decod(iplane,ipln,ie,idum,lista,nx,6,nelm)
         srf_list(i) = 6*(ie-1) + ipln
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine interp_all(uvw,xi,yi,zi,ierr)
c
c     routine to compute average quantities along a line, x0---x1.
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real uvw(13)
c
      call rzero(uvw,13)
c
      call interp(uvw(1),xi,yi,zi,idum,u   ,ierr)
      if (ierr.eq.1) then
         call prs  ('Coud not interpolate point in interp_all.$')
         call prrrr(xi,yi,zi)
         return
      endif
c
      call interp(uvw(2),xi,yi,zi,idum,v   ,ierr)
      call interp(uvw(3),xi,yi,zi,idum,w   ,ierr)
      call interp(uvw(4),xi,yi,zi,idum,p   ,ierr)
      call interp(uvw(5),xi,yi,zi,idum,t   ,ierr)
c     write(6,*) xi,yi,uvw(5)
      call interp(uvw(6),xi,yi,zi,idum,work,ierr)
      call interp(uvw(7),xi,yi,zi,idum,wkv1,ierr)
      call interp(uvw(8),xi,yi,zi,idum,wkv2,ierr)
      call interp(uvw(9),xi,yi,zi,idum,wkv3,ierr)
      call interp(uvw(10),xi,yi,zi,idum,vortex,ierr)
      IF (lvrt.eq.maxpts) THEN
         call interp(uvw(11),xi,yi,zi,idum,vrt1,ierr)
         call interp(uvw(12),xi,yi,zi,idum,vrt2,ierr)
         call interp(uvw(13),xi,yi,zi,idum,vrt3,ierr)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine line_avg(uvwavg,x0,x1)
c
c     routine to compute average quantities along a line, x0---x1.
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real uvwavg(9,1),x0(3),x1(3)
c
c     Line objects are defined as an ordered set of points in R^d, d=2 or 3.
c     More than one object may be defined.
c
c
c     Relevant arrays/scalars are:
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/    x_l_objs(mopts),y_l_objs(mopts),z_l_objs(mopts)
     $  ,wkl_objs(mopts),w1l_objs(mopts),w2l_objs(mopts),w3l_objs(mopts)
     $  ,u_l_objs(mopts),v_l_objs(mopts),w_l_objs(mopts),p_l_objs(mopts)
     $  ,t_l_objs(mopts)
c
      integer  num_l_pts(0:100)
c
      real uvw(13)
c
c
      num_l_objs    = 1
      num_l_pts(0)  = 1
      num_l_pts(1)  = 101
      num_l_pts(2)  = 301
      num_l_pts(3)  = 701
      xcenter = 0.
      ycenter = 0.
      radius  = 1.01
      z0      = 0.01
      one     = 1.
      pi      = 4.*atan(one)
      do l=1,num_l_objs
c
         i0   = num_l_pts(l-1)
         npt  = num_l_pts(l)-num_l_pts(l-1)
c
         call rzero(uvwavg(1,l),9)
         sum  = 0.
         ds   = 1./npt
         do i=num_l_pts(l-1),num_l_pts(l)
c
            j  = i-num_l_pts(l-1)
            ss = j*ds
            x_l_objs(i) =  x0(1) + ss*(x1(1)-x0(1))
            y_l_objs(i) =  x0(2) + ss*(x1(2)-x0(2))
            z_l_objs(i) =  x0(3) + ss*(x1(3)-x0(3))
c
c           Interpolate current fields onto object points
c
            call interp_all(uvw,x_l_objs(i),y_l_objs(i),z_l_objs(i)
     $                     ,ierr)
            if (ierr.eq.1) then
               call prsi('Coud not interpolate point in line_obj.$',i)
               call prrrr(x_l_objs(i),y_l_objs(i),z_l_objs(i))
               wt = 0.0
            elseif (i.eq.num_l_pts(l-1).or.i.eq.num_l_pts(l)) then
               wt = 0.5
            else
               wt = 1.0
            endif
            sum = sum + wt
c
            call add2s2(uvwavg(1,l),uvw,wt,9)
         enddo
c
c        Scale
c
         if (sum.ne.0.) sum  = 1./sum
         call cmult(uvwavg(1,l),sum,9)
c
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine time_trace
c
c     Compute stuff for Miles, full groove + flat
c
c
c     Objectives:
c
c      .compute Nu_b(x), using trapezoidal rule + Richardson
c      .compute Fanning friction factor, avg'd over x
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real uvw(13)
c
c
      call prs  ('Input start,stop, skip fld numbers:$')
      call reiii (istart_fld,iend_fld,iskip)
c
      ntot = nx*ny*nz*nel
c
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      xmid = xmin + 0.9*(xmax-xmin)
c
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      ymid = ymin + 0.5*(ymax-ymin)
c
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      zmid = zmin + 0.5*(zmax-zmin)
c
c
      open(unit=14,file='trace.dat')
c
      iskip = max(iskip,1)
c
      iframe = 0
      do ifld=istart_fld,iend_fld,iskip
c
         iframe = iframe+1
         ndumps = ifld
c
         quanty = 'PRESSURE'
         compon = 'X'
         deriv  = 'N'
         call getfld(ndumps,ierr,.true.,.true.)
c
         call interp_all(uvw,xmid,ymid,zmid,ierr)
c
         write(14,14) ifld,time,(uvw(k),k=1,9)
         write(6 ,14) ifld,time,(uvw(k),k=1,9)
   14    format(i4,1p9e12.4)
c
      enddo
      close(unit=14)
      call pris  (iframe,' values written to trace.dat.$')
      call prsrrr('XYZ loc:$',xmid,ymid,zmid)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine integrate_uut(q1i,q2i,q3i,x0,x1,q1,q2,q3)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real x0(3),x1(3),q1(1),q2(1),q3(1)
c
      n=256
c
      ds = (x1(1)-x0(1))**2 + (x1(2)-x0(2))**2 + (x1(3)-x0(3))**2
      if (ds.gt.0) ds = sqrt(ds)
      ds = ds/n
      dx = (x1(1)-x0(1))/n
      dy = (x1(2)-x0(2))/n
      dz = (x1(3)-x0(3))/n
c
      sum1 = 0.
      sum2 = 0.
      sum3 = 0.
c
      do i=0,n
         xi = x0(1) + i*dx
         yi = x0(2) + i*dy
         zi = x0(3) + i*dz
         call interp(w1,xi,yi,zi,idum,q1,ierr)
         call interp(w2,xi,yi,zi,idum,q2,ierr)
         call interp(w3,xi,yi,zi,idum,q3,ierr)
         if (ierr.eq.1) then
            write(6,1) i,'Interp fail:',xi,yi,zi
    1       format(i3,1x,a12,1p7e12.4)
         else
c           write(6,1) i,'Interp good:',xi,yi,zi,w1,w2,sum1,sum2
            if (i.eq.0 .or. i.eq.n ) then
               wt = 1.*ds/3.
            elseif (mod(i,2).eq.1) then
               wt = 4.*ds/3.
            else
               wt = 2.*ds/3.
            endif
c
            sum1 = sum1+w1*wt
            sum2 = sum2+w2*wt
            sum3 = sum3+w3*wt
         endif
      enddo
      q1i = sum1
      q2i = sum2
      q3i = sum3
c     call prs('continue?$')
c     call res(line,1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_component
c   
      include 'basics.inc'
      include 'basicsp.inc'
c
      CALL PRS(' ENTER COMPONENT (For scalar plots)$')
      NCHOIC=1
      ITEM(NCHOIC)='Magnitude'
      ITEM(2)='X'
      ITEM(3)='Y'
      NCHOIC=3
      IF(IF3D)THEN
         ITEM(4)='Z'
         NCHOIC=NCHOIC+1
      ENDIF
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET COMPONENT')
      COMPON=CHOICE(1:1)
c
      ntot = nx*ny*nz*nel
      if (compon.eq.'M') then
         do i=1,ntot
            work(i) = wkv1(i)**2 + wkv2(i)**2 + wkv3(i)**2
            if (work(i).gt.0.) work(i) = sqrt(work(i))
         enddo
      elseif (compon.eq.'X') then
         call copy(work,wkv1,ntot)
      elseif (compon.eq.'Y') then
         call copy(work,wkv2,ntot)
      elseif (compon.eq.'Z') then
         call copy(work,wkv3,ntot)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine add_const
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c     Add a constant to the x-component of velocity (for hemi paper)
c
c
      call prs('Enter constant shift for X component of vector field:$')
      call rer(shift)
c
      ntot = nel*nx*ny*nz
      call cadd(wkv1,shift,ntot)
      call copy(work,wkv1,ntot)
c
      WKMAX = GLMAX(WORK,NTOT)
      WKMIN = GLMIN(WORK,NTOT)
      WKDLT = WKMAX-WKMIN
      write(6,*) 'wkmax,delt:',quanty,wkmax,wkmin,wkdlt
c
      SPMAX = WKMAX
      SPMIN = WKMIN
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ col(12,0:mopts-1)
      real uvw(13),x0(3),x1(3)
      logical iftmp
c
      call rzero(col,12*mopts)
c
c     Compute stuff for Miles, full groove + flat
c
c
c     call prs('Would you like to generate a time trace file? (y/n)$')
c     call res(ans,1)
c     if (ans.eq.'y'  .or.  ans.eq.'Y') call time_trace
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     4. ~q"(x)  :=   q"(x) * ds/dx , where s is the surface area
c                                     (ds/dx = sqrt(2) in groove)
c                                     (ds/dx = 1 in channel)
c     5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c     6. Tc(x) -- z-avg'd centerline temperature
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c
c     NOTE:  This is currently hardwired for the 1-groove+flat geometry,
c            due to the ds/dx.
c
c     Set up quantities
c
c
c     compute
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     6. Tc(x) -- z-avg'd centerline temperature
c
c     Offset zmid so we're not right on an element bdry.
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      zmid = 0.625*(zmax+zmin)
      ymid = 0.500*(ymax+ymin)
c
      call interp_all(uvw,xmin,ymid,zmid,ierr)
c
c
c
c
      call prs('Enter .fld number for file containing <uT> as W:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
      call setwrk(.true.)
c
c     dp/dx is now in work,  after the next setwrk call, dpdx will be in wrk2
c
c
c     n  = 2400
      call prs('Input number of points in x-direction:$')
      call rei(n)
      dx = (xmax-xmin)/n
c
      do i=0,n
         xx = xmin + i*dx
         call interp_all(uvw,xx,ymid,zmid,ierr)
c
c        - x -
c
         col(1,i) = xx
c
c
c        - dp/dx -
c
         col(2,i) = uvw(6)
c
c
c        - (P(L)-P0)/L -
c
         if (i.eq.0) then
            p0       = uvw(4)
            col(3,i) = uvw(6)
         else
            col(3,i) = (p0-uvw(4))/(col(1,0)-col(1,i))
         endif
c
c
c        - Tc -
c
         col(6,i) = uvw(5)
c
      enddo
c
c
c
c
c     Now compute q"
c
c     Set quantity to magnitude of grad T
c
      quanty = 'TEMPERATURE'
      deriv  = 'GRAD'
      compon = 'M'
      call setwrk(.true.)
c
      zmid = 0.53*(zmax+zmin)
c
c      4. ~q"(x)  :=   q"(x) * ds/dx
c
      do jpass=1,2
c
         m = 2*n
         if (jpass.eq.2) m = n
         do i=0,m
            col(4,i) = 0. 
         enddo
         dxx = (xmax-xmin)/m
c
         do ipass=1,2
         do i=0,m
            xx = xmin + i*dxx
            eps = 1.e-6
            if (xx.le.0.012) then
               ytop = 0.022 + xx-eps
               ybot = 0.012 - xx+eps
               two  = 2.
               dsdx = sqrt(two)
            elseif (xx.le.0.024) then
               ytop = 0.034 - (xx-.012)-eps
               ybot =         (xx-.012)+eps
               two  = 2.
               dsdx = sqrt(two)
            else
               ytop = 0.022
               ybot = 0.012
               dsdx = 1.0
            endif
            if (ipass.eq.1) call interp_all(uvw,xx,ytop,zmid,ierr)
            if (ipass.eq.2) call interp_all(uvw,xx,ybot,zmid,ierr)
            col(4,i) = col(4,i) + dsdx*uvw(6)
c           write(6,*) i,' INTERP: ',xx,ytop,ybot,zmid,eps,ierr
c
         enddo
         enddo
c
         if (jpass.eq.1) then
c
c           5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c              via trapezoidal integration in x
c
            col(5,0) = col(4,0)
            sum = 0.
            do i=1,n
               sum = sum + (col(4,2*i-2)+4.*col(4,2*i-1)+col(4,2*i))/6.
               avg = sum/i
               col(5,i) = avg
            enddo
         endif
      enddo
c
c
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
      do i=0,n
         xx = xmin + i*dx
         if (xx.le.0.) xx = 1.e-9
c
         eps = 1.e-6
         if (xx.le.0.012) then
            ytop = 0.022 + xx - eps
            ybot = 0.012 - xx + eps
            two  = 2.
         elseif (xx.le.0.024) then
            ytop = 0.034 - (xx-.012) - eps
            ybot =          xx-.012  + eps
            two  = 2.
         else
            ytop = 0.022
            ybot = 0.012
         endif
         x0(1) = xx
         x0(2) = ybot
         x0(3) = zmid
         x1(1) = xx
         x1(2) = ytop
         x1(3) = zmid
         call integrate_uut(ui,uti,ti,x0,x1,u,w,wkv1)
         col( 7,i) = uti/ui
         col( 8,i) = uti
         col( 9,i) = ui
         col(12,i) = ti
      enddo
c
c     A quick check to see if dTb/dx = ~q"
c
c     .col(10) -- int_0^L (q")
c     .col(11) -- heat_in - heat_out   (convective flux)
c     .col(12) -- heat_in - heat_out   (conductive flux)
c
c
c
      i0 = 0
c     call pris(n,' values in list, input i0 for heat check:$')
c     call rei (i0)
      rhocp   = param(7)
      conduct = param(8)
      do i=0,n
         col(10,i) = dz*conduct*col(5,i)*(col(1,i)-col(1,i0))
         col(11,i) = dz*rhocp*(col(8,i0)-col(8,i))
     $             - dz*conduct*(col(12,i0)-col(12,i))
      enddo
c
c   x      dp/dx    Pout(L)-Pin  ~q"  int(~q")/L   Tc(x)     Tb(x)
c
c
      open(unit=14,file='xprofw.dat')
      do i=0,n
         write(14,7) (col(j,i),j=1,11)
      enddo
  7   format(1p11e12.4)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprofw.dat.$')
c
c
      open(unit=14,file='xprof.dat')
      do i=0,n
         write(14,6) (col(j,i),j=1,7)
      enddo
  6   format(1p11e11.3)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprof.dat.$')
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles2
c
      include 'basics.inc'
      include 'basicsp.inc'
      character*10 pname
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ avg(6,0:mopts-1,3)
     $              , rms(6,0:mopts-1,3)
      real uvw(13),x0(3),x1(3)
      logical iftmp
c
      call rzero(rms,6*mopts)
      call rzero(avg,6*mopts)
c
c     Compute stuff for Miles, full groove + flat
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. u   u'
c     3. v   v'
c     4. w   w'
c     5. p   p'
c     6. t   t'
c
c     First pass, compute avg.  Second pass, compute rms
c
      call prs
     $    ('Enter .fld number for file containing avgerages:$')
      call rei(jfld)
c
c     Get .fld file
c
      quanty = 'PRESSURE'
      deriv  = 'NO'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
c
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      zmid = 0.630*(zmax+zmin)
      ymid = 0.500*(ymax+ymin)
c
      n  = 2400
c     call prs('Input number of points in x-direction:$')
c     call rei(n)
      dx = (xmax-xmin)/n
c
c
c     Avg
c
      do iy=1,3
c        Hardwired for channel
         ym = ymid - .005  + .0025*iy
         ym = ym   + .5e-4
         do i=0,n
            xx = xmin + i*dx
            call interp_all(uvw,xx,ym,zmid,ierr)
            avg(1,i,iy) = xx
            avg(2,i,iy) = uvw(1)   ! u
            avg(3,i,iy) = uvw(2)   ! v
            avg(4,i,iy) = uvw(3)   ! w
            avg(5,i,iy) = uvw(4)   ! p
            avg(6,i,iy) = uvw(5)   ! t
         enddo
      enddo
c
c     RMS
c
      call prs
     $   ('Enter .fld number for file containing <u^2> quantities:$')
      call rei(jfld)
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
c
c
      do iy=1,3
c        Hardwired for channel
         ym = ymid - .005  + .0025*iy
         ym = ym + .5e-4
         do i=0,n
            xx = xmin + i*dx
            call interp_all(uvw,xx,ym,zmid,ierr)
            rms(1,i,iy) = xx
            do k=1,5
               rms(k+1,i,iy) = uvw(k) - avg(k+1,i,iy)*avg(k+1,i,iy)
               if (rms(k+1,i,iy).gt.0) then
                  rms(k+1,i,iy) = sqrt(rms(k+1,i,iy))
               elseif (rms(k+1,i,iy).lt.0) then
                  write(6,*) 'RMS < 0 ?',xx,rms(k+1,i,iy),avg(k+1,i,iy)
     $                       ,k,i,iy
                  rms(k+1,i,iy) = 0.
               endif
            enddo
         enddo
      enddo
c
c
      do iy=1,3
         write(pname,3) iy
  3      format('x_avg',i1,'.dat')
         open(unit=14,file=pname)
         do i=0,n
            write(14,7) (avg(j,i,iy),j=1,6)
         enddo
  7      format(1p11e12.4)
         close(unit=14)
      enddo
      n1 = n+1
      call pris  (n1,' values written to x_avg.dat.$')
c
c
c
      do iy=1,3
         write(pname,4) iy
  4      format('x_rms',i1,'.dat')
         open(unit=14,file=pname)
         do i=0,n
            write(14,7) (rms(j,i,iy),j=1,6)
         enddo
         close(unit=14)
      enddo
      n1 = n+1
      call pris  (n1,' values written to x_rms.dat.$')
c
c     Output the data in another format for miles
c
      x_off = 0.
      dxmax = xmax-xmin
      if (dxmax.gt. 0.030) x_off = .024
c
      do iy=2,2
         jy = 4
         write(pname,4) jy
         open(unit=14,file=pname)
         do i=0,n
            xx = x_off + avg(1,i,iy)
            write(14,7) xx,(avg(j,i,iy),j=2,4),(rms(j,i,iy),j=2,4)
         enddo
         close(unit=14)
      enddo
      n1 = n+1
      call pris  (n1,' values written to x_rms4.dat.$')
c
c
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles3
c
c     This is like "compute_miles" but is for steady state results --
c     Thus, no "<uT>" file is required.
c
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ col(12,0:mopts-1)
      real uvw(13),x0(3),x1(3)
      logical iftmp
c
      call rzero(col,12*mopts)
c
c     Compute stuff for Miles, full groove + flat
c
c
      call prs('Would you like to generate a time trace file? (y/n)$')
      call res(ans,1)
      if (ans.eq.'y'  .or.  ans.eq.'Y') call time_trace
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     4. ~q"(x)  :=   q"(x) * ds/dx , where s is the surface area
c                                     (ds/dx = sqrt(2) in groove)
c                                     (ds/dx = 1 in channel)
c     5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c     6. Tc(x) -- z-avg'd centerline temperature
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c
c     NOTE:  This is currently hardwired for the 1-groove+flat geometry,
c            due to the ds/dx.
c
c     Set up quantities
c
c
c
      call prs('Enter .fld number for file containing <uT> as W:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
c
      call setwrk(.true.)
c
c     dp/dx is now in work,  after the next setwrk call, dpdx will be in wrk2
c
c
c     compute
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     6. Tc(x) -- z-avg'd centerline temperature
c
c     Offset zmid so we're not right on an element bdry.
c
      ntot = nx*ny*nz*nel
c
      call col3(w,u,t,ntot)
c
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      zmid = 0.625*(zmax+zmin)
      ymid = 0.500*(ymax+ymin)
c
      n  = 2400
c     call prs('Input number of points in x-direction:$')
c     call rei(n)
      dx = (xmax-xmin)/n
c
      do i=0,n
         xx = xmin + i*dx
         call interp_all(uvw,xx,ymid,zmid,ierr)
c
c        - x -
c
         col(1,i) = xx
c
c
c        - dp/dx -
c
         col(2,i) = uvw(6)
c
c
c        - (P(L)-P0)/L -
c
         if (i.eq.0) then
            p0       = uvw(4)
            col(3,i) = uvw(6)
         else
            col(3,i) = (p0-uvw(4))/(col(1,0)-col(1,i))
         endif
c
c
c        - Tc -
c
         col(6,i) = uvw(5)
c
      enddo
c
c
c
c
c     Now compute q"
c
c     Set quantity to magnitude of grad T
c
      quanty = 'TEMPERATURE'
      deriv  = 'GRAD'
      compon = 'M'
      call setwrk(.true.)
c
      zmid = 0.53*(zmax+zmin)
c
c      4. ~q"(x)  :=   q"(x) * ds/dx
c
      do jpass=1,2
         m = 2*n
         if (jpass.eq.2) m = n
c
         do i=0,m
            col(4,i) = 0. 
         enddo
c
         dxx = (xmax-xmin)/m
         do ipass=1,2
         do i=0,m
            xx = xmin + i*dxx
            eps = 1.e-6
            if (xx.le.0.012) then
               ytop = 0.022 + xx-eps
               ybot = 0.012 - xx+eps
               two  = 2.
               dsdx = sqrt(two)
            elseif (xx.le.0.024) then
               ytop = 0.034 - (xx-.012)-eps
               ybot =         (xx-.012)+eps
               two  = 2.
               dsdx = sqrt(two)
            else
               ytop = 0.022
               ybot = 0.012
               dsdx = 1.0
            endif
            if (ipass.eq.1) call interp_all(uvw,xx,ytop,zmid,ierr)
            if (ipass.eq.2) call interp_all(uvw,xx,ybot,zmid,ierr)
            col(4,i) = col(4,i) + dsdx*uvw(6)
c           write(6,*) i,' INTERP: ',xx,ytop,ybot,zmid,eps,ierr
c
         enddo
         enddo
c
         if (jpass.eq.1) then
c
c           5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c              via trapezoidal integration in x
c
            col(5,0) = col(4,0)
            sum = 0.
            do i=1,n
               sum = sum + (col(4,2*i-2)+4.*col(4,2*i-1)+col(4,2*i))/6.
               avg = sum/i
               col(5,i) = avg
            enddo
         endif
      enddo
c
c
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
      do i=0,n
         xx = xmin + i*dx
         if (xx.le.0.) xx = 1.e-9
c
         eps = 1.e-6
         if (xx.le.0.012) then
            ytop = 0.022 + xx - eps
            ybot = 0.012 - xx + eps
            two  = 2.
         elseif (xx.le.0.024) then
            ytop = 0.034 - (xx-.012) - eps
            ybot =          xx-.012  + eps
            two  = 2.
         else
            ytop = 0.022
            ybot = 0.012
         endif
         x0(1) = xx
         x0(2) = ybot
         x0(3) = zmid
         x1(1) = xx
         x1(2) = ytop
         x1(3) = zmid
         call integrate_uut(ui,uti,ti,x0,x1,u,w,wkv1)
         col( 7,i) = uti/ui
         col( 8,i) = uti
         col( 9,i) = ui
         col(12,i) = ti
      enddo
c
c     A quick check to see if dTb/dx = ~q"
c
c     .col(10) -- int_0^L (q")
c     .col(11) -- heat_in - heat_out   (convective flux)
c     .col(12) -- heat_in - heat_out   (conductive flux)
c
c
c
      i0 = 0
c     call pris(n,' values in list, input i0 for heat check:$')
c     call rei (i0)
      rhocp   = param(7)
      conduct = param(8)
      do i=0,n
         col(10,i) = dz*conduct*col(5,i)*(col(1,i)-col(1,i0))
         col(11,i) = dz*rhocp*(col(8,i0)-col(8,i))
     $             - dz*conduct*(col(12,i0)-col(12,i))
      enddo
c
c   x      dp/dx    Pout(L)-Pin  ~q"  int(~q")/L   Tc(x)     Tb(x)
c
c
      open(unit=14,file='xprofw.dat')
      do i=0,n
         write(14,7) (col(j,i),j=1,11)
      enddo
  7   format(1p11e12.4)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprofw.dat.$')
c
c
      open(unit=14,file='xprof.dat')
      do i=0,n
         write(14,7) (col(j,i),j=1,7)
      enddo
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprof.dat.$')
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine azi_avg
c
c     This subroutine sets up and generates the azimuthal avg's for
c     Anil's problem.    2/28/00  pff.
c
c     Input is a 2D .fld file corresponding to x-y points of interest.
c
c     Output is a 2D .fld file with u,v,p,t at those points,
c     computed as the integrated azimuthal average.
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
c
c     Get input file, strip header, and write header to output file.
      call get_file(npt,10,11,ierr)
      write(6,*) 'npt:',npt,ierr
      if (ierr.eq.1) return
c
c
      do i=1,npt
         read(10,*) xstart,zstart
         call azi_int(uaz,vaz,waz,paz,taz,xstart,zstart,ierr)
         write(11,11) uaz,vaz,waz,taz
         write(6 ,12) i,npt,ierr,uaz,vaz,waz,taz,xstart,zstart
      enddo
   11 format(1p6e14.6)
   12 format(2i8,i4,1p6e12.4)
      close(10)
      close(11)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine trapzdu(s,m,a,b,n,sum,f,it)
c
c     Algebraic avg, using trapezoidal rule
c
c     j>>n
      integer n
      real s(m),sum(m),f(m)
      real a, b
c
      integer it, j, it0
      save    it0
      real del, tnm, x
      real tnmb
      save tnmb
c
      if (n.eq.1) then
         it0 = 17
         it = it0
         tnm = it
         del = (b-a)/tnm
         x = a
         call rzero(s,m)
         tnm = 0.
         do j = 1, it
            call func_eval(f,m,x,ierr)
            if (ierr.eq.0) then
               call add2(s,f,m)
               tnm = tnm + 1.
            endif
            x = x + del
         enddo
         if (tnm.gt.0) then
            tnmi = 1./tnm
            call cmult(s,tnmi,m)
         endif
c        Save base count
         tnb = tnm
      else
         it = it0*(2**(n-2))
         tnm = it
         del = (b-a)/tnm
         x = a + 0.5*del
c
         tnm = 0.
         call rzero(sum,m)
c
         do j = 1, it
            call func_eval(f,m,x,ierr)
            if (ierr.eq.0) then
c              don't count drop-outs
               call add2(sum,f,m)
               tnm = tnm + 1.
            endif
            x = x + del
         enddo
         if (tnm.gt.0) then
            tnmi = 1./tnm
            call cmult(sum,tnmi,m)
         endif
         tnms = tnmb + tnm
         if (tnms.gt.0) then
            do i=1,m
               s(i) = (tnmb*s(i) + tnm*sum(i))/tnms
            enddo
         endif
         tnmb = tnms
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine qtrapu(uint,m,a,b,s,olds,w,f,ierr)
      real uint(m),s(m),olds(m),w(m),f(m)
      logical ifdone
c
c     2**(jmax-1) = max. # of steps allowed
      integer jmax
c     parameter (jmax = 11)
c     parameter (jmax = 8)
      parameter (jmax = 5)
c
      integer j
c
c     eps = 1.e-10
c     one = 1.
c     if (one+eps.eq.1.) eps = 5.e-5
c
      eps = 1.e-3
c
      call cfill(olds,-1.e30,m)
      call rzero(s,m)
      do j = 1,jmax
c
         call trapzdu(s,m,a,b,j,w,f,it)
c
         ierr   = 0
         ifdone = .true.
c
         do i=1,m
            dsi = abs(s(i)-olds(i))
c           ept = eps*abs(olds(i))
            ept = eps*abs(s(i))
            if (i.ne.4) write(6,1) i,j,dsi,ept,s(i)
    1       format(2i4,' err:',1p3e12.4)
            if (dsi.lt.ept) then
               uint(i) = s(i)
            else
               ifdone = .false.
               ierr   = 1
            endif
         enddo
c
         if (ifdone) return
         write(6,*) 'm',m,j
         write(6,*) 's',(s(k),k=1,m)
         call copy(olds,s,m)
c
      enddo
      call copy(uint,s,m)
      return
      end
c-----------------------------------------------------------------------
      subroutine azi_int(uaz,vaz,waz,paz,taz,xstart,zstart,ierr)
c
c     Azimuthal integration
c     Anil's problem.    2/28/00  pff.
c
      common /azicom/ z2,r2
c
      real uint(6),ss(6),olds(6),wk1(6),wk2(6)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real rmin,rmax
      save rmin,rmax
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
c     Check if we're on the inner or outer radii
c
      if (icalld.eq.0) then
c
         icalld = 1
         rmin =  9.e20
         rmax = -9.e20
c
         ntot = nx*ny*nz*nel
         do i=1,ntot
            rrr = xp(i)*xp(i) + yp(i)*yp(i) + zp(i)*zp(i)
            rmin = min(rrr,rmin)
            rmax = max(rrr,rmax)
         enddo
      endif
c
      z2 = zstart
      r2 = xstart
c
c     Check if we're on the inner or outer radii since interpolation
c     is very expensive on these surfaces
c
      r3 = z2*z2 + r2*r2
      d1 = abs(r3-rmin)/rmax
      d2 = abs(r3-rmax)/rmax
c
      eps = 4.e-5
      if (d1.lt.eps) then
         uaz = 0.
         vaz = 0.
         waz = 0.
         paz = 0.
         taz = 1.
         return
      elseif (d2.lt.eps) then
         uaz = 0.
         vaz = 0.
         waz = 0.
         paz = 0.
         taz = 0.
         return
      endif
c
      one   = 1.
      twopi = 8.*atan(one)
c
      a = 0.5
      b = a + twopi
c
      call qtrapu(uint,5,a,b,ss,olds,wk1,wk2,ierr)
c     write(6,*) 'uint:',(uint(k),k=1,5)
      uaz = uint(1)
      vaz = uint(2)
      waz = uint(3)
      paz = uint(4)
      taz = uint(5)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine func_eval(ss,m,tht,ierr)
c
c     Azimuthal function evaluation
c     Anil's problem.    2/28/00  pff.
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real ss(m)
c
      common /azicom/ z2,r2
c

      real    uumin,uumax
      save    uumin,uumax
      integer icalld
      save    icalld
      data    icalld  /0/
c
      real rll(3)
c
      ierr = 0
      if (icalld.eq.0) then
         icalld = 1
         ntot = nx*ny*nz*nel
         uumin = glmin(u,ntot)
         uumax = glmax(u,ntot)
      endif
      nxyz = nx*ny*nz
c
      xt = r2*cos(tht)
      yt = r2*sin(tht)
      call interp_new(ss(1),xt,yt,z2,idum,u,ierr,rll,ieo)
      if (ierr.eq.0) then
c
         ict = 2. + 9.*(ss(1)-uumin)/(uumax-uumin)
         call color(ict)
         call draw_circ1(xt,yt,z2,0.03)
c
         ieoff = nxyz*(ieo-1) + 1
         call evalsc(ss(2),v(ieoff),rll,0)
         call evalsc(ss(3),w(ieoff),rll,0)
c        call evalsc(ss(4),p(ieoff),rll,0)
         call evalsc(ss(5),t(ieoff),rll,0)
c
c        call interp(ss(2),xt,yt,z2,idum,v,ierr)
c        call interp(ss(3),xt,yt,z2,idum,w,ierr)
cc       call interp(ss(4),xt,yt,z2,idum,p,ierr)
c        call interp(ss(5),xt,yt,z2,idum,t,ierr)
c
         ss(4) = 1.
      else
         call color(14)
         call draw_circ1(xt,yt,z2,0.03)
         call rzero(ss,5)
      endif
c
      ur = ss(1)*cos(tht)+ss(2)*sin(tht)
      ut = ss(2)*cos(tht)-ss(1)*sin(tht)
      uz = ss(3)
      ss(1) = ur
      ss(2) = uz
      ss(3) = ut
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_file(npt,ii,io,ierr)
c
c     .Get asci 2D .fld file for meridional average
c
c     .Output header info to io
c
      include 'basics.inc'
      include 'basicsp.inc'
      CHARACTER FILE*40
      CHARACTER*1 FILE1(40)
      EQUIVALENCE (FILE,FILE1)
      DATA FILE /'                                        '/
C
c
      ierr = 0
c
      CALL PRS('  Name of file containing xy points?:$')
      CALL RES(FILE,40)
      write(6,*) 'Attempting to open formatted file ',file
      open (unit=ii,file=file,status='old',err=999)
      open (unit=io,file='s2d.fld')
c
      read (ii,*,err=999) neli,nxi,nyi,nzi,time,istep
      write(io,1,err=999) neli,nxi,nyi,nzi,time
    1 format(4i4,e14.7,i5,'    U P T',40x)
c
      npt = neli*nxi*nyi
      read (ii,6) (dum,k=1,neli)
      write(io,6) (dum,k=1,neli)
    6 format(6e11.4)
c
      return
  999 continue
      call prs('Unable to open file'//FILE//'$')
      ierr = 1
      return
      end
c-----------------------------------------------------------------------
      subroutine set_anil
c
c     Driver for anil's problem
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      CALL PRS(' ENTER CHOICE for spherical analysis$')
      ITEM(1)='radial average'
      ITEM(2)='meridional plane'
      ITEM(3)='Stream Fct (for 2D only)'
      NCHOIC=3
c
      CALL MENU(XMOUSE,YMOUSE,BUTTON,'SET SPHERE')
c
c
      if (choice.eq.'radial average') then
         call prs('Start avg. data in radial direction$')
         call avg_uvwpt_radial
         call prs('Done avg. data in radial direction$')
      elseif (choice.eq.'meridional plane') then
         call azi_avg
      elseif (choice.eq.'Stream Fct (for 2D only)') then
         call strfct_spec(work)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine interp_new(val,xex,yex,zex,idum,tt,IERR,rrl,ieo)
      include 'basics.inc'
      include 'basicsp.inc'
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      REAL TT(1)
      REAL RRL(3),XXL(3)
      SAVE XXL
C
      REAL    r1last,r2last,r3last
      SAVE    r1last,r2last,r3last
      DATA    r1last,r2last,r3last /3*0./
c
      INTEGER ICALLD,IEOLD,IEOFF,NXYZ
      SAVE    ICALLD,IEOLD,IEOFF,NXYZ
      DATA    ICALLD,IEOLD,IEOFF,NXYZ/4*0/
C
      IF (ICALLD.le.2) THEN
         CALL locglob
         CALL COEF
         NXYZ=NX*NY*NZ
         XXL(1)=1.0e14
         XXL(2)=1.0e14
         XXL(3)=1.0e14
      ENDIF
      ICALLD = ICALLD+1
C
C     Check to see if point is same as last point:
C
      IF (XXL(1).EQ.XEX.AND.XXL(2).EQ.YEX.AND.XXL(3).EQ.ZEX
     $   .AND. IEOLD.NE.0) THEN
         IE = IEOLD
         GOTO 3000
      ENDIF
C
C     Initilize the "last" found data
C
      IELAST=IEOLD
      R1LAST=RRL(1)
      R2LAST=RRL(2)
      R3LAST=RRL(3)
C
      XXL(1) = XEX
      XXL(2) = YEX
      XXL(3) = ZEX
C
C     Check old element first
C
      IF (IEOLD.NE.0) THEN
         IE = IEOLD
         RRL(1)=r1last
         RRL(2)=r2last
         RRL(3)=r3last
         IF(   XBMIN(IE).LE.XXL(1) .AND. XXL(1).LE.XBMAX(IE)
     $   .AND. YBMIN(IE).LE.XXL(2) .AND. XXL(2).LE.YBMAX(IE)
     $   .AND. ZBMIN(IE).LE.XXL(3) .AND. XXL(3).LE.ZBMAX(IE) )
     $   THEN
C
C           Potentially in this element, check to be sure.
C
            call findr(RRL(1),XXL(1),IE,1,ierr)
            IF (ABS(RRL(1)).LE.1.00001 .AND.
     $          ABS(RRL(2)).LE.1.00001 .AND.
     $          ABS(RRL(3)).LE.1.00001       ) GOTO 3000
         ENDIF
      ENDIF
C
C     Now check all elements
C
      call findre(ie,rrl,xxl,rminmax,ierr)
c     WRITE(S,2021) IELAST,xex,yex,zex
c2021 FORMAT(' Point IS INSIDE?    an element. (',I5,3F8.3,')$')
c     CALL PRS(S)
      if (rminmax.le.1.002) goto 3000
c
      WRITE(S,2001) IELAST,xex,yex,zex
 2001 FORMAT(' Point is not inside an element. (',I5,3F8.3,')$')
      CALL PRS(S)
c
      VAL=0.
      IEOLD=0
      IEO  =1
      IENTRP = IEOLD
      IDUM   = IENTRP
      ierr = 1
      return
C
C     Found the element containing this point
C
 3000 CONTINUE
      IF (IEOLD.EQ.IE) THEN
         CALL EVALSC( VAL , TT(IEOFF) ,RRL , 0 )
      ELSE
         IEOLD = IE
         IEOFF=NXYZ*(IE-1)+1
         CALL EVALSC( VAL , TT(IEOFF) ,RRL , 1 )
      ENDIF
      IENTRP = IEOLD
      IEO    = IEOLD
      IDUM   = IENTRP
      ierr = 0
      return
      END
c-----------------------------------------------------------------------
      subroutine strfct_spec(psi)
      include 'basics.inc'
      include 'basicsp.inc'
      REAL PSI(1)
C
      COMMON /PFFLG/  ILGRNG,ISCND,IENTRP,INEWTX,isdiag
      COMMON /CTMP1/  ARCR(NXM,NYM),ARCS(NXM,NYM)
     $               ,PINTX(NXM,NXM),PINTY(NYM,NYM),FLAG(NELM)
     $ ,XRM1(NXM,NYM),YRM1(NXM,NYM),XSM1(NXM,NYM),YSM1(NXM,NYM)
      DIMENSION II1(4),JJ1(4),II2(4),JJ2(4)
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
C     Set ILGRNG=0 so that we know that the Lagrangian (streamline) data
C     in COMMON /CTMP1/ has been clobbered and will need to be recomputed.
C
      ILGRNG=0
C
C     Set matching side indices to set streamfunctions equivalent.
C
C              3
C          +------+
C          |      |
C        4 |      | 2
C          |      |
C          +------+
C             1
C
      II1(1)=1
      JJ1(1)=1
      II1(2)=NX
      JJ1(2)=1
      II1(3)=NX
      JJ1(3)=NY
      II1(4)=1
      JJ1(4)=NY
C
      II2(1)=NX
      JJ2(1)=1
      II2(2)=NX
      JJ2(2)=NY
      II2(3)=1
      JJ2(3)=NY
      II2(4)=1
      JJ2(4)=1
C
C     Set up partial integral operators, PINTX, PINTY:
C
      CALL SPINT(PINTX,ZPTS,WGHT,NX)
      CALL SPINT(PINTY,ZPTS,WGHT,NY)
C
C     Compute PSI locally for each element
C
      DO 100 IE=1,NEL
         call strem2_spec(psi,ie)
  100 CONTINUE
C
C     Adjust the streamfunction level for all elements, starting with IE=1.
C
C     Flag = 0     -     not set
C     Flag = 1     -     set
C     Flag = 2     -     saturated, i.e. - all neighbors are set.
C
      CALL RZERO(FLAG,NEL)
c
c     SPECIAL
      call find_x_flag(k)
      flag(k) = 1.
c     FLAG(1)=1.0
c
      DO 1000 IEE=1,NEL
      DO 1000 IE=1,NEL
         IF (FLAG(IE).EQ.1.) THEN
         DO 800 ISIDE=1,4
            IF (CBC(ISIDE,IE,1).EQ.'E') THEN
               IEN=IFIX(BC(1,ISIDE,IE,1))
               IF (FLAG(IEN).EQ.0.) THEN
C
C                 Adjust neighboring element.
C
                  ISN=IFIX(BC(2,ISIDE,IE,1))
                  I1=II1(ISIDE)
                  J1=JJ1(ISIDE)
                  I2=II2(ISN)
                  J2=JJ2(ISN)
                  DIF=PSI(IND(I1,J1,1,IE))-PSI(IND(I2,J2,1,IEN))
                  DO 600 I=1,NX
                  DO 600 J=1,NY
                     PSI(IND(I,J,1,IEN))=PSI(IND(I,J,1,IEN))+DIF
  600             CONTINUE
                  FLAG(IEN)=1.
               ENDIF
            ENDIF
  800       CONTINUE
            FLAG(IE)=2.
         ENDIF
 1000 CONTINUE
c
      return
      END
c-----------------------------------------------------------------------
      subroutine strem2_spec(psi,ie)
C
      include 'basics.inc'
      include 'basicsp.inc'
C
      REAL PSI(1)
      COMMON /CTMP1/  ARCR(NXM,NYM),ARCS(NXM,NYM)
     $               ,PINTX(NXM,NXM),PINTY(NYM,NYM),FLAG(NELM)
     $ ,XRM1(NXM,NYM),YRM1(NXM,NYM),XSM1(NXM,NYM),YSM1(NXM,NYM)
C
      IND(I,J,K,IEL)=I+NX*(J-1)+NX*NY*(K-1)+NX*NY*NZ*(IEL-1)
C
      CALL MXM(DGDR,NX,XP(IND(1,1,1,IE)),NX,XRM1,NY)
      CALL MXM(DGDR,NX,YP(IND(1,1,1,IE)),NX,YRM1,NY)
C
      CALL MXM(XP(IND(1,1,1,IE)),NX,DGDRT,NY,XSM1,NY)
      CALL MXM(YP(IND(1,1,1,IE)),NX,DGDRT,NY,YSM1,NY)
C
      NXY=NX*NY
C
      CALL RZERO(PSI(IND(1,1,1,IE)),NXY)
C
C        Begin by computing psi along the left border:
C
c     write(6,*) 'this is ifaxis:',ifaxis,ie
      if (.not.ifaxis) then
c
C        X - axisymmetric case        (pff 11/29/98)
C        X - axisymmetric case        (pff 02/29/99)
c
         one   = 1.
         twopi = 8.*atan(one)
         DO 135 IP=1,NY
            IPP = NX*(IP-1) + 1
            FLUX=U(IND(1,IP,1,IE))*YSM1(IPP,1)
     $          -V(IND(1,IP,1,IE))*XSM1(IPP,1)
cSPECIAL    FLUX=flux*twopi*yp(ind(1,ip,1,ie))
            FLUX=flux*twopi*xp(ind(1,ip,1,ie))
C
            DO 125 IY=2,NY
               IYP = IY + NX*(IP-1)
               PSI(IND(1,IY,1,IE))=PSI(IND(1,IY,1,IE))
     $                            +PINTY(IYP,1)*FLUX
  125       CONTINUE
  135    CONTINUE
C
C        Integrate in R direction for each row, IY.
C
         DO 305 IY=1,NY
            DO 205 IX=2,NX
               PSI(IND(IX,IY,1,IE))=PSI(IND(1,IY,1,IE))
  205       CONTINUE
            DO 235 IP=1,NX
               IPY = IP + NX*(IY-1)
               FLUX=U(IND(IP,IY,1,IE))*YRM1(IPY,1)
     $             -V(IND(IP,IY,1,IE))*XRM1(IPY,1)
cSPECIAL       FLUX=flux*twopi*yp(ind(ip,iy,1,ie))
               FLUX=flux*twopi*xp(ind(ip,iy,1,ie))
C
               DO 225 IX=2,NX
                  IXP = IX + NX*(IP-1)
                  PSI(IND(IX,IY,1,IE))=PSI(IND(IX,IY,1,IE))
     $                                +PINTX(IXP,1)*FLUX
  225          CONTINUE
  235       CONTINUE
  305    CONTINUE
      endif
C
      return
      END
c-----------------------------------------------------------------------
      subroutine find_x_flag(iemin)
c
c     Find an element with minimal "x" value
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      n    = nx*ny*nz*nel
      nxyz = nx*ny*nz
c
      xmin = glmax(xp,n)
c
      l = 1
      do ie=1,nel
         xx = vlmin(xp(l),nxyz)
         if (xx.lt.xmin) then
            xmin  = xx
            iemin = ie
         endif
         l = l + nxyz
      enddo
c
      write(6,*) 'this is iemin,xx:',iemin,xx
c
      return
      end
c-----------------------------------------------------------------------
      subroutine integrate_uut6(wint,x0,x1,q1,q2,q3,q4,q5,q6)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real x0(3),x1(3)
      real wint(6),sumi(6)
      real q1(1),q2(1),q3(1),q4(1),q5(1),q6(1)
c
      n=256
c
      ds = (x1(1)-x0(1))**2 + (x1(2)-x0(2))**2 + (x1(3)-x0(3))**2
      if (ds.gt.0) ds = sqrt(ds)
      ds = ds/n
      dx = (x1(1)-x0(1))/n
      dy = (x1(2)-x0(2))/n
      dz = (x1(3)-x0(3))/n
c
      call rzero(sumi,6)
c
      eps = .002
      do i=0,n
         ip = 0
         xi = x0(1) + i*dx
         yi = x0(2) + i*dy
         zi = x0(3) + i*dz
    1    call interp(wint(1),xi,yi,zi,idum,q1,ierr)
         if (ierr.eq.0) then
            call interp(wint(2),xi,yi,zi,idum,q2,ierr)
            call interp(wint(3),xi,yi,zi,idum,q3,ierr)
            call interp(wint(4),xi,yi,zi,idum,q4,ierr)
            call interp(wint(5),xi,yi,zi,idum,q5,ierr)
            call interp(wint(6),xi,yi,zi,idum,q6,ierr)
            if (i.eq.0 .or. i.eq.n ) then
               wt = 1.*ds/3.
            elseif (mod(i,2).eq.1) then
               wt = 4.*ds/3.
            else
               wt = 2.*ds/3.
            endif
c
            call add2s2(sumi,wint,wt,6)
         elseif (i.eq.0.and.ip.le.5) then
            xi = xi+dx*eps
            yi = yi+dy*eps
            zi = zi+dz*eps
            ip = ip+1
            goto1
         elseif (i.eq.n.and.ip.le.5) then
            xi = xi-dx*eps
            yi = yi-dy*eps
            zi = zi-dz*eps
            ip = ip+1
            goto1
         else
            write(6,2) i,'Interp fail:',xi,yi,zi
    2       format(i3,1x,a12,1p7e12.4)
         endif
      enddo
      call copy(wint,sumi,6)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine integrate_uut2(wint,x0,x1,q1,q2)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real x0(3),x1(3)
      real wint(9),sumi(9)
      real q1(1),q2(1)
c
      n=256
      n=128
c
      ds = (x1(1)-x0(1))**2 + (x1(2)-x0(2))**2 + (x1(3)-x0(3))**2
      if (ds.gt.0) ds = sqrt(ds)
      ds = ds/n
      dx = (x1(1)-x0(1))/n
      dy = (x1(2)-x0(2))/n
      dz = (x1(3)-x0(3))/n
c
      nq = 2
      call rzero(sumi,nq)
c
      eps = .002
      write(22,*) 
      do i=0,n
         ip = 0
         xi = x0(1) + i*dx
         yi = x0(2) + i*dy
         zi = x0(3) + i*dz
    1    call interp(wint(1),xi,yi,zi,idum,q1,ierr)
         write(22,*) xi,yi,wint(1)
         if (ierr.eq.0) then
            call interp(wint(2),xi,yi,zi,idum,q2,ierr)
            if (i.eq.0 .or. i.eq.n ) then
               wt = 1.*ds/3.
            elseif (mod(i,2).eq.1) then
               wt = 4.*ds/3.
            else
               wt = 2.*ds/3.
            endif
c
            call add2s2(sumi,wint,wt,nq)
         elseif (i.eq.0.and.ip.le.5) then
            xi = xi+dx*eps
            yi = yi+dy*eps
            zi = zi+dz*eps
            ip = ip+1
            goto1
         elseif (i.eq.n.and.ip.le.5) then
            xi = xi-dx*eps
            yi = yi-dy*eps
            zi = zi-dz*eps
            ip = ip+1
            goto1
         else
            write(6,2) i,'Interp fail:',xi,yi,zi
    2       format(i3,1x,a12,1p7e12.4)
         endif
      enddo
      call copy(wint,sumi,nq)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine integrate_uut9(wint,x0,x1,q1,q2,q3,q4,q5,q6,q7,q8,q9)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      real x0(3),x1(3)
      real wint(9),sumi(9)
      real q1(1),q2(1),q3(1)
      real q4(1),q5(1),q6(1)
      real q7(1),q8(1),q9(1)
c
      n=256
      n=128
c
      ds = (x1(1)-x0(1))**2 + (x1(2)-x0(2))**2 + (x1(3)-x0(3))**2
      if (ds.gt.0) ds = sqrt(ds)
      ds = ds/n
      dx = (x1(1)-x0(1))/n
      dy = (x1(2)-x0(2))/n
      dz = (x1(3)-x0(3))/n
c
      call rzero(sumi,9)
c
      eps = .002
      write(22,*) 
      do i=0,n
         ip = 0
         xi = x0(1) + i*dx
         yi = x0(2) + i*dy
         zi = x0(3) + i*dz
    1    call interp(wint(1),xi,yi,zi,idum,q1,ierr)
         write(22,*) xi,yi,wint(1)
         if (ierr.eq.0) then
            call interp(wint(2),xi,yi,zi,idum,q2,ierr)
            call interp(wint(3),xi,yi,zi,idum,q3,ierr)
            call interp(wint(4),xi,yi,zi,idum,q4,ierr)
            call interp(wint(5),xi,yi,zi,idum,q5,ierr)
            call interp(wint(6),xi,yi,zi,idum,q6,ierr)
            call interp(wint(7),xi,yi,zi,idum,q7,ierr)
            call interp(wint(8),xi,yi,zi,idum,q8,ierr)
            call interp(wint(9),xi,yi,zi,idum,q9,ierr)
            if (i.eq.0 .or. i.eq.n ) then
               wt = 1.*ds/3.
            elseif (mod(i,2).eq.1) then
               wt = 4.*ds/3.
            else
               wt = 2.*ds/3.
            endif
c
            call add2s2(sumi,wint,wt,9)
         elseif (i.eq.0.and.ip.le.5) then
            xi = xi+dx*eps
            yi = yi+dy*eps
            zi = zi+dz*eps
            ip = ip+1
            goto1
         elseif (i.eq.n.and.ip.le.5) then
            xi = xi-dx*eps
            yi = yi-dy*eps
            zi = zi-dz*eps
            ip = ip+1
            goto1
         else
            write(6,2) i,'Interp fail:',xi,yi,zi
    2       format(i3,1x,a12,1p7e12.4)
         endif
      enddo
      call copy(wint,sumi,9)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles4
c
c     Similar to compute_miles, save that it's designed for the
c     intermittent grooved-flat (periodic) case, assuming L_grooves = L_flat
c
c     ALSO -- Note that T is now scaled by user supplied "c" value
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (lxyz=nxm*nym*nzm)
      common /ctmp4/ work1(lxyz),work2(lxyz)
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ col(22,0:mopts-1)
      real uvw(13),x0(3),x1(3)
      real wint(9)
      logical iftmp
c
      call rzero(col,22*mopts)
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR miles4 -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
c
c     Compute stuff for Miles, full groove + flat
c
c
c     call prs('Would you like to generate a time trace file? (y/n)$')
c     call res(ans,1)
c     if (ans.eq.'y'  .or.  ans.eq.'Y') call time_trace
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     4. ~q"(x)  :=   q"(x) * ds/dx , where s is the surface area
c                                     (ds/dx = sqrt(2) in groove)
c                                     (ds/dx = 1 in channel)
c     5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c     6. Tc(x) -- z-avg'd centerline temperature
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c
c     NOTE:  This is currently hardwired for the 1-groove+flat geometry,
c            due to the ds/dx.
c
c     Set up quantities
c
c
c     compute
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     6. Tc(x) -- z-avg'd centerline temperature
c
c     Offset zmid so we're not right on an element bdry.
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      xmid = 0.500*(xmax+xmin)
      ymid = 0.500*(ymax+ymin)
      zmid = 0.625*(zmax+zmin)
c
c     Scale T and P
c
      call prs('Enter average c value from logfile:$')
      call rer(cscl)
      call prs('Enter average dpdx from logfile:$')
      call rer(dpdx)
      do i=1,ntot
         xl = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
         p(i) = p(i)+dpdx*xl
      enddo
c
      call interp_all(uvw,xmin,ymid,zmid,ierr)
c
c
c
c     Compute ubar^2
c
      do i=1,ntot
         vrt2(i) = u(i)**2
      enddo
c
c
      call prs('Enter .fld number for file containing <u">^2:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
c     Get u'^2
      quanty = 'VELOCITY'
      deriv  = 'NO'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
      do i=1,ntot
         vrt3(i) = u(i)
      enddo
c
c
c
      call prs('Enter .fld number for file containing <uT> as W:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
      if (ndim.eq.2) call copy(w,v,ntot)
      do i=1,ntot
         xl = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
         w(i) = w(i)*exp(-cscl*xl)
         p(i) = p(i)+dpdx*xl
      enddo
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
      call setwrk(.true.)
c
c     dp/dx is now in work,  after the next setwrk call, dpdx will be in wrk2
c
c
c     n  = 2400
      call prs('Input number of points in x-direction:$')
      call rei(n)
      dx = (xmax-xmin)/n
c
      do i=0,n
         xx = xmin + i*dx
         call interp_all(uvw,xx,ymid,zmid,ierr)
c
c        - x -
c
         col(1,i) = xx
c
c
c        - dp/dx -
c
         col(2,i) = uvw(6)
c
c
c        - (P(L)-P0)/L -
c
         if (i.eq.0) then
            p0       = uvw(4)
            col(3,i) = uvw(6)
         else
            col(3,i) = (p0-uvw(4))/(col(1,0)-col(1,i))
         endif
c
c
c        - Tc -
c
         col(6,i) = uvw(5)
c
      enddo
c
c
c     Set quantity to vorticity, and save in "vrt1"
c
      quanty = 'VORTICITY'
      deriv  = 'NO'
      compon = 'X'
      call setwrk(.true.)
      call copy(vrt1,work,ntot)
c
c     Now compute q"
c
c     Set quantity to magnitude of grad T
c
      quanty = 'TEMPERATURE'
      deriv  = 'GRAD'
      compon = 'M'
      call setwrk(.true.)
c
      zmid = 0.53*(zmax+zmin)
c
c      4. ~q"(x)  :=   q"(x) * ds/dx
c
      do jpass=1,2
c
         m = 2*n
         if (jpass.eq.2) m = n
         do i=0,m
            col(4,i) = 0. 
         enddo
         dxx = (xmax-xmin)/m
c
         do ipass=1,2
         do i=0,m
            xx = xmin + i*dxx
            eps = 1.e-5
c
c           For g7f7 geometry, grooves are < xmid, flat is > xmid
c
            if (xx.lt.xmid) then
               xl = i*dxx + .024
  111          xl = xl-.024
               if (xl.gt.0.024) goto 111
               if (xl.le.0.012) then
                  ytop = 0.022 + xl-eps
                  ybot = 0.012 - xl+eps
                  two  = 2.
                  dsdx = sqrt(two)
               elseif (xl.le.0.024) then
                  ytop = 0.034 - (xl-.012)-eps
                  ybot =         (xl-.012)+eps
                  two  = 2.
                  dsdx = sqrt(two)
               endif
            else
               ytop = 0.022-eps
               ybot = 0.012+eps
               dsdx = 1.0
            endif
            if (ipass.eq.1) call interp_all(uvw,xx,ytop,zmid,ierr)
            if (ipass.eq.2) call interp_all(uvw,xx,ybot,zmid,ierr)
            col(4,i) = col(4,i) + dsdx*uvw(6)
c           write(6,*) i,' INTERP: ',xx,ytop,ybot,zmid,eps,ierr
c
c
            if (jpass.eq.2) then
c              This is the integrated wall shear stress (from vorticity)
               if (ipass.eq.1) then
                  col(13,i) = col(13,i) + uvw(11)/dsdx
               else
                  col(13,i) = col(13,i) - uvw(11)/dsdx
               endif
            endif
c
         enddo
         enddo
c
         if (jpass.eq.1) then
c
c           5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c              via trapezoidal integration in x
c
            col(5,0) = col(4,0)
            sum = 0.
            do i=1,n
               sum = sum + (col(4,2*i-2)+4.*col(4,2*i-1)+col(4,2*i))/6.
               avg = sum/i
               col(5,i) = avg
            enddo
         endif
      enddo
c
c
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c     Save pb, ub^2 and u'^2 prior to differentiation.
      call copy(wkv1  ,p   ,ntot)
      call copy(vrt1  ,vrt2,ntot)
      call copy(vortex,vrt3,ntot)
c
      nxyz=nx*ny*nz
      j   =1
      do ie=1,nel
         call dudxyz(work2,p   ,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (p(j)   ,work2,nxyz)
         call dudxyz(work2,vrt2,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (vrt2(j),work2,nxyz)
         call dudxyz(work2,vrt3,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (vrt3(j),work2,nxyz)
         j=j+nxyz
      enddo
c
      do i=0,n
         xx = xmin + i*dx
         if (xx.le.0.) xx = 1.e-9
c
c        For g7f7 geometry, grooves are < xmid, flat is > xmid
c
         if (xx.lt.xmid) then
            xl = i*dxx + .024
  211       xl = xl-.024
            if (xl.gt.0.024) goto 211
            if (xl.le.0.012) then
               ytop = 0.022 + xl-eps
               ybot = 0.012 - xl+eps
            elseif (xl.le.0.024) then
               ytop = 0.034 - (xl-.012)-eps
               ybot =         (xl-.012)+eps
            endif
         else
            ytop = 0.022-eps
            ybot = 0.012+eps
         endif
         x0(1) = xx
         x0(2) = ybot
         x0(3) = zmid
         x1(1) = xx
         x1(2) = ytop
         x1(3) = zmid
         call integrate_uut9
     $      (wint,x0,x1,u,w,t,p,vrt2,vrt3,wkv1,vrt1,vortex)
         col( 7,i) = wint(2)/wint(1)
         col( 8,i) = wint(2)
         col( 9,i) = wint(1)
         col(12,i) = wint(3)
c
c        dist = ytop-ybot
         dist = 1.
c
         col(14,i) = wint(4)/dist             !d/dx pbar
         col(15,i) = wint(5)/dist             !d/dx ubar^2
         col(16,i) = wint(6)/dist             !d/dx u'^2
c
         col(17,i) = wint(7)/dist             ! pbar
         col(18,i) = wint(8)/dist             ! ubar^2
         col(19,i) = wint(9)/dist             ! u'^2
c
      enddo
c
c     A quick check to see if dTb/dx = ~q"
c
c     .col(10) -- int_0^L (q")
c     .col(11) -- heat_in - heat_out   (convective flux)
c     .col(12) -- heat_in - heat_out   (conductive flux)
c
c
c
      i0 = 0
c     call pris(n,' values in list, input i0 for heat check:$')
c     call rei (i0)
      rhocp   = param(7)
      conduct = param(8)
      do i=0,n
         col(10,i) = dz*conduct*col(5,i)*(col(1,i)-col(1,i0))
         col(11,i) = dz*rhocp*(col(8,i0)-col(8,i))
     $             - dz*conduct*(col(12,i0)-col(12,i))
      enddo
c
c   x      dp/dx    Pout(L)-Pin  ~q"  int(~q")/L   Tc(x)     Tb(x)
c
c
      open(unit=14,file='xprofw.dat')
      do i=0,n
         write(14,7) (col(k,i),k=1,2),(col(j,i),j=13,19)
      enddo
  7   format(1p11e12.4)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprofw.dat.$')
c
c
      open(unit=14,file='xprof.dat')
      do i=0,n
         write(14,6) (col(j,i),j=1,7)
      enddo
  6   format(1p11e11.3)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprof.dat.$')
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles5_old
c
c     Similar to compute_miles4, save that it's designed for the
c     groove only case.
c
c     Also, it's been modified to look for data generated by subuser.f
c
c     ALSO -- Note that T is now scaled by user supplied "c" value
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (lxyz=nxm*nym*nzm)
      common /ctmp4/ work1(lxyz),work2(lxyz)
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ col(22,0:mopts-1)
      real uvw(13),x0(3),x1(3)
      real wint(9)
      logical iftmp
c
      call rzero(col,22*mopts)
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR miles5 -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
c
c     Compute stuff for Miles, full groove + flat
c
c
c     call prs('Would you like to generate a time trace file? (y/n)$')
c     call res(ans,1)
c     if (ans.eq.'y'  .or.  ans.eq.'Y') call time_trace
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     4. ~q"(x)  :=   q"(x) * ds/dx , where s is the surface area
c                                     (ds/dx = sqrt(2) in groove)
c                                     (ds/dx = 1 in channel)
c     5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c     6. Tc(x) -- z-avg'd centerline temperature
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c
c     NOTE:  This is currently hardwired for the 1-groove+flat geometry,
c            due to the ds/dx.
c
c     Set up quantities
c
c
c     compute
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     6. Tc(x) -- z-avg'd centerline temperature
c
c     Offset zmid so we're not right on an element bdry.
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      xmid = 0.500*(xmax+xmin)
      xmid = xmid + (xmax-xmin)   ! g4 geometry
      ymid = 0.500*(ymax+ymin)
      zmid = 0.625*(zmax+zmin)
c
c     Scale T and P
c
      call prs('Enter average c value from logfile:$')
      call rer(cscl)
      call prs('Enter average dpdx from logfile:$')
      call rer(dpdx)
      do i=1,ntot
         xl = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
         p(i) = p(i)+dpdx*xl
      enddo
c
      call interp_all(uvw,xmin,ymid,zmid,ierr)
c
c
c
c     Compute ubar^2
c
      do i=1,ntot
         vrt2(i) = u(i)**2
      enddo
c
c
      call prs('Enter .fld number for file containing <u">^2:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
c     Get u'^2
      quanty = 'VELOCITY'
      deriv  = 'NO'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
      do i=1,ntot
         vrt3(i) = u(i)
      enddo
c
c
c
      call prs('Enter .fld number for file containing <uT> as W:$')
      call rei(jfld)
c
c     Get .fld file, and compute dpdx
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
      if (ndim.eq.2) call copy(w,v,ntot)
      do i=1,ntot
         xl = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
c        w(i) = w(i)*exp(-cscl*xl)
         w(i) = t(i)*u(i)                 ! quick hack  4/19/01
         p(i) = p(i)+dpdx*xl
      enddo
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
      call setwrk(.true.)
c
c     dp/dx is now in work,  after the next setwrk call, dpdx will be in wrk2
c
c
      n  = 600
c     call prs('Input number of points in x-direction:$')
c     call rei(n)
      dx = (xmax-xmin)/n
c
      do i=0,n
         xx = xmin + i*dx
         call interp_all(uvw,xx,ymid,zmid,ierr)
c
c        - x -
c
         col(1,i) = xx
c
c
c        - dp/dx -
c
         col(2,i) = uvw(6)
c
c
c        - (P(L)-P0)/L -
c
         if (i.eq.0) then
            p0       = uvw(4)
            col(3,i) = uvw(6)
         else
            col(3,i) = (p0-uvw(4))/(col(1,0)-col(1,i))
         endif
c
c
c        - Tc -
c
         col(6,i) = uvw(5)
c
      enddo
c
c
c     Set quantity to vorticity, and save in "vrt1"
c
      quanty = 'VORTICITY'
      deriv  = 'NO'
      compon = 'X'
      call setwrk(.true.)
      call copy(vrt1,work,ntot)
c
c     Now compute q"
c
c     Set quantity to magnitude of grad T
c
      quanty = 'TEMPERATURE'
      deriv  = 'GRAD'
      compon = 'M'
      call setwrk(.true.)
c
      zmid = 0.53*(zmax+zmin)
c
c      4. ~q"(x)  :=   q"(x) * ds/dx
c
      do jpass=1,2
c
         m = 2*n
         if (jpass.eq.2) m = n
         do i=0,m
            col(4,i) = 0. 
         enddo
         dxx = (xmax-xmin)/m
c
         do ipass=1,2
         do i=0,m
            xx = xmin + i*dxx
            eps = 1.e-5
c
c           For g4 geometry, grooves are < xmid, flat is > xmid
c
            if (xx.lt.xmid) then
               xl = i*dxx + .024
  111          xl = xl-.024
               if (xl.gt.0.024) goto 111
               if (xl.le.0.012) then
                  ytop = 0.022 + xl-eps
                  ybot = 0.012 - xl+eps
                  two  = 2.
                  dsdx = sqrt(two)
               elseif (xl.le.0.024) then
                  ytop = 0.034 - (xl-.012)-eps
                  ybot =         (xl-.012)+eps
                  two  = 2.
                  dsdx = sqrt(two)
               endif
            else
               ytop = 0.022-eps
               ybot = 0.012+eps
               dsdx = 1.0
            endif
            if (ipass.eq.1) call interp_all(uvw,xx,ytop,zmid,ierr)
            if (ipass.eq.2) call interp_all(uvw,xx,ybot,zmid,ierr)
            col(4,i) = col(4,i) + dsdx*uvw(6)
c           write(6,*) i,' INTERP: ',xx,ytop,ybot,zmid,eps,ierr
c
c
            if (jpass.eq.2) then
c              This is the integrated wall shear stress (from vorticity)
               if (ipass.eq.1) then
                  col(13,i) = col(13,i) + uvw(11)/dsdx
               else
                  col(13,i) = col(13,i) - uvw(11)/dsdx
               endif
            endif
c
         enddo
         enddo
c
         if (jpass.eq.1) then
c
c           5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c              via trapezoidal integration in x
c
            col(5,0) = 0.
            sum = 0.
            h   = (xmax-xmin)/n
            do i=1,n
               sum = sum + (col(4,2*i-2)+4.*col(4,2*i-1)+col(4,2*i))/6.
c
               avg = sum/i
               col(5,i) = avg   !  This is the mean heat flux, avg'd from 0 to x
c
c              hfx = h*sum      !  This is the integrated heat flux from 0 to x
c              col(5,i) = hfx
c
            enddo
         endif
      enddo
c
c
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c     Save pb, ub^2 and u'^2 prior to differentiation.
      call copy(wkv1  ,p   ,ntot)
      call copy(vrt1  ,vrt2,ntot)
      call copy(vortex,vrt3,ntot)
c
      nxyz=nx*ny*nz
      j   =1
      do ie=1,nel
         call dudxyz(work2,p   ,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (p(j)   ,work2,nxyz)
         call dudxyz(work2,vrt2,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (vrt2(j),work2,nxyz)
         call dudxyz(work2,vrt3,rxm1,sxm1,txm1,jacm1,ie)
         call copy  (vrt3(j),work2,nxyz)
         j=j+nxyz
      enddo
c
      do i=0,n
         xx = xmin + i*dx
         if (xx.le.0.) xx = 1.e-9
c
c        For g4 geometry, grooves are < xmid, flat is > xmid
c
         if (xx.lt.xmid) then
            xl = i*dxx + .024
  211       xl = xl-.024
            if (xl.gt.0.024) goto 211
            if (xl.le.0.012) then
               ytop = 0.022 + xl-eps
               ybot = 0.012 - xl+eps
            elseif (xl.le.0.024) then
               ytop = 0.034 - (xl-.012)-eps
               ybot =         (xl-.012)+eps
            endif
         else
            ytop = 0.022-eps
            ybot = 0.012+eps
         endif
         x0(1) = xx
         x0(2) = ybot
         x0(3) = zmid
         x1(1) = xx
         x1(2) = ytop
         x1(3) = zmid
         call integrate_uut9
     $      (wint,x0,x1,u,w,t,p,vrt2,vrt3,wkv1,vrt1,vortex)
         col( 7,i) = wint(2)/wint(1)
         col( 8,i) = wint(2)
         col( 9,i) = wint(1)
         col(12,i) = wint(3)
c
c        dist = ytop-ybot
         dist = 1.
c
         col(14,i) = wint(4)/dist             !d/dx pbar
         col(15,i) = wint(5)/dist             !d/dx ubar^2
         col(16,i) = wint(6)/dist             !d/dx u'^2
c
         col(17,i) = wint(7)/dist             ! pbar
         col(18,i) = wint(8)/dist             ! ubar^2
         col(19,i) = wint(9)/dist             ! u'^2
c
      enddo
c
c     A quick check to see if dTb/dx = ~q"
c
c     .col(10) -- int_0^L (q")
c     .col(11) -- heat_in - heat_out   (convective flux)
c     .col(12) -- heat_in - heat_out   (conductive flux)
c
c
c
      i0 = 0
c     call pris(n,' values in list, input i0 for heat check:$')
c     call rei (i0)
      rhocp   = param(7)
      conduct = param(8)
      do i=0,n
         col(10,i) = dz*conduct*col(5,i)*(col(1,i)-col(1,i0))
         col(11,i) = dz*rhocp*(col(8,i0)-col(8,i))
     $             - dz*conduct*(col(12,i0)-col(12,i))
      enddo
c
c   x      dp/dx    Pout(L)-Pin  ~q"  int(~q")/L   Tc(x)     Tb(x)
c
c
      open(unit=14,file='xprofw.dat')
      do i=0,n
c        write(14,7) (col(k,i),k=1,2),(col(j,i),j=13,19)
         write(14,7) (col(k,i),k=1,2),(col(j,i),j=8,9)
      enddo
  7   format(1p11e12.4)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprofw.dat.$')
c
c
      open(unit=14,file='xprof.dat')
      do i=0,n
         write(14,6) (col(j,i),j=1,7)
      enddo
  6   format(1p11e11.3)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprof.dat.$')
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine compute_miles5
c
c     Similar to compute_miles4, save that it's designed for the
c     groove only case.
c
c     Also, it's been modified to look for data generated by subuser.f
c
c     ALSO -- Note that T is now scaled by user supplied "c" value
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter (lxyz=nxm*nym*nzm)
      common /ctmp4/ work1(lxyz),work2(lxyz)
c
      parameter (mopts=9) ! = 9000)
      common /rlobjs/ col(22,0:mopts-1)
      real uvw(13),x0(3),x1(3)
      real wint(9)
      logical iftmp
c
      call rzero(col,22*mopts)
      IF (lvrt.lt.maxpts) THEN
        call prs('ERROR miles5 -- set lvrt=maxpts in basicsp.inc$')
        return
      endif
c
c     Compute stuff for Miles, full groove + flat
c
c
c     call prs('Would you like to generate a time trace file? (y/n)$')
c     call res(ans,1)
c     if (ans.eq.'y'  .or.  ans.eq.'Y') call time_trace
c
c
c
c     SPATIAL DATA --- blah vs X.
c
c
c     Objectives:
c
c
c     Compute the following as a function of x, and put into col i, where i=1...7
c
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     4. ~q"(x)  :=   q"(x) * ds/dx , where s is the surface area
c                                     (ds/dx = sqrt(2) in groove)
c                                     (ds/dx = 1 in channel)
c     5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c     6. Tc(x) -- z-avg'd centerline temperature
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c
c     NOTE:  This is currently hardwired for the groove geometry,
c            due to the ds/dx.
c
c     Set up quantities
c
c
c     compute
c     1. x
c     2. dp/dx
c     3. P(x) - P(0)
c     6. Tc(x) -- z-avg'd centerline temperature
c
c     Offset zmid so we're not right on an element bdry.
c
      ntot = nx*ny*nz*nel
      xmin = glmin(xp,ntot)
      xmax = glmax(xp,ntot)
      ymin = glmin(yp,ntot)
      ymax = glmax(yp,ntot)
      zmin = glmin(zp,ntot)
      zmax = glmax(zp,ntot)
      dz   = zmax-zmin
      if (ndim.eq.2) dz = 1.
c
      xmid = 0.500*(xmax+xmin)
      xmid = xmid + (xmax-xmin)   ! g4 geometry
      ymid = 0.500*(ymax+ymin)
      zmid = 0.625*(zmax+zmin)
c
      call interp_all(uvw,xmin,ymid,zmid,ierr)  ! trigger interp setup
c
c     Scale T and P
c
      call prs('Enter average c value from logfile:$')
      call rer(cscl)
      call prs('Enter average dpdx from logfile:$')
      call rer(dpdx)
c
c     Get avg .fld file
c
      quanty = 'VELOCITY'
      deriv  = 'NO'
      compon = 'X'
c
c     Turn on Z-averaging .... requires Z-planar .rea file
      iftmp    = ifavgupt
      ifavgupt = .true.
      call prs('Enter .fld number for file containing time avg data:$')
      call rei(jfld)
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
c
c
      do i=1,ntot
         xl = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
         p(i) = p(i)+dpdx*xl
      enddo
c
c
      quanty = 'PRESSURE'
      deriv  = 'GRAD'
      compon = 'X'
      call setwrk(.true.)
c     dp/dx is now in work,  after the next setwrk call, dpdx will be in wrk2
c
c     call prs('Input number of points in x-direction:$')
c     call rei(n)
      n  = 600
      dx = (xmax-xmin)/n
c
      do i=0,n
         xx = xmin + i*dx
         call interp_all(uvw,xx,ymid,zmid,ierr)
c
c        - x -
         col(1,i) = xx
c
c        centerline - dp/dx -
         col(2,i) = uvw(6)
c
c        centerline - (P(L)-P0)/L -
         if (i.eq.0) then
            p0       = uvw(4)
            col(3,i) = uvw(6)
         else
            col(3,i) = (p0-uvw(4))/(col(1,0)-col(1,i))
         endif
c
c        - Tc -
         col(6,i) = uvw(5)
      enddo
c
c
c
c     Now compute q"
c
c     Set quantity to magnitude of grad T
c
      quanty = 'TEMPERATURE'
      deriv  = 'GRAD'
      compon = 'M'
      call setwrk(.true.)
c
      zmid = 0.53*(zmax+zmin)
c
c      4. ~q"(x)  :=   q"(x) * ds/dx
c
      do jpass=1,2
c
         m = 2*n
         if (jpass.eq.2) m = n
         do i=0,m
            col(4,i) = 0. 
         enddo
         dxx = (xmax-xmin)/m
c
         do ipass=1,2
         do i=0,m
            xx = xmin + i*dxx
            eps = 1.e-5
c
c           For g4 geometry, grooves are < xmid, flat is > xmid
c
            if (xx.lt.xmid) then
               xl = i*dxx + .024
  111          xl = xl-.024
               if (xl.gt.0.024) goto 111
               if (xl.le.0.012) then
                  ytop = 0.022 + xl-eps
                  ybot = 0.012 - xl+eps
                  two  = 2.
                  dsdx = sqrt(two)
               elseif (xl.le.0.024) then
                  ytop = 0.034 - (xl-.012)-eps
                  ybot =         (xl-.012)+eps
                  two  = 2.
                  dsdx = sqrt(two)
               endif
            else
               ytop = 0.022-eps
               ybot = 0.012+eps
               dsdx = 1.0
            endif
            if (ipass.eq.1) call interp_all(uvw,xx,ytop,zmid,ierr)
            if (ipass.eq.2) call interp_all(uvw,xx,ybot,zmid,ierr)
            col(4,i) = col(4,i) + dsdx*uvw(6)
c           write(6,*) i,' INTERP: ',xx,ytop,ybot,zmid,eps,ierr
c
         enddo
         enddo
c
         if (jpass.eq.1) then
c
c           5. q(L) := W int(~q"(x) dx) - Net heat flux vs L
c              via trapezoidal integration in x
c
            col(5,0) = 0.
            sum = 0.
            h   = (xmax-xmin)/n
            do i=1,n
               sum = sum + (col(4,2*i-2)+4.*col(4,2*i-1)+col(4,2*i))/6.
c
               avg = sum/i
               col(5,i) = avg   !  This is the mean heat flux, avg'd from 0 to x
c
c              hfx = h*sum      !  This is the integrated heat flux from 0 to x
c              col(5,i) = hfx
c
            enddo
         endif
      enddo
c
c
c     7. Tb(x) := int( <uT> dy ) / int( <u> ), where <.> denotes temporal avg.
c                                             NOTE: this reqr's hmt's new 
c                                             temporal avg. data for <uT>,
c                                             which is stored in w of sp.out
c
c     Save u,T prior to loading next .fld file.
      call copy(wkv1  ,u   ,ntot)
      call copy(wkv2  ,T   ,ntot)
c
c     Get .fld file to compute bulk temperature
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'X'
c
      iftmp    = ifavgupt !  Turn on Z-averaging 
      ifavgupt = .true.   !  requires Z-planar .rea file
      call prs('Enter .fld number for file containing <uT> as W:$')
      call rei   (jfld)
      call getfld(jfld,ierr,.false.,.true.)
      ifavgupt = iftmp
      do i=1,ntot
         xl   = xp(i)-xmin
         t(i) = t(i)*exp(-cscl*xl)
      enddo
c
c
      do i=0,n
         xx = xmin + i*dx
         if (xx.le.0.) xx = 1.e-9
c
c        For g4 geometry, grooves are < xmid, flat is > xmid
c
         if (xx.lt.xmid) then
            xl = i*dxx + .024
  211       xl = xl-.024
            if (xl.gt.0.024) goto 211
            if (xl.le.0.012) then
               ytop = 0.022 + xl-eps
               ybot = 0.012 - xl+eps
            elseif (xl.le.0.024) then
               ytop = 0.034 - (xl-.012)-eps
               ybot =         (xl-.012)+eps
            endif
         else
            ytop = 0.022-eps
            ybot = 0.012+eps
         endif
         x0(1) = xx
         x0(2) = ybot
         x0(3) = zmid
         x1(1) = xx
         x1(2) = ytop
         x1(3) = zmid
         call integrate_uut2
     $      (wint,x0,x1,wkv1,t)
         col( 7,i) = wint(2)/wint(1)
         col( 8,i) = wint(2)
         col( 9,i) = wint(1)
      enddo
c
c
c   x      dp/dx    Pout(L)-Pin  ~q"  int(~q")/L   Tc(x)     Tb(x)
c
c
      open(unit=14,file='xprofw.dat')
      do i=0,n
         write(14,7) (col(k,i),k=1,2),(col(j,i),j=8,9)
      enddo
  7   format(1p11e12.4)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprofw.dat.$')
c
c
      open(unit=14,file='xprof.dat')
      do i=0,n
         write(14,6) (col(j,i),j=1,7)
      enddo
  6   format(1p11e11.3)
      close(unit=14)
      n1 = n+1
      call pris  (n1,' values written to xprof.dat.$')
c
      quanty = 'TEMPERATURE'
      deriv  = 'NO'
      compon = 'M'
      call setwrk(.true.)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine shift_comp
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      CALL PRS(' ENTER component to be shifted (x,y, or z)$')
      call res(ans,1)
c
      CALL PRS(' ENTER shift amount$')
      call rer(shift)
c
      ntot = nx*ny*nz*nel
      if (ans.eq.'x'.or.ans.eq.'X') call cadd(wkv1,shift,ntot)
      if (ans.eq.'y'.or.ans.eq.'Y') call cadd(wkv2,shift,ntot)
      if (ans.eq.'z'.or.ans.eq.'Z') call cadd(wkv3,shift,ntot)
c
      call blank(line,70)
      write(line,1) ntot,ans,shift
    1 format(i9,' pts of ',a1,'-component shifted by',1pe11.3,'$')
      CALL PRS(line)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine scale_comp
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      CALL PRS(' ENTER component to be scaled (x,y, or z)$')
      call res(ans,1)
c
      CALL PRS(' ENTER scale factor$')
      call rer(scale)
c
      ntot = nx*ny*nz*nel
      if (ans.eq.'x'.or.ans.eq.'X') call cmult(wkv1,scale,ntot)
      if (ans.eq.'y'.or.ans.eq.'Y') call cmult(wkv2,scale,ntot)
      if (ans.eq.'z'.or.ans.eq.'Z') call cmult(wkv3,scale,ntot)
c
      call blank(line,70)
      write(line,1) ntot,ans,scale
    1 format(i9,' pts of ',a1,'-component scaled by',1pe11.3,'$')
      CALL PRS(line)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine facind (kx1,kx2,ky1,ky2,kz1,kz2,nx1,ny1,nz1,iface)
      kx1=1
      ky1=1
      kz1=1
      kx2=nx1
      ky2=ny1
      kz2=nz1
      if (iface.eq.1) ky2=1
      if (iface.eq.2) kx1=nx1
      if (iface.eq.3) ky1=ny1
      if (iface.eq.4) kx2=1
      if (iface.eq.5) kz2=1
      if (iface.eq.6) kz1=nz1
      return
      end
c-----------------------------------------------------------------------
      subroutine facev(a,ie,iface,val,nx,ny,nz)
c
c     Assign the value VAL to face(IFACE,IE) of array A.
c     IFACE is the input in the pre-processor ordering scheme.
c
      real a(nx,ny,nz,1)
      call facind (kx1,kx2,ky1,ky2,kz1,kz2,nx,ny,nz,iface)
      do 100 iz=kz1,kz2
      do 100 iy=ky1,ky2
      do 100 ix=kx1,kx2
         a(ix,iy,iz,ie)=val
  100 continue
      return
      end
c-----------------------------------------------------------------------
      subroutine getnormals(unx,uny,unz,ie,is,x,y,z,ld,ndim)
      real x(ld,8),y(ld,8),z(ld,8)
c
      real u(3),v(3),w(3)
c
c     Base normals on cross-product of diagonals
c
      if (is.eq.1) then
         u(1) = x(ie,6)-x(ie,1)
         u(2) = y(ie,6)-y(ie,1)
         u(3) = z(ie,6)-z(ie,1)
c
         v(1) = x(ie,5)-x(ie,2)
         v(2) = y(ie,5)-y(ie,2)
         v(3) = z(ie,5)-z(ie,2)
      elseif(is.eq.2) then
         u(1) = x(ie,7)-x(ie,2)
         u(2) = y(ie,7)-y(ie,2)
         u(3) = z(ie,7)-z(ie,2)
c
         v(1) = x(ie,6)-x(ie,3)
         v(2) = y(ie,6)-y(ie,3)
         v(3) = z(ie,6)-z(ie,3)
      elseif(is.eq.3) then
         u(1) = x(ie,8)-x(ie,3)
         u(2) = y(ie,8)-y(ie,3)
         u(3) = z(ie,8)-z(ie,3)
c
         v(1) = x(ie,7)-x(ie,4)
         v(2) = y(ie,7)-y(ie,4)
         v(3) = z(ie,7)-z(ie,4)
      elseif(is.eq.4) then
         u(1) = x(ie,5)-x(ie,4)
         u(2) = y(ie,5)-y(ie,4)
         u(3) = z(ie,5)-z(ie,4)
c
         v(1) = x(ie,8)-x(ie,1)
         v(2) = y(ie,8)-y(ie,1)
         v(3) = z(ie,8)-z(ie,1)
      elseif(is.eq.5) then
         u(1) = x(ie,4)-x(ie,2)
         u(2) = y(ie,4)-y(ie,2)
         u(3) = z(ie,4)-z(ie,2)
c
         v(1) = x(ie,3)-x(ie,1)
         v(2) = y(ie,3)-y(ie,1)
         v(3) = z(ie,3)-z(ie,1)
      elseif(is.eq.6) then
         u(1) = x(ie,7)-x(ie,5)
         u(2) = y(ie,7)-y(ie,5)
         u(3) = z(ie,7)-z(ie,5)
c
         v(1) = x(ie,8)-x(ie,6)
         v(2) = y(ie,8)-y(ie,6)
         v(3) = z(ie,8)-z(ie,6)
      endif
c
      call vcross_normal(w,u,v)
      unx = w(1)
      uny = w(2)
      unz = w(3)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine vcross_normal(u,v,w)
C
C     Compute a Cartesian vector cross product.
C
      real u(3),v(3),w(3)
c
      u(1) = v(2)*w(3) - v(3)*w(2)
      u(2) = v(3)*w(1) - v(1)*w(3)
      u(3) = v(1)*w(2) - v(2)*w(1)
c
c     Normalize
c
      r2 = u(1)*u(1) + u(2)*u(2) + u(3)*u(3)
      if (r2.gt.0.) then
         r2 = sqrt(r2)
         u(1) = u(1)/r2
         u(2) = u(2)/r2
         u(3) = u(3)/r2
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mkside1(xs,ys,zs,iside,ie)
      include 'basics.inc'
C
C     Find side midpoint
C
      ic1=iside
      ic2=iside+1
      if(iside.eq.4)ic2=1
c     This stuff only relevant for 3d
      ic3=ic1+4
      ic4=ic2+4
      if (iside.eq.5) then
         ic1=1
         ic2=2
         ic3=3
         ic4=4
      elseif (iside.eq.6) then
         ic1=1+4
         ic2=2+4
         ic3=3+4
         ic4=4+4
      endif
      if (if3d) then
         xs =( x(ie,ic1)+x(ie,ic2)
     $       +  x(ie,ic3)+x(ie,ic4) )/4.
         ys =( y(ie,ic1)+y(ie,ic2)
     $       +  y(ie,ic3)+y(ie,ic4) )/4.
         zs =( z(ie,ic1)+z(ie,ic2)
     $       +  z(ie,ic3)+z(ie,ic4) )/4.
      else
         xs =( x(ie,ic1)+x(ie,ic2) )/2.
         ys =( y(ie,ic1)+y(ie,ic2) )/2.
         zs = 0.0
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine set_element_cluster_param(par_cluster,nbface)
c
      include 'basics.inc'
c
      real par_cluster       (9,nelm)
c
c     Discrimination Parameters:
c
c     Normalized Position:  
c
c     1. xp := (x-x0)/DX
c     2. yp := (y-y0)/DY
c     3. zp := (z-z0)/DZ
c
c     Unit normal of face:
c
c     4. xn
c     5. yn
c     6. zn
c
c     Face number: 
c
c     7. fn  (1--6)
c
c     Extra parameters:  
c
c     8.  element number
c     9.  cluster id   (1--ncluster)
c
c
c     For every element that is not an "E-E" or "J-SP" bc, do:
c
      nfaces = 2*ndim
c
      ifld=2
      if (ifflow) ifld=1
c
      ntot = nx*ny*nz*nel
      xmin = glmin(x,ntot)
      ymin = glmin(y,ntot)
      zmin = glmin(z,ntot)
      xmax = glmax(x,ntot)
      ymax = glmax(y,ntot)
      zmax = glmax(z,ntot)
c
      dxi = 1./(xmax-xmin)
      dyi = 1./(ymax-ymin)
      dzi = 1.
      if (if3d) dzi = 1./(zmax-zmin)
c
      
c
      nbface=0
      do je=1,nel
      do jf=1,nfaces
         if (cbc(jf,je,ifld).ne.'E  '  .and.
     $       cbc(jf,je,ifld).ne.'J  '  .and.
     $       cbc(jf,je,ifld).ne.'SP ') then
            nbface=nbface+1
            call mkside1(xs,ys,zs,jf,je)
            par_cluster(1,nbface) = xs
            par_cluster(2,nbface) = ys
            par_cluster(3,nbface) = zs
            call getnormals(xn,yn,zn,je,jf,x,y,z,nelm,ndim)
            par_cluster(4,nbface) = xn
            par_cluster(5,nbface) = yn
            par_cluster(6,nbface) = zn
            par_cluster(7,nbface) = jf
            par_cluster(8,nbface) = je
            par_cluster(9,nbface) = 0.
         endif
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine init_cluster_means
     $     (cls_means,npar,ncluster,par_cluster,ldp,nbface)
c
      real cls_means         (npar,ncluster)
      real par_cluster       (ldp ,nbface)
c
c
      do k=1,ncluster
         cls_means(k,k) = 1.e20*((-1)**k)
      enddo
c
      do i=1,nbface
      do k=1,ncluster
         j=k
         if (mod(k,2).eq.1) then  ! MAX
            if (par_cluster(j,i).gt.cls_means(j,k)) then
               call copy(cls_means(1,k),par_cluster(1,i),npar)
            endif
         else                     ! MIN
            if (par_cluster(j,i).lt.cls_means(j,k)) then
               call copy(cls_means(1,k),par_cluster(1,i),npar)
            endif
         endif
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      function ran1(idum)
c
      integer idum,ia,im,iq,ir,ntab,ndiv
      real    ran1,am,eps,rnmx
c
      parameter (ia=16807,im=2147483647,am=1./im,iq=127773,ir=2836)
      parameter (ntab=32,ndiv=1+(im-1)/ntab,eps=1.2e-7,rnmx=1.-eps)
c
c     Numerical Rec. in Fortran, 2nd eD.  P. 271
c
      integer j,k
      integer iv(ntab),iy
      save    iv,iy
      data    iv,iy /ntab*0,0/
c
      if (idum.le.0.or.iy.eq.0) then
         idum=max(-idum,1)
         do j=ntab+8,1,-1
            k    = idum/iq
            idum = ia*(idum-k*iq)-ir*k
            if(idum.lt.0) idum = idum+im
            if (j.le.ntab) iv(j) = idum
         enddo
         iy = iv(1)
      endif
      k    = idum/iq
      idum = ia*(idum-k*iq)-ir*k
      if(idum.lt.0) idum = idum+im
      j     = 1+iy/ndiv
      iy    = iv(j)
      iv(j) = idum
      ran1  = min(am*iy,rnmx)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine cluster_bc
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      parameter(max_clusters=10)
      real wgt_cluster       (7,max_clusters)
      real cls_means         (7,max_clusters)
      real cls_means_new     (7,max_clusters)
      real par_cluster       (9,nelm)
      integer n_in_cluster   (max_clusters)
      integer fn
c
      logical ifconverged
c
c     Identify clusters of boundary faces
c
      ncluster = 4
c
c     Discrimination Parameters:
c
c     Normalized Position:  
c
c     1. xp := (x-x0)/DX
c     2. yp := (y-y0)/DY
c     3. zp := (z-z0)/DZ
c
c     Unit normal of face:
c
c     4. xn
c     5. yn
c     6. zn
c
c     Face number: 
c
c     7. fn  (1--6)
c
c     Extra parameters:  
c
c     8.  element number
c     9.  cluster id   (1--ncluster)
c
      call set_element_cluster_param(par_cluster,nbface)
c
      call rone(wgt_cluster,7*max_clusters)
c     do k=1,4
c     do j=1,3
c        wgt_cluster(j,k) = 0.0    ! Light emphasis on position
c     enddo
c     enddo
c     do j=4,6
c        wgt_cluster(j,4) = 1.0    ! Light emphasis on normal
c     enddo
   
c
      nsides = 2*ndim
      do k=1,4
         wgt_cluster(7,k) = 5.0/(nsides*nsides)
      enddo
c
c
c     Begin K-means iteration
c
c      (First, a bit of diagnostic prep)
      ntot = nx*ny*nz*nel
      call copy(work,z,ntot)
      wkmax = glmax(work,ntot)
      wkmin = glmin(work,ntot)
      wkdlt = wkmax-wkmin
c
      ldp =9
      npar=7
c
      call init_cluster_means
     $     (cls_means_new,npar,ncluster,par_cluster,ldp,nbface)
c
      idum=0.
      do iter=1,1000
         ifconverged=.true.
         call copy (cls_means,cls_means_new,npar*ncluster)
         call rzero(cls_means_new,npar*ncluster)
         call izero(n_in_cluster ,     ncluster)
c
         do i=1,nbface
            dmin = 1.e20
            kmin = 1
            do k=1,ncluster
               d2=0.
               do j=1,npar
                  d2 = d2 + wgt_cluster(j,k)
     $               * (cls_means(j,k)-par_cluster(j,i))**2
               enddo
               dot = 0.
               do j=4,6
                  dot=dot+cls_means(j,k)*par_cluster(j,i)
               enddo
               d2 = d2+(1.-dot)
               if (d2.lt.dmin) then
                  dmin=d2
                  kmin=k
               endif
            enddo
            kold = par_cluster(9,i)
            if (kmin.ne.kold) then
               ifconverged=.false.
               par_cluster(9,i)=kmin
               write(6,*) 'swap',i,kold,kmin,(par_cluster(jj,i),jj=1,9)
            endif
c
c           Recompute cluster means
c
            k=kmin
            do j=1,npar
               cls_means_new(j,k)=cls_means_new(j,k) + par_cluster(j,i)
               n_in_cluster (k)  = n_in_cluster (k)  + 1
            enddo
         enddo
c
         if (ifconverged) goto 1001
         do k=1,ncluster
      write(6,*) 'nclu:',k,n_in_cluster(k),npar,(cls_means(j,k),j=1,7)
            if (n_in_cluster(k).gt.0) then
               do j=1,npar
                  cls_means_new(j,k)=cls_means_new(j,k)/n_in_cluster(k)
               enddo
            else   ! force cluster to not disappear
               if (idum.eq.0) idum=iter
               i = (nbface+1)*ran1(idum)
               i = min(i,nbface)
               i = max(i,1)
               call copy(cls_means_new(1,k),par_cluster(1,i),npar)
            endif
         enddo
c
c        Diagnostics
         do i=1,nbface
            fn  = par_cluster(7,i)
            ie  = par_cluster(8,i)
            cls = par_cluster(9,i)
            val = wkmin + wkdlt*(cls-1.)/(ncluster-1.)
c           write(6,*) 'val:',i,val,cls,ie,fn
            call facev(work,ie,fn,val,nx,ny,nz)
         enddo
         CALL PLOTIT(1)
         call prs('Continue? (y/n)$')
         call res(ans,1)
         if (ans.eq.'n'.or.ans.eq.'N') goto 1001
c
      enddo
 1001 continue
c
      return
      end
c-----------------------------------------------------------------------
      subroutine wave_space_z
c
c     Compute current work array into wave-space (Az,Bz)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /ctmp1/ wwz(nxm*nym*nzm),bfh(nzm*nzm),lam(nzm)

      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.eq.0) then
         icalld = icalld + 1
         call mapz(zp,regdir)
      endif
c
      mz = nz-1
      call semhat(aht,bht,cht,dht,dpht,jpht,zht,mz,wht)
c
      nxyz=nx*ny*nz
      zl2 = (zp(nxyz-1)-zp(1))/2.
      zli = 1./zl2
c
      call cmult(aht,zli,nz*nz)
      call cmult(bht,zl2,nz)
c
      call rzero(bfh,nz*nz)
      l = 1
      do i=1,nz
         bfh(l) = bht(i)
         l = l+nz+1
      enddo
c
      call generalev(aht,bfh,lam,nz,wwz)
c
      call transpose(wwz,nz,aht,nz)
      call copy     (aht,wwz,nz*nz)
c
      call ztransform(work,aht,wwz,nx,ny,nz,nel)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine ztransform(u,st,wk,nx,ny,nz,nel)
      real u(nx*ny*nz,nel)
      real st(nz,nz)
      real wk(nx*ny*nz)
c
      integer e
c
      nxy  = nx*ny
      nxyz = nx*ny*nz
      do e=1,nel
         call mxm(u(1,e),nxy,st,nz,wk,nz)
         call copy(u(1,e),wk,nxyz)
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine read_fld
c
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      call get_fname80_open_old (80,ierr)
c
c
      if (ierr.eq.0) then
         ntot = nx*ny*nz*nel
         do i=1,ntot
            read(80,*) work(i),wkv1(i),wkv2(i),wkv3(i)
         enddo
         close(unit=80)
         return
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine get_fname80_open_old (io,ierr)
c
      character*80 fname
      character*1  fname1(80)
      equivalence (fname,fname1)
c
      call get_fname80 (fname)
c
      ierr = 0
      open(unit=io,file=fname,status='old',err=99)
      return
c
   99 continue
      call prs('Error, could not open file:$')
      length = ltrunc(fname,80) 
      fname1(length+1) = '$'
      call prs(fname)
      ierr = 1
      return
      end
c-----------------------------------------------------------------------
      subroutine get_fname80(fname)
c
      character*80 fname
c
      call prs('Input name of file:$')
      call blank(fname,80)
      call res(fname,80)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine big_image
c     
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
      logical ifdrm,iftmp,iftmh
c
      common /quantc/ quanto
      character*20 quanto
c
      ifauto = .true.
      quanty = quanto
c
      IF (PLFORM.EQ.'X-Y PLOT'.and.XYATTR.EQ.'SET OF PROFILES') 
     $   open (unit=46,file='profile.out')
c
c
      IFTMP  = IFCLRB
      IFCLRB = .FALSE.
      IFTMH  = IFHARD
      write(6,*) 'ifhard:',ifhard
c
      nframe = 1
c
      call prs  ('Trim the ouput frame (1=yes, 0=no, <0=no dump)?$')
      call rei  (itrim)
c
c
      iww = windoww
      iwh = windowh
      write(line,70) iww,iwh
   70 format
     $('Input window scale factor (curent w h:',i5,',',i4,'):$')
      call prs(line)
      call rer(wscale)
      iwws = iww
      iwhs = iwh
      fudge = .9
      iww  = wscale*iww
      iwh  = wscale*iwh*fudge
      call set_ww_wh(iww,iwh)
      call post_reset_window
c
      if (itrim.gt.0) call auto_size_pixel_window
      jstat = 1
      nstat = 1
      call savstate(jstat,nstat,ir,ic,il,ii,'nmsh')
c
      kframe = 0
      do iframe = 1,1,1
         IFHARD = IFTMH
c
         write(6,*) 'ifhard:',ifhard,iframe
         CALL CLEAR
c        CALL DRMESH
         CALL HARD
         CALL HEJECT
         CALL DRMESH
         CALL DRFRAM
         CALL NOHARD
c
         visc = param(2)
         if (visc.le.0) then
            rey = -visc
         else
            rey = 1./visc
         endif
         call blank(line,70)
         write(line,8) rey,nx
    8    format('Reynolds Number =',f7.0,'  N=',i2,'$')
         CALL GSWRIT(.02,.965,1.0,line)
         call blank(line,70)
c
         time_frame = time
         write(line,71) time_frame,kframe
   71    format('Time = ',f12.7,'  Frame =',i5,'$')
c        CALL GSWRIT(.02,.565,1.0,line)
         CALL HARD
c
c        Plot all requested forms...
         write(6,*) 'This is nstate:',nstate
         DO I=1,nstate
            write(6,*) 'This is istate:',i,nstate
c
            CALL SETSTATE(I,ifdrm,ir,ic,il,ii,'mlti')
            IFHARD = IFTMH
c
c           Note, getfld does nothing if "ndumps" is unchanged between
c           successive calls
c
            CALL SETWRK(.false.)
            if (ifdrm) CALL DRMESH
            CALL PLOTIT(0)
         ENDDO
c
         kframe = kframe+1
c        call grab_window('x12_movie\0',kframe)
         if (itrim.ge.0) call grab_window_pix('movie',5,iframe)
         CALL NOHARD
c
      ENDDO
      iww  = iwws
      iwh  = iwhs
      call set_ww_wh(iww,iwh)
      call post_reset_window
      CALL PRS('WAKE UP!!  The movie is over!!$')
      IFCLRB=IFTMP
      PLFORM='MULTIPLOT'
C
C     Pause...  (before putting menu bar back)
C
      IFTMP=IFHARD
      IFHARD=.FALSE.
c     IF(.NOT.IFDEMO) THEN
         CALL PRS('Hit mouse button for menu$')
         CALL MOUSE(XMOUSE,YMOUSE,BUTTON)
         IF(BUTTON.EQ.'RIGHT')  call grab_window_pix('frame',5,-1)
c     ENDIF
c
      IFHARD=IFTMP
      ifauto = .false.
c
      IF (PLFORM.EQ.'X-Y PLOT'.and.XYATTR.EQ.'SET OF PROFILES') 
     $   close (unit=46)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine enstrophy_out
c     
c     Dump out enstrophy
c     
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
      logical ifdrm,iftmp,iftmh
c
      call comp_vorticity
c
      n=nx*ny*nz*nel
c
      if (if3d) then
         do i=1,n
            wkv1(i) = wkv1(i)*wkv1(i)
     $              + wkv2(i)*wkv2(i)
     $              + wkv3(i)*wkv3(i)
         enddo
      else
         do i=1,n
            wkv1(i) = work(i)*work(i)
         enddo
      endif
c
      call prs   ('Input nelx,nely,nelz:$')
      call reiii (nelx,nely,nelz)
c
      call sem2lex(work,wkv1,nelx,nely,nelz)
c
      open(unit=72,file='enstrophy',form='unformatted')
      write(72) (work(k),k=1,n)
      close(72)
c
      do i=1,n
         wkv1(i) = xp(i)*xp(i)+yp(i)*yp(i)
         wkv1(i) = sqrt(wkv1(i))
         wkv2(i) = atan2(yp(i),xp(i))
      enddo
c
      call sem2lex(work,wkv1,nelx,nely,nelz)
      call outlex (work,nelx,nely,nelz,1,'rdata')
c
      call sem2lex(work,wkv2,nelx,nely,nelz)
      call outlex (work,nelx,nely,nelz,2,'tdata')
c
      call sem2lex(work,zp,nelx,nely,nelz)
      call outlex (work,nelx,nely,nelz,3,'zdata')
c
      return
      end
c-----------------------------------------------------------------------
      subroutine sem2lex(ul,us,nelx,nely,nelz)
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
c
      real ul(nx,nelx,ny,nely,nz,nelz)
      real us(nx,ny,nz,nel)
c     
      le = 0
      do ke=1,nelz
      do je=1,nely
      do ie=1,nelx
         le = le+1
         do iz=1,nz
         do iy=1,ny
         do ix=1,nx
            ul(ix,ie,iy,je,iz,ke) = us(ix,iy,iz,le)
         enddo
         enddo
         enddo
      enddo
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine outlex (ul,nelx,nely,nelz,idim,name5)
c
      include 'basics.inc'
      include 'basicsp.inc'
      include 'state.inc'
c
      real ul(nx,nelx,ny,nely,nz,nelz)
      character*5 name5
c     
      open(unit=72,file=name5)
c
      if (idim.eq.1) then
         do ie=1,nelx
         do ix=1,nx
            write(72,1) ul(ix,ie,1,1,1,1)
         enddo
         enddo
      elseif (idim.eq.2) then
         do ie=1,nely
         do ix=1,ny
            write(72,1) ul(1,1,ix,ie,1,1)
         enddo
         enddo
      else
         do ie=1,nelz
         do ix=1,nz
            write(72,1) ul(1,1,1,1,ix,ie)
         enddo
         enddo
      endif
    1 format(1pe15.7)
c
      close(72)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine const_per_elem
c
c     read a constant for each element from a file and put into work
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      nxyz = nx*ny*nz
      open (unit=73,file='p.dat',err=999)
      do i=1,nel
         read(73,*,end=99,err=99) const
         k = nxyz*(i-1) + 1
         call cfill(work(k),const,nxyz)
         call cfill(u(k)   ,const,nxyz)
      enddo
   99 return
      close (73)
  999 call prs('Could not open "t.dat" - return.$')
      return
      end
c-----------------------------------------------------------------------
      subroutine get_map(pmap,nel,depth,cell,nv,nrnk,npts,noutflow,io)
      integer pmap(nel),depth,cell(nv,nel)
      integer d2,e,cmin,cmax

      integer h2s(8) ! hypercube to strange ordering
      save    h2s
      data    h2s / 1,2,4,3,5,6,8,7 /

      read(io,*,end=99,err=99) nel,nactive,depth,d2,npts,nrnk,noutflow

      do e=1,nel
c        read(io,*,end=99,err=99) pmap(e),(cell(h2s(k),e),k=1,nv)
         read(io,*,end=99,err=99) pmap(e),(cell(k,e),k=1,nv)
      enddo

      ntot = nv*nel
      cmin = iglmin(cell,ntot)
      cmax = iglmax(cell,ntot)
      write(6,1) nv,nel,ntot,nrnk,cmin,cmax 
    1 format('NV: ',6i9)

      return

  99  call prsi('Error reading .map file$',e)
      return
      end
c-----------------------------------------------------------------------
      subroutine view_map
c
      include 'basics.inc'
      include 'basicsp.inc'
      character*80 fname
      character*4  c4
c
      integer pmap(nelm),depth,cell(4,nelm),e,pnew
c
      call prs('Input name of .map file: (blah.map)$')
      call blank(fname,80)
      call res(fname,80)
      open (unit=73,file=fname,err=999)
c
      nv = 4 ! 2D

      call get_map(pmap,nelp,depth,cell,nv,nrnk,npts,noutflow,73)
      close (unit=73)
c
      mnp = 2**depth
      do kpass=1,depth
         call clear
         call drmesh
         call drfram
         nnp = 2**kpass

         do e=1,nelp
            xav = 0
            yav = 0
            zav = 0
            kk  = 0
            do j=0,1
            do i=0,1
               ii = 1 + i*(nx-1) + j*nx*(ny-1) + (e-1)*nx*ny
               kk = kk+1
               write(c4,1) cell(kk,e)
    1          format(i3,'$')
               call color(0)
               call g3writ(xp(ii),yp(ii),zp(ii),2.0,c4)
               xav = xav+xp(ii)*.25
               yav = yav+yp(ii)*.25
            enddo
            enddo

            pnew = (pmap(e)*nnp)/mnp
            write(c4,1) pnew
            call color(15)
            call g3writ(xav,yav,zav,2.0,c4)

         enddo

         call prs('continue?$')
         call res(ans,1)
      enddo

      return
  999 return
      end
c-----------------------------------------------------------------------
      subroutine xy_average
c
c     Compute current work array into wave-space (Az,Bz)
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /ctmp1/ uavg(nxm*nelm),vavg(nxm*nelm),wavg(nxm*nelm)
     $             , zavg(nxm*nelm)

      integer kelz
      save    kelz
      data    kelz /0/


      if (kelz.eq.0) then
         call prs('Input number of elements in z:$')
         call rei(kelz)
         nelxy = nel/kelz
      endif

      call xy_avg1(zavg,zp,area,nx,nelxy,kelz)
      call xy_avg1(uavg,u ,area,nx,nelxy,kelz)
      call xy_avg1(vavg,v ,area,nx,nelxy,kelz)
      call xy_avg1(wavg,w ,area,nx,nelxy,kelz)
      call xy_avgo(zavg,uavg,vavg,wavg,nx,kelz)

      return
      end
c-----------------------------------------------------------------------
      subroutine xy_avg1(ua,u,area,nx,nelxy,nelz)
      real u(nx,nx,nx,nelxy,nelz),area(nx,nx,6,nelxy,nelz)
      real ua(nx,nelz)

      do iez=1,nelz
      do iz =1,nx

         ua(iz,iez) = 0.
         wt         = 0.

         do ie =1,nelxy
         do iy =1,nx
         do ix =1,nx
            ua(iz,iez) = ua(iz,iez) 
     $                 + area(ix,iy,5,ie,1)*u(ix,iy,iz,ie,iez)
            wt         = wt + area(ix,iy,5,ie,1)
         enddo
         enddo
         enddo
         ua(iz,iez) = ua(iz,iez) / wt

      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine xy_avgo(za,ua,va,wa,nx,nelz)
      real za(nx,nelz),ua(nx,nelz),va(nx,nelz),wa(nx,nelz)

      write(55,*)
      k=0
      do iez=1,nelz
      do iz =1,nx-1
         k=k+1
         write(55,5) k,za(iz,iez),ua(iz,iez),va(iz,iez),wa(iz,iez)
    5    format(i8,1p4e15.7)
      enddo
      enddo
      k  = k+1
      iz  = nx
      iez = nelz
      write(55,5) k,za(iz,iez),ua(iz,iez),va(iz,iez),wa(iz,iez)

      return
      end
c-----------------------------------------------------------------------
      subroutine xz_average
c
      include 'basics.inc'
      include 'basicsp.inc'
c
      common /ctmp1/ uavg(nxm*nelm),vavg(nxm*nelm),wavg(nxm*nelm)
     $             , yavg(nxm*nelm)

      integer kely
      save    kely
      data    kely /0/


      if (kely.eq.0) then
         call prs('Input number of elements in x,y,z:$')
         call reiii(kelx,kely,kelz)
      endif

      call xz_avg1(yavg,yp,area,nx,kelx,kely,kelz)
      call xz_avg1(uavg,u ,area,nx,kelx,kely,kelz)
      call xz_avg1(vavg,v ,area,nx,kelx,kely,kelz)
      call xz_avg1(wavg,w ,area,nx,kelx,kely,kelz)
      call xy_avgo(yavg,uavg,vavg,wavg,nx,kely)

      return
      end
c-----------------------------------------------------------------------
      subroutine xz_avg1(ua,u,area,nx,nelx,nely,nelz)
      real u(nx,nx,nx,nelx,nely,nelz),area(nx,nx,6,nelx,nely,nelz)
      real ua(nx,nely)

      do iey=1,nely
      do iy =1,nx

         ua(iy,iey) = 0.
         wt         = 0.

         do iez=1,nelz
         do iex=1,nelx
         do iz =1,nx
         do ix =1,nx
            ua(iy,iey) = ua(iy,iey) 
     $                 + area(ix,iz,3,iex,1,iez)*u(ix,iy,iz,iex,iey,iez)
            wt         = wt + area(ix,iz,3,iex,1,iez)
         enddo
         enddo
         enddo
         enddo
         ua(iy,iey) = ua(iy,iey) / wt

      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine xz6_average
c
c     6-way average of hex-fuel pin flow variables
c
      include 'basics.inc'
      include 'basicsp.inc'

      integer kely
      save    kely
      data    kely /0/


      kelx = 54
      kely = 54
      kelz = 120

      if (kely.eq.0) then
         call prs('Input number of elements in x,y,z (4,54,120):$')
         call reiii(kelx,kely,kelz)
      endif

      call xz_avg6(u,nx,kelx,kely,kelz)
      call xz_avg6(v,nx,kelx,kely,kelz)
      call xz_avg6(w,nx,kelx,kely,kelz)

      return
      end
c-----------------------------------------------------------------------
      subroutine xz_avg6(u,nx,nelx,nely,nelz)
c
c     6-way average of hex-fuel pin flow variables
c
      real u(nx*nx*nx,nelx,nely,nelz)
      integer ex,ey,ez,eys,ezs,ei


      n3  = nx**3
      ny6 = nely/6
      nz6 = nelz/6

      do ei=1,5
         do ez=1,nz6
         do ey=1,nely
            eys = ey + ei*nelx*ny6
            eys = mod1(eys,nely)
            ezs = ez + ei*nelx*nely*nz6
            do ex=1,nelx
            do i=1,n3
               u(i,ex,ey,ez) = u(i,ex,ey,ez)+u(i,ex,eys,ezs)
            enddo
            enddo
         enddo
         enddo
      enddo

      s6 = 1.
      s6 = s6/6.

      do ez=1,nz6
      do ey=1,nely
      do ex=1,nelx
      do i=1,n3
         u(i,ex,ey,ez) = u(i,ex,ey,ez)*s6
      enddo
      enddo
      enddo
      enddo

      do ei=1,5
         do ez=1,nz6
         do ey=1,nely
            eys = ey + ei*nelx*ny6
            eys = mod1(eys,nely)
            ezs = ez + ei*nelx*nely*nz6
            do ex=1,nelx
            do i=1,n3
               u(i,ex,eys,ezs) = u(i,ex,ey,ez)
            enddo
            enddo
         enddo
         enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine mapleg1(uh,u,nx,ndim,w)
c
c     Convert u to Legendre space --- USING ORTHONORMAL polynomials
c
      real uh(nx**ndim),u(nx**ndim),w(nx**ndim,2)
c
      parameter (lx1=50)
      common /errcmn/ Lj(lx1*lx1),Ljt(lx1*lx1),zgm1(lx1)
      real Lj,Ljt
c
      integer icalld
      save    icalld
      data    icalld  /0/
c
      if (icalld.ne.nx) then
         icalld = nx
         call zwgll(zgm1,w,nx)
         call build_legend_transform(Lj,Ljt,zgm1,nx)
      endif


      ! Go to Legendre space
      call tensr3(uh,nx,u,nx,Lj,Ljt,Ljt,w,ndim)

      return
      end
c-----------------------------------------------------------------------
      subroutine build_legend_transform(Lj,Ljt,zpts,nx)
c
      real Lj(nx*nx),Ljt(nx*nx),zpts(nx)
c
      parameter (lm=90)
      integer   indr(lm),indc(lm),ipiv(lm)
c
      if (nx.gt.lm) then
         write(6,*) 'ABORT in build_legend_transform:',nx,lm
         call exitt
      endif
c
      j = 1
      n = nx-1
      do i=1,nx
         z = zpts(i)
         call legendre_poly(Lj(j),z,n)  ! Return Lk(z), k=0,...,n
         j = j+nx
      enddo
      call transpose1(Lj,nx)

      j = 1
      do n=0,nx-1
         s = 2*n+1
         s = 0.5*s
         s = sqrt(s)
         call cmult(Lj(j),s,nx)  ! normalize Lj
         j = j+nx
      enddo

      call gaujordf  (Lj,nx,nx,indr,indc,ipiv,ierr,rmult)
      call transpose (Ljt,nx,Lj,nx)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine tensr3(v,nv,u,nu,A,Bt,Ct,w,ndim)
C
C     -  Tensor product application of v = (C x B x A) u
C        NOTE -- the transpose of B & C must be input, rather than B & C.
C
C     -  scratch arrays: w(nu*nu*nv)
C
C
      real v(nv,nv,nv),u(nu,nu,nu)
      real A(1),B(1),C(1)
      real w(1)
C
C
      if (ndim.eq.3) then
         nuv = nu*nv
         nvv = nv*nv
         call mxm(A,nv,u,nu,v,nu*nu)
         k=1
         l=1
         do iz=1,nu
            call mxm(v(k,1,1),nv,Bt,nu,w(l),nv)
            k=k+nuv
            l=l+nvv
         enddo
         call mxm(w,nvv,Ct,nu,v,nv)
      else
         call mxm(A,nv,u,nu,w,nu)
         call mxm(w,nv,Bt,nu,v,nv)
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine transpose1(a,n)
      real a(n,n)
c
      do j=1,n
      do i=j+1,n
         ta     = a(i,j)
         a(i,j) = a(j,i)
         a(j,i) = ta
      enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine error_dist
c
c     Shell-based error distribution
c
      include 'basics.inc'
      include 'basicsp.inc'
      integer e
      common /ctmp1/ ue(nxm*nym*nzm),ve(nxm*nym*nzm),we(nxm*nym*nzm)
     $             , wk(2*nxm*nym*nzm)


      open(unit=84,file='err.dat')

      nxyz = nx**ndim
      k=1
      do e=1,nel
         call mapleg1(ue,u(k),nx,ndim,wk)
         call mapleg1(ve,v(k),nx,ndim,wk)
         call mapleg1(we,w(k),nx,ndim,wk)
         do i=1,nxyz
            ue(i) = ue(i)**2+ve(i)**2+we(i)**2
         enddo
         call out_dist3(ue,nx,e)
         k=k+nxyz
      enddo
      close(84)

      return
      end
c-----------------------------------------------------------------------
      subroutine out_dist3(u,nx,e)
c
      real u(nx,nx,nx)
      integer e

      real err(100)

      call rzero(err,nx)

      do k=1,nx
      do j=1,nx
      do i=1,nx
         l = max(k,j)
         l = max(l,i)
         err(l) = err(l) + u(i,j,k)
      enddo
      enddo
      enddo

      write(84,*)
      do i=1,nx
         i0 = i-1
         write(84,1) i0,e,err(i)
      enddo
    1 format(i3,i7,1pe12.4)

      return
      end
c-----------------------------------------------------------------------
