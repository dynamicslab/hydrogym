c-----------------------------------------------------------------------
      program nekmerge

c     Merge multiple .rea files together.  Output is in re2 or rea format
c
c     this program will take a nek all-ascii rea file and
c     extract geometry, bcs, and curve data to a new binary re2 file, plus
c     an ascii rea file for just the parameters

c
c     Input:
c
c     a or b, for ascii vs binary output
c
c     filename for output  (e.g., blah, for blah.rea blah.re2)
c
c     filename for input .rea file 1
c     filename for input .rea file 2
c     filename for input .rea file 3
c     etc.
c
c

      include 'SIZE'
      include 'INPUT'

      character*1 ans
      write(6,*) 'ascii or binary output ? (a/b):'
      read (5,*) ans

      call load_rea_files(ans)
      call set_ee_bcs
      call output_mesh(ans)

      call exitt
      end
c-----------------------------------------------------------------------
      subroutine load_rea_files(ans)
      character*1 ans

c     Merge multiple .rea files together.  
c
c     this program will take a nek all-ascii rea file and
c     extract geometry, bcs, and curve data to a new binary re2 file, plus
c     an ascii rea file for just the parameters

      include 'SIZE'
      include 'INPUT'

      integer e,f

      nelv_all = 0
      nelt_all = 0
      ncurve   = 0

      call open_file_out (ifile,ans)

      nfld = 1        ! CHANGE THIS IFHEAT etc.

      call blank(cbc,3*6*lelt*ldimt1)

      do ifile=1,1000

!        .Tacitly assumes that all .rea files have the same ndim
!        .and the same nfld
!        .Also, writes out params for ifile=1

         call open_file_in  (ifile,iend)
         if (iend.eq.1) goto 99


         nelo = nelt_all
         call rw_param(nelt,nelv,ndim,nelt_all,nelv_all,nfld,ifile)
         nel  = nelt
         e    = nelo + 1

         if(nelt_all.gt.lelt) then
           write(6,*) 
     &      'ABORT: Too many elements, increase LELT and recompile'
           call exitt 
         endif


         call rd_xyz (x(1,e),y(1,e),z(1,e),nel,ndim)

         call rd_curve(ncurvn,ccurve(1,e),curve(1,1,e),nel,ndim)
         ncurve = ncurve + ncurvn
c         write(6,*) ncurve,ncurvn,e,' NCURVE'

         call rd_bdry(cbc(1,e,1),bc(1,1,e,1),ibc(1,e,1),string
     $               ,nel,nelo,ndim,nfld,lelt)
c        write(6,66) (cbc(k,1,1),k=1,12)
c 66     format(6(1x,a3))

      enddo
   99 continue

      nelt = nelt_all
      nelv = nelv_all
      nel  = nelt_all

      return
      end
c-----------------------------------------------------------------------
      subroutine open_file_in(ifile,iend)

      character*80 file,fout,fbout,string
      character*1  file1(80),fout1(80),fbout1(80),string1(80)
      equivalence (file1,file)
      equivalence (fout1,fout)
      equivalence (fbout1,fbout)
      equivalence (string,string1)


      write(6,*) 
     &  'Input source .rea file name or press enter to continue:'  
      call blank(file,80)
      read(5,80,err=99,end=99) file
      len = indx1(file,' ',1)-1
      if (len.eq.0) goto 99

      nb = 80-len
      call blank(file1(len+1),nb)

   80 format(a80)
   81 format(80a1)

      call chcopy(file1(len+1),'.rea',4)

      len = ltrunc(file,80)
c      write(6,*) 'Opening input file: ',(file1(k),k=1,len)

      open(unit=10, file=file)
      iend = 0
      return

   99 iend = 1
      return

      end
c-----------------------------------------------------------------------
      subroutine open_file_out(ifile,ans)

      character*1 ans

      character*80 file,fout,fbout,string
      character*1  file1(80),fout1(80),fbout1(80),string1(80)
      equivalence (file1,file)
      equivalence (fout1,fout)
      equivalence (fbout1,fbout)
      equivalence (string,string1)


      call blank(fout,80)

      write(6,*) 'Input new (output) file name:'      
      read(5,80) fout
   80 format(a80)

      lou = indx1(fout,' ',1)-1
      nb = 80-lou
      call blank(fout1(lou+1),nb)
      fbout = fout

      call chcopy(fout1(lou+1),'.rea',4)

      len = ltrunc(fout,80)
c      write(6,*) 'Opening output file: ',(fout1(k),k=1,len)

      open(unit=11, file=fout)

      if (ans.eq.'b') then ! binary output
         call chcopy(fbout1(lou+1),'.re2\0',5)

         len = ltrunc(fbout,80)
c         write(6,*) 'Opening binary file: ',(fout1(k),k=1,len)

         call byte_open(fbout)

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine set_ee_bcs ! connectivity 

      include 'SIZE'
      include 'INPUT'

      parameter (lf = 6*lelt)
      common /cwork/ dx(4*lf),face(lf),ind(lf),ninseg(lf),ifseg(lf)
      real    dx
      integer face,ind,ninseg
      logical ifseg

      call get_side    (dx,x,y,z,nel,ndim)
      call global_fnbr (face,dx,ndim,nel,ind,ninseg,ifseg)
      call f_pairing(cbc,face,ndim,nel,nfld,lelt,ind)

      return
      end
c-----------------------------------------------------------------------
      subroutine global_fnbr(face,dx,ndim,nel,ind,ninseg,ifseg)
c
c     Establish global numbering for faces, based on dx
c
      integer face(1),ind(1),ninseg(1)
      logical ifseg(1)
      real dx(0:ndim,1)
      integer e
      real dxt(4),t1(4),t2(4)

      nface = 2*ndim
      n     = nface*nel

      do i=1,n
         face  (i) = i
         ifseg (i) = .false.
      enddo

c     Sort by directions

      lda         = 1+ndim
      nseg        = 1
      ifseg(1)    = .true.
      ninseg(1)   = n

      do ipass=1,ndim   ! Multiple passes eliminates false positives
      do j=1,ndim       ! Sort within each segment
         write(6,*) 'locglob:',ipass,j,nseg,n
         i =1
         j1=j+1
         do iseg=1,nseg
            call tuple_sort(dx(0,i),lda,ninseg(iseg),j1,1,ind,dxt) ! key=j1
            call iswap_ip  (face(i),ind,ninseg(iseg)) ! Swap position 
            i  =   i + ninseg(iseg)
         enddo
 
         q=0.2
         q=0.0010   ! Smaller is better
         do i=2,n
           if ((dx(j,i)-dx(j,i-1))**2.gt.q*min(dx(0,i),dx(0,i-1)))
     $        ifseg(i)=.true.
         enddo

         nseg = 0              !  Count up number of different segments
         do i=1,n
            if (ifseg(i)) then
               nseg = nseg+1
               ninseg(nseg) = 1
            else
               ninseg(nseg) = ninseg(nseg) + 1
            endif
         enddo
      enddo
      enddo
c
c     Unshuffle geometry
c
      call tuple_swapt_ip(dx,lda,n,face,t1,t2)
c
c     Assign global node numbers (now sorted lexigraphically!)
c
      ig = 0
      do i=1,n
         if (ifseg(i)) ig=ig+1
         ind(face(i)) = ig
      enddo
      nglb = ig
      call icopy(face,ind,n)
c          do ik=1,n
c             write(15,*) dx(1,ik),dx(2,ik),ik
c          enddo

      write(6,6) nseg,nglb,n
    6 format('done locglob_lexico:',3i12)

      return
      end
c-----------------------------------------------------------------------
      subroutine get_side   (dx,x,y,z,nel,ndim)

      real dx(0:ndim,2*ndim,nel),x(8,nel),y(8,nel),z(8,nel)

      integer ivtx(4,6)
      save    ivtx
      data    ivtx / 1,3,5,7 , 2,4,6,8     ! faces 1 and 2:  r=-1,1
     $             , 1,2,5,6 , 3,4,7,8     ! faces 3 and 4:  s=-1,1
     $             , 1,2,3,4 , 5,6,7,8 /   ! faces 5 and 6:  t=-1,1

      integer ivt2(2,4)
      save    ivt2
      data    ivt2 / 1,3 , 2,4     ! faces 1 and 2:  r=-1,1
     $             , 1,2 , 3,4 /   ! faces 3 and 4:  s=-1,1

      integer vinv(8)              ! vertex map: preproc <-> symmetric
      save    vinv
      data    vinv / 1,2 , 4,3 , 5,6 , 8,7 /


      integer e,f

      nface = 2*ndim
      nvf   = 2**(ndim-1)
      scale = 1./nvf

      if (ndim.eq.2) then
         do f=1,4
         do i=1,2
            ivtx(i,f) = ivt2(i,f)
         enddo
         enddo
      endif


      do e=1,nel

         do f=1,nface                       ! f in symmetric notation
            call rzero(dx(1,f,e),3)
            do i=1,nvf
               iv = ivtx(i,f)
               iv = vinv(iv)   ! convert to preproc notation
               dx(1,f,e) = dx(1,f,e) + scale*x(iv,e)
               dx(2,f,e) = dx(2,f,e) + scale*y(iv,e)
               dx(3,f,e) = dx(3,f,e) + scale*z(iv,e)
            enddo
         enddo

         b = 1.e22               ! assign min face_gap^2 to dx(0,.)
         do j=1,nface
         do i=1,j-1
            s = 0.
            do k=1,ndim
               s = s+(dx(k,i,e)-dx(k,j,e))**2
               b = min(b,s)
            enddo
         enddo
         enddo

         do f=1,nface                       ! f in symmetric notation
            dx(0,f,e) = b
         enddo

      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine f_pairing(cbc,face,ndim,nel,nfld,lelt,work)

      character*3 cbc(6,lelt,ndim)
      integer face(2*ndim,nel),work(1)

      integer eface(6)
      save    eface
      data    eface  / 4,2,1,3,5,6 /

      integer e,f

      nface = 2*ndim
      n     = nface*nel
      call izero(work,n)
      do e=1,nel
      do f=1,nface
         i = face(f,e)
         work(i) = work(i)+1
      enddo
      enddo

      do e=1,nel
      do f=1,nface
         i = face(f,e)

c        write(6,4) i,work(i),f,e,cbc(f,e,1)
c  4     format(4i9,2x,a3)

         if (work(i).eq.2) then    ! we have adjoining faces
            do k=1,nfld
               cbc(eface(f),e,k) = 'E  '  ! This is equivalent to E-E
            enddo
         elseif (work(i).gt.2) then    ! we have adjoining faces
            write(6,*) 'FACE ERROR:',e,f,i,work(i)
            call exitt
         endif
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine exitt
      stop
      return
      end
c-----------------------------------------------------------------------
      subroutine output_mesh(ans)
      character*1 ans

      if (ans.eq.'a'.or.ans.eq.'A') then
         call output_mesh_ascii
      else
         call output_mesh_bin
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine output_mesh_bin

c     output remainder of mesh: .rea/.re2 format

      include 'SIZE'
      include 'INPUT'

      character*80 hdr
      real*4 test,buf(8)
      save   test
      data   test  / 6.54321 /

      integer e,f


      nels = -nelt
      write(11,11) nels, ndim, nelv
      write(6 ,11) nels, ndim, nelv
   11 format(i12,i3,i12,1x,'NEL,NDIM,NELV')

      call blank(hdr,80)
      write(hdr,1) nel,ndim,nelv
    1 format('#v001',i9,i3,i9,' hdr')
      call byte_write(hdr,20)   ! assumes byte_open() already issued
      call byte_write(test,1)   ! write the endian discriminator

c      call strg_write(hdr,20)   ! assumes byte_open() already issued
c      call strg_write(test,1)   ! write the endian discriminator

      do e=1,nel      
         if (mod(e,10000).eq.0) write(6,*) e,nel,' mesh'

         igroup = 0
         call byte_write(igroup, 1)
c         call  int_write(igroup, 1)

         if (ndim.eq.3) then
            call byte_write(x(1,e),8)
            call byte_write(y(1,e),8)
            call byte_write(z(1,e),8)

c            call real_write(x(1,e),8)
c            call real_write(y(1,e),8)
c            call real_write(z(1,e),8)

         else
            call byte_write(x(1,e),4)
            call byte_write(y(1,e),4)

c            call real_write(x(1,e),4)
c            call real_write(y(1,e),4)
         endif

      enddo

      nface = 2*ndim

      call byte_write(ncurve,1)
      if (ncurve.gt.0) then
         do e=1,nel      
         do f=1,nface
            if (ccurve(f,e).ne.' ') then
               call byte_write(e     ,1)
               call byte_write(f     ,1)
               call byte_write(curve(1,f,e),5)
               call byte_write(ccurve(f,e),1) ! writing 4 bytes, but ok
            
c               call  int_write(e     ,1)
c               call  int_write(f     ,1)
c               call real_write(curve(1,f,e),5)
c               call strg_write(ccurve(f,e),1) ! writing 4 bytes, but ok
            endif
         enddo
         enddo
      endif
         

      do ifld = 1,nfld
         nbc = 0
         do e=1,nel
         do f=1,nface
            if (cbc(f,e,ifld).ne.'E  '.and.cbc(f,e,ifld).ne.'   ')
     $         nbc = nbc+1
         enddo
         enddo

         write(6,*) ifld,nbc,' Number of bcs'
         call byte_write(nbc,1)
c         call  int_write(nbc,1)

         do e=1,nel
         do f=1,nface
            if (cbc(f,e,ifld).ne.'E  '.and.cbc(f,e,ifld).ne.'   ') then
               call icopy      (buf(1),e,1)
               call icopy      (buf(2),f,1)
               call copy       (buf(3),bc(1,f,e,ifld),5)
               call blank      (buf(8),4)
               call chcopy     (buf(8),cbc(f,e,ifld),3)
               if(nel.ge.1000000) call icopy(buf(3),ibc(f,e,ifld),1)
               call byte_write (buf,8)
c               call bdry_write (buf,8)
            endif
         enddo
         enddo

      enddo


      call scanout(string,'PRESOLVE',8,10,0) ! clear any remaining BC
      call lineout(11,string,80)
      call scanout(string,'end',3,10,11)

      close (unit=10)
      close (unit=11)
      call byte_close (fbout)

      return
      end
c-----------------------------------------------------------------------
      subroutine output_mesh_ascii

c     output remainder of mesh: ascii format

      include 'SIZE'
      include 'INPUT'

      write(11,11) nelt, ndim, nelv
   11 format(i12,i3,i12,1x,'NEL,NDIM,NELV')

      call out_xyz_ascii
      call out_curve_ascii
      call out_bdry_ascii

      call scanout(string,'PRESOLVE',8,10,0) ! clear any remaining BC
      call lineout(11,string,80)
      call scanout(string,'end',3,10,11)

      close (unit=10)
      close (unit=11)

      return
      end
c-----------------------------------------------------------------------
      subroutine out_xyz_ascii

      include 'SIZE'
      include 'INPUT'

      integer e


      call blank(string,80)
      write(string,1)
c             123456789 123456789 123456789 123456789 123456789 
    1 format('        E:         1 [    1a]    GROUP     0')
      len = ltrunc(string,80)

      do e=1,nelt

         write(string1(15),'(i12)') e
         write(11,81) (string1(k),k=1,len)
   81    format(80a1)

         if (ndim.eq.2) then
            write(11,90)  (x(k,e),k=1,4)
            write(11,90)  (y(k,e),k=1,4)
         else
            write(11,90)  (x(k,e),k=1,4)
            write(11,90)  (y(k,e),k=1,4)
            write(11,90)  (z(k,e),k=1,4)
            write(11,90)  (x(k,e),k=5,8)
            write(11,90)  (y(k,e),k=5,8)
            write(11,90)  (z(k,e),k=5,8)
         endif
      enddo
c  90 format(1p4e14.6)
   90 format(4e14.6)

      return
      end
c-----------------------------------------------------------------------
      subroutine out_curve_ascii

c     .Ouput curve side data in ascii to unit 11

      include 'SIZE'
      include 'INPUT'

      integer e,f


      write(11,11)
      write(11,12) ncurve
      write(6 ,12) ncurve
   11 format(' ***** CURVED SIDE DATA *****')
   12 format(i12
     $ ,' Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE')

      if (ncurve.gt.0) then
         do e=1,nel
         do k=1,12
            if (ccurve(k,e).ne.' ') then
               call cleanr(curve(1,k,e),5)
               if (nelt.lt.1000) then
                  write(11,60) k,e,(curve(j,k,e),j=1,5),ccurve(k,e)
               elseif (nelt.lt.1 000 000) then
                  write(11,61) k,e,(curve(j,k,e),j=1,5),ccurve(k,e)
               else
                  write(11,62) k,e,(curve(j,k,e),j=1,5),ccurve(k,e)
               endif
            endif
         enddo
         enddo
   50    continue
   60    format(i3,i3,1p5g14.6,1x,a1)
   61    format(i2,i6,1p5g14.6,1x,a1)
   62    format(i2,i12,1p5g14.6,1x,a1)
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine out_bdry_ascii

c     .Ouput bdry data

      include 'SIZE'
      include 'INPUT'

      integer e,f,fld

      integer eface(6)
      save    eface
      data    eface  / 4,2,1,3,5,6 /

      real*8 bc8(6)
c
c
      write(11,31) 
   31 format('  ***** BOUNDARY CONDITIONS *****')
c
c
      do fld=1,nfld
         if2 = fld-2
         if (fld.eq.1) write(11,41)
         if (fld.eq.2) write(11,42)
         if (fld.ge.3) write(11,43) if2
   41    format('  ***** FLUID   BOUNDARY CONDITIONS *****')
   42    format('  ***** THERMAL BOUNDARY CONDITIONS *****')
   43    format('  ***** PASSIVE SCALAR',i2
     $                       ,'  BOUNDARY CONDITIONS *****')

         nface = 2*ndim
         do e=1,nel
         do f=1,nface
            if (nel.lt.1000) then
               write(11,20) cbc(f,e,fld),e,f,(bc(j,f,e,fld),j=1,5)
            elseif (nel.lt.100000) then
               write(11,21) cbc(f,e,fld),e,f,(bc(j,f,e,fld),j=1,5)
            elseif (nel.lt.1000000) then
               write(11,22) cbc(f,e,fld),e,(bc(j,f,e,fld),j=1,5)
            else
               bc8(f) = ibc(f,e,fld)
               write(11,23) cbc(f,e,fld),e,bc8(f),(bc(j,f,e,fld),j=2,5)
            endif

   20       format(1x,a3,2i3,5g14.6)
   21       format(1x,a3,i5,i1,5g14.6)
   22       format(1x,a3,i6,5g14.6)
   23       format(1x,a3,i12,5g18.11)
         enddo
         enddo
      enddo
      if(nfld.eq.1) write(11,44)
   44    format('  ***** NO THERMAL BOUNDARY CONDITIONS *****')

      return
      end
c-----------------------------------------------------------------------
