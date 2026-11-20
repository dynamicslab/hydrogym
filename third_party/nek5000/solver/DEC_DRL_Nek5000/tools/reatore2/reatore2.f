c-----------------------------------------------------------------------
      program trans

c     this program will take a nek all-ascii rea file and
c     extract geometry, bcs, and curve data to a new binary re2 file, plus
c     an ascii rea file for just the parameters

      
      character*4 ans
      real*4      buf(10)
      real*8      buf2(30)
      real*8      rgroup,rcurve,rbc

      character*80 file,fout,fbout,string
      character*1  file1(80),fout1(80),fbout1(80),string1(80)
      equivalence (file1,file)
      equivalence (fout1,fout)
      equivalence (fbout1,fbout)
      equivalence (string,string1)

      parameter(lelt=1000000)
      common /array/ x(8),y(8),z(8),bc(5,6,lelt),curve(6,8)
      common /arrai/ nel,ncurve
      common /arrac/ cbc
      character*3 cbc(6,lelt)
      integer e,f
      character*1 ccurve(4)
      logical ifflow,ifheat

      character*80 hdr
      real*4 test
      data   test  / 6.54321 /

      integer ibc(6,lelt)
      real*8  bc8(5)

      write(6,*) 'Input old (source) file name:'      
      call blank(file,80)
      read(5,80) file
      len = ltrunc(file,80)
   80 format(a80)
   81 format(80a1)

      write(6,*) 'Input new (output) file name:'      
      call blank(fout,80)
      read(5,80) fout
      fbout = fout
      lou = ltrunc(fout,80)

      write(6,*)
      write(6,'(A)') 'Start converting ...'
      write(6,*)

      call chcopy(file1(len+1),'.rea',4)
      call chcopy(fout1(lou+1),'.rea',4)
      call chcopy(fbout1(lou+1),'.re2' // char(0),5)

      open(unit=10, file=file)
      open(unit=11, file=fout)
      call byte_open(fbout)

      call scanout(string,'MESH DATA',9,10,11)
      call lineout(11,string,80)
      write(6,'(A,A)') 'read: ',string

      read (10,*)   nel, ndim, nelv
      write(11,11) -nel, ndim, nelv
   11 format(i12,2x,i1,2x,i12,5x,'NELT,NDIM,NELV')
   
      if(nel.ge.lelt) then
        write(6,*) 'Abort: number of elements too large'
        write(6,*) 'maximum number of elements ',lelt
        write(6,*) 'change lelt and recompile'
        stop
      endif

      call blank(hdr,80)
      write(hdr,1) nel,ndim,nelv
    1 format('#v002',i9,i3,i9,' this is the hdr')
      call byte_write(hdr,20)   ! assumes byte_open() already issued
      call byte_write(test,1)   ! write the endian discriminator


C MESH
      igroup = 0
      do ie=1,nel      
         read(10,*) ! for now skip igroup (not used at the moment)
         rgroup=igroup
         call byte_write(rgroup, 2)

         if (ndim.eq.3) then
            read (10,*)   (x(k),k=1,4)
            read (10,*)   (y(k),k=1,4)
            read (10,*)   (z(k),k=1,4)
            read (10,*)   (x(k),k=5,8)
            read (10,*)   (y(k),k=5,8)
            read (10,*)   (z(k),k=5,8)

            call copy48(buf2(1) ,x,8)
            call copy48(buf2(9) ,y,8)
            call copy48(buf2(17),z,8)
            call byte_write(buf2(1) ,16)
            call byte_write(buf2(9) ,16)
            call byte_write(buf2(17),16)
         else
            read (10,*)   (x(k),k=1,4)
            read (10,*)   (y(k),k=1,4)
           
            call copy48(buf2(1) ,x,4)
            call copy48(buf2(5) ,y,4)
            call byte_write(buf2(1),8)
            call byte_write(buf2(5),8)
         endif
      enddo
      write(6,'(i12,1x,i12,1x,i1,A)') nel,nelv,ndim, 
     &     ' nelt/nelv/ndim'

c CURVED SIDES
      read (10,80) string !  curved sides
      write(6,'(A,A)') 'read: ',string
      read(10,*) ncurve
      rcurve=ncurve
      call byte_write(rcurve,2)

      call blank(ccurve,4)
      if (ncurve.gt.0) then
         do icurve=1,ncurve
c           if (mod(icurve,10000).eq.0) write(6,*) icurve,' curve'
            if (nel.lt.1000) then
               read(10,60) f,e,(buf(k),k=1,5),ccurve(1)
            elseif (nel.lt.1 000 000) then
               read(10,61) f,e,(buf(k),k=1,5),ccurve(1)
            else
               read(10,62) f,e,(buf(k),k=1,5),ccurve(1)
            endif
            buf2(1)=e
            buf2(2)=f
            call copy48(buf2(3),buf(1),5)
            call blank(buf2(8),8)
            call chcopy(buf2(8),ccurve,1)
            call byte_write(buf2,16)
         enddo
   60    format(i3,i3 ,5g14.6,1x,a1)
   61    format(i2,i6 ,5g14.6,1x,a1)
   62    format(i2,i12,5g14.6,1x,a1)

          write(6,*) ncurve,' Number of curved sides'
      endif

C BOUNDARY CONDITIONS
      read (10,80) string ! ***** BOUNDARY CONDITIONS *****
      write(6,'(A,A)') 'read: ',string

      nface = 2*ndim
      do kpass = 1,50   ! for now, at most 50 fields
         read (10,80) string
         if (indx1(string,'BOUN',4).ne.0) then ! we might have bcs
            if (indx1(string,'NO ',3).eq.0)  then ! we have bcs, read and count
               write(6,'(A,A)') 'read: ',string
               nbc = 0
               nelb = nelv
               if(kpass.eq.2) nelb=nel     ! only ifield2 is a T mesh 
               do e=1,nelb
               do f=1,nface
                  if (nel.lt.1 000 000) then
                     read(10,20) cbc(f,e),(bc(k,f,e),k=1,5)
                  else
                     read(10,21) cbc(f,e),(bc8(k),k=1,5)
                     do ii = 1,5
                        bc(ii,f,e)=bc8(ii)
                     enddo
                     ibc(f,e) = bc8(1)
                  endif
                  if (cbc(f,e).ne.'E  ') nbc = nbc+1
               enddo
               enddo
   20          format(1x,a3,6x,5g14.6)  
   21          format(1x,a3,12x,5g18.11)  

               write(6,*) kpass,nbc,' Number of bcs'
               rbc=nbc
               call byte_write(rbc,2)

               do e=1,nelb
               do f=1,nface
                  if (cbc(f,e).ne.'E  ') then
                     buf2(1)=e
                     buf2(2)=f
                     call copy48(buf2(3),bc(1,f,e),5)
                     call blank      (buf2(8),8)
                     call chcopy     (buf2(8),cbc(f,e),3)
                     if(nelb.ge.1000000) buf2(3)=ibc(f,e)
                     call byte_write (buf2,16)
                  endif
               enddo
               enddo

            endif
         else
            goto 51
         endif

      enddo
   51 continue

      len = ltrunc(string,80)
      write(11,81) (string1(k),k=1,len)
c     write(6 ,81) (string1(k),k=1,len)

      call scanout(string,'end',3,10,11)

      close (unit=10)
      close (unit=11)
      call byte_close (fbout)

  999 continue
      write(6,*) 'done'
      stop

 9999 continue
      write(6,*) 'could not find file "indat". abort.'
      stop
      end
c-----------------------------------------------------------------------
      subroutine exitt

      stop 
 
      return
      end
