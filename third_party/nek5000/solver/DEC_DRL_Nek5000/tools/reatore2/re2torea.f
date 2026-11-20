c-----------------------------------------------------------------------
      program re2torea

c     this program will take a nek all-ascii rea file and
c     extract geometry, bcs, and curve data to a new binary re2 file, plus
c     an ascii rea file for just the parameters

      
      character*80 fin
      character*1  fin1(80)
      equivalence (fin1,fin)

      character*80 fout
      character*1  fout1(80)
      equivalence (fout1,fout)

      character*80 fbin
      character*1  fbin1(80)
      equivalence (fbin1,fbin)

      character*80 string
      character*1  string1(80)
      equivalence (string,string1)
c
      character*80 sstring

      parameter(nelm=1000000)
      common /array/ x(8),y(8),z(8),bc(5),curve(6,8)
      common /arrai/ nel,ncurve
      common /arrac/ cbc,ccurve,ca
      character*3 cbc
      integer id, jd
      character*1 ccurve(8,nelm)
      character*1 ca(nelm)
      logical ifflow,ifheat

      character*80 hdr
      real test

      integer buf(30)

      real*8 bc8(5)
      integer ibc
c     
c     for workstation:
      in = 5
c
c     Get file name
      write(6,*) 'Input old (source) file name:'      
      call blank(fin,80)
      read(in,80) fin
      len = ltrunc(fin,80)
   80 format(a80)
c
c     Get file name
      write(6,*) 'Input new (output) file name:'      
      call blank(fout,80)
      read(in,80) fout
      fbin = fin
      lou = ltrunc(fout,80)

      call chcopy(fin1(len+1),'.rea',4)
      call chcopy(fout1(lou+1),'.rea',4)
      call chcopy(fbin1(len+1),'.re2\0',5)
c
      open(unit=10, file=fin)
      open(unit=11, file=fout)
      call byte_open(fbin)

      call scanout(sstring,'IFCHAR',6,10,11)
      write(11,80) sstring      

      call blank(hdr,80)
      call byte_read(hdr,20)
      read (hdr,1) nelv,ndim,nel

      write(11,*) 
     & ' 30.0000  20.0000  -6.00000  -10.0000 XFAC,YFAC,XZERO,YZERO'
      write(11,*) 
     & '**MESH DATA** 6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8'

      write(11,*) abs(nelv),ndim,nel


 1    format('#v001',i9,i3,i9,' this is the hdr')

      call byte_read(test,1)
      write(*,*) 'read: endian test flag ', test

      print *, nel, ndim, nelv


c mesh

      do ie=1,nel      

         call byte_read(igroup, 1)

         if(ndim.eq.3) then 
           call byte_read(x,8)
           call byte_read(y,8)
           call byte_read(z,8)
         else
           call byte_read(x,4)
           call byte_read(y,4)
         endif

         if(nel.lt.100000) then
           write (11,12) ie, ie, 'a', igroup
         else
           write (11,'(A,I12,A,I1)') 
     &        '  ELEMENT ', ie, '  GROUP  ', igroup
         endif
         write (11,*)   (x(k),k=1,4)
         write (11,*)   (y(k),k=1,4)
 
         if(ndim.eq.3) then 
           write (11,*)   (z(k),k=1,4)
           write (11,*)   (x(k),k=5,8)
           write (11,*)   (y(k),k=5,8)
           write (11,*)   (z(k),k=5,8)
         endif
      enddo

 12   format(
     $     '            ELEMENT',i12,' [',i5,a1,']    GROUP     ',i1)


c curved sides

      call byte_read(ncurve,1)
      write(6,*) 'number of curved sides', ncurve

      write(11,13)
      write(11,14) ncurve
 13   format(' ***** CURVED SIDE DATA *****')
 14   format(i12
     $     ,' Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE')

      IF (NCURVE.GT.0) THEN
         DO 50 ICURVE=1,NCURVE
            call byte_read(ieg, 1)
            call byte_read(iedg, 1)
            call byte_read(r1, 1)
            call byte_read(r2, 1)
            call byte_read(r3, 1)
            call byte_read(r4, 1)
            call byte_read(r5, 1)
            call byte_read(ans, 1)

            IF (nel.LT.1000) THEN
               write(11,60) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ELSEIF (nel.LT.1 000 000) then
               write(11,61) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ELSE
               write(11,62) IEDG,IEG,R1,R2,R3,R4,R5,ANS
            ENDIF
 50      CONTINUE
 60      FORMAT(I3,I3 ,5G14.6,1X,A1)
 61      FORMAT(I2,I6 ,5G14.6,1X,A1)
 62      format(i2,i12,5g14.6,1x,a1)
      endif


c boundary conditions

      write(11,*) '***** BOUNDARY CONDITIONS *****'

      do ifld=1,1
         if(ifld.eq.1) write(11,*) 
     &    '***** FLUID   BOUNDARY CONDITIONS *****'
         if(ifld.eq.2) write(11,*) 
     &    '***** THERMAL  BOUNDARY CONDITIONS *****'
         if(ifld.gt.2) write(11,*) 
     &    '***** PASSIVE SCALAR  BOUNDARY CONDITIONS *****'
         call byte_read(nbc,1)
         write(6,*) ifld, nbc, 'number of bc'

         do ie = 1,nbc
            call byte_read(id,1) !element
            call byte_read(jd,1)
            call byte_read(ibc,1)
            call byte_read(bc(2),4)
            call byte_read(buf,1)
            call chcopy(cbc,buf,3)

            bc(1)  = ibc
            bc8(1) = ibc
            do ii = 2,5
               bc8(ii) = bc(ii)
            enddo

            if (nel.lt.1 000) then
                  write (11,20) cbc,id,jd,(bc(k),j=1,5)
            elseif(nel.lt.100 000) then
                  write (11,21) cbc,id,jd,(bc(k),j=1,5)
            elseif(nel.lt.1 000 000) then
                  write (11,22) cbc,id,(bc(k),j=1,5)
            else
                  write (11,23) cbc,id,(bc8(k),j=1,5)
            endif
         enddo
      enddo
   20 FORMAT(1x,A3,2I3,5G14.6)
   21 FORMAT(1x,A3,i5,i1,5G14.6)
   22 FORMAT(1x,A3,i6,5G14.7)
   23 FORMAT(1x,A3,i12,5G18.11)

      rewind(10) 
      call scanout(sstring,'PRESOLVE',8,10,99)
      write(11,80) sstring    
      call scanout(string,'endendend',9,10,11)
c
      close (unit=10)
      close (unit=11)
      call byte_close (fbin)

  999 continue
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
