!=======================================================================
!     Adam Peplinski; 2015.10.25
!     Set of subroutines reated to IO
!     
!=======================================================================
!***********************************************************************
!     get free unit number
      subroutine IO_freeid(iunit, ierr)
      implicit none

!     argument list
      integer iunit
      integer ierr
!     local variables
      logical ifcnnd            ! is unit connected
!-----------------------------------------------------------------------
!     find free unit
      ierr=0
!     to not interact with the whole nek5000 I/O
      iunit = 200
      do
         inquire(unit=iunit,opened=ifcnnd,iostat=ierr)
         if(ifcnnd) then
            iunit = iunit +1
         else
            exit
         endif
      enddo

      return
      end
!***********************************************************************
c     To create file name; based on mfo_open_files from
      subroutine IO_mfo_fname(prefix,fname,bname,k) 
      implicit none

cc MA:       include 'SIZE_DEF'
      include 'SIZE'
cc MA      include 'INPUT_DEF'
      include 'INPUT'
cc MA      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA      include 'RESTART_DEF'
      include 'RESTART'

!     argument list
      character*1 prefix(3)
      character*132  fname, bname
      integer k

!     local variables
      character*3 prefx
      character*1   fnam1(132)

      character*6  six
      save         six
      data         six / "??????" /

      character*1 slash,dot
      save        slash,dot
      data        slash,dot  / '/' , '.' /

      integer len
      real rfileo
!     functions
      integer ltrunc, ndigit
!-----------------------------------------------------------------------
      call blank(fname,132)     !  zero out
      
      call chcopy(prefx,prefix,3)
      

#ifdef MPIIO
      rfileo = 1
#else
      rfileo = nfileo
#endif

      ndigit = log10(rfileo) + 1
     
      k = 1
      if (ifdiro) then          !  Add directory
         call chcopy(fnam1(1),'A',1)
         call chcopy(fnam1(2),six,ndigit)  ! put ???? in string
         k = 2 + ndigit
         call chcopy(fnam1(k),slash,1)
         k = k+1
      endif

      if (prefix(1).ne.' '.and.prefix(2).ne.' '.and. !  Add prefix
     $    prefix(3).ne.' ') then
         call chcopy(fnam1(k),prefix,3)
         k = k+3
      endif
      
      len=ltrunc(bname,132)   !  Add SESSION
      call chcopy(fnam1(k),bname,len)
      k = k+len
     
      if (ifreguo) then
         len=4
         call chcopy(fnam1(k),'_reg',len)
         k = k+len
      endif
      
      call chcopy(fnam1(k),six,ndigit) !  Add file-id holder
      k = k + ndigit
      
      call chcopy(fnam1(k  ),dot,1) !  Add .f appendix
      

      call chcopy(fnam1(k+1),'f',1)
      k = k + 2
      

      call chcopy(fname,fnam1,k)

      return
      end
!***********************************************************************
!     it is a modified version of mbyte_open from ic.f but without 
!     equivalence and MPIIO part; I need for some tools 
!     because processor independent part is saved only by master.
!***********************************************************************
      subroutine IO_mbyte_open_srl(hname,fid,ierr) ! open  blah000.fldnn
      implicit none

cc MA:       include 'SIZE_DEF'
      include 'SIZE'
cc MA      include 'TSTEP_DEF'
      include 'TSTEP'

!     argumnt list
      integer fid, ierr
      character*132 hname

!     local variables
      character (LEN=8) eight,fmt,s8
      save         eight
      data         eight / "????????" /

      character(LEN=132) fname

      integer i1, ipass, k, len
!     functions
      integer ltrunc, indx1
!-----------------------------------------------------------------------
      call blank (fname,132)
      len = ltrunc(hname,132)
      call chcopy (fname,hname,len)

      do ipass=1,2              ! 2nd pass, in case 1 file/directory
         kloop: do k=8,1,-1
         i1 = index(fname,eight(1:k))
         if (i1.ne.0) then      ! found k??? string
            write(fmt,1) k,k
 1          format('(i',i1,'.',i1,')')
            write(s8,fmt) fid
            fname(i1:i1+k-1) = s8(1:k)
            exit kloop
         endif
         enddo kloop
      enddo

!     add ending character	
      len = ltrunc(fname,132)
      write(*,*) 'TEST',len
      fname(len+1:len+1) = CHAR(0)

      call byte_open(fname,ierr)
      write(6,6) nid,istep,trim(fname(1:len))
    6 format(2i8,' OPEN: ',A)

      return
      end
!***********************************************************************
