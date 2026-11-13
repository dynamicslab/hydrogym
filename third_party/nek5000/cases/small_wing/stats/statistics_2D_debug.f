!=======================================================================
! Name        : statistics_2D
! Author      : Prabal Singh Negi, Adam Peplinski
! Version     : last modification 2015.05.22
! Copyright   : GPL
! Description : Some debugging routines for the 2D statistics toolbox
!=======================================================================
c----------------------------------------------------------------------

      subroutine vis_ownership

      implicit none
     
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
cc MA:      include 'SOLN_DEF'
      include 'SOLN'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'

      integer i,j,k,e,el
      integer lt

      real utmp(lx1,ly1,lz1,lelt)
      real vtmp(lx1,ly1,lz1,lelt)
      real wtmp(lx1,ly1,lz1,lelt)
      real tmp_own

      lt = lx1*ly1*lz1*lelt
      call rzero(utmp,lt)

      lt = lx1*ly1*lz1
      do e=1,nelt
           el = stat_lmap(e)
           tmp_own = stat_own(el) + 0. 
!           call ifill(utmp(1,1,1,e),tmp_own,lt)
      do k=1,lz1
      do j=1,ly1
      do i=1,lx1
          utmp(i,j,k,e) = tmp_own
          vtmp(i,j,k,e) = nid
          wtmp(i,j,k,e) = STAT_RECNO+0.
      enddo
      enddo
      enddo
      
      enddo

      call outpost(utmp,vtmp,wtmp,pr,t,'own')

      end subroutine vis_ownership 

c----------------------------------------------------------------------

      subroutine seq_out 

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
      include 'STATS'

      integer ii
      integer p_recid
      integer tmp_rec_len
      integer msg_id
      integer len

!    Temporary declarations
      integer tosend

      p_recid = 0
     
      if  (nid.eq.0) then

      call get_send_no(192,tosend,stat_own,
     $            stat_lnum,nid)
      write(6,*) 'NID - Status for 192:',nid,tosend

      do ii=1,np-1

          msg_id = 0 
          len=1*isize
          call csend(msg_id,1,len,ii,0)        ! send handshake
          call crecv(msg_id,tosend,len)   ! rec data 

          write(6,*) 'NID - done:',ii, tosend 

      enddo

      else

          call get_send_no(192,tosend,stat_own,
     $            stat_lnum,nid)

          msg_id = 0
          len=1*isize
          call crecv(msg_id,tmp_rec_len,len)      ! rec handshake
          call csend(msg_id,tosend,len,p_recid,0)     ! send data

      endif

      end subroutine seq_out
 
c----------------------------------------------------------------------

      subroutine stat_output_mapping

      implicit none

cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'STATS'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
!      include 'mpif.h'

      integer ii,jj
      integer p_recid
      integer tmp_rec_id(lelt)
      integer tmp_rec_pos(lelt)
      integer tmp_rec_len(3)
      integer msg_id
      integer len
      
      character*5 str
      p_recid = 0

      write(str,'(i5.5)') np     
      if  (nid.eq.0) then
      open(unit=1001,file='comm_map_'//str//'.out')
      
      tmp_rec_len=STAT_SND_CNT
      if (tmp_rec_len(1).eq.0) tmp_rec_len(1)=1
      write(1001,*) 'NID/SND_CNT/RECNO/MAXREC:',nid,STAT_SND_CNT,
     $    STAT_RECNO,STAT_MAXREC
      write(1001,*) 'PROCID:',(STAT_PROCID(jj), jj=1,tmp_rec_len(1))
      write(1001,*) 'PROCPOS:',(STAT_PROCPOS(jj), jj=1,tmp_rec_len(1))

      do ii=1,np-1

          msg_id = 0 
          len=1*isize 
          call csend(msg_id,1,len,ii,0)        ! send handshake
          len=3*isize 
          call crecv(msg_id,tmp_rec_len,len)   ! rec data size

          msg_id = 1
          len=lelt*isize
          call crecv(msg_id,tmp_rec_id,len)       ! rec proc_id
     
          msg_id = 2
          call crecv(msg_id,tmp_rec_pos,len)      ! rec proc_pos

          write(1001,*) 'NID/SND_CNT/RECNO/MAXREC:',ii,tmp_rec_len(1),
     $         tmp_rec_len(2), tmp_rec_len(3)
          if (tmp_rec_len(1).eq.0) tmp_rec_len(1)=1
          write(1001,*) 'PROCID:',(tmp_rec_id(jj),jj=1,tmp_rec_len(1))
          write(1001,*) 'PROCPOS:',(tmp_rec_pos(jj),jj=1,tmp_rec_len(1))

      enddo

      close(1001)
      else

          msg_id = 0
          len=1*isize
          call crecv(msg_id,tmp_rec_len,len)      ! rec handshake
          len=3*isize
          tmp_rec_len(1) = STAT_SND_CNT
          tmp_rec_len(2) = STAT_RECNO
          tmp_rec_len(3) = STAT_MAXREC
          call csend(msg_id,tmp_rec_len,len,p_recid,0)     ! send data length

          msg_id  = 1
          len=lelt*isize
          call csend(msg_id,STAT_PROCID,len,p_recid,0)      ! send procid

          msg_id = 2
          call csend(msg_id,STAT_PROCPOS,len,p_recid,0)     ! send procpos    

      endif

      end subroutine stat_output_mapping

c----------------------------------------------------------------------



