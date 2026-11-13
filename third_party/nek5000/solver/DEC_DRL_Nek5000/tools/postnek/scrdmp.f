      SUBROUTINE SCRDMP
      INCLUDE 'basics.inc'
C
      character*80 command
      common /iscdmp/ nsdump
      integer icalld
      save    icalld
      data    icalld/0/
      icalld=icalld+1
      nsdump=icalld
C
      CALL BLANK(COMMAND,80)
      IF(IFVMS)THEN
C        VMS
         IERR=0
      ELSE
C        Ultrix
         write(command,10) icalld
   10    format('xwd > t.dmp',i2.2)
         CALL SYSTEM(command)
      ENDIF
      RETURN
      END
      SUBROUTINE SCRuDMP
      INCLUDE 'basics.inc'
C
      character*80 command
      common /iscdmp/ nsdump
C
      IF(IFVMS)THEN
C        VMS
         IERR=0
      ELSE
C        Ultrix
         CALL PRSI(' Input dump number:$',nsdump )
         CALL REI(isdump)
         write(command,10) isdump
   10    format('xwud -in t.dmp',i2.2)
         CALL SYSTEM(command)
      ENDIF
      RETURN
      END
