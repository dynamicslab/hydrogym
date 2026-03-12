      subroutine vprops 
C-----------------------------------------------------------------------
C
C     Set material properties
C
C     Material type: 0 for default  (PARAM and PCOND/PRHOCP)
C                    1 for constant props; 
C                    2 for fortran function;
C
C-----------------------------------------------------------------------
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      LOGICAL  IFKFLD,IFEFLD
C
      if (nio.eq.0.and.loglevel.gt.2)
     $   write(6,*) 'vprops', ifield

      NXYZ1 = lx1*ly1*lz1
      NEL   = NELFLD(IFIELD)
      NTOT1 = NXYZ1*NEL
C
      IF (ISTEP.EQ.0) THEN
C
C        First time around, set defaults
C
         ifvarp(ifield) = .false.
         if (iflomach) ifvarp(ifield) = .true.

         if (.not.ifvarp(ifield)) then ! check all groups
            do iel=1,nel
               igrp  = igroup(iel)
               itype = matype(igrp,ifield)
               if(itype.ne.0) ifvarp(ifield) = .true.
            enddo
         endif

         itest = 0                        ! test against all processors
         if (ifvarp(ifield)) itest = 1
         itest = iglmax(itest,1)
         if (itest.gt.0) ifvarp(ifield) = .true.

      endif         
C
C     Fill up property arrays every time step
      DO 1000 IEL=1,NEL
C
         IGRP=IGROUP(IEL)

         if (ifuservp) then
C
C           User specified fortran function   (pff 2/13/01)
            CALL NEKUVP (IEL)
            DIFMIN = VLMIN(VDIFF(1,1,1,IEL,IFIELD),NXYZ1)
            IF (DIFMIN .LE. 0.0) THEN
               WRITE (6,100) DIFMIN,IFIELD,IGRP
               CALL EXITT
            endif
C
         ELSE IF(MATYPE(IGRP,IFIELD).EQ.1)THEN
C
C           Constant property within groups of elements
C
            CDIFF  = CPGRP(IGRP,IFIELD,1)
            CTRANS = CPGRP(IGRP,IFIELD,2)
            CALL CFILL(VDIFF (1,1,1,IEL,IFIELD),CDIFF,NXYZ1)
            CALL CFILL(VTRANS(1,1,1,IEL,IFIELD),CTRANS,NXYZ1)
            IF (CDIFF.LE.0.0) THEN
               WRITE(6,100) CDIFF,IFIELD,IGRP
  100          FORMAT(2X,'ERROR:  Non-positive diffusivity ('
     $        ,G12.3,') specified for field',I2,', group',I2
     $        ,' element',I4,'.'
     $        ,/,'ABORTING in VPROPS',//)
               CALL EXITT
            endif
C
         ELSE IF(MATYPE(IGRP,IFIELD).EQ.2)THEN
C
C           User specified fortran function
C
            CALL NEKUVP (IEL)
C
            if(optlevel.le.2) then
              DIFMIN = VLMIN(VDIFF(1,1,1,IEL,IFIELD),NXYZ1)
              IF (DIFMIN .LE. 0.0) THEN
                 WRITE (6,100) DIFMIN,IFIELD,IGRP
                 CALL EXITT
              endif
            endif
C
         ELSE IF(MATYPE(IGRP,IFIELD).EQ.0)THEN
C
C           Default constant property
C
            CDIFF  = CPFLD(IFIELD,1)
            CTRANS = CPFLD(IFIELD,2)
c           write(6,*) 'vdiff:',ifield,cdiff,ctrans
            CALL CFILL(VDIFF (1,1,1,IEL,IFIELD),CDIFF,NXYZ1)
            CALL CFILL(VTRANS(1,1,1,IEL,IFIELD),CTRANS,NXYZ1)
            IF (CDIFF.LE.0.0) THEN
               WRITE(6,200) CDIFF,IFIELD
  200          FORMAT(2X,'ERROR:  Non-positive diffusivity ('
     $        ,G12.3,') specified for field',I2,'.',/
     $        ,'ABORTING in VPROPS',//)
               CALL EXITT
            endif
         endif
C
 1000 CONTINUE
C
      return
      end
C
