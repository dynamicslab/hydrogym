c-----------------------------------------------------------------------
      subroutine plan4 (igeom)

C     Splitting scheme A.G. Tomboulides et al.
c     Journal of Sci.Comp.,Vol. 12, No. 2, 1998
c
C     NOTE: QTL denotes the so called thermal
c           divergence and has to be provided
c           by userqtl.
c
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'GEOM'
      INCLUDE 'MASS'
      INCLUDE 'SOLN'
      INCLUDE 'MVGEOM'
      INCLUDE 'TSTEP'
      INCLUDE 'ORTHOP'
      INCLUDE 'CTIMER'
C
      COMMON /SCRNS/ RES1  (LX1,LY1,LZ1,LELV)
     $ ,             RES2  (LX1,LY1,LZ1,LELV)
     $ ,             RES3  (LX1,LY1,LZ1,LELV)
     $ ,             DV1   (LX1,LY1,LZ1,LELV)
     $ ,             DV2   (LX1,LY1,LZ1,LELV)
     $ ,             DV3   (LX1,LY1,LZ1,LELV)
     $ ,             RESPR (LX2,LY2,LZ2,LELV)
      common /scrvh/ h1    (lx1,ly1,lz1,lelv)
     $ ,             h2    (lx1,ly1,lz1,lelv)
 
      REAL           DPR   (LX2,LY2,LZ2,LELV)
      EQUIVALENCE   (DPR,DV1)
      LOGICAL        IFSTSP

      REAL DVC (LX1,LY1,LZ1,LELV), DFC(LX1,LY1,LZ1,LELV)
      REAL DIV1, DIV2, DIF1, DIF2, QTL1, QTL2
c
      INTYPE = -1
      NTOT1  = lx1*ly1*lz1*NELV

      if (igeom.eq.1) then

         ! compute explicit contributions bfx,bfy,bfz 
         call makef 

         call sumab(vx_e,vx,vxlag,ntot1,ab,nab)
         call sumab(vy_e,vy,vylag,ntot1,ab,nab)
         if (if3d) call sumab(vz_e,vz,vzlag,ntot1,ab,nab)

      else

         if(iflomach) call opcolv(bfx,bfy,bfz,vtrans)

         ! add user defined divergence to qtl 
         call add2 (qtl,usrdiv,ntot1)

         if (igeom.eq.2) call lagvel

         ! mask Dirichlet boundaries
         call bcdirvc  (vx,vy,vz,v1mask,v2mask,v3mask) 

         ! compute pressure
         call copy(prlag,pr,ntot1)
         if (icalld.eq.0) tpres=0.0
         icalld=icalld+1
         npres=icalld
         etime1=dnekclock()

         call crespsp  (respr)
         call invers2  (h1,vtrans,ntot1)
         call rzero    (h2,ntot1)
         call ctolspl  (tolspl,respr)
         napproxp(1) = laxtp
         call hsolve   ('PRES',dpr,respr,h1,h2 
     $                        ,pmask,vmult
     $                        ,imesh,tolspl,nmxp,1
     $                        ,approxp,napproxp,binvm1)
         call add2    (pr,dpr,ntot1)
         call ortho   (pr)

         tpres=tpres+(dnekclock()-etime1)

         ! compute velocity
         if(ifstrs .and. .not.ifaxis) then
            call bcneutr
            call cresvsp_weak(res1,res2,res3,h1,h2)
         else
            call cresvsp     (res1,res2,res3,h1,h2)
         endif
         call ophinv       (dv1,dv2,dv3,res1,res2,res3,h1,h2,tolhv,nmxv)
         call opadd2       (vx,vy,vz,dv1,dv2,dv3)

      endif

      return
      END

c-----------------------------------------------------------------------
      subroutine crespsp (respr)

C     Compute startresidual/right-hand-side in the pressure

      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      REAL           RESPR (LX1*LY1*LZ1,LELV)
c
      COMMON /SCRNS/ TA1   (LX1*LY1*LZ1,LELV)
     $ ,             TA2   (LX1*LY1*LZ1,LELV)
     $ ,             TA3   (LX1*LY1*LZ1,LELV)
     $ ,             WA1   (LX1*LY1*LZ1*LELV)
     $ ,             WA2   (LX1*LY1*LZ1*LELV)
     $ ,             WA3   (LX1*LY1*LZ1*LELV)
      COMMON /SCRMG/ W1    (LX1*LY1*LZ1,LELV)
     $ ,             W2    (LX1*LY1*LZ1,LELV)
     $ ,             W3    (LX1*LY1*LZ1,LELV)

      common /scruz/         sij (lx1*ly1*lz1,6,lelv)
      parameter (lr=lx1*ly1*lz1)
      common /scrvz/         ur(lr),us(lr),ut(lr)
     $                     , vr(lr),vs(lr),vt(lr)
     $                     , wr(lr),ws(lr),wt(lr)

      CHARACTER CB*3
      
      NXYZ1  = lx1*ly1*lz1
      NTOT1  = NXYZ1*NELV
      NFACES = 2*ldim

c     -mu*curl(curl(v))
      call op_curl (ta1,ta2,ta3,vx_e,vy_e,vz_e,
     &              .true.,w1,w2)
      if(IFAXIS) then  
         CALL COL2 (TA2, OMASK,NTOT1)
         CALL COL2 (TA3, OMASK,NTOT1)
      endif
      call op_curl  (wa1,wa2,wa3,ta1,ta2,ta3,.true.,w1,w2)
      if(IFAXIS) then  
         CALL COL2  (WA2, OMASK,NTOT1)
         CALL COL2  (WA3, OMASK,NTOT1)
      endif
      call opcolv   (wa1,wa2,wa3,bm1)
c
      call opgrad   (ta1,ta2,ta3,QTL)
      if(IFAXIS) then  
         CALL COL2  (ta2, OMASK,ntot1)
         CALL COL2  (ta3, OMASK,ntot1)
      endif
      scale = -4./3. 
      call opadd2cm (wa1,wa2,wa3,ta1,ta2,ta3,scale)

c compute stress tensor for ifstrs formulation - variable viscosity Pn-Pn
      if (ifstrs .and. ifvvisp) then
         call opgrad   (ta1,ta2,ta3,vdiff)
         call invcol2  (ta1,vdiff,ntot1)
         call invcol2  (ta2,vdiff,ntot1)
         call invcol2  (ta3,vdiff,ntot1)

         nij = 3
         if (if3d.or.ifaxis) nij=6

         call comp_sij   (sij,nij,vx_e,vy_e,vz_e,
     &                    ur,us,ut,vr,vs,vt,wr,ws,wt)
         call col_mu_sij (w1,w2,w3,ta1,ta2,ta3,sij,nij)

         call opcolv   (ta1,ta2,ta3,QTL)
         scale2 = -2./3. 
         call opadd2cm (w1,w2,w3,ta1,ta2,ta3,scale2)
         call opsub2   (wa1,wa2,wa3,w1,w2,w3)

      endif

      call invcol3  (w1,vdiff,vtrans,ntot1)
      call opcolv   (wa1,wa2,wa3,w1)

c     add old pressure term because we solve for delta p 
      call invers2 (ta1,vtrans,ntot1)
      call rzero   (ta2,ntot1)

      call bcdirpr (pr)

      call axhelm  (respr,pr,ta1,ta2,imesh,1)
      call chsign  (respr,ntot1)

c     add explicit (NONLINEAR) terms 
      n = lx1*ly1*lz1*nelv
      do i=1,n
         ta1(i,1) = bfx(i,1,1,1)/vtrans(i,1,1,1,1)-wa1(i)
         ta2(i,1) = bfy(i,1,1,1)/vtrans(i,1,1,1,1)-wa2(i)
         ta3(i,1) = bfz(i,1,1,1)/vtrans(i,1,1,1,1)-wa3(i)
      enddo
      call opdssum (ta1,ta2,ta3)
      do i=1,n
         ta1(i,1) = ta1(i,1)*binvm1(i,1,1,1)
         ta2(i,1) = ta2(i,1)*binvm1(i,1,1,1)
         ta3(i,1) = ta3(i,1)*binvm1(i,1,1,1)
      enddo

      if (if3d) then
         call cdtp    (wa1,ta1,rxm1,sxm1,txm1,1)
         call cdtp    (wa2,ta2,rym1,sym1,tym1,1)
         call cdtp    (wa3,ta3,rzm1,szm1,tzm1,1)
         do i=1,n
            respr(i,1) = respr(i,1)+wa1(i)+wa2(i)+wa3(i)
         enddo
      else
         call cdtp    (wa1,ta1,rxm1,sxm1,txm1,1)
         call cdtp    (wa2,ta2,rym1,sym1,tym1,1)

         do i=1,n
            respr(i,1) = respr(i,1)+wa1(i)+wa2(i)
         enddo
      endif

C     add thermal divergence
      dtbd = BD(1)/DT
      call admcol3(respr,QTL,bm1,dtbd,ntot1)
 
C     surface terms
      DO 100 IEL=1,NELV
         DO 300 IFC=1,NFACES
            CALL RZERO  (W1(1,IEL),NXYZ1)
            CALL RZERO  (W2(1,IEL),NXYZ1)
            IF (ldim.EQ.3)
     $      CALL RZERO  (W3(1,IEL),NXYZ1)
            CB = CBC(IFC,IEL,IFIELD)
            IF (CB(1:1).EQ.'V'.OR.CB(1:1).EQ.'v'.or.
     $         cb.eq.'MV '.or.cb.eq.'mv '.or.cb.eq.'shl') then
               CALL FACCL3
     $         (W1(1,IEL),VX(1,1,1,IEL),UNX(1,1,IFC,IEL),IFC)
               CALL FACCL3
     $         (W2(1,IEL),VY(1,1,1,IEL),UNY(1,1,IFC,IEL),IFC)
               IF (ldim.EQ.3)
     $          CALL FACCL3
     $         (W3(1,IEL),VZ(1,1,1,IEL),UNZ(1,1,IFC,IEL),IFC)
            ELSE IF (CB(1:3).EQ.'SYM') THEN
               CALL FACCL3
     $         (W1(1,IEL),TA1(1,IEL),UNX(1,1,IFC,IEL),IFC)
               CALL FACCL3
     $         (W2(1,IEL),TA2(1,IEL),UNY(1,1,IFC,IEL),IFC)
               IF (ldim.EQ.3)
     $          CALL FACCL3
     $         (W3(1,IEL),TA3(1,IEL),UNZ(1,1,IFC,IEL),IFC)
            ENDIF
            CALL ADD2   (W1(1,IEL),W2(1,IEL),NXYZ1)
            IF (ldim.EQ.3)
     $      CALL ADD2   (W1(1,IEL),W3(1,IEL),NXYZ1)
            CALL FACCL2 (W1(1,IEL),AREA(1,1,IFC,IEL),IFC)
            IF (CB(1:1).EQ.'V'.OR.CB(1:1).EQ.'v'.or.
     $         cb.eq.'MV '.or.cb.eq.'mv ') then
              CALL CMULT(W1(1,IEL),dtbd,NXYZ1)
            endif
            CALL SUB2 (RESPR(1,IEL),W1(1,IEL),NXYZ1)
  300    CONTINUE
  100 CONTINUE

C     Assure that the residual is orthogonal to (1,1,...,1)T 
C     (only if all Dirichlet b.c.)
      CALL ORTHO (RESPR)

      return
      END
c----------------------------------------------------------------------
      subroutine cresvsp (resv1,resv2,resv3,h1,h2)

C     Compute the residual for the velocity

      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      real resv1(lx1,ly1,lz1,lelv)
     $   , resv2(lx1,ly1,lz1,lelv)
     $   , resv3(lx1,ly1,lz1,lelv)
     $   , h1   (lx1,ly1,lz1,lelv)
     $   , h2   (lx1,ly1,lz1,lelv)

      COMMON /SCRUZ/ TA1   (LX1,LY1,LZ1,LELV)
     $ ,             TA2   (LX1,LY1,LZ1,LELV)
     $ ,             TA3   (LX1,LY1,LZ1,LELV)
     $ ,             TA4   (LX1,LY1,LZ1,LELV)

      NTOT = lx1*ly1*lz1*NELV
      INTYPE = -1

      CALL SETHLM  (H1,H2,INTYPE)

      CALL OPHX    (RESV1,RESV2,RESV3,VX,VY,VZ,H1,H2)
      CALL OPCHSGN (RESV1,RESV2,RESV3)

      scale = -1./3.
      if (ifstrs) scale =  2./3.

      call col3    (ta4,vdiff,qtl,ntot)
      call add2s1  (ta4,pr,scale,ntot)
      call opgrad  (ta1,ta2,ta3,TA4)
      if(IFAXIS) then
         CALL COL2 (TA2, OMASK,NTOT)
         CALL COL2 (TA3, OMASK,NTOT)
      endif
c
      call opsub2  (resv1,resv2,resv3,ta1,ta2,ta3)
      call opadd2  (resv1,resv2,resv3,bfx,bfy,bfz)

      return
      end

c----------------------------------------------------------------------
      subroutine cresvsp_weak (resv1,resv2,resv3,h1,h2)

C     Compute the residual for the velocity

      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      real resv1(lx1,ly1,lz1,lelv)
     $   , resv2(lx1,ly1,lz1,lelv)
     $   , resv3(lx1,ly1,lz1,lelv)
     $   , h1   (lx1,ly1,lz1,lelv)
     $   , h2   (lx1,ly1,lz1,lelv)

      COMMON /SCRUZ/ TA1   (LX1,LY1,LZ1,LELV)
     $ ,             TA2   (LX1,LY1,LZ1,LELV)
     $ ,             TA3   (LX1,LY1,LZ1,LELV)
     $ ,             TA4   (LX1,LY1,LZ1,LELV)
      COMMON /SCRMG/ wa1   (LX1*LY1*LZ1,LELV)
     $ ,             wa2   (LX1*LY1*LZ1,LELV)
     $ ,             wa3   (LX1*LY1*LZ1,LELV)

      NTOT = lx1*ly1*lz1*NELV
      INTYPE = -1

      CALL OPRZERO (RESV1,RESV2,RESV3)
      CALL OPRZERO (wa1  ,wa2  ,wa3  )
      CALL OPRZERO (ta1  ,ta2  ,ta3  )

      CALL SETHLM  (H1,H2,INTYPE)

      CALL OPHX    (RESV1,RESV2,RESV3,VX,VY,VZ,H1,H2)
      CALL OPCHSGN (RESV1,RESV2,RESV3)

      scale = -1./3.
      if (ifstrs) scale =  2./3.

      call col3    (ta4,vdiff,qtl,ntot)
      call cmult   (ta4,scale,ntot)
      call opgrad  (ta1,ta2,ta3,TA4)

      call cdtp    (wa1,pr ,rxm1,sxm1,txm1,1)
      call cdtp    (wa2,pr ,rym1,sym1,tym1,1)
      if(if3d) call cdtp    (wa3,pr ,rzm1,szm1,tzm1,1)

      call sub2    (ta1,wa1,ntot)
      call sub2    (ta2,wa2,ntot)
      if(if3d) call sub2    (ta3,wa3,ntot)

      if(IFAXIS) then
         CALL COL2 (TA2, OMASK,NTOT)
         CALL COL2 (TA3, OMASK,NTOT)
      endif
c
      call opsub2  (resv1,resv2,resv3,ta1,ta2,ta3)
      call opadd2  (resv1,resv2,resv3,bfx,bfy,bfz)

      return
      end

c-----------------------------------------------------------------------
      subroutine op_curl(w1,w2,w3,u1,u2,u3,ifavg,work1,work2)
c
      include 'SIZE'
      include 'TOTAL'
c
      real duax(lx1), ta(lx1,ly1,lz1,lelv)

      logical ifavg
c
      real w1(1),w2(1),w3(1),work1(1),work2(1),u1(1),u2(1),u3(1)
c
      ntot  = lx1*ly1*lz1*nelv
      nxyz  = lx1*ly1*lz1
c     work1=dw/dy ; work2=dv/dz
        call dudxyz(work1,u3,rym1,sym1,tym1,jacm1,1,2)
        if (if3d) then
           call dudxyz(work2,u2,rzm1,szm1,tzm1,jacm1,1,3)
           call sub3(w1,work1,work2,ntot)
        else
           call copy(w1,work1,ntot)

           if(ifaxis) then
              call copy (ta,u3,ntot)
              do iel = 1,nelv
                if(IFRZER(iel)) then
                  call rzero (ta(1,1,1,iel),lx1)
                  call MXM   (ta(1,1,1,iel),lx1,DATM1,ly1,duax,1)
                  call copy  (ta(1,1,1,iel),duax,lx1)
                endif
                call col2    (ta(1,1,1,iel),yinvm1(1,1,1,iel),nxyz)
              enddo
              call add2      (w1,ta,ntot)
           endif

        endif
c     work1=du/dz ; work2=dw/dx
        if (if3d) then
           call dudxyz(work1,u1,rzm1,szm1,tzm1,jacm1,1,3)
           call dudxyz(work2,u3,rxm1,sxm1,txm1,jacm1,1,1)
           call sub3(w2,work1,work2,ntot)
        else
           call rzero (work1,ntot)
           call dudxyz(work2,u3,rxm1,sxm1,txm1,jacm1,1,1)
           call sub3(w2,work1,work2,ntot)
        endif
c     work1=dv/dx ; work2=du/dy
        call dudxyz(work1,u2,rxm1,sxm1,txm1,jacm1,1,1)
        call dudxyz(work2,u1,rym1,sym1,tym1,jacm1,1,2)
        call sub3(w3,work1,work2,ntot)
c
c    Avg at bndry
c
c     if (ifavg) then
      if (ifavg .and. .not. ifcyclic) then

         ifielt = ifield
         ifield = 1
       
         call opcolv  (w1,w2,w3,bm1)
         call opdssum (w1,w2,w3)
         call opcolv  (w1,w2,w3,binvm1)

         ifield = ifielt

      endif
c
      return
      end

c-----------------------------------------------------------------------
      subroutine opadd2cm (a1,a2,a3,b1,b2,b3,c)
      INCLUDE 'SIZE'
      REAL A1(1),A2(1),A3(1),B1(1),B2(1),B3(1),C
      NTOT1=lx1*ly1*lz1*NELV
      if (ldim.eq.3) then
         do i=1,ntot1
            a1(i) = a1(i) + b1(i)*c
            a2(i) = a2(i) + b2(i)*c
            a3(i) = a3(i) + b3(i)*c
         enddo
      else
         do i=1,ntot1
            a1(i) = a1(i) + b1(i)*c
            a2(i) = a2(i) + b2(i)*c
         enddo
      endif
      return
      END

c-----------------------------------------------------------------------
      subroutine split_vis

c     Split viscosity into a constant implicit (VDIFF) and variable 
c     explicit (VDIFF_E) part.

      include 'SIZE'
      include 'TOTAL'

      n = lx1*ly1*lz1*nelv

      dnu_star = -nu_star
      call cadd2 (vdiff_e,vdiff,dnu_star,n)   ! set explicit part

      call cfill (vdiff,nu_star,n)            ! set implicit part

      return
      end
c-----------------------------------------------------------------------
      subroutine redo_split_vis     !     Redo split viscosity

      include 'SIZE'
      include 'TOTAL'

      n = lx1*ly1*lz1*nelv
      call add2(vdiff,vdiff_e,n) ! sum up explicit and implicit part

      return
      end
c-----------------------------------------------------------------------
      subroutine col_mu_sij(w1,w2,w3,ta1,ta2,ta3,sij,nij)

      include 'SIZE'
      include 'TOTAL'

      parameter (lr=lx1*ly1*lz1)
      real w1 (lr,1),w2 (lr,1),w3 (lr,1)
      real ta1(lr,1),ta2(lr,1),ta3(lr,1)
      real sij(lr,nij,1)
      integer e

      nxyz1 = lx1*ly1*lz1

      if (if3d.or.ifaxis) then
        do e=1,nelv
          call vdot3 (w1(1,e),ta1(1,e),ta2(1,e),ta3(1,e)
     $               ,sij(1,1,e),sij(1,4,e),sij(1,6,e),nxyz1)
          call vdot3 (w2(1,e),ta1(1,e),ta2(1,e),ta3(1,e)
     $                  ,sij(1,4,e),sij(1,2,e),sij(1,5,e),nxyz1)
          call vdot3 (w3(1,e),ta1(1,e),ta2(1,e),ta3(1,e)
     $                  ,sij(1,6,e),sij(1,5,e),sij(1,3,e),nxyz1)
        enddo

      else

        do e=1,nelv
          call vdot2 (w1(1,e),ta1(1,e),ta2(1,e)
     $               ,sij(1,1,e),sij(1,3,e),nxyz1)
          call vdot2 (w2,ta1(1,e),ta2(1,e)
     $               ,sij(1,3,e),sij(1,2,e),nxyz1)
          call rzero (w3(1,e),nxyz1)
        enddo

      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine sumab(v,vv,vvlag,ntot,ab_,nab_)
c
c     sum up AB/BDF contributions 
c
      include 'SIZE'

      real vvlag(lx1*ly1*lz1*lelv,*)
      real ab_(*)

      ab0 = ab_(1)
      ab1 = ab_(2)
      ab2 = ab_(3)

      call add3s2(v,vv,vvlag(1,1),ab0,ab1,ntot)
      if(nab_.eq.3) call add2s2(v,vvlag(1,2),ab2,ntot)

      return
      end
c-----------------------------------------------------------------------
      subroutine userqtl_scig
c
c     Compute the thermal divergence QTL for an ideal single component gas 
c     QTL := div(v) = -1/rho * Drho/Dt
c
      include 'SIZE'
      include 'TOTAL'
      include 'CVODE'

      common /scrns/ w1(lx1,ly1,lz1,lelt)
     $              ,w2(lx1,ly1,lz1,lelt)
     $              ,w3(lx1,ly1,lz1,lelt)
     $              ,tx(lx1,ly1,lz1,lelt)
     $              ,ty(lx1,ly1,lz1,lelt)
     $              ,tz(lx1,ly1,lz1,lelt)

      nxyz = lx1*ly1*lz1
      ntot = nxyz*nelv

      ifld_save = ifield

c - - Assemble RHS of T-eqn
      ifield=2

      call makeuq
      call copy(qtl,bq,ntot)

      ifield=1     !set right gs handle (QTL is only defined on the velocity mesh)
      call opgrad  (tx,ty,tz,t)
      call opdssum (tx,ty,tz)
      call opcolv  (tx,ty,tz,binvm1)
      call opcolv  (tx,ty,tz,vdiff(1,1,1,1,2))
      call opdiv   (w2,tx,ty,tz)

      call add2    (qtl,w2,ntot)
      call dssum   (qtl,lx1,ly1,lz1)
      call col2    (qtl,binvm1,ntot)

      ! QTL = T_RHS/(rho*cp**T)
      call col3    (w2,vtrans(1,1,1,1,2),t,ntot)
      call invcol2 (qtl,w2,ntot)

      dp0thdt = 0.0
      if (ifdp0dt) then
         dd = (1.0 - gamma0)/gamma0
         call rone(w1,ntot)
         call cmult(w1,dd,ntot)

         call invcol3(w2,vtrans(1,1,1,1,2),vtrans,ntot)
         call invcol2(w1,w2,ntot)

         call cadd(w1,1.0,ntot)
         call copy(w2,w1,ntot)
         call col2(w1,bm1,ntot)
  
         p0alph1 = 1. / glsum(w1,ntot)
  
         call copy   (w1,qtl,ntot)
         call col2   (w1,bm1,ntot)

         termQ = glsum(w1,ntot)
         if (ifcvfun) then
            termV = glcflux(vx,vy,vz)
            prhs  = p0alph1*(termQ - termV)
            pcoef =(cv_bd(1) - cv_dtNek*prhs)
            call add3s2(Saqpq,p0thn,p0thlag(1),cv_bd(2),cv_bd(3),1)
            if(nbd.eq.3) call add2s2(Saqpq,p0thlag(2),cv_bd(4),1)
            p0th  = Saqpq / pcoef
         else
            termV = glcflux(vx_e,vy_e,vz_e)
            prhs  = p0alph1*(termQ - termV)
            pcoef =(bd(1) - dt*prhs)
            call add3s2(Saqpq,p0thn,p0thlag(1),bd(2),bd(3),1)
            if(nbd.eq.3) call add2s2(Saqpq,p0thlag(2),bd(4),1)
            p0th  = Saqpq / pcoef
            p0thlag(2) = p0thlag(1)
            p0thlag(1) = p0thn
            p0thn      = p0th
         endif

         dp0thdt= prhs*p0th

         dd =-prhs
         call cmult(w2,dd,ntot)
         call add2 (qtl,w2,ntot)
      endif

      ifield = ifld_save

      return
      end
c-----------------------------------------------------------------------
      subroutine qthermal
c
c     generic qtl wrapper
c
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      ntot = lx1*ly1*lz1*nelv

      call rzero(qtl,ntot)
      call userqtl()

      return
      end
c-----------------------------------------------------------------------
      subroutine printdiverr
c
      INCLUDE 'SIZE'
      INCLUDE 'TOTAL'

      COMMON /SCRNS/ DVC  (LX1,LY1,LZ1,LELV),
     $               DV1  (LX1,LY1,LZ1,LELV),
     $               DV2  (LX1,LY1,LZ1,LELV),
     $               DFC  (LX1,LY1,LZ1,LELV)
 
      ntot1 = lx1*ly1*lz1*nelv

      !Calculate Divergence norms of new VX,VY,VZ
      CALL OPDIV   (DVC,VX,VY,VZ)
      CALL DSSUM   (DVC,lx1,ly1,lz1)
      CALL COL2    (DVC,BINVM1,NTOT1)

      CALL COL3    (DV1,DVC,BM1,NTOT1)
      DIV1 = GLSUM (DV1,NTOT1)/VOLVM1

      CALL COL3    (DV2,DVC,DVC,NTOT1)
      CALL COL2    (DV2,BM1   ,NTOT1)
      DIV2 = GLSUM (DV2,NTOT1)/VOLVM1
      DIV2 = SQRT  (DIV2)

      !Calculate Divergence difference norms
      CALL SUB3    (DFC,DVC,QTL,NTOT1)
      CALL COL3    (DV1,DFC,BM1,NTOT1)
      DIF1 = GLSUM (DV1,NTOT1)/VOLVM1

      CALL COL3    (DV2,DFC,DFC,NTOT1)
      CALL COL2    (DV2,BM1   ,NTOT1)
      DIF2 = GLSUM (DV2,NTOT1)/VOLVM1
      DIF2 = SQRT  (DIF2)

      CALL COL3    (DV1,QTL,BM1,NTOT1)
      QTL1 = GLSUM (DV1,NTOT1)/VOLVM1

      CALL COL3    (DV2,QTL,QTL,NTOT1)
      CALL COL2    (DV2,BM1   ,NTOT1)
      QTL2 = GLSUM (DV2,NTOT1)/VOLVM1
      QTL2 = SQRT  (QTL2)

      IF (NIO.EQ.0) THEN
         WRITE(6,'(13X,A,1p2e13.4)')
     &         'L1/L2 DIV(V)        ',DIV1,DIV2
         WRITE(6,'(13X,A,1p2e13.4)') 
     &         'L1/L2 QTL           ',QTL1,QTL2
         WRITE(6,'(13X,A,1p2e13.4)')
     &         'L1/L2 DIV(V)-QTL    ',DIF1,DIF2
         IF (DIF2.GT.0.1) WRITE(6,'(13X,A)') 
     &         'WARNING: DIV(V)-QTL too large!'
      ENDIF

      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE BCDIRPR(S)
C
C     Apply Dirichlet boundary conditions to surface of Pressure.
C     Use IFIELD=1.
C
      INCLUDE 'SIZE'
      INCLUDE 'TSTEP'
      INCLUDE 'INPUT'
      INCLUDE 'SOLN'
      INCLUDE 'TOPOL'
      INCLUDE 'CTIMER'
C
      DIMENSION S(LX1,LY1,LZ1,LELT)
      COMMON /SCRSF/ TMP(LX1,LY1,LZ1,LELT)
     $             , TMA(LX1,LY1,LZ1,LELT)
     $             , SMU(LX1,LY1,LZ1,LELT)
      common  /nekcb/ cb
      CHARACTER CB*3

      if (icalld.eq.0) then
         tusbc=0.0
         nusbc=0
         icalld=icalld+1
      endif
      nusbc=nusbc+1
      etime1=dnekclock()
C
      IFLD   = 1
      NFACES = 2*ldim
      NXYZ   = lx1*ly1*lz1
      NEL    = NELFLD(IFIELD)
      NTOT   = NXYZ*NEL
      NFLDT  = NFIELD - 1
C
      CALL RZERO(TMP,NTOT)
C
C     pressure boundary condition
C
      DO 2100 ISWEEP=1,2
C
         DO 2010 IE=1,NEL
         DO 2010 IFACE=1,NFACES
            CB=CBC(IFACE,IE,IFIELD)
            BC1=BC(1,IFACE,IE,IFIELD)
            IF (cb.EQ.'O  ' .or. cb.eq.'ON ' .or.
     $          cb.eq.'o  ' .or. cb.eq.'on ') 
     $          CALL FACEIS (CB,TMP(1,1,1,IE),IE,IFACE,lx1,ly1,lz1)
 2010    CONTINUE
C
C        Take care of Neumann-Dirichlet shared edges...
C
         IF (ISWEEP.EQ.1) CALL DSOP(TMP,'MXA',lx1,ly1,lz1)
         IF (ISWEEP.EQ.2) CALL DSOP(TMP,'MNA',lx1,ly1,lz1)
 2100 CONTINUE
C
C     Copy temporary array to temperature array.
C
      CALL COL2(S,PMASK,NTOT)
      CALL ADD2(S,TMP,NTOT)

      tusbc=tusbc+(dnekclock()-etime1)

      RETURN
      END
C
c-----------------------------------------------------------------------
