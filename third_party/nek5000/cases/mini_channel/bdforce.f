ccc Implementation of Body-Force Damping For yielding H12
ccc=======================================================
      subroutine init_bodyforce
!! Initialization of the body-force damping tensor 
C---------------------------------
c  Define variables
C---------------------------------
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      ! include 'INPUT'
      include 'BDFORCE'
      
      real Ret, ypctrl, ynorm,alpha 
      
      ! YW: Fix the param here 

      integer ie,iface,ix,iy,iz ! Iteration
      integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2,NTOT ! Face related
      integer icounter
      real xf,yf,zf

      ypctrl=UPARAM(5) ! Sensing plane location for Body-Force
      alpha=UPARAM(6)  ! Amplitude for Body-Force
      Ret = UPARAM(7) ! This is already determined by the user, but we need to store it here for the calculation

C---------------------------------
c Function
C---------------------------------

      if (NID.eq.0) print *, "[BDFD] INIT START, Y, Alpha,",ypctrl,alpha
      
      NTOT   = LX1*LY1*LZ1*LELT
      NEL    = NELFLD(IFIELD)
      nfaces = 2*NDIM 
      ynorm  = ypctrl / Ret ! Calculate the wall-normal distance 
      if (NID.eq.0) print *, "[BDFD] ReTau=",Ret,", Ynorm=",ynorm
      icounter = 0 
      ! Initialize the mask tensor 
      call rzero(bdf_mask(1,1,1,1),NTOT)

      do ie=1,NEL
      do iface=1,nfaces 
        
        ! Traversing the grid points in an element.
        call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
        do iz=KZ1,KZ2
        do iy=KY1,KY2
        do ix=KX1,KX2
        xf=xm1(ix,iy,iz,ie)
        yf=ym1(ix,iy,iz,ie)
        zf=zm1(ix,iy,iz,ie)
        
        ! Mask the tensor if it is within the volume 
        if (yf .le. ynorm) then 
          bdf_mask(ix,iy,iz,ie) = 1.0 * alpha 
          icounter = icounter + 1 
        endif 

        enddo ! do ix = KX1, KX2 
        enddo ! do iy = KY1, KY2 
        enddo ! do iz = KZ1, KZ2  

      enddo ! do iface=1,nface
      enddo ! do ie=1,NEL 
      
      ! A inspection
      if (icounter.ge.0) then 
        print *, "[BDFD] NID=",NID,",Number of grid=",icounter
      endif 
      
      if (NID.eq.0) print *, "[BDFD] INIT END!"
      
      end subroutine init_bodyforce
ccc=======================================================


ccc=======================================================
      subroutine bodyforce_damping(ix,iy,iz,iel,bdfy)
     
      implicit none 
      include 'SIZE'
      include 'BDFORCE'   !! Body-force damping
      include 'INPUT'  !! cc MA: PARAM(77) BODY FORCE
      include 'SOLN'   !! MA: velocity fields (mask buffer)
      include 'GEOM'   !! MA: grid coordinates
      include 'TSTEP'                   ! ISTEP
C---------------------------------
c  Define variables
C---------------------------------
      !! Iterative variable
      integer ix,iy,iz,iel 
      !! Forcing term 
      real bdfy, vwny, imsk

C---------------------------------
c Function
C---------------------------------
      ! Get Mask
      imsk = bdf_mask(ix,iy,iz,iel)
      
      ! Get wall-normal velocity from solution
      vwny = vy(ix,iy,iz,iel)
      
      ! Generate body foce, if mask=0.0 then force=0.0 accordingly. 
      bdfy = -1.0 * imsk * vwny 

#ifdef YWDEBUG
      test_bdf(ix,iy,iz,iel)=bdfy
#endif 

      end subroutine bodyforce_damping

ccc=======================================================
      subroutine bodyforce_output 
     
      implicit none 
      include 'SIZE'
      include 'BDFORCE'   !! Body-force damping
      include 'INPUT'  !! cc MA: PARAM(77) BODY FORCE
      include 'SOLN'   !! MA: velocity fields (mask buffer)
      include 'GEOM'   !! MA: grid coordinates
      include 'TSTEP'                   ! ISTEP

#ifdef YWDEBUG
      call outpost(test_bdf,vy,vz,pr,t,'bdf')
#endif 

      end subroutine bodyforce_output