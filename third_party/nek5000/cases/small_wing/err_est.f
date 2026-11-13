!=======================================================================
! Name        : err_est
! Author      : Adam Peplinski
! Version     :
! Copyright   : GPL
! Description : set of subroutines to implement error estimator
!               Subroutines calculating error estimator based on variable spectra
!               are adopted from Catherine Mavriplis code.
!=======================================================================
!=======================================================================
!     Initialise error estimator
      subroutine err_est_init
      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
      include 'ERR_EST'

!     local variables
      integer i
!-----------------------------------------------------------------------
!     initalise coefficient mapping
      call err_est_cff_init

!     set cutoff parameters
!     used for values
      EEST_SMALL = 1.e-14
!     used for ratios
      EEST_SMALLR = 1.e-10
!     used for gradients
      EEST_SMALLG = 1.e-5
!     used for sigma and rtmp in error calculations
      EEST_SMALLS = 0.2

!     refinement and derefinement thresholds
!     This should be taken from .rea file, but for now I just set it here.
      EEST_REFT  = 1.0e-6
      EEST_DREFT = 5.0e-7

!     logical flags for variables that will be taken into account
!     velx - 1
!     vely - 2
!     velz - 3
!     temp - 4
!     ps   - 5...
      do i =1,LDIMT3
        EEST_IFESTV(i) = .FALSE.
      enddo

!     for this setup only temperature is tested
      EEST_IFESTV(4) = .TRUE.

      return
      end
!=======================================================================
!     Initialise spectral coefficient mapping
!     Modified by Prabal for relaxation term implementation.
      subroutine err_est_cff_init
      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
cc MA:      include 'WZ_DEF'
      include 'WZ'
      include 'ERR_EST'

!     local variables
      integer i, j, k, n
!     Legendre polynomial
      real plegx(LX1),plegy(LY1),plegz(LZ1)
      real z, rtmp
!-----------------------------------------------------------------------
!     check polynomial order and numer of point for extrapolation
      if (min(NX1,NY1).le.(EEST_NP+EEST_ELR)) then
        if (NID.eq.0) write(*,*) 'Error: increase L[XYZ]1'
        call exitt
      endif

      if (IF3D.and.(NZ1.le.(EEST_NP+EEST_ELR))) then
        if (NID.eq.0) write(*,*)'Error: increase L[XYZ]1'
        call exitt
      endif

!     initialise arrays
!     X - direction
      n = NX1-1
      do j= 1, NX1
!     Legendre polynomial
        z = ZGM1(j,1)
        call legendre_poly(plegx,z,n)
        do i=1, NX1
            EEST_XMAP(i,j) = plegx(i)*WXM1(j)
        enddo
      enddo
!     Y - direction; transposed
      n = NY1-1
      do j= 1, NY1
!     Legendre polynomial
        z = ZGM1(j,2)
        call legendre_poly(plegy,z,n)
        do i=1, NY1
            EEST_YTMAP(j,i) = plegy(i)*WYM1(j)
        enddo
      enddo
!     Z - direction; transposed
      if(IF3D) then
        n = NZ1-1
        do j= 1, NZ1
!     Legendre polynomial
            z = ZGM1(j,3)
            call legendre_poly(plegz,z,n)
            do i=1, NZ1
                EEST_ZTMAP(j,i) = plegz(i)*WZM1(j)
            enddo
        enddo
      endif

!     multiplicity factor
      rtmp = 1.0/2.0**NDIM
      do k=1,NZ1
        do j=1,NY1
            do i=1,NX1
                EEST_FAC(i,j,k) = (2.0*(i-1)+1.0)*(2.0*(j-1)+1.0)*
     $             (2.0*(k-1)+1.0)*rtmp
            enddo
        enddo
      enddo

      return
      end
!=======================================================================
!     get polynomial coefficients for error estimator in single element
!     finally square
      subroutine err_est_el_lget(coeff,var)
      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
      include 'ERR_EST'

!     input/output
      real coeff(LX1,LY1,LZ1), var(LX1,LY1,LZ1)

!     local variables
      integer nxy, nyz, iz

!     work array
      real xa(LX1,LY1,LZ1), xb(LX1,LY1,LZ1)
      common /CTMP0/ xa, xb
!     test
      integer i,j,k
!-----------------------------------------------------------------------
      nxy = NX1*NY1
      nyz = NY1*NZ1

!     test
!      call copy(coeff,EEST_YTMAP,nxy)
!      return

      if (IF3D) then
         call mxm(EEST_XMAP,NX1,var,NX1,xa,nyz)
         do iz = 1,NZ1
            call mxm(xa(1,1,iz),NX1,EEST_YTMAP,NY1,
     $           xb(1,1,iz),NY1)
         enddo
         call mxm(xb,nxy,EEST_ZTMAP,NZ1,coeff,NZ1)
      else
         call mxm(EEST_XMAP,NX1,var,NX1,xa,nyz)
         call mxm(xa,NX1,EEST_YTMAP,NY1,coeff,NY1)
      endif

!     multiply by factor
      iz = NX1*NY1*NZ1
      call col2(coeff,EEST_FAC,iz)

!     square coefficients
      call vsq(coeff, iz)

      return
      end
!=======================================================================
!     get error estimater and sigma for single variable on a whole mesh for all directions
      subroutine err_est_var(est,sig,var,nell)
      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'
      include 'SIZE'
cc MA:      include 'INPUT_DEF'
      include 'INPUT'
      include 'ERR_EST'

!     input/output
      real est(nell), sig(nell)
      real var(LX1,LY1,LZ1,nell)
      integer nell

!     local variables
      integer i, j, k, l, j_st, j_en
!     initialisation marker
      integer icalled
      save icalled
      data icalled /0/
!     polynomial coefficients
      real coeff(LX1,LY1,LZ1)
!     Legendre coefficients; first value coeff(1,1,1)
      real coef11
!     copy of last EEST_NP collumns of coefficients
      real coefx(EEST_NP,LY1,LZ1),coefy(EEST_NP,LX1,LZ1),
     $     coefz(EEST_NP,LX1,LY1)
!     estimated error
      real estx, esty, estz
!     estimated decay rate
      real sigx, sigy, sigz
      real third
      parameter (third = 1.0/3.0)

!     for testing only
!      integer ntt
!      real ncn_test(lx1,ly1,lz1,lelt,10)
!      common /test_ncn/ ncn_test
!-----------------------------------------------------------------------

      if (NID.eq.0) write(*,*) 'Get error estimate.'

!     this is bad way of initialisation, but for now I leave it here
      if(icalled.eq.0) then
        icalled = 1
!     initialise error estimator
        call err_est_init
      endif

!     loop over elements
      do i=1,nell
!     get square of polynomial coefficients for given variable
        call err_est_el_lget(coeff(1,1,1),var(1,1,1,i))

!     lower left corner
        coef11 = coeff(1,1,1)

!     small value; nothing to od
        if (coef11.ge.EEST_SMALL) then

!     extrapolate coefficients
!     X - direction
!     copy last EEST_NP collumns (or less if NX1 is smaller)
!     EEST_ELR allows to exclude last row
            j_st = max(1,NX1-EEST_NP+1-EEST_ELR)
            j_en = max(1,NX1-EEST_ELR)
            do l=1,NZ1
                do k=1,NY1
                    do j = j_st,j_en
                        coefx(j_en-j+1,k,l) = coeff(j,k,l)
                    enddo
                enddo
            enddo

!     get extrapolated values
            call err_est_extrap(estx,sigx,coef11,coefx,
     $           j_st,j_en,NY1,NZ1)


!     Y - direction
!     copy last EEST_NP collumns (or less if NY1 is smaller)
!     EEST_ELR allows to exclude last row
            j_st = max(1,NY1-EEST_NP+1-EEST_ELR)
            j_en = max(1,NY1-EEST_ELR)
            do l=1,NZ1
                do k=j_st,j_en
                    do j =  1,NX1
                        coefy(j_en-k+1,j,l) = coeff(j,k,l)
                    enddo
                enddo
            enddo

!     get extrapolated values
            call err_est_extrap(esty,sigy,coef11,coefy,
     $           j_st,j_en,NX1,NZ1)

            if (IF3D) then
!     Z - direction
!     copy last EEST_NP collumns (or less if NZ1 is smaller)
!     EEST_ELR allows to exclude last row
                j_st = max(1,NZ1-EEST_NP+1-EEST_ELR)
                j_en = max(1,NZ1-EEST_ELR)
                do l=j_st,j_en
                    do k= 1,NY1
                        do j =  1,NX1
                            coefz(j_en-l+1,j,k) = coeff(j,k,l)
                        enddo
                    enddo
                enddo

!     get extrapolated values
                call err_est_extrap(estz,sigz,coef11,coefz,
     $              j_st,j_en,NX1,NY1)

!     average
                est(i) =  sqrt(estx + esty + estz)
                sig(i) =  third*(sigx + sigy + sigz)
            else
                est(i) =  sqrt(estx + esty)
                sig(i) =  0.5*(sigx + sigy)
            endif

        else
!     for testing
            estx = 0.0
            esty = 0.0
            estz = 0.0
            sigx = -1.0
            sigy = -1.0
            sigz = -1.0
!     for testing; end

            est(i) =  0.0
            sig(i) = -1.0
        endif

!     for testing only
!        ntt = lx1*ly1*lz1
!        call copy (ncn_test(1,1,1,i,1) ,coeff(1,1,1),ntt)
!        call cfill(ncn_test(1,1,1,i,2) ,coef11,ntt)
!        call cfill(ncn_test(1,1,1,i,3) ,est(i),ntt)
!        call cfill(ncn_test(1,1,1,i,4) ,sig(i),ntt)
!        call cfill(ncn_test(1,1,1,i,5) ,estx,ntt)
!        call cfill(ncn_test(1,1,1,i,6) ,sigx,ntt)
!        call cfill(ncn_test(1,1,1,i,7) ,esty,ntt)
!        call cfill(ncn_test(1,1,1,i,8) ,sigy,ntt)
!        call cfill(ncn_test(1,1,1,i,9) ,estz,ntt)
!        call cfill(ncn_test(1,1,1,i,10),sigz,ntt)
!     for testing only; end

      enddo

      return
      end
!=======================================================================
!     get extrapolated values of sigma and error estimator
!     We assume coef(n) = c*exp(-sigma*n)
!     Error estimate eest = sqrt(2*(coef(N)**2/(2*N+1+\int_(N+1)^\infty coef(n)**2/(n+1) dn)))
!     We estimate sigma and eest
      subroutine err_est_extrap(estx,sigx,coef11,coef,
     $           ix_st,ix_en,nyl,nzl)
      implicit none

!      include 'NEKP4EST_DEF' ! variable declaration for include files
cc MA:      include 'SIZE_DEF'    
      include 'SIZE'
      include 'ERR_EST'

!     input/output
      integer ix_st,ix_en,nyl,nzl
!     Legendre coefficients; last EEST_NP columns
      real coef(EEST_NP,nyl,nzl)
!     Legendre coefficients; first value coeff(1,1,1)
      real coef11
!     estimated error and decay rate
      real estx, sigx

!     local variables
      integer i, j, k, l
      integer nsigt, pnr, nzlt
      real sigt, smallr, cmin, cmax, cnm, rtmp, rtmp2, rtmp3
      real sumtmp(4), cffl(EEST_NP)
      real stmp, estt, clog, ctmp, cave, erlog
      logical cuse(EEST_NP)
!-----------------------------------------------------------------------
!     initial values
      estx =  0.0
      sigx = -1.0

!     relative cutoff
      smallr = coef11*EEST_SMALLR

!     number of points
      pnr = ix_en - ix_st +1

!     to few points to interpolate
!      if ((ix_en - ix_st).le.1) return

!     for averaging, initial values
      sigt = 0.0
      nsigt = 0

!     loop over all face points
      nzlt = max(1,nzl - EEST_ELR) !  for 2D runs
      do i=1,nzlt
!     weight
        rtmp3 = 1.0/(2.0*(i-1)+1.0)
        do j=1,nyl - EEST_ELR

!     find min and max coef along single row
            cffl(1) = coef(1,j,i)
            cmin = cffl(1)
            cmax = cmin
            do k =2,pnr
                cffl(k) = coef(k,j,i)
                cmin = min(cmin,cffl(k))
                cmax = max(cmax,cffl(k))
            enddo

!     are coefficients sufficiently big
            if((cmin.gt.0.0).and.(cmax.gt.smallr)) then
!     mark array position we use in iterpolation
                do k =1,pnr
                    cuse(k) = .TRUE.
                enddo
!     max n for polynomial order
                cnm = real(ix_en)

!     check if all the points should be taken into account
!     in original code by Catherine Mavriplis this part is written
!     for 4 points, so I place if statement first
                if (pnr.eq.4) then
!     should we neglect last values
                    if ((cffl(1).lt.smallr).and.
     &                  (cffl(2).lt.smallr)) then
                        if (cffl(3).lt.smallr) then
                            cuse(1) = .FALSE.
                            cuse(2) = .FALSE.
                            cnm = real(ix_en-2)
                        else
                            cuse(1) = .FALSE.
                            cnm = real(ix_en-1)
                        endif
                    else
!     should we take stronger gradient
                        if ((cffl(1)/cffl(2).lt.EEST_SMALLG).and.
     $                  (cffl(3)/cffl(4).lt.EEST_SMALLG)) then
                            cuse(1) = .FALSE.
                            cuse(3) = .FALSE.
                            cnm = real(ix_en-1)
                         elseif ((cffl(2)/cffl(1).lt.EEST_SMALLG).and.
     $                  (cffl(4)/cffl(3).lt.EEST_SMALLG)) then
                            cuse(2) = .FALSE.
                            cuse(4) = .FALSE.
                        endif
                    endif
                endif

!     get sigma for given face point
                do k =1,4
                    sumtmp(k) = 0.0
                enddo
!     find new min and count noumbero of points
                cmin = cmax
                cmax = 0.0
                do k =1,pnr
                    if(cuse(k)) then
                        rtmp  = real(ix_en-k+1)
                        rtmp2 = log(cffl(k))
                        sumtmp(1) = sumtmp(1) +rtmp2
                        sumtmp(2) = sumtmp(2) +rtmp
                        sumtmp(3) = sumtmp(3) +rtmp*rtmp
                        sumtmp(4) = sumtmp(4) +rtmp2*rtmp
!     find new min and count used points
                        cmin = min(cffl(k),cmin)
                        cmax = cmax + 1.0
                    endif
                enddo
!     decay rate along single row
                stmp = (sumtmp(1)*sumtmp(2) - sumtmp(4)*cmax)/
     $                 (sumtmp(3)*cmax - sumtmp(2)*sumtmp(2))
!     for averaging
                sigt = sigt + stmp
                nsigt = nsigt + 1

!     get error estimator depending on calculated decay rate
                estt = 0.0
                if (stmp.lt.EEST_SMALLS) then
                    estt = cmin
                else
!     get averaged constant in front of c*exp(-sig*n)
                    clog = (sumtmp(1)+stmp*sumtmp(2))/cmax
                    ctmp = exp(clog)
!     average exponent
                    cave = sumtmp(1)/cmax
!     check quality of approximation comparing is to the constant cave
                    do k =1,2
                        sumtmp(k) = 0.0
                    enddo
                    do k =1,pnr
                        if(cuse(k)) then
                            erlog = clog - stmp*real(ix_en-k+1)
                            sumtmp(1) = sumtmp(1)+
     $                          (erlog-log(cffl(k)))**2
                            sumtmp(2) = sumtmp(2)+
     $                          (erlog-cave)**2
                        endif
                    enddo
                    rtmp = 1.0 - sumtmp(1)/sumtmp(2)
                    if (rtmp.lt.EEST_SMALLS) then
                        estt = cmin
                    else
                        estt = ctmp/stmp*exp(-stmp*(cnm+1.0))
                    endif
                endif
!     add contribution to error estimator; variable weight
                estx = estx + estt/(2.0*(j-1)+1.0)*rtmp3
            endif  ! if((cmin.gt.0.0).and.(cmax.gt.smallr))
        enddo
      enddo
!     constant weight
!     in original code by Catherine Mavriplis this weight is multiplied
!     by 4.0, but this is later divided by 4.0, so I skip it
      estx = estx/(2.0*(ix_en-1)+1.0)

!     final everaging
      if (nsigt.gt.0) then
        sigx = sigt/nsigt
      endif

      return
      end
!=======================================================================
