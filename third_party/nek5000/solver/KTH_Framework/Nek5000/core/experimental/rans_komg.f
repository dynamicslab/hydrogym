      real function rans_komg_mut(ix,iy,iz,iel)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      save ifldla
      data ifldla /ldimt1/ 

      if(ix*iy*iz*iel.eq.1 .and. ifield.le.ifldla) then
         if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'updating komg_mut'
         if(ifrans_komg_stndrd)      call rans_komg_stndrd_eddy
         if(ifrans_komg_lowRe)       call rans_komg_lowRe_eddy
         if(ifrans_komgSST_stndrd)   call rans_komgSST_stndrd_eddy
         if(ifrans_komgSST_lowRe)    call rans_komgSST_lowRe_eddy
         if(ifrans_komg_stndrd_noreg)call rans_komg_stndrd_eddy_noreg
      endif

      ifldla = ifield
      rans_komg_mut = mut(ix,iy,iz,iel)

      return
      end
c-----------------------------------------------------------------------
      real function rans_komg_mutsk(ix,iy,iz,iel)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      rans_komg_mutsk = mutsk(ix,iy,iz,iel)

      return
      end
c-----------------------------------------------------------------------
      real function rans_komg_mutso(ix,iy,iz,iel)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      rans_komg_mutso = mutso(ix,iy,iz,iel)

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_getCoeffs(coeffs_in)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      real coeffs_in(1)

      do i=1,ncoeffs
         coeffs_in(i) = coeffs(i)
      enddo
      
      return
      end
c-----------------------------------------------------------------------
      real function rans_komg_kSrc(ix,iy,iz,iel)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      logical ifevalsrc
      data ifevalsrc /.true./
      common /komgifsrc/ ifevalsrc

      if(ix*iy*iz*iel.eq.1 .and. ifevalsrc) then
         if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'updating komg_src'
         if(ifrans_komg_stndrd)      call rans_komg_stndrd_compute
         if(ifrans_komg_lowRe)       call rans_komg_lowRe_compute
         if(ifrans_komgSST_stndrd)   call rans_komgSST_stndrd_compute
         if(ifrans_komgSST_lowRe)    call rans_komgSST_lowRe_compute
         if(ifrans_komg_stndrd_noreg)call rans_komg_stndrd_compute_noreg
         ifevalsrc = .false.
      endif

      if(ifld_k.gt.ifld_omega) ifevalsrc = .true.
      rans_komg_kSrc = kSrc(ix,iy,iz,iel)

      return
      end
c-----------------------------------------------------------------------
      real function rans_komg_omgSrc(ix,iy,iz,iel)
      include 'SIZE'
      include 'TSTEP'
      include 'RANS_KOMG'

      logical ifevalsrc
      data ifevalsrc /.true./
      common /komgifsrc/ ifevalsrc

      if(ix*iy*iz*iel.eq.1 .and. ifevalsrc) then
         if(nid.eq.0 .and. loglevel.gt.2) write(6,*) 'updating komg_src'
         if(ifrans_komg_stndrd)      call rans_komg_stndrd_compute
         if(ifrans_komg_lowRe)       call rans_komg_lowRe_compute
         if(ifrans_komgSST_stndrd)   call rans_komgSST_stndrd_compute
         if(ifrans_komgSST_lowRe)    call rans_komgSST_lowRe_compute
         if(ifrans_komg_stndrd_noreg)call rans_komg_stndrd_compute_noreg
         ifevalsrc = .false.
      endif

      if(ifld_omega.gt.ifld_k) ifevalsrc = .true.
      rans_komg_omgSrc = omgSrc(ix,iy,iz,iel)

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_stndrd_compute
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      real           tempv(lxyz), rhoalpk (lxyz)
     $              ,omwom(lxyz), rhoalpfr(lxyz)

      real           term1(lxyz), term2   (lxyz)
     $              ,term3(lxyz), term4   (lxyz)
     $              ,g    (lxyz), div     (lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

      real mu_omeg(lxyz), mu_omegx(lxyz), mu_omegy(lxyz), mu_omegz(lxyz)
      real extra_src_omega(lxyz)

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call copy   (g,   St_mag2(1,e),       lxyz)
        call copy   (div, DivQ   (1,e),       lxyz)

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

c solve for omega_pert

c ---------------------
c        call check_omwall_behavior
c ---------------------
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
c             write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
c              write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0 
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d) 
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          omwom (i)= 1.0/(1.0+t(i,1,1,e,ifld_omega-1)/f_omegb(i,1,1,e))

c See equations from eqns_k_omega1.pdf from Eq. (3) onwards
c Eq.(1) and (2) in eqns_k_omega1.pdf are the governing equations
c no source terms Sk or S_w are added

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))
          sum_xx  =      OiOjSk (i,e)

! calculate del k * del omega / omega

          if(if3d)then
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                         / (omega**3+tiny)
          else
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                         / (omega**3+tiny)
          endif

          alp_str = alpinf_str 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!
          rhoalpk(i)= rho*alp_str*k

          factr = 1.0
          sigma_omega1 = 1.0/sigma_omega
          rhoalpfr(i) = expn * rho * alp_str * factr * omwom(i)
     $                                           * sigma_omega1

c nu_t is kinematic turbulent viscosity
c units of k = m2/s2, units of omega = 1/s, nu units = m2/s
c set limit for nu_t
c           nu_t    = max(tiny,nu_t)
c	    if(nu_t.gt.5000*mu)nu_t = 5000*mu

          betai_str = betainf_str

          if (xk3.le.0)then
            f_beta_str = 1.0
          else
            f_beta_str = (1.0 + 680.0*xk3*xk3)/(1.0 + 400.0*xk3*xk3)
          endif
          Y_k = rho * betai_str * f_beta_str * k * omega

c betai_str = beta_star in 12.5.15 for incompressible flow
 

          extra_prod = 0.
          if(iflomach) extra_prod = twothird*div(i)
          G_k0= mu_t*g(i) - ( rho*k + mu_t*div(i) )*extra_prod
          G_k = min(G_k0, 10.*Y_k)

c g(i) is S**2 in equation sheet

          kSrc  (i,1,1,e) = G_k - Y_k

c Compute production of omega
          alpha = (alp_inf/alp_str)

c          G_w = alpha*alp_str*rho*G_k/mu_t !g(i)
          G_w0 = alpha*alp_str*rho*(g(i)-(omega+div(i))*extra_prod)

c Compute dissipation of omega
          beta = beta_0
c no compressibility correction M < 0.25

          x_w = abs((sum_xx)/(betainf_str*omega + tiny)**3)
          f_b = 1.0
          if(if3d) f_b = (1.0 + 70.0*x_w)/(1.0 + 80.0*x_w)

          Y_w = rho*beta*f_b * omega * omega

          G_w = min(G_w0, 10.0*Y_w)
          omgSrc(i,1,1,e) = G_w - Y_w

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

c solve for omega_pert

c add mut*delsqf
        sigma_omega1 = 1.0/sigma_omega
        call copy   (mu_omeg,rhoalpk,lxyz)
        call cmult  (mu_omeg,sigma_omega1,lxyz)
        call col4   (extra_src_omega,mu_omeg ,omwom
     $                                ,delsqf_omegb(1,1,1,e),lxyz)

c add mu*delsqf
        call copy   (tempv,delsqf_omegb(1,1,1,e),lxyz)
        call cmult  (tempv,mu,lxyz)
        call col2   (tempv,f_omegb(1,1,1,e),lxyz)
        call add2   (extra_src_omega, tempv,lxyz)

c Form (1/sigma_w) del_mut * del_omw
c  form 1: (del_yw/yw) del_k
        call col3   (term1,dfdx_omegb(1,1,1,e),k_x,lxyz)
        call addcol3(term1,dfdy_omegb(1,1,1,e),k_y,lxyz)
        if(if3d)
     $  call addcol3(term1,dfdz_omegb(1,1,1,e),k_z,lxyz)

c  form 2: 2(omw/om) (del_omw/yw)^2
        expm = -expn
        call col3   (term2,omwom,delfsq_omegb(1,1,1,e),   lxyz)
        call col2   (term2           ,t(1,1,1,e,ifld_k-1),lxyz)
        call cmult  (term2,expm,lxyz)
        call add3   (tempv, term1, term2, lxyz)

c  form 3: -(omw/om) k (del_yw/yw) \del_omp/omw
        call col3   (term3     ,omwom,t(1,1,1,e,ifld_k-1),lxyz)
        call invcol2(term3     ,f_omegb(1,1,1,e)         ,lxyz)
        call col3   (term4,  dfdx_omegb(1,1,1,e),omp_x,   lxyz)
        call addcol3(term4,  dfdy_omegb(1,1,1,e),omp_y,   lxyz)
        if(if3d)
     $  call addcol3(term4,  dfdz_omegb(1,1,1,e),omp_z,   lxyz)
        call subcol3(tempv, term3, term4, lxyz)

        call addcol3(extra_src_omega, rhoalpfr, tempv,    lxyz)

c add rho v * del_omw
        call col3   (tempv,dfdx_omegb(1,1,1,e),VX(1,1,1,e),lxyz)
        call addcol3(tempv,dfdy_omegb(1,1,1,e),VY(1,1,1,e),lxyz)
        if(if3d)
     $  call addcol3(tempv,dfdz_omegb(1,1,1,e),VZ(1,1,1,e),lxyz)
        call col2   (tempv,vtrans(1,1,1,e,1),lxyz)
        call admcol3(extra_src_omega,tempv,f_omegb(1,1,1,e),expm,lxyz)

        call add2   (omgSrc(1,1,1,e), extra_src_omega           ,lxyz)

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0) 
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0) 
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_lowRe_compute
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      real           tempv(lxyz), rhoalpk (lxyz)
     $              ,omwom(lxyz), rhoalpfr(lxyz)

      real           term1(lxyz), term2   (lxyz)
     $              ,term3(lxyz), term4   (lxyz)
     $              ,g    (lxyz), div     (lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

      real mu_omeg(lxyz), mu_omegx(lxyz), mu_omegy(lxyz), mu_omegz(lxyz)
      real extra_src_omega(lxyz)

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call copy   (g,   St_mag2(1,e),       lxyz)
        call copy   (div, DivQ   (1,e),       lxyz)

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

c solve for omega_pert

c ---------------------
c        call check_omwall_behavior
c ---------------------
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0 
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d) 
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          omwom (i)= 1.0/(1.0+t(i,1,1,e,ifld_omega-1)/f_omegb(i,1,1,e))

c See equations from eqns_k_omega1.pdf from Eq. (3) onwards
c Eq.(1) and (2) in eqns_k_omega1.pdf are the governing equations
c no source terms Sk or S_w are added

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))
          sum_xx  =      OiOjSk (i,e)

! calculate del k * del omega / omega

          if(if3d)then
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                         / (omega**3+tiny)
          else
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                         / (omega**3+tiny)
          endif

          re_t    = rho * k /(mu * omega + tiny) 
          alp_str = alpinf_str * (alp0_str + (re_t/r_k))
     $                                         / (1.+(re_t/r_k)) 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!
          rhoalpk(i)= rho*alp_str*k

          rtld = re_t / r_k
          funcr = (1.0 - alp0_str) / (alp0_str + rtld)/(1.0 + rtld)
          factr = (1.0 + rtld * funcr)
          sigma_omega1 = 1.0/sigma_omega
          rhoalpfr(i) = expn * rho * alp_str * factr * omwom(i)
     $                                           * sigma_omega1

c nu_t is kinematic turbulent viscosity
c units of k = m2/s2, units of omega = 1/s, nu units = m2/s
c set limit for nu_t
c           nu_t    = max(tiny,nu_t)
c	    if(nu_t.gt.5000*mu)nu_t = 5000*mu

          betai_str = betainf_str * (akk + (re_t/r_b)**4)
     $              / (1.0 + (re_t/r_b)**4)


          if (xk3.le.0)then
            f_beta_str = 1.0
          else
            f_beta_str = (1.0 + 680.0*xk3*xk3)/(1.0 + 400.0*xk3*xk3)
          endif
          Y_k = rho * betai_str * f_beta_str * k * omega

c betai_str = beta_star in 12.5.15 for incompressible flow
 
          extra_prod = 0.
          if(iflomach) extra_prod = twothird*div(i)
          G_k0= mu_t*g(i) - ( rho*k + mu_t*div(i) )*extra_prod
          G_k = min(G_k0, 10.*Y_k)

c g(i) is S**2 in equation sheet

          kSrc  (i,1,1,e) = G_k - Y_k

c Compute production of omega
          alpha = (alp_inf/alp_str) * 
     $          ((alpha_0 + (re_t/r_w))/(1.0 + (re_t/r_w)))

c          G_w = alpha*alp_str*rho*G_k/mu_t !g(i)
          G_w0 = alpha*alp_str*rho*(g(i)-(omega+div(i))*extra_prod)

c Compute dissipation of omega
          beta = beta_0
c no compressibility correction M < 0.25

          x_w = abs((sum_xx)/(betainf_str*omega + tiny)**3)
          f_b = 1.0
          if(if3d) f_b = (1.0 + 70.0*x_w)/(1.0 + 80.0*x_w)

          Y_w = rho*beta*f_b * omega * omega

          G_w = min(G_w0, 10.0*Y_w)
          omgSrc(i,1,1,e) = G_w - Y_w

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

c solve for omega_pert

c add mut*delsqf
        sigma_omega1 = 1.0/sigma_omega
        call copy   (mu_omeg,rhoalpk,lxyz)
        call cmult  (mu_omeg,sigma_omega1,lxyz)
        call col4   (extra_src_omega,mu_omeg ,omwom
     $                                ,delsqf_omegb(1,1,1,e),lxyz)

c add mu*delsqf
        call copy   (tempv,delsqf_omegb(1,1,1,e),lxyz)
        call cmult  (tempv,mu,lxyz)
        call col2   (tempv,f_omegb(1,1,1,e),lxyz)
        call add2   (extra_src_omega, tempv,lxyz)

c Form (1/sigma_w) del_mut * del_omw
c  form 1: (del_yw/yw) del_k
        call col3   (term1,dfdx_omegb(1,1,1,e),k_x,lxyz)
        call addcol3(term1,dfdy_omegb(1,1,1,e),k_y,lxyz)
        if(if3d)
     $  call addcol3(term1,dfdz_omegb(1,1,1,e),k_z,lxyz)

c  form 2: 2(omw/om) (del_omw/yw)^2
        expm = -expn
        call col3   (term2,omwom,delfsq_omegb(1,1,1,e),   lxyz)
        call col2   (term2           ,t(1,1,1,e,ifld_k-1),lxyz)
        call cmult  (term2,expm,lxyz)
        call add3   (tempv, term1, term2, lxyz)

c  form 3: -(omw/om) k (del_yw/yw) \del_omp/omw
        call col3   (term3     ,omwom,t(1,1,1,e,ifld_k-1),lxyz)
        call invcol2(term3     ,f_omegb(1,1,1,e)         ,lxyz)
        call col3   (term4,  dfdx_omegb(1,1,1,e),omp_x,   lxyz)
        call addcol3(term4,  dfdy_omegb(1,1,1,e),omp_y,   lxyz)
        if(if3d)
     $  call addcol3(term4,  dfdz_omegb(1,1,1,e),omp_z,   lxyz)
        call subcol3(tempv, term3, term4, lxyz)

        call addcol3(extra_src_omega, rhoalpfr, tempv,    lxyz)

c add rho v * del_omw
        call col3   (tempv,dfdx_omegb(1,1,1,e),VX(1,1,1,e),lxyz)
        call addcol3(tempv,dfdy_omegb(1,1,1,e),VY(1,1,1,e),lxyz)
        if(if3d)
     $  call addcol3(tempv,dfdz_omegb(1,1,1,e),VZ(1,1,1,e),lxyz)
        call col2   (tempv,vtrans(1,1,1,e,1),lxyz)
        call admcol3(extra_src_omega,tempv,f_omegb(1,1,1,e),expm,lxyz)

        call add2   (omgSrc(1,1,1,e), extra_src_omega           ,lxyz)

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komgSST_stndrd_compute
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      real           tempv(lxyz), rhoalpk (lxyz)
     $              ,omwom(lxyz), rhoalpfr(lxyz)

      real           term1(lxyz), term2   (lxyz)
     $              ,term3(lxyz), term4   (lxyz)
     $              ,g    (lxyz), div     (lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

      real mu_omeg(lxyz), mu_omegx(lxyz), mu_omegy(lxyz), mu_omegz(lxyz)
      real extra_src_omega(lxyz)

c====turbulence constants==========

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigk1        = coeffs( 2)
        sigom1       = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta1        = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        beta_str     = coeffs( 8)
        gamma1       = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)

c additional SST and k and epsilon constants
        alp1         = coeffs(17)
        beta2        = coeffs(18)
        sigk2        = coeffs(19)
        sigom2       = coeffs(20)
        gamma2       = coeffs(21)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call copy   (g,   St_mag2(1,e),       lxyz)
        call copy   (div, DivQ   (1,e),       lxyz)

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

c solve for omega_pert

c ---------------------
c        call check_omwall_behavior
c ---------------------
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0 
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d) 
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          omwom (i)= 1.0/(1.0+t(i,1,1,e,ifld_omega-1) /f_omegb(i,1,1,e))

c See equations from eqns_k_omega1.pdf from Eq. (3) onwards
c Eq.(1) and (2) in eqns_k_omega1.pdf are the governing equations
c no source terms Sk or S_w are added
c ----------

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))

! calculate del k * del omega / omega

          if(if3d)then
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                            / (omega+tiny)
          else
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                            / (omega+tiny)
          endif

          rhoalpk(i)= rho*k
          rhoalpfr(i) = expn * rho * omwom(i) * sigom1

! calculate F2 based on arg2

          yw   = ywd  (i,1,1,e)
          ywm1 = ywdm1(i,1,1,e)
          ywm2 = ywm1*ywm1
          arg2_1 =      sqrt(abs(k)) * ywm1 / omega / beta_str
          arg2_2 =          500.0*nu * ywm2 / omega
          arg2   = max(2.0*arg2_1, arg2_2)
          Fun2   = tanh(arg2 * arg2)

! calculate F1 based on arg1

          argCD  =     2.0 * rho * sigom2 * xk
          CDkom  = max(argCD, 1.0e-10)
          arg1_1 = max(    arg2_1, arg2_2)
          arg1_2 =     4.0 * rho * sigom2 * k * ywm2 / CDkom
          arg1   = min(    arg1_1, arg1_2)
          Fun1   = tanh(arg1 * arg1 * arg1 * arg1)

! calculate mu_t 
          argn_1 = alp1*(omega + tiny)
          argn_2 = Fun2*St_magn ! this can also be Om_magn

          mu_t = 0.0
          denom = max(argn_1, argn_2)
          if(yw.ne.0) mu_t = rho * alp1 * k / denom 

! Compute Y_k = dissipation of k

          Y_k = rho * beta_str * k * omega

! Compute G_k = production of  k and limit it to 10*Y_k (the dissipation of k)

          extra_prod = 0.
          if(iflomach) extra_prod = twothird*div(i)
          G_k0= mu_t*g(i) - ( rho*k + mu_t*div(i) )*extra_prod
          G_k = min(G_k0, 10.*Y_k)

! Compute Source term for k 

          kSrc  (i,1,1,e) = G_k - Y_k

c Compute production of omega

          beta  = Fun1 * beta1  + (1.0 - Fun1) * beta2
          gamma = Fun1 * gamma1 + (1.0 - Fun1) * gamma2
          sigk  = Fun1 * sigk1  + (1.0 - Fun1) * sigk2
          sigom = Fun1 * sigom1 + (1.0 - Fun1) * sigom2

! Compute production of omega

          G_w0= rho * gamma * (g(i)-(div(i)+denom/alp1)*extra_prod)

! Compute dissipation of omega
 
          Y_w = rho * beta * omega * omega

          G_w = min(G_w0, 10.0*Y_w)

! Compute additional SST term for omega

          S_w = (1.0 - Fun1) * argCD

! Compute Source term for omega

          omgSrc(i,1,1,e) = G_w - Y_w + S_w
          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t * sigk
          mutso(i,1,1,e)   = mu_t * sigom

        enddo

c solve for omega_pert

c add mut*delsqf
        call copy   (mu_omeg,rhoalpk,lxyz)
        call cmult  (mu_omeg,sigom1, lxyz)
        call col4   (extra_src_omega,mu_omeg ,omwom
     $                                ,delsqf_omegb(1,1,1,e),lxyz)

c add mu*delsqf
        call copy   (tempv,delsqf_omegb(1,1,1,e),lxyz)
        call cmult  (tempv,mu,lxyz)
        call col2   (tempv,f_omegb(1,1,1,e),lxyz)
        call add2   (extra_src_omega, tempv,lxyz)

c Form (1/sigma_w) del_mut * del_omw
c  form 1: (del_yw/yw) del_k
        call col3   (term1,dfdx_omegb(1,1,1,e),k_x,lxyz)
        call addcol3(term1,dfdy_omegb(1,1,1,e),k_y,lxyz)
        if(if3d)
     $  call addcol3(term1,dfdz_omegb(1,1,1,e),k_z,lxyz)

c  form 2: 2(omw/om) (del_omw/yw)^2
        expm = -expn
        call col3   (term2,omwom,delfsq_omegb(1,1,1,e),   lxyz)
        call col2   (term2           ,t(1,1,1,e,ifld_k-1),lxyz)
        call cmult  (term2,expm,lxyz)
        call add3   (tempv, term1, term2, lxyz)

c  form 3: -(omw/om) k (del_yw/yw) \del_omp/omw
        call col3   (term3     ,omwom,t(1,1,1,e,ifld_k-1),lxyz)
        call invcol2(term3     ,f_omegb(1,1,1,e)         ,lxyz)
        call col3   (term4,  dfdx_omegb(1,1,1,e),omp_x,   lxyz)
        call addcol3(term4,  dfdy_omegb(1,1,1,e),omp_y,   lxyz)
        if(if3d)
     $  call addcol3(term4,  dfdz_omegb(1,1,1,e),omp_z,   lxyz)
        call subcol3(tempv, term3, term4, lxyz)

        call addcol3(extra_src_omega, rhoalpfr, tempv,    lxyz)

c add rho v * del_omw
        call col3   (tempv,dfdx_omegb(1,1,1,e),VX(1,1,1,e),lxyz)
        call addcol3(tempv,dfdy_omegb(1,1,1,e),VY(1,1,1,e),lxyz)
        if(if3d)
     $  call addcol3(tempv,dfdz_omegb(1,1,1,e),VZ(1,1,1,e),lxyz)
        call col2   (tempv,vtrans(1,1,1,e,1),lxyz)
        call admcol3(extra_src_omega,tempv,f_omegb(1,1,1,e),expm,lxyz)

        call add2   (omgSrc(1,1,1,e), extra_src_omega           ,lxyz)

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komgSST_lowRe_compute
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      real           tempv(lxyz), rhoalpk (lxyz)
     $              ,omwom(lxyz), rhoalpfr(lxyz)

      real           term1(lxyz), term2   (lxyz)
     $              ,term3(lxyz), term4   (lxyz)
     $              ,g    (lxyz), div     (lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

      real mu_omeg(lxyz), mu_omegx(lxyz), mu_omegy(lxyz), mu_omegz(lxyz)
      real extra_src_omega(lxyz)

c====turbulence constants==========

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigk1        = coeffs( 2)
        sigom1       = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta1        = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        beta_str     = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)

c additional SST and k and epsilon constants
        alp1         = coeffs(17)
        beta2        = coeffs(18)
        sigk2        = coeffs(19)
        sigom2       = coeffs(20)
        gamma2       = coeffs(21)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call copy   (g,   St_mag2(1,e),       lxyz)
        call copy   (div, DivQ   (1,e),       lxyz)

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

c solve for omega_pert

c ---------------------
c        call check_omwall_behavior
c ---------------------
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0 
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d) 
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          omwom (i)= 1.0/(1.0+t(i,1,1,e,ifld_omega-1)/f_omegb(i,1,1,e))

c See equations from eqns_k_omega1.pdf from Eq. (3) onwards
c Eq.(1) and (2) in eqns_k_omega1.pdf are the governing equations
c no source terms Sk or S_w are added
c ----------

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))
          sum_xx  =      OiOjSk (i,e)

! calculate del k * del omega / omega

          if(if3d)then
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                            / (omega+tiny)
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                         / (omega**3+tiny)
          else
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                            / (omega+tiny)
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                         / (omega**3+tiny)
          endif

c ------------ begin low Re part of k-omega -------------------

          re_t    = rho * k /(mu * omega + tiny) 
          alp_str = alpinf_str * (alp0_str + (re_t/r_k))
     $                                         / (1.+(re_t/r_k)) 
          rhoalpk(i)= rho*alp_str*k

          rtld = re_t / r_k
          funcr = (1.0 - alp0_str) / (alp0_str + rtld)/(1.0 + rtld)
          factr = (1.0 + rtld * funcr)
          rhoalpfr(i) = expn * rho * alp_str * factr * omwom(i) * sigom1

c nu_t is kinematic turbulent viscosity
c units of k = m2/s2, units of omega = 1/s, nu units = m2/s
c set limit for nu_t
c           nu_t    = max(tiny,nu_t)
c	    if(nu_t.gt.5000*mu)nu_t = 5000*mu

          betai_str = beta_str * (akk + (re_t/r_b)**4)
     $              / (1.0 + (re_t/r_b)**4)

          if (xk3.le.0)then
            f_beta_str = 1.0
          else
            f_beta_str = (1.0 + 680.0*xk3*xk3)/(1.0 + 400.0*xk3*xk3)
          endif

c Compute production of omega
          alpha = (alp_inf/alp_str) *
     $          ((alpha_0 + (re_t/r_w))/(1.0 + (re_t/r_w)))

          gammai = alpha*alp_str

c Compute dissipation of omega

          x_w = abs((sum_xx)/(beta_str*omega + tiny)**3)
          f_b = 1.0
          if(if3d) f_b = (1.0 + 70.0*x_w)/(1.0 + 80.0*x_w)
          betai= beta1 * f_b

c ------------ end   low Re part of k-omega -------------------

! calculate F2 based on arg2

          yw   = ywd  (i,1,1,e)
          ywm1 = ywdm1(i,1,1,e)
          ywm2 = ywm1*ywm1
          arg2_1 =      sqrt(abs(k)) * ywm1 / omega / beta_str
          arg2_2 =          500.0*nu * ywm2 / omega
          arg2   = max(2.0*arg2_1, arg2_2)
          Fun2   = tanh(arg2 * arg2)

! calculate F1 based on arg1

          argCD  =     2.0 * rho * sigom2 * xk
          CDkom  = max(argCD, 1.0e-10)
          arg1_1 = max(    arg2_1, arg2_2)
          arg1_2 =     4.0 * rho * sigom2 * k * ywm2 / CDkom
          arg1   = min(    arg1_1, arg1_2)
          Fun1   = tanh(arg1 * arg1 * arg1 * arg1)

! calculate mu_t 
          argn_1 = alp1*(omega + tiny) / alp_str
          argn_2 = Fun2*St_magn ! this can also be St_magn

          mu_t = 0.0
          denom = max(argn_1, argn_2)
          if(yw.ne.0) mu_t = rho * alp1 * k / denom 

! Compute Y_k = dissipation of k

c          Y_k = rho * beta_str * k * omega
          Y_k = rho * betai_str * f_beta_str * k * omega

! Compute G_k = production of  k and limit it to 10*Y_k (the dissipation of k)

          extra_prod = 0.
          if(iflomach) extra_prod = twothird*div(i)
          G_k0= mu_t*g(i) - ( rho*k + mu_t*div(i) )*extra_prod
          G_k = min(G_k0, 10.*Y_k)

! Compute Source term for k 

          kSrc  (i,1,1,e) = G_k - Y_k

c Compute production of omega

          beta  = Fun1 * betai  + (1.0 - Fun1) * beta2
          gamma = Fun1 * gammai + (1.0 - Fun1) * gamma2
          sigk  = Fun1 * sigk1  + (1.0 - Fun1) * sigk2
          sigom = Fun1 * sigom1 + (1.0 - Fun1) * sigom2

! Compute production of omega

          G_w0= rho * gamma * (g(i)-(div(i)+denom/alp1)*extra_prod)

! Compute dissipation of omega

          Y_w = rho * beta * omega * omega

          G_w = min(G_w0, 10.0*Y_w)

! Compute additional SST term for omega

          S_w = (1.0 - Fun1) * argCD

! Compute Source term for omega

          omgSrc(i,1,1,e) = G_w - Y_w + S_w
          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t * sigk
          mutso(i,1,1,e)   = mu_t * sigom

        enddo

c solve for omega_pert

c add mut*delsqf
        call copy   (mu_omeg,rhoalpk,lxyz)
        call cmult  (mu_omeg,sigom1, lxyz)
        call col4   (extra_src_omega,mu_omeg ,omwom
     $                                ,delsqf_omegb(1,1,1,e),lxyz)

c add mu*delsqf
        call copy   (tempv,delsqf_omegb(1,1,1,e),lxyz)
        call cmult  (tempv,mu,lxyz)
        call col2   (tempv,f_omegb(1,1,1,e),lxyz)
        call add2   (extra_src_omega, tempv,lxyz)

c Form (1/sigma_w) del_mut * del_omw
c  form 1: (del_yw/yw) del_k
        call col3   (term1,dfdx_omegb(1,1,1,e),k_x,lxyz)
        call addcol3(term1,dfdy_omegb(1,1,1,e),k_y,lxyz)
        if(if3d)
     $  call addcol3(term1,dfdz_omegb(1,1,1,e),k_z,lxyz)

c  form 2: 2(omw/om) (del_omw/yw)^2
        expm = -expn
        call col3   (term2,omwom,delfsq_omegb(1,1,1,e),   lxyz)
        call col2   (term2           ,t(1,1,1,e,ifld_k-1),lxyz)
        call cmult  (term2,expm,lxyz)
        call add3   (tempv, term1, term2, lxyz)

c  form 3: -(omw/om) k (del_yw/yw) \del_omp/omw
        call col3   (term3     ,omwom,t(1,1,1,e,ifld_k-1),lxyz)
        call invcol2(term3     ,f_omegb(1,1,1,e)         ,lxyz)
        call col3   (term4,  dfdx_omegb(1,1,1,e),omp_x,   lxyz)
        call addcol3(term4,  dfdy_omegb(1,1,1,e),omp_y,   lxyz)
        if(if3d)
     $  call addcol3(term4,  dfdz_omegb(1,1,1,e),omp_z,   lxyz)
        call subcol3(tempv, term3, term4, lxyz)

        call addcol3(extra_src_omega, rhoalpfr, tempv,    lxyz)

c add rho v * del_omw
        call col3   (tempv,dfdx_omegb(1,1,1,e),VX(1,1,1,e),lxyz)
        call addcol3(tempv,dfdy_omegb(1,1,1,e),VY(1,1,1,e),lxyz)
        if(if3d)
     $  call addcol3(tempv,dfdz_omegb(1,1,1,e),VZ(1,1,1,e),lxyz)
        call col2   (tempv,vtrans(1,1,1,e,1),lxyz)
        call admcol3(extra_src_omega,tempv,f_omegb(1,1,1,e),expm,lxyz)

        call add2   (omgSrc(1,1,1,e), extra_src_omega           ,lxyz)

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_stndrd_compute_noreg
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      real           tempv(lxyz), rhoalpk (lxyz)
     $              ,omwom(lxyz), rhoalpfr(lxyz)

      real           term1(lxyz), term2   (lxyz)
     $              ,term3(lxyz), term4   (lxyz)
     $              ,g    (lxyz), div     (lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

      real mu_omeg(lxyz), mu_omegx(lxyz), mu_omegy(lxyz), mu_omegz(lxyz)
      real extra_src_omega(lxyz)

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call copy   (g,   St_mag2(1,e),       lxyz)
        call copy   (div, DivQ   (1,e),       lxyz)
c        call rzero  (div,                     lxyz)

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

c solve for omega_pert

c ---------------------
c        call check_omwall_behavior
c ---------------------
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0 
          o_x(i)= omp_x(i)
          o_y(i)= omp_y(i)
          if(if3d) 
     $    o_z(i)= omp_z(i)

c See equations from eqns_k_omega1.pdf from Eq. (3) onwards
c Eq.(1) and (2) in eqns_k_omega1.pdf are the governing equations
c no source terms Sk or S_w are added

          if(St_mag2(i,e).lt.0.) write(*,*) '  St_mag2', i, e, St_mag2
          if(Om_mag2(i,e).lt.0.) write(*,*) '  Om_mag2', i, e, Om_mag2
          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))
          sum_xx  =      OiOjSk (i,e)
c         if(sum_xx .ne.0.) write(*,*) '  sum_xx ', i, e, sum_xx

! calculate del k * del omega / omega

          if(if3d)then
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                         / (omega**3+tiny)
          else
            xk3= (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                         / (omega**3+tiny)
          endif

          alp_str = alpinf_str 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!

c nu_t is kinematic turbulent viscosity
c units of k = m2/s2, units of omega = 1/s, nu units = m2/s
c set limit for nu_t
c           nu_t    = max(tiny,nu_t)
c	    if(nu_t.gt.5000*mu)nu_t = 5000*mu

          betai_str = betainf_str

          if (xk3.le.0)then
            f_beta_str = 1.0
          else
            f_beta_str = (1.0 + 680.0*xk3*xk3)/(1.0 + 400.0*xk3*xk3)
          endif
          Y_k = rho * betai_str * f_beta_str * k * omega
 
c betai_str = beta_star in 12.5.15 for incompressible flow
 
          extra_prod = 0.
          if(iflomach) extra_prod = twothird*div(i)
          G_k0= mu_t*g(i) - ( rho*k + mu_t*div(i) )*extra_prod
          G_k = min(G_k0, 10.*Y_k)

c g(i) is S**2 in equation sheet

          kSrc  (i,1,1,e) = G_k - Y_k

c Compute production of omega
          alpha = (alp_inf/alp_str)

c          G_w0 = alpha*alp_str*rho*g(i)
          G_w0 = alpha*alp_str*rho*(g(i)-(omega+div(i))*extra_prod)

c Compute dissipation of omega
          beta = beta_0
c no compressibility correction M < 0.25

          x_w = abs((sum_xx)/(betainf_str*omega + tiny)**3)
          f_b = 1.0
          if(if3d) f_b = (1.0 + 70.0*x_w)/(1.0 + 80.0*x_w)

          Y_w = rho*beta*f_b * omega * omega

          G_w = min(G_w0, 10.0*Y_w)
          omgSrc(i,1,1,e) = G_w - Y_w

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_init(ifld_k_in,ifld_omega_in,ifcoeffs
     $                       ,coeffs_in,wall_id,ywd_in,model_id)
c
c     Initialize values ifld_omega & ifld_k for RANS k-omega turbulence
c     modeling
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      real w1,w2,w3,w4,w5
      common /SCRNS/
     & w1(lx1*ly1*lz1*lelv)
     &,w2(lx1*ly1*lz1*lelv)
     &,w3(lx1*ly1*lz1*lelv)
     &,w4(lx1*ly1*lz1*lelv)
     &,w5(lx1*ly1*lz1*lelv)

      integer n,wall_id,ifld_mx
      real coeffs_in(1),ywd_in(1)
      logical ifcoeffs

      character*3 bcw
      character*36 mname(6)

      data mname
     &/'regularized standard k-omega        '
     &,'regularized low-Re k-omega          '
     &,'regularized standard k-omega SST    '
     &,'regularized low-Re k-omega SST      '
     &,'non-regularized standard k-omega    '
     &,'non-regularized standard k-omega SST'/

      n=nx1*ny1*nz1*nelv

      if(nid.eq.0) write(6,*) 'init RANS model'

      if(iflomach) then
        if(nid.eq.0) write(6,*)
     &          "ERROR: K-OMEGA NOT SUPPORTED WITH LOW MACH FORMULATION"
        call exitt
      endif

      ifrans_komg_stndrd       = .FALSE.
      ifrans_komg_lowRe        = .FALSE.
      ifrans_komgSST_stndrd    = .FALSE.
      ifrans_komgSST_lowRe     = .FALSE.
      ifrans_komg_stndrd_noreg = .FALSE.
      if(model_id .eq.0) ifrans_komg_stndrd          = .TRUE.
      if(model_id .eq.1) ifrans_komg_lowRe           = .TRUE.
      if(model_id .eq.2) ifrans_komgSST_stndrd       = .TRUE.
      if(model_id .eq.3) ifrans_komgSST_lowRe        = .TRUE.
      if(model_id .eq.4) ifrans_komg_stndrd_noreg    = .TRUE.

      if(nid.eq.0) write(*,'(a,a)') 
     &                      '  model: ',mname(model_id+1)
      ifld_k     = ifld_k_in
      ifld_omega = ifld_omega_in
      ifld_mx=max(ifld_k,ifld_omega)
      if (ifld_mx.gt.ldimt1) 
     $  call exitti('nflds gt ldimt+1, recompile with ldimt > ',
     $  ifld_mx+1)

! specify k-omega model coefficients

c      if(ncoeffs_in.lt.ncoeffs) 
c     $  call exitti('dim of user provided komg coeffs array 
c     $               should be >=$',ncoeffs)

      if(ifcoeffs) then
         do i=1,ncoeffs
            coeffs(i) =coeffs_in(i)
         enddo
      else
         if(ifrans_komg_stndrd .or. ifrans_komg_lowRe .or.
     $   ifrans_komg_stndrd_noreg)call rans_komg_set_defaultcoeffs
         if(ifrans_komgSST_stndrd .or. ifrans_komgSST_lowRe)
     $                            call rans_komgSST_set_defaultcoeffs
      endif

c solve for omega_pert
      if(wall_id.eq.0) then
        if(nid.eq.0) write(6,*) ' user supplied wall distance'
        call copy(ywd,ywd_in,n)
      else
        bcw    = 'W  '
        ifld   = 1
        if(nid.eq.0) write(6,*) 'BC for distance ',bcw
        if(wall_id.eq.1) call cheap_dist(ywd,ifld,bcw)
        if(wall_id.eq.2) call distf(ywd,ifld,bcw,w1,w2,w3,w4,w5)
        call copy(ywd_in,ywd,n)
      endif

      call rans_komg_omegabase

      if(nid.eq.0) write(6,*) 'done :: init RANS'

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_omegabase
c
c     Compute Omega base solution which diverges at walls
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (nprof_max = 10000)
      integer i,j,k,e
      real    nu

      real kv_min,omeg_max

      omeg_max   = coeffs(15)
      beta0      = coeffs(6)
      nu  = param(2)/param(1)
      ntot1 = nx1*ny1*nz1*nelv

      betainf_str = coeffs(11)
      yw_min = glmin(ywd,ntot1)

      Cfcon = 6.0 * nu / beta0 ! 2.0 * nu0 / betainf_str ! 
      expn  = -2.0

      call gradm1 (dfdx_omegb,dfdy_omegb,dfdz_omegb,   ywd)
      call opcolv (dfdx_omegb,dfdy_omegb,dfdz_omegb,   bm1)
      call opdssum(dfdx_omegb,dfdy_omegb,dfdz_omegb)
      call opcolv (dfdx_omegb,dfdy_omegb,dfdz_omegb,binvm1)

c     call gradm1 (dudx,      dudy      ,dudz,  dfdx_omegb)
c     call copy   (delsqf_omegb, dudx, ntot1)
c     call gradm1 (dudx,      dudy      ,dudz,  dfdy_omegb)
c     call add2   (delsqf_omegb, dudy, ntot1)
c     call gradm1 (dudx,      dudy      ,dudz,  dfdz_omegb)
c     call add2   (delsqf_omegb, dudz, ntot1)

      toll = 1.0e-08

c     call vdot3  (delfsq_omegb,dfdx_omegb,dfdy_omegb,dfdz_omegb 
c    $                         ,dfdx_omegb,dfdy_omegb,dfdz_omegb,ntot1)

      call rzero  (delsqf_omegb, ntot1)
      call rone   (delfsq_omegb, ntot1)

      do e = 1,nelv
      do k = 1,nz1
      do j = 1,ny1
      do i = 1,nx1

         ieg = lglel(e)
         yw   = ywd(i,j,k,e)       ! 1.0 - abs(y)
         ywmin=sqrt(Cfcon/omeg_max)
         if(yw.gt.ywmin) then
            ywm1 = 1.0 /yw
         else
           if(yw.ne.0) write(*,'(a,3G14.7,4I5)') 
     $                   'ywmin and yw ',ywmin,yw,omeg_max, i, j, k, ieg
            ywm1 = 1.0 /ywmin
         endif
         ywdm1(i,j,k,e) = ywm1
         ywm2 = ywm1*ywm1
         ywm3 = ywm2*ywm1
         ywm4 = ywm2*ywm2

         f_omegb     (i,j,k,e) =        Cfcon * ywm2
         delfpart              = expn * ywm1
         dfdx_omegb  (i,j,k,e) = dfdx_omegb(i,j,k,e)  * ywm1
         dfdy_omegb  (i,j,k,e) = dfdy_omegb(i,j,k,e)  * ywm1
         dfdz_omegb  (i,j,k,e) = dfdz_omegb(i,j,k,e)  * ywm1
         delsqfpart            = expn * (expn - 1.0)  * ywm2
         delsqf_omegb(i,j,k,e) =(delsqfpart  * delfsq_omegb(i,j,k,e)
     $                         + delfpart    * delsqf_omegb(i,j,k,e))
         delfsq_omegb(i,j,k,e) = delfsq_omegb(i,j,k,e)* ywm2

      enddo
      enddo
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_set_defaultcoeffs
c
      include 'SIZE'
      include 'RANS_KOMG'

c ====various problem-specific turbulence constants
c omeg_max = value of omega on the walls
c kv_min = value of K on the walls
c Pr_t is the turbulent prandtl number

        vkappa = 0.41

c Turbulent viscosity constants
        Pr_t         = 0.85
        coeffs( 1)   = Pr_t
        sigma_k      = 2.0
        coeffs( 2)   = sigma_k
        sigma_omega  = 2.0
        coeffs( 3)   = sigma_omega

c Low Reynolds number correction constants

c Production of K constants
        alpinf_str   = 1.0
        coeffs( 4)   = alpinf_str
        r_k          = 6.0
        coeffs( 5)   = r_k
        beta_0       = 0.072 ! should be 0.075 for SST
        coeffs( 6)   = beta_0
        alp0_str     = beta_0/3.0
        coeffs( 7)   = alp0_str

c Dissipation of K constants
        betainf_str  = 0.09
        coeffs( 8)   = betainf_str
        alp_inf      = 0.52
c        alp_inf      = beta_0/betainf_str 
c     $               - vkappa**2/sqrt(betainf_str)/sigma_omega ! should be 0.52 for k-omega
        coeffs( 9)   = alp_inf
        r_b          = 8.0
        coeffs(10)   = r_b
        akk          = 4.0/15.0
        coeffs(11)   = akk

c Production of omega constants
        alpha_0      = 1.0/9.0
        coeffs(12)   = alpha_0
        r_w          = 2.95
        coeffs(13)   = r_w

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = 0.0
        coeffs(14)   = kv_min
        omeg_max     = 2.0e8 ! 400.0 Lan
        coeffs(15)   = omeg_max
        tiny         = 1.0e-20
        coeffs(16)   = tiny

        return
        end
c-----------------------------------------------------------------------
      subroutine rans_komgSST_set_defaultcoeffs
c
      include 'SIZE'
      include 'RANS_KOMG'

c ====various problem-specific turbulence constants
c omeg_max = value of omega on the walls
c kv_min = value of K on the walls
c Pr_t is the turbulent prandtl number

        vkappa = 0.41

c Turbulent viscosity constants
        Pr_t         = 0.85
        coeffs( 1)   = Pr_t
        sigk1        = 0.5 ! should be 0.85 for SST
        coeffs( 2)   = sigk1
        sigom1       = 0.5
        coeffs( 3)   = sigom1

c Low Reynolds number correction constants

c Production of K constants
        alpinf_str   = 1.0
        coeffs( 4)   = alpinf_str
        r_k          = 6.0
        coeffs( 5)   = r_k
        beta_0       = 0.072 ! should be 0.075 for SST
        coeffs( 6)   = beta_0
        alp0_str     = beta_0/3.0
        coeffs( 7)   = alp0_str

c Dissipation of K constants
        betainf_str  = 0.09
        coeffs( 8)   = betainf_str
        alp_inf      = beta_0/betainf_str 
     $               - sigom1 * vkappa**2/sqrt(betainf_str) ! should be 0.52 for k-omega
c        alp_inf      = 0.52
        coeffs( 9)   = alp_inf
        r_b          = 8.0
        coeffs(10)   = r_b
        akk          = 4.0/15.0
        coeffs(11)   = akk

c Production of omega constants
        alpha_0      = 1.0/9.0
        coeffs(12)   = alpha_0
        r_w          = 2.95
        coeffs(13)   = r_w

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = 0.0
        coeffs(14)   = kv_min
        omeg_max     = 2.0e8 ! 400.0 
        coeffs(15)   = omeg_max
        tiny         = 1.0e-20
        coeffs(16)   = tiny

c additional SST and k and epsilon constants
        alp1         = 0.31
        coeffs(17)   = alp1
        beta2        = 0.0828
        coeffs(18)   = beta2
        sigk2        = 1.0
        coeffs(19)   = sigk2
        sigom2       = 0.856
        coeffs(20)   = sigom2
        gamma2       = beta2/betainf_str 
     $               - sigom2 * vkappa**2/sqrt(betainf_str) ! should be 0.44 for k-epsilon
c        gamma2       = 0.44
        coeffs(21)   = gamma2

        return
        end
c-----------------------------------------------------------------------
      subroutine rans_komg_stndrd_eddy
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert

          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          alp_str = alpinf_str 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_lowRe_eddy
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          re_t    = rho * k /(mu * omega + tiny) 
          alp_str = alpinf_str * (alp0_str + (re_t/r_k))
     $                                         / (1.+(re_t/r_k)) 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komgSST_stndrd_eddy
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

c====turbulence constants==========

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigk1        = coeffs( 2)
        sigom1       = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta1        = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        beta_str     = coeffs( 8)
        gamma1       = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)

c additional SST and k and epsilon constants
        alp1         = coeffs(17)
        beta2        = coeffs(18)
        sigk2        = coeffs(19)
        sigom2       = coeffs(20)
        gamma2       = coeffs(21)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d)
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))

! calculate del k * del omega / omega

          if(if3d)then
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                            / (omega+tiny)
          else
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                            / (omega+tiny)
          endif

! calculate F2 based on arg2

          yw   = ywd  (i,1,1,e)
          ywm1 = ywdm1(i,1,1,e)
          ywm2 = ywm1*ywm1
          arg2_1 =      sqrt(abs(k)) * ywm1 / omega / beta_str
          arg2_2 =          500.0*nu * ywm2 / omega
          arg2   = max(2.0*arg2_1, arg2_2)
          Fun2   = tanh(arg2 * arg2)

! calculate F1 based on arg1

          argCD  =     2.0 * rho * sigom2 * xk
          CDkom  = max(argCD, 1.0e-10)
          arg1_1 = max(    arg2_1, arg2_2)
          arg1_2 =     4.0 * rho * sigom2 * k * ywm2 / CDkom
          arg1   = min(    arg1_1, arg1_2)
          Fun1   = tanh(arg1 * arg1 * arg1 * arg1)

! calculate mu_t 
          argn_1 = alp1*(omega + tiny)
          argn_2 = Fun2*St_magn ! this can also be St_magn

          mu_t = 0.0
          denom = max(argn_1, argn_2)
          if(yw.ne.0) mu_t = rho * alp1 * k / denom 

          sigk  = Fun1 * sigk1  + (1.0 - Fun1) * sigk2
          sigom = Fun1 * sigom1 + (1.0 - Fun1) * sigom2

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t * sigk
          mutso(i,1,1,e)   = mu_t * sigom

        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komgSST_lowRe_eddy
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      real           k_x(lxyz),k_y(lxyz),k_z(lxyz)
     $             , o_x(lxyz),o_y(lxyz),o_z(lxyz)
     $              ,omp_x(lxyz), omp_y(lxyz), omp_z(lxyz)

      common /storesom/ St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

c====turbulence constants==========

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigk1        = coeffs( 2)
        sigom1       = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta1        = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        beta_str     = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)

c additional SST and k and epsilon constants
        alp1         = coeffs(17)
        beta2        = coeffs(18)
        sigk2        = coeffs(19)
        sigom2       = coeffs(20)
        gamma2       = coeffs(21)
c================================

      call comp_StOm (St_mag2, Om_mag2, OiOjSk, DivQ)

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv

        call gradm11(k_x,  k_y,  k_z,  t(1,1,1,1,ifld_k    -1),e)
        call gradm11(omp_x,omp_y,omp_z,t(1,1,1,1,ifld_omega-1),e)

        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) + f_omegb(i,1,1,e) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          expn = -2.0
          o_x(i)= omp_x(i)+expn * dfdx_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          o_y(i)= omp_y(i)+expn * dfdy_omegb(i,1,1,e) *f_omegb(i,1,1,e)
          if(if3d)
     $    o_z(i)= omp_z(i)+expn * dfdz_omegb(i,1,1,e) *f_omegb(i,1,1,e)

          St_magn = sqrt(St_mag2(i,e))
          Om_magn = sqrt(Om_mag2(i,e))

! calculate del k * del omega / omega

          if(if3d)then
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i) + k_z(i)*o_z(i))
     $                                            / (omega+tiny)
          else
            xk = (k_x(i)*o_x(i) + k_y(i)*o_y(i))
     $                                            / (omega+tiny)
          endif

c ------------ begin low Re part of k-omega -------------------

          re_t    = rho * k /(mu * omega + tiny) 
          alp_str = alpinf_str * (alp0_str + (re_t/r_k))
     $                                         / (1.+(re_t/r_k)) 
c ------------ end   low Re part of k-omega -------------------

! calculate F2 based on arg2

          yw   = ywd  (i,1,1,e)
          ywm1 = ywdm1(i,1,1,e)
          ywm2 = ywm1*ywm1
          arg2_1 =      sqrt(abs(k)) * ywm1 / omega / beta_str
          arg2_2 =          500.0*nu * ywm2 / omega
          arg2   = max(2.0*arg2_1, arg2_2)
          Fun2   = tanh(arg2 * arg2)

! calculate F1 based on arg1

          argCD  =     2.0 * rho * sigom2 * xk
          CDkom  = max(argCD, 1.0e-10)
          arg1_1 = max(    arg2_1, arg2_2)
          arg1_2 =     4.0 * rho * sigom2 * k * ywm2 / CDkom
          arg1   = min(    arg1_1, arg1_2)
          Fun1   = tanh(arg1 * arg1 * arg1 * arg1)

! calculate mu_t 
          argn_1 = alp1*(omega + tiny) / alp_str
          argn_2 = Fun2*St_magn ! this can also be St_magn

          mu_t = 0.0
          denom = max(argn_1, argn_2)
          if(yw.ne.0) mu_t = rho * alp1 * k / denom 

          sigk  = Fun1 * sigk1  + (1.0 - Fun1) * sigk2
          sigom = Fun1 * sigom1 + (1.0 - Fun1) * sigom2

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t * sigk
          mutso(i,1,1,e)   = mu_t * sigom

        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine rans_komg_stndrd_eddy_noreg
c
c     Compute RANS source terms and diffusivities on an 
c     element-by-element basis
c
      include 'SIZE'
      include 'TOTAL'
      include 'RANS_KOMG'

      parameter (lxyz=lx1*ly1*lz1)

      integer e
      real k,mu_t,nu_t,mu,nu, kv_min,omeg_max

c Turbulent viscosity constants
        Pr_t         = coeffs( 1)
        sigma_k      = coeffs( 2)
        sigma_omega  = coeffs( 3)

c Low Reynolds number correction constants
        alpinf_str   = coeffs( 4)
        r_k          = coeffs( 5)
        beta_0       = coeffs( 6)
        alp0_str     = coeffs( 7)

c Dissipation of K constants
        betainf_str  = coeffs( 8)
        alp_inf      = coeffs( 9)
        r_b          = coeffs(10)
        akk          = coeffs(11)

c Production of omega constants
        alpha_0      = coeffs(12)
        r_w          = coeffs(13)

c Dissipation of omega constants
c         beta_0 = defined earlier	     

        kv_min       = coeffs(14)
        omeg_max     = coeffs(15)
        tiny         = coeffs(16)
c================================

      nome_neg = 0
      nkey_neg = 0
      xome_neg = 0.
      xkey_neg = 0.

      do e=1,nelv
        do i=1,lxyz

          rho = param(1) ! vtrans(i,1,1,e,1)
          mu  = param(2) ! vdiff (i,1,1,e,1)
          nu  = mu/rho

c limits for k, omega

c solve for omega_pert
          omega   = t(i,1,1,e,ifld_omega-1) ! Current k & omega values
          k       = t(i,1,1,e,ifld_k  -1)   ! from previous timestep

          iflim_omeg = 0 ! limit omega^{prime} 1 limit omega_total

          if    (iflim_omeg.eq.0) then
            if(t(i,1,1,e,ifld_omega-1).lt.0.0) then
c             write(*,*) 'Zero OMEG ', t(i,1,1,e,ifld_omega-1)
              xome_neg = min(xome_neg,t(i,1,1,e,ifld_omega-1))
              nome_neg = nome_neg + 1
c             write(*,*) 'Neg  OMEG ', nome_neg, xome_neg
              t(i,1,1,e,ifld_omega-1) =0.01*abs(t(i,1,1,e,ifld_omega-1))
              omega = t(i,1,1,e,ifld_omega-1) ! Current k & omega values
            endif

            if(t(i,1,1,e,ifld_k-1).lt.0.0) then
c             write(*,*) 'Zero K    ', t(i,1,1,e,ifld_k-1)
              xkey_neg = min(xkey_neg,t(i,1,1,e,ifld_k-1))
              nkey_neg = nkey_neg + 1
c             write(*,*) 'Neg  KEY  ', nkey_neg, xkey_neg
              t(i,1,1,e,ifld_k-1) = 0.01*abs(t(i,1,1,e,ifld_k-1))
              k     = t(i,1,1,e,ifld_k  -1)   ! from previous timestep
            endif

          elseif(iflim_omeg.eq.1) then

            if(omega.lt.0.0) then
              write(*,*) 'OMEG tot is neg', omega
              omega = 0.01*abs(omega)
            endif

            if(k.lt.0.0) then
               write(*,*) 'K  is neg', k
               k = 0.01*abs(k)
            endif

          endif

          alp_str = alpinf_str 
          mu_t    = rho * alp_str*k/(omega + tiny) ! should multiply these by rho!!!

          mut  (i,1,1,e)   = mu_t
          mutsk(i,1,1,e)   = mu_t / sigma_k
          mutso(i,1,1,e)   = mu_t / sigma_omega
        enddo

      enddo

      if(loglevel.gt.2) then
        nome_neg =iglsum(nome_neg,1)
        nkey_neg =iglsum(nkey_neg,1)
        xome_neg = glmin(xome_neg,1)
        xkey_neg = glmin(xkey_neg,1)

        if(nid.eq.0 .and. nome_neg.gt.0)
     $    write(*,*) 'Neg Omega ', nome_neg, xome_neg
        if(nid.eq.0 .and. nkey_neg.gt.0)
     $    write(*,*) 'Neg TKE   ', nkey_neg, xkey_neg
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine comp_StOm(St_mag2,Om_mag2,OiOjSk,DivQ)
c
c     Compute the square of the magnitude of the stress and rotation tensors
c     St_mag2=2Sij*Sij=S'ij*S'ij/2, S'ij=2Sij, Sij=(dui/dxj+duj/dxi)/2
c     Om_mag2=2Oij*Oij=O'ij*O'ij/2, O'ij=OSij, Oij=(dui/dxj-duj/dxi)/2
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e

      parameter(lt  =lx1*ly1*lz1*lelv)
      parameter(lxyz=lx1*ly1*lz1)

      common /scruz/    sij  (lx1*ly1*lz1,6,lelv)
     $                , oij  (lx1*ly1*lz1,lelv,3)

      real              St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)
     $                , DivQ   (lx1*ly1*lz1,lelv)

      logical iflmc, ifdss
c
      thqrt    = 0.75

      iflmc = .false.! to add lowmach correction to 2Sij (2Sij - 2Q/3*delta_ij) 
      ifdss = .true. ! to dssum sij and oij
      nij = 3
      if (if3d.or.ifaxis) nij=6

      if(nid.eq.0 .and. loglevel.gt.2) write(*,*) 'updating StOm '
      call comp_sij_oij     (sij, oij, nij, vx, vy, vz, iflmc, ifdss) ! S'ij=2Sij, O'ij=2Oij
      call comp_sij_oij_mag2(sij, oij, nij, St_mag2, Om_mag2, OiOjSk) ! St_mag2=2S^2=S'^2/2
c                                                                     ! Om_mag2=2O^2=O'^2/2
      do e = 1, nelv
         call add3   (DivQ(1,e), sij(1,1,e), sij(1,2,e), lxyz)
         if(if3d.or.ifaxis)
     $   call add2   (DivQ(1,e), sij(1,3,e),             lxyz)
         if(iflmc)     call cmult(DivQ(1,e), thqrt,      lxyz)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine comp_sij_oij(sij,oij,nij,u,v,w,iflmc,ifdss)
c                                       du_i       du_j
c     Compute the stress tensor S_ij := ----   +   ----
c                                       du_j       du_i
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e
      logical iflmc, ifdss
c
      parameter(lt  =lx1*ly1*lz1*lelv)
      parameter(lxyz=lx1*ly1*lz1)

      real sij(lx1*ly1*lz1,nij,lelv), oij(lx1*ly1*lz1,lelv,3)
      real u  (lx1*ly1*lz1,lelv)
      real v  (lx1*ly1*lz1,lelv)
      real w  (lx1*ly1*lz1,lelv)

      real ur(lxyz),us(lxyz),ut(lxyz)
     $    ,vr(lxyz),vs(lxyz),vt(lxyz)
     $    ,wr(lxyz),ws(lxyz),wt(lxyz)

      real j ! Inverse Jacobian

      n    = lx1-1      ! Polynomial degree
      nxyz = lx1*ly1*lz1

      if (if3d) then     ! 3D CASE
       do e=1,nelv
        call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
        call local_grad3(vr,vs,vt,v,N,e,dxm1,dxtm1)
        call local_grad3(wr,ws,wt,w,N,e,dxm1,dxtm1)

        do i=1,nxyz

         j = jacmi(i,e)

         sij(i,1,e) = j*  ! du/dx + du/dx
     $   2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e)+ut(i)*txm1(i,1,1,e))

         sij(i,2,e) = j*  ! dv/dy + dv/dy
     $   2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e)+vt(i)*tym1(i,1,1,e))

         sij(i,3,e) = j*  ! dw/dz + dw/dz
     $   2*(wr(i)*rzm1(i,1,1,e)+ws(i)*szm1(i,1,1,e)+wt(i)*tzm1(i,1,1,e))

         sij(i,4,e) = j*  ! du/dy + dv/dx
     $   (ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e)+ut(i)*tym1(i,1,1,e) +
     $    vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e)+vt(i)*txm1(i,1,1,e) )
         oij(i,e,3) = j*  ! dv/dx - du/dy
     $   (vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e)+vt(i)*txm1(i,1,1,e) -
     $    ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e)+ut(i)*tym1(i,1,1,e) )

         sij(i,5,e) = j*  ! dv/dz + dw/dy
     $   (wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e)+wt(i)*tym1(i,1,1,e) +
     $    vr(i)*rzm1(i,1,1,e)+vs(i)*szm1(i,1,1,e)+vt(i)*tzm1(i,1,1,e) )
         oij(i,e,1) = j*  ! dw/dy - dv/dz
     $   (wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e)+wt(i)*tym1(i,1,1,e) -
     $    vr(i)*rzm1(i,1,1,e)+vs(i)*szm1(i,1,1,e)+vt(i)*tzm1(i,1,1,e) )

         sij(i,6,e) = j*  ! du/dz + dw/dx
     $   (ur(i)*rzm1(i,1,1,e)+us(i)*szm1(i,1,1,e)+ut(i)*tzm1(i,1,1,e) +
     $    wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e)+wt(i)*txm1(i,1,1,e) )
         oij(i,e,2) = j*  ! du/dz - dw/dx
     $   (ur(i)*rzm1(i,1,1,e)+us(i)*szm1(i,1,1,e)+ut(i)*tzm1(i,1,1,e) -
     $    wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e)+wt(i)*txm1(i,1,1,e) )


        enddo
       enddo

      elseif (ifaxis) then  ! AXISYMMETRIC CASE  

c
c        Notation:                       ( 2  x  Acheson, p. 353)
c                     Cylindrical
c            Nek5k    Coordinates
c
c              x          z
c              y          r
c              z          theta
c

         do e=1,nelv
            call setaxdy ( ifrzer(e) )  ! change dytm1 if on-axis
            call local_grad2(ur,us,u,N,e,dxm1,dytm1)
            call local_grad2(vr,vs,v,N,e,dxm1,dytm1)
            call local_grad2(wr,ws,w,N,e,dxm1,dytm1)

            do i=1,nxyz
               j = jacmi(i,e)
               r = ym1(i,1,1,e)                              ! Cyl. Coord:

               sij(i,1,e) = j*  ! du/dx + du/dx              ! e_zz
     $           2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e))

               sij(i,2,e) = j*  ! dv/dy + dv/dy              ! e_rr
     $           2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))

               if (r.gt.0) then                              ! e_@@
                  sij(i,3,e) = 2.*v(i,e)/r  ! v / r  ! corrected AT: factor of 2, 10/30/18
               else
                  sij(i,3,e) = j*  ! L'Hopital's rule: e_@@ = 2dv/dr
     $            2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))
               endif

               sij(i,4,e) = j*  ! du/dy + dv/dx             ! e_zr
     $            ( ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) +
     $              vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) )
               oij(i,e,3) = j*  ! dv/dx - du/dy
     $            ( vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) -
     $              ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) )

               if (r.gt.0) then                             ! e_r@
                  sij(i,5,e) = j*  ! dw/dy 
     $              ( wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e) )
     $              - w(i,e) / r
                  oij(i,e,1) = j*  ! dw/dy 
     $              ( wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e) )
     $              + w(i,e) / r
               else
                  sij(i,5,e) = 0
                  oij(i,e,2) = j*  ! L'Hopital's rule: e_r@ = 2dw/dr
     $            2*(wr(i)*rym1(i,1,1,e)+ws(i)*sym1(i,1,1,e))
               endif

               sij(i,6,e) = j*  ! dw/dx                     ! e_@z
     $            ( wr(i)*rxm1(i,1,1,e)+ws(i)*sxm1(i,1,1,e) )
               oij(i,e,2) = j*  !-dw/dx
     $            (-wr(i)*rxm1(i,1,1,e)-ws(i)*sxm1(i,1,1,e) )

            enddo
         enddo

      else              ! 2D CASE

         do e=1,nelv
            call local_grad2(ur,us,u,N,e,dxm1,dxtm1)
            call local_grad2(vr,vs,v,N,e,dxm1,dxtm1)

            do i=1,nxyz
               j = jacmi(i,e)

               sij(i,1,e) = j*  ! du/dx + du/dx
     $           2*(ur(i)*rxm1(i,1,1,e)+us(i)*sxm1(i,1,1,e))
               oij(i,e,2) = 0.

               sij(i,2,e) = j*  ! dv/dy + dv/dy
     $           2*(vr(i)*rym1(i,1,1,e)+vs(i)*sym1(i,1,1,e))
               oij(i,e,3) = 0.

               sij(i,3,e) = j*  ! du/dy + dv/dx
     $           (ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) +
     $            vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) )
               oij(i,e,1) = j*  ! dv/dx - du/dy
     $           (vr(i)*rxm1(i,1,1,e)+vs(i)*sxm1(i,1,1,e) -
     $            ur(i)*rym1(i,1,1,e)+us(i)*sym1(i,1,1,e) )

            enddo
         enddo
      endif

      if(iflomach .and. iflmc) then

        onethird = 1./3.
        if(if3d .or. ifaxis) then
          do e=1,nelv
            do i=1,nxyz
             trS = sij(i,1,e) + sij(i,2,e) + sij(i,3,e) ! 2(du/dx + dv/dy + dw/dz or v/r) in axisym
             sij(i,1,e) = sij(i,1,e) - onethird*trS     ! 2S - (2/3)Q = S'-(1/3)tr(S')
            enddo
          enddo
        else
          do e=1,nelv
            do i=1,nxyz
             trS = sij(i,1,e) + sij(i,2,e)              ! 2(du/dx + dv/dy)
             sij(i,1,e) = sij(i,1,e) - onethird*trS     ! 2S - (2/3)Q = S'-(1/3)tr(S')
            enddo
           enddo
        endif

      endif

      ifielt = ifield
      ifield = 1
      if(ifdss) call dssum_sij_oij(sij, oij, nij)
      ifield = ifielt

      return
      end
c-----------------------------------------------------------------------
      subroutine dssum_sij_oij(sij,oij,nij)
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e
c
      parameter(lt  =lx1*ly1*lz1*lelv)
      parameter(lxyz=lx1*ly1*lz1)

      real sij(lx1*ly1*lz1,nij,lelv), oij(lx1*ly1*lz1,lelv,3)

      common /scrsij/  work1(lx1*ly1*lz1,lelv), work2(lx1*ly1*lz1,lelv)
     $                                        , work3(lx1*ly1*lz1,lelv)

      nxyz = lx1*ly1*lz1
      ntot = nxyz*nelv

      if (if3d .or. ifaxis) then
         call opcolv  (oij(1,1,1), oij(1,1,2), oij(1,1,3),   bm1)
         call opdssum (oij(1,1,1), oij(1,1,2), oij(1,1,3)       )
         call opcolv  (oij(1,1,1), oij(1,1,2), oij(1,1,3),binvm1)
         do e=1,nelv
            call copy (work1(1,e), sij(1,1,e),nxyz)
            call copy (work2(1,e), sij(1,2,e),nxyz)
            call copy (work3(1,e), sij(1,3,e),nxyz)
         enddo
         call opcolv  (work1, work2, work3,   bm1)
         call opdssum (work1, work2, work3       )
         call opcolv  (work1, work2, work3,binvm1)
         do e=1,nelv
            call copy (sij(1,1,e),work1(1,e), nxyz)
            call copy (sij(1,2,e),work2(1,e), nxyz)
            call copy (sij(1,3,e),work3(1,e), nxyz)
         enddo
         do e=1,nelv
            call copy (work1(1,e), sij(1,4,e),nxyz)
            call copy (work2(1,e), sij(1,5,e),nxyz)
            call copy (work3(1,e), sij(1,6,e),nxyz)
         enddo
         call opcolv  (work1, work2, work3,   bm1)
         call opdssum (work1, work2, work3       )
         call opcolv  (work1, work2, work3,binvm1)
         do e=1,nelv
            call copy (sij(1,4,e),work1(1,e), nxyz)
            call copy (sij(1,5,e),work2(1,e), nxyz)
            call copy (sij(1,6,e),work3(1,e), nxyz)
         enddo
      else
         call col2    (oij(1,1,1),bm1   ,ntot)
         call dssum   (oij(1,1,1),lx1,ly1,lz1)
         call col2    (oij(1,1,1),binvm1,ntot)
         do e=1,nelv
            call copy (work1(1,e), sij(1,1,e),nxyz)
            call copy (work2(1,e), sij(1,2,e),nxyz)
            call copy (work3(1,e), sij(1,3,e),nxyz)
         enddo
         call opcolv  (work1, work2, work3,   bm1)
         call opdssum (work1, work2, work3       )
         call opcolv  (work1, work2, work3,binvm1)
         do e=1,nelv
            call copy (sij(1,1,e),work1(1,e), nxyz)
            call copy (sij(1,2,e),work2(1,e), nxyz)
            call copy (sij(1,3,e),work3(1,e), nxyz)
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine comp_sij_oij_mag2(sij,oij,nij,St_mag2,Om_mag2,OiOjSk)
c
c     Compute the square of the magnitude of the stress and rotation tensors
c     St_mag2=2Sij*Sij=S'ij*S'ij/2, S'ij=2Sij, Sij=(dui/dxj+duj/dxi)/2
c     Om_mag2=2Oij*Oij=O'ij*O'ij/2, O'ij=OSij, Oij=(dui/dxj-duj/dxi)/2
c
      include 'SIZE'
      include 'TOTAL'
c
      integer e
c
      parameter(lt  =lx1*ly1*lz1*lelv)
      parameter(lxyz=lx1*ly1*lz1)

      real sij(lx1*ly1*lz1,nij,lelv), oij(lx1*ly1*lz1,lelv,3)

      real              St_mag2(lx1*ly1*lz1,lelv)
     $                , Om_mag2(lx1*ly1*lz1,lelv)
     $                , OiOjSk (lx1*ly1*lz1,lelv)

      common /scrsij/  work1(lx1*ly1*lz1,lelv), work2(lx1*ly1*lz1,lelv)
     $                                        , work3(lx1*ly1*lz1,lelv)
      real tmp1(lxyz), tmp2(lxyz), tmp3(lxyz)

      nxyz    = lx1*ly1*lz1
      ntot    = nxyz*nelv
      two     = 2.
      onehalf = 0.5
      onethird= 1./3.
      oneeight= 1./8.
      beta    = onehalf-onethird

      if (if3d .or. ifaxis) then     ! 3D CASE or axisymmetric

       do e=1,nelv
          call    col3 (St_mag2(1,e), sij(1,1,e), sij(1,1,e),nxyz)
          call addcol3 (St_mag2(1,e), sij(1,2,e), sij(1,2,e),nxyz)
          call addcol3 (St_mag2(1,e), sij(1,3,e), sij(1,3,e),nxyz)
          call    col3 (work1  (1,e), sij(1,4,e), sij(1,4,e),nxyz)
          call addcol3 (work1  (1,e), sij(1,5,e), sij(1,5,e),nxyz)
          call addcol3 (work1  (1,e), sij(1,6,e), sij(1,6,e),nxyz)
          call  add2s2 (St_mag2(1,e), work1(1,e), two,       nxyz)
c
          call    col3 (work1  (1,e), oij(1,e,1), oij(1,e,1),nxyz)
          call    col3 (work2  (1,e), oij(1,e,2), oij(1,e,2),nxyz)
          call    col3 (work3  (1,e), oij(1,e,3), oij(1,e,3),nxyz)
          call    add4 (Om_mag2(1,e), work1(1,e), work2(1,e)
     $                                          , work3(1,e),nxyz)

          call    add3 (tmp1, work2(1,e), work3(1,e), nxyz)
          call    add3 (tmp2, work1(1,e), work3(1,e), nxyz)
          call    add3 (tmp3, work1(1,e), work2(1,e), nxyz)
          call    col3 (OiOjSk(1,e),tmp1, sij(1,1,e), nxyz)
          call addcol3 (OiOjSk(1,e),tmp2, sij(1,2,e), nxyz)
          call addcol3 (OiOjSk(1,e),tmp3, sij(1,3,e), nxyz)

          call    col4 (tmp1, oij(1,e,1), oij(1,e,2), sij(1,4,e), nxyz)
          call addcol4 (tmp1, oij(1,e,2), oij(1,e,3), sij(1,5,e), nxyz)
          call subcol4 (tmp1, oij(1,e,1), oij(1,e,3), sij(1,6,e), nxyz)
          call add2s2  (OiOjSk(1,e), tmp1, two,                   nxyz)

       enddo
           
      else              ! 2D CASE

       do e=1,nelv
          call    col3 (St_mag2(1,e), sij(1,1,e), sij(1,1,e),nxyz)
          call addcol3 (St_mag2(1,e), sij(1,2,e), sij(1,2,e),nxyz)
          call    col3 (work1  (1,e), sij(1,3,e), sij(1,3,e),nxyz)
          call  add2s2 (St_mag2(1,e), work1(1,e), two,       nxyz)
c
          call    col3 (Om_mag2(1,e), oij(1,e,1), oij(1,e,1),nxyz)
          call   rzero (OiOjSk (1,e),                        nxyz)
       enddo

      endif

      call  cmult (St_mag2, onehalf, ntot) ! St_mag2=2*Sij*Sij=S'ij*S'ij/2

      return
      end
c-----------------------------------------------------------------------
