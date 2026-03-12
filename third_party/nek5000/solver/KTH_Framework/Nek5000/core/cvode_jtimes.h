      subroutine fcvjtimes (v,fjv,tt,y,fy,h,ipar,rpar,work,ier)
c
c     Compute Jacobian Vetor product FJV
c     approximated by 1st-order fd quotient 
c
      real v(*), fjv(*), tt, y(*), fy(*), h, rpar(1), work(*)

      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      INCLUDE 'CVODE'

      integer*8 ipar(1)

      if (nio.eq.0.and.loglevel.gt.2)
     $   write(6,*) 'fcvjtimes'
      
      ifdqj = .true.
  
      ! compute weighted rms norm ||v||
      call fcvgeterrweights(work,ier)
      sum = 0.0
      do i = 1,cv_nlocal
         dnorm = v(i)*work(i)
         sum = sum + dnorm*dnorm
      enddo
      sum = sqrt(glsum(sum,1)/cv_nglobal)
      sig =  1./sum
      sig = cv_sigs * sig

      ! set FJV = f(t, y + sigs*v/||v||)
      do i = 1,cv_nlocal
         work(i) = y(i) + sig*v(i)
      enddo
      call fcvfun(tt,work,fjv,ipar,rpar,ier)

      siginv = 1./sig
      do i = 1,cv_nlocal
         fjv(i) = fjv(i)*siginv - fy(i)*siginv
      enddo

      ifdqj = .false.
      ier = 0

      return
      end
