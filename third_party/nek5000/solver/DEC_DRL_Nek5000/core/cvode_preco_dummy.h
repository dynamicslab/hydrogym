      subroutine fcvpsol(tcv,y,fy,r,z,gam,delta,lr,ipar,rpar,w,ier)

      real tcv,y(*),fy(*),r(*),z(*),gam,delta,rpar(*),w(*)
      integer*8 ipar(1)
      integer lr, ier

      return
      end
c----------------------------------------------------------------------
      subroutine fcvpset(tcv,y,fy,jok,jcur,gam,h,ipar,rpar,w1,w2,w3,ier)

      real tcv,y(*),fy(*),gam,h,rpar(*),w1(*),w2(*),w3(*)
      integer jok,jcur
      integer*8 ipar(*)
      integer ier

      return
      end
