c=======================================================================
c Actuation power of the wall transpiration (hydrogym.power interface)
c
c At every GLL node of the actuated wall the power put into the fluid
c is (Bewley, Moin & Temam 2001; Hasegawa & Kasagi 2011, App. A)
c
c     pi_p = v_n (p - p_ref),     pi_k = rho v_n |u_w|^2 / 2,
c
c with u_w the imposed wall velocity, v_n = u_w . n (n into the
c fluid, v_n > 0 = blowing) and p_ref the instantaneous spanwise mean
c of the wall pressure at each (x, y) station. The cost forms are
c taken per node before the surface quadrature (weights: area):
c
c     signed   = sum w (pi_p + pi_k)
c     positive = sum w (max(pi_p, 0) + max(pi_k, 0))
c     abs      = sum w (|pi_p| + |pi_k|)
c
c PARAM(96) (HydroGym power section):
c     0  off,
c     1  sampled after every solver step of an action (power_sampling
c        mean, time-step weighted) and at its end,
c     2  at the end of an action only (power_sampling end).
c The values are sent on the request POWER. They are dimensional;
c hydrogym.nek divides them by the reference power 0.5 rho U^3 A.
c=======================================================================

c-----------------------------------------------------------------------
      subroutine drl_power_kernel(n,w,vn,u2,p,pref,rho,s,pn)
c Cost forms of n surface nodes; no solver state (unit-tested).
c s(1) signed, s(2) positive, s(3) abs, s(4) signed with the full
c pressure (p_ref = 0), s(5) area, s(6) wall flux sum w v_n (> 0:
c net blowing); pn(i) = signed node power w*pi.
      implicit none
      integer n,i,k
      real w(n),vn(n),u2(n),p(n),pref(n),rho,s(6),pn(n)
      real pip,pik
      do k=1,6
         s(k)=0.0
      enddo
      do i=1,n
         pip=vn(i)*(p(i)-pref(i))
         pik=0.5*rho*vn(i)*u2(i)
         s(1)=s(1)+w(i)*(pip+pik)
         s(2)=s(2)+w(i)*(max(pip,0.0)+max(pik,0.0))
         s(3)=s(3)+w(i)*(abs(pip)+abs(pik))
         s(4)=s(4)+w(i)*(vn(i)*p(i)+pik)
         s(5)=s(5)+w(i)
         s(6)=s(6)+w(i)*vn(i)
         pn(i)=w(i)*(pip+pik)
      enddo
      return
      end

c-----------------------------------------------------------------------
      logical function drl_power_face(f,e)
c Faces of the power integral: velocity faces on the wing (all wing
c faces of the case carry the velocity condition; nodes without
c actuation have u_w = 0 and add nothing). The far-field velocity
c faces lie outside the box.
      implicit none
      include 'SIZE'
      include 'TOTAL'
      integer f,e,kx1,kx2,ky1,ky2,kz1,kz2,ix,iy,iz
      drl_power_face=.false.
      if (cbc(f,e,1).ne.'v  ') return
      call facind(kx1,kx2,ky1,ky2,kz1,kz2,lx1,ly1,lz1,f)
      do iz=kz1,kz2
      do iy=ky1,ky2
      do ix=kx1,kx2
         if (xm1(ix,iy,iz,e).lt.-0.05.or.xm1(ix,iy,iz,e).gt.1.05)
     $      return
         if (abs(ym1(ix,iy,iz,e)).gt.0.3) return
      enddo
      enddo
      enddo
      drl_power_face=.true.
      return
      end

c-----------------------------------------------------------------------
      subroutine drl_power_inst(s,pwagt)
c Instantaneous power: global totals s(6) (see drl_power_kernel) and
c the signed power of every agent of this rank: the power of its GLL
c node (sum over the face copies of the node), shared equally by the
c agents at that node (an agent is one element-local copy of a wall
c node), so that the agents sum to the power of the agents' nodes.
c Collective (all ranks).
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'
      real s(6),pwagt(totctrl)
      real pwpm(lx1,ly1,lz1,lelv),pwpref(lx1,ly1,lz1,lelv)
      real pwnod(lx1,ly1,lz1,lelv),pwavz(lx1,ly1,xnel,ynel)
      real pwwa(lx1,ly2,lz2),pwwb(lx1,ly1,lz2)
      real pwfw(lx1*lz1),pwfvn(lx1*lz1),pwfu2(lx1*lz1)
      real pwfp(lx1*lz1),pwfpr(lx1*lz1),pwfpn(lx1*lz1)
      real pwsl(6),pwux,pwuy,pwuz
      integer e,f,i,nf,k,il,ie,ix,iy,iz,ntot,iface
      integer js1,jf1,jskip1,js2,jf2,jskip2,j1,j2
      real pwcnt(lx1,ly1,lz1,lelv)
      logical drl_power_face,pwcnt_ok
      save pwpm,pwpref,pwnod,pwavz,pwcnt,pwcnt_ok
      data pwcnt_ok /.false./
      ntot=lx1*ly1*lz1*nelv

c     wall pressure on the velocity mesh and its spanwise mean
      call mappr(pwpm,pr,pwwa,pwwb)
      call copy(pwpref,pwpm,ntot)
      call z_averaging(pwpref,pwavz)
      call z_avg_reshape(pwpref,pwavz)

      call rzero(pwnod,ntot)
      do k=1,6
         s(k)=0.0
      enddo
      call dsset(lx1,ly1,lz1)
      do e=1,nelv
      do f=1,2*ldim
         if (drl_power_face(f,e)) then
            iface =eface1(f)
            js1   =skpdat(1,iface)
            jf1   =skpdat(2,iface)
            jskip1=skpdat(3,iface)
            js2   =skpdat(4,iface)
            jf2   =skpdat(5,iface)
            jskip2=skpdat(6,iface)
            i=0
            do j2=js2,jf2,jskip2
            do j1=js1,jf1,jskip1
               i=i+1
               pwux=vx(j1,j2,1,e)
               pwuy=vy(j1,j2,1,e)
               pwuz=vz(j1,j2,1,e)
               pwfw(i) =area(i,1,f,e)
c              unx: outward normal of the fluid element (into the wing)
               pwfvn(i)=-(pwux*unx(i,1,f,e)+pwuy*uny(i,1,f,e)
     $                 +pwuz*unz(i,1,f,e))
               pwfu2(i)=pwux*pwux+pwuy*pwuy+pwuz*pwuz
               pwfp(i) =pwpm(j1,j2,1,e)
               pwfpr(i)=pwpref(j1,j2,1,e)
            enddo
            enddo
            nf=i
            call drl_power_kernel(nf,pwfw,pwfvn,pwfu2,pwfp,pwfpr,
     $                            param(1),pwsl,pwfpn)
            do k=1,6
               s(k)=s(k)+pwsl(k)
            enddo
            i=0
            do j2=js2,jf2,jskip2
            do j1=js1,jf1,jskip1
               i=i+1
               pwnod(j1,j2,1,e)=pwnod(j1,j2,1,e)+pwfpn(i)
            enddo
            enddo
         endif
      enddo
      enddo
      call gop(s,pwsl,'+  ',6)

c     number of agents at each GLL node (once)
      ifield=1
      if (.not.pwcnt_ok) then
         call rzero(pwcnt,ntot)
         do il=1,numctrl
            ie=gllel(info_agt(1,il))
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            pwcnt(ix,iy,iz,ie)=pwcnt(ix,iy,iz,ie)+1.0
         enddo
         call dssum(pwcnt,lx1,ly1,lz1)
         pwcnt_ok=.true.
      endif
c     node power of the agents: sum over all copies of the GLL node
      call dssum(pwnod,lx1,ly1,lz1)
      call rzero(pwagt,totctrl)
      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         pwagt(il)=pwnod(ix,iy,iz,ie)/max(pwcnt(ix,iy,iz,ie),1.0)
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine drl_power_reset
c Close an open action interval (new episode).
      implicit none
      include 'SIZE'
      include 'DRL'
      include 'DRL_POWER'
      pw_open=.false.
      return
      end

c-----------------------------------------------------------------------
      subroutine drl_power_sample(i_evolv)
c Every-step sampling (PARAM(96) = 1), called after drl_reward. The
c call with i_evolv = 1 comes before the first solver step of the
c action and opens the interval; the calls 2..ndrl sample the states
c after the solver steps 1..ndrl-1, the request POWER after the last.
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'
      include 'DRL_POWER'
      integer i_evolv,k,il
      real s(6),pwagt(totctrl),pwwt
      if (nint(param(96)).ne.1) return
      if (i_evolv.eq.1) then
         do k=1,6
            pw_sum(k)=0.0
         enddo
         call rzero(pw_agsum,totctrl)
         pw_wsum=0.0
         pw_nsamp=0
         pw_open=.true.
         return
      endif
      if (.not.pw_open) return
      call drl_power_inst(s,pwagt)
      pwwt=abs(dt)
      do k=1,6
         pw_sum(k)=pw_sum(k)+pwwt*s(k)
      enddo
      do il=1,numctrl
         pw_agsum(il)=pw_agsum(il)+pwwt*pwagt(il)
      enddo
      pw_wsum=pw_wsum+pwwt
      pw_nsamp=pw_nsamp+1
      return
      end

c-----------------------------------------------------------------------
      subroutine drl_power_out
c Request POWER: instantaneous power now (end of the action) and, if an
c interval was sampled (PARAM(96) = 1), its time-step weighted mean.
c NID 0 sends pwhdr(16) with tag 1995:
c   1-5 end (signed, positive, abs, signed full pressure, area),
c   6-10 mean (same order), 11 mean available (1/0), 12 samples,
c   13 sum of the weights, 14 time, 15/16 wall flux end/mean.
c Every rank with agents sends its agents' signed power, end (tag
c 110000+NID) and mean (tag 120000+NID), TOTCTRL values each.
      implicit none
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'
      include 'DRL_POWER'
      include 'mpif.h'
      real s(6),pwagt(totctrl),pwmean(totctrl),pwhdr(16),pwwt
      integer k,il,ierr
      call drl_power_inst(s,pwagt)
      call rzero(pwhdr,16)
      call rzero(pwmean,totctrl)
      do k=1,5
         pwhdr(k)=s(k)
      enddo
      pwhdr(15)=s(6)
      if (pw_open.and.nint(param(96)).eq.1) then
         pwwt=abs(dt)
         do k=1,6
            pw_sum(k)=pw_sum(k)+pwwt*s(k)
         enddo
         do il=1,numctrl
            pw_agsum(il)=pw_agsum(il)+pwwt*pwagt(il)
         enddo
         pw_wsum=pw_wsum+pwwt
         pw_nsamp=pw_nsamp+1
         do k=1,5
            pwhdr(5+k)=pw_sum(k)/pw_wsum
         enddo
         do il=1,numctrl
            pwmean(il)=pw_agsum(il)/pw_wsum
         enddo
         pwhdr(16)=pw_sum(6)/pw_wsum
         pwhdr(11)=1.0
         pwhdr(12)=pw_nsamp
         pwhdr(13)=pw_wsum
         pw_open=.false.
      endif
      pwhdr(14)=time
      if (nid.eq.0) then
         call MPI_SEND(pwhdr,16,MPI_DOUBLE_PRECISION,0,1995,
     $                 DRL_COMM,ierr)
      endif
      if (numctrl.ne.0) then
         call MPI_SEND(pwagt,totctrl,MPI_DOUBLE_PRECISION,0,
     $                 nid+110000,DRL_COMM,ierr)
         call MPI_SEND(pwmean,totctrl,MPI_DOUBLE_PRECISION,0,
     $                 nid+120000,DRL_COMM,ierr)
      endif
      if (nid.eq.0) write(6,'(a,1p6e13.5)')
     $   ' [POWER] end signed/pos/abs, mean signed/pos/abs',
     $   pwhdr(1),pwhdr(2),pwhdr(3),pwhdr(6),pwhdr(7),pwhdr(8)
      return
      end
c-----------------------------------------------------------------------
