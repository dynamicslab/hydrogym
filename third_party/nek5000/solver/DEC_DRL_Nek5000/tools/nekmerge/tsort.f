c-----------------------------------------------------------------------
      subroutine ituple_sort(a,lda,n,key,nkey,ind,aa)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer a(lda,n),aa(lda)
      integer ind(1),key(nkey)
      logical iftuple_ialtb
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
c           aa  = a  (l)
            call icopy(aa,a(1,l),lda)
            ii  = ind(l)
         else
c           aa =   a(ir)
            call icopy(aa,a(1,ir),lda)
            ii = ind(ir)
c           a(ir) =   a( 1)
            call icopy(a(1,ir),a(1,1),lda)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
c              a(1) = aa
               call icopy(a(1,1),aa,lda)
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if (iftuple_ialtb(a(1,j),a(1,j+1),key,nkey)) j=j+1
            endif
            if (iftuple_ialtb(aa,a(1,j),key,nkey)) then
c              a(i) = a(j)
               call icopy(a(1,i),a(1,j),lda)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
c        a(i) = aa
         call icopy(a(1,i),aa,lda)
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      subroutine tuple_sort(a,lda,n,key,nkey,ind,aa)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      real a(lda,n),aa(lda)
      integer ind(1),key(nkey)
      logical iftuple_altb
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
c           aa  = a  (l)
            call copy(aa,a(1,l),lda)
            ii  = ind(l)
         else
c           aa =   a(ir)
            call copy(aa,a(1,ir),lda)
            ii = ind(ir)
c           a(ir) =   a( 1)
            call copy(a(1,ir),a(1,1),lda)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
c              a(1) = aa
               call copy(a(1,1),aa,lda)
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
c              if ( a(j).lt.a(j+1) ) j=j+1
               if (iftuple_altb(a(1,j),a(1,j+1),key,nkey)) j=j+1
            endif
c           if (aa.lt.a(j)) then
            if (iftuple_altb(aa,a(1,j),key,nkey)) then
c              a(i) = a(j)
               call copy(a(1,i),a(1,j),lda)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
c        a(i) = aa
         call copy(a(1,i),aa,lda)
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      logical function iftuple_ialtb(a,b,key,nkey)
      integer a(1),b(1)
      integer key(nkey)
c
      do i=1,nkey
         k=key(i)
         if (a(k).lt.b(k)) then
            iftuple_ialtb = .true.
            return
         elseif (a(k).gt.b(k)) then
            iftuple_ialtb = .false.
            return
         endif
      enddo
      iftuple_ialtb = .false.
      return
      end
c-----------------------------------------------------------------------
      logical function iftuple_altb(a,b,key,nkey)
      real a(1),b(1)
      integer key(nkey)
c
      do i=1,nkey
         k=key(i)
         if (a(k).lt.b(k)) then
            iftuple_altb = .true.
            return
         elseif (a(k).gt.b(k)) then
            iftuple_altb = .false.
            return
         endif
      enddo
      iftuple_altb = .false.
      return
      end
c-----------------------------------------------------------------------
      logical function iftuple_ianeb(a,b,key,nkey)
      integer a(1),b(1)
      integer key(nkey)
c
      iftuple_ianeb = .false.
      do i=1,nkey
         k=key(i)
         if (a(k).ne.b(k)) then
            iftuple_ianeb = .true.
            return
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      logical function iftuple_iaeqb(a,b,key,nkey)
      integer a(1),b(1)
      integer key(nkey)
c
      iftuple_iaeqb = .true.
      do i=1,nkey
         k=key(i)
         if (a(k).ne.b(k)) then
            iftuple_iaeqb = .false.
            return
         endif
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine tuple_swapt_ip(x,m,n,p,t1,t2)
      real x(m,n),t1(m),t2(m)
      integer p(1)
c
c     In-place permutation: x'(p) = x
c

      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            loop_start = k
            next       = p(loop_start)
            call copy(t1,x(1,loop_start),m)
            do j=1,n
               if (next.lt.0) then
                  write(6,*) 'Hey! iswapt_ip problem.',j,k,n,next
                  call exitt(1)
               elseif (next.eq.loop_start) then
                  call copy(x(1,next),t1,m)
                  p(next) = -p(next)
                  goto 10
               else
                  call copy(t2,x(1,next),m)
                  call copy(x(1,next),t1,m)
                  call copy(t1,t2       ,m)
                  nextp   =  p(next)
                  p(next) = -p(next)
                  next    =  nextp
               endif
            enddo
   10       continue
         endif
      enddo
c
      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine isort(a,ind,n)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer a(1),ind(1)
      integer aa
C
      dO 10 j=1,n
         ind(j)=j
   10 continue
C
      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
            aa  = a  (l)
            ii  = ind(l)
         else
                 aa =   a(ir)
                 ii = ind(ir)
              a(ir) =   a( 1)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
                 a(1) = aa
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if ( a(j).lt.a(j+1) ) j=j+1
            endif
            if (aa.lt.a(j)) then
                 a(i) = a(j)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
           a(i) = aa
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      subroutine iswap_ip(x,p,n)
      integer x(1),xstart
      integer p(1)
c
c     In-place permutation: x' = x(p)
c
      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            xstart     = x(k)
            loop_start = k
            last       = k
            do j=k,n
               next    = p(last)
               if (next.lt.0) then
                  write(6,*) 'Hey! iswap_ip problem.',j,k,n,next
                  call exitt(0)
               elseif (next.eq.loop_start) then
                  x(last) = xstart
                  p(last) = -p(last)
                  goto 10
               else
                  x(last) = x(next)
                  p(last) = -p(last)
                  last    = next
               endif
            enddo
   10       continue
         endif
      enddo
c
      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine iswapt_ip(x,p,n)
      integer x(1),t1,t2
      integer p(1)
c
c     In-place permutation: x'(p) = x
c

      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            loop_start = k
            next       = p(loop_start)
            t1         = x(loop_start)
            do j=1,n
               if (next.lt.0) then
                  write(6,*) 'Hey! iswapt_ip problem.',j,k,n,next
                  call exitt(1)
               elseif (next.eq.loop_start) then
                  x(next) = t1
                  p(next) = -p(next)
                  goto 10
               else
                  t2      =  x(next)
                  x(next) =  t1
                  t1      =  t2
                  nextp   =  p(next)
                  p(next) = -p(next)
                  next    =  nextp
               endif
            enddo
   10       continue
         endif
      enddo
c
      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
