        subroutine mxm_bgq_8(a,n1,b,n2,c,n3)
          implicit none

          integer  n1, n2, n3
          real(8)  a(n1,n2),b(n2,n3)
          real(8)  c(n1,n3)

          integer i, j
          vector(real(8)) av1, av2, av3, av4, av5, av6, av7, av8
          vector(real(8)) bv1, bsv1, bsv2, bsv3, bsv4
          vector(real(8)) bv2, bsv5, bsv6, bsv7, bsv8
          vector(real(8)) cv

          call alignx(32, a(1,1))
          call alignx(32, b(1,1))
          call alignx(32, c(1,1))

          do i = 1, n1, 4
            av1 = vec_ld(0,a(i,1))
            av2 = vec_ld(0,a(i,2))
            av3 = vec_ld(0,a(i,3))
            av4 = vec_ld(0,a(i,4))
            av5 = vec_ld(0,a(i,5))
            av6 = vec_ld(0,a(i,6))
            av7 = vec_ld(0,a(i,7))
            av8 = vec_ld(0,a(i,8))

            do j = 1, n3
              bv1 = vec_ld(0,b(1,j))
              bv2 = vec_ld(0,b(5,j))
              bsv1 = vec_splat(bv1, 0)
              bsv2 = vec_splat(bv1, 1)
              bsv3 = vec_splat(bv1, 2)
              bsv4 = vec_splat(bv1, 3)
              bsv5 = vec_splat(bv2, 0)
              bsv6 = vec_splat(bv2, 1)
              bsv7 = vec_splat(bv2, 2)
              bsv8 = vec_splat(bv2, 3)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)
              cv =  vec_madd(av7, bsv7, cv)
              cv =  vec_madd(av8, bsv8, cv)

              call vec_st(cv, 0, c(i,j))
            end do
          end do
          return
        end subroutine mxm_bgq_8

        subroutine mxm_bgq_16(a,n1,b,n2,c,n3)
          implicit none

          integer n1, n2, n3
          real(8) a(n1,n2),b(n2,n3)
          real(8) c(n1,n3)

          integer i, j

          vector(real(8)) av1, av2, av3, av4, av5, av6, av7, av8
          vector(real(8)) av9, av10, av11, av12, av13, av14, av15, av16
          vector(real(8)) bv1, bsv1, bsv2, bsv3, bsv4
          vector(real(8)) bv2, bsv5, bsv6, bsv7, bsv8
          vector(real(8)) bv3, bsv9, bsv10, bsv11, bsv12
          vector(real(8)) bv4, bsv13, bsv14, bsv15, bsv16

          vector(real(8)) cv

          call alignx(32, a(1,1))
          call alignx(32, b(1,1))
          call alignx(32, c(1,1))

          do i = 1, n1, 4
            av1 = vec_ld(0,a(i,1))
            av2 = vec_ld(0,a(i,2))
            av3 = vec_ld(0,a(i,3))
            av4 = vec_ld(0,a(i,4))
            av5 = vec_ld(0,a(i,5))
            av6 = vec_ld(0,a(i,6))
            av7 = vec_ld(0,a(i,7))
            av8 = vec_ld(0,a(i,8))
            av9 = vec_ld(0,a(i,9))
            av10 = vec_ld(0,a(i,10))
            av11 = vec_ld(0,a(i,11))
            av12 = vec_ld(0,a(i,12))
            av13 = vec_ld(0,a(i,13))
            av14 = vec_ld(0,a(i,14))
            av15 = vec_ld(0,a(i,15))
            av16 = vec_ld(0,a(i,16))

            do j = 1, n3
              bv1 = vec_ld(0,b(1,j))
              bv2 = vec_ld(0,b(5,j))
              bv3 = vec_ld(0,b(9,j))
              bv4 = vec_ld(0,b(13,j))

              bsv1 = vec_splat(bv1, 0)
              bsv2 = vec_splat(bv1, 1)
              bsv3 = vec_splat(bv1, 2)
              bsv4 = vec_splat(bv1, 3)
              bsv5 = vec_splat(bv2, 0)
              bsv6 = vec_splat(bv2, 1)
              bsv7 = vec_splat(bv2, 2)
              bsv8 = vec_splat(bv2, 3)
              bsv9 = vec_splat(bv3, 0)
              bsv10 = vec_splat(bv3, 1)
              bsv11 = vec_splat(bv3, 2)
              bsv12 = vec_splat(bv3, 3)
              bsv13 = vec_splat(bv4, 0)
              bsv14 = vec_splat(bv4, 1)
              bsv15 = vec_splat(bv4, 2)
              bsv16 = vec_splat(bv4, 3)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)
              cv =  vec_madd(av7, bsv7, cv)
              cv =  vec_madd(av8, bsv8, cv)
              cv =  vec_madd(av9, bsv9, cv)
              cv =  vec_madd(av10, bsv10, cv)
              cv =  vec_madd(av11, bsv11, cv)
              cv =  vec_madd(av12, bsv12, cv)
              cv =  vec_madd(av13, bsv13, cv)
              cv =  vec_madd(av14, bsv14, cv)
              cv =  vec_madd(av15, bsv15, cv)
              cv =  vec_madd(av16, bsv16, cv)

              call vec_st(cv, 0, c(i,j))
            end do
          end do
          return
        end subroutine mxm_bgq_16


        subroutine mxm_bgq_6(a,n1,b,n2,c,n3)
          implicit none

          integer n1, n2, n3
          real(8) a(n1,n2),b(n2,n3)
          real(8) c(n1,n3)

          integer i, j

          vector(real(8)) av1, av2, av3, av4, av5, av6
          vector(real(8)) bv1, bsv1, bsv2, bsv3, bsv4
          vector(real(8)) bv2, bsv5, bsv6

          vector(real(8)) cv

          call alignx(32, a(1,1))
          call alignx(32, b(1,1))
          call alignx(32, c(1,1))

          do i = 1, n1, 4
            av1 = vec_ld(0,a(i,1))
            av2 = vec_ld(0,a(i,2))
            av3 = vec_ld(0,a(i,3))
            av4 = vec_ld(0,a(i,4))
            av5 = vec_ld(0,a(i,5))
            av6 = vec_ld(0,a(i,6))

            do j = 1, n3, 2
              bv1 = vec_ld(0,b(1,j))
              bv2 = vec_ld(0,b(5,j))

              bsv1 = vec_splat(bv1, 0)
              bsv2 = vec_splat(bv1, 1)
              bsv3 = vec_splat(bv1, 2)
              bsv4 = vec_splat(bv1, 3)
              bsv5 = vec_splat(bv2, 0)
              bsv6 = vec_splat(bv2, 1)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)

              call vec_st(cv, 0, c(i,j))

              bv1 = vec_ld(0,b(9,j))

              bsv1 = vec_splat(bv2, 2)
              bsv2 = vec_splat(bv2, 3)
              bsv3 = vec_splat(bv1, 0)
              bsv4 = vec_splat(bv1, 1)
              bsv5 = vec_splat(bv1, 2)
              bsv6 = vec_splat(bv1, 3)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)

              call vec_st(cv, 0, c(i,j+1))
            end do
          end do
          return
        end subroutine mxm_bgq_6

        subroutine mxm_bgq_10(a,n1,b,n2,c,n3)
          implicit none

          integer n1, n2, n3
          real(8) a(n1,n2),b(n2,n3)
          real(8) c(n1,n3)

          integer i, j

          vector(real(8)) av1, av2, av3, av4, av5, av6, av7, av8
          vector(real(8)) av9, av10
          vector(real(8)) bv1, bsv1, bsv2, bsv3, bsv4
          vector(real(8)) bv2, bsv5, bsv6, bsv7, bsv8
          vector(real(8)) bv3, bsv9, bsv10
          vector(real(8)) cv

          call alignx(32, a(1,1))
          call alignx(32, b(1,1))
          call alignx(32, c(1,1))

          do i = 1, n1, 4
            av1 = vec_ld(0,a(i,1))
            av2 = vec_ld(0,a(i,2))
            av3 = vec_ld(0,a(i,3))
            av4 = vec_ld(0,a(i,4))
            av5 = vec_ld(0,a(i,5))
            av6 = vec_ld(0,a(i,6))
            av7 = vec_ld(0,a(i,7))
            av8 = vec_ld(0,a(i,8))
            av9 = vec_ld(0,a(i,9))
            av10 = vec_ld(0,a(i,10))

            do j = 1, n3, 2
              bv1 = vec_ld(0,b(1,j))
              bv2 = vec_ld(0,b(5,j))
              bv3 = vec_ld(0,b(9,j))

              bsv1 = vec_splat(bv1, 0)
              bsv2 = vec_splat(bv1, 1)
              bsv3 = vec_splat(bv1, 2)
              bsv4 = vec_splat(bv1, 3)
              bsv5 = vec_splat(bv2, 0)
              bsv6 = vec_splat(bv2, 1)
              bsv7 = vec_splat(bv2, 2)
              bsv8 = vec_splat(bv2, 3)
              bsv9 = vec_splat(bv3, 0)
              bsv10 = vec_splat(bv3, 1)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)
              cv =  vec_madd(av7, bsv7, cv)
              cv =  vec_madd(av8, bsv8, cv)
              cv =  vec_madd(av9, bsv9, cv)
              cv =  vec_madd(av10, bsv10, cv)

              call vec_st(cv, 0, c(i,j))

              bv1 = vec_ld(0,b(13,j))
              bv2 = vec_ld(0,b(17,j))

              bsv1 = vec_splat(bv3, 2)
              bsv2 = vec_splat(bv3, 3)
              bsv3 = vec_splat(bv1, 0)
              bsv4 = vec_splat(bv1, 1)
              bsv5 = vec_splat(bv1, 2)
              bsv6 = vec_splat(bv1, 3)
              bsv7 = vec_splat(bv2, 0)
              bsv8 = vec_splat(bv2, 1)
              bsv9 = vec_splat(bv2, 2)
              bsv10 = vec_splat(bv2, 3)

              cv =  vec_mul(av1, bsv1)
              cv =  vec_madd(av2, bsv2, cv)
              cv =  vec_madd(av3, bsv3, cv)
              cv =  vec_madd(av4, bsv4, cv)
              cv =  vec_madd(av5, bsv5, cv)
              cv =  vec_madd(av6, bsv6, cv)
              cv =  vec_madd(av7, bsv7, cv)
              cv =  vec_madd(av8, bsv8, cv)
              cv =  vec_madd(av9, bsv9, cv)
              cv =  vec_madd(av10, bsv10, cv)

              call vec_st(cv, 0, c(i,j+1))
            end do
          end do
          return
        end subroutine mxm_bgq_10

