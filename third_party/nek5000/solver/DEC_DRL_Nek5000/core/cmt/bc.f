      subroutine bcflux(flux)
!JH031615 assuming that ujump has included Dirichlet boundary conditions,
!         the only things the viscous terms need are enforced adiabatic
!         conditions and recognition that the gs_op_fields in viscousf
!         has not affected the boundary faces, which have been
!         premultiplied by 1/2
      include 'SIZE'
      include 'INPUT'
      include 'DG'
      include 'TSTEP' ! wait how do we know what ifield is?
      integer e,eq,f
      real flux(nx1*nz1,2*ndim,nelt,toteq)

      nface=2*ndim
      nxz=nx1*nz1
      ifield=1

      do e=1,nelt
         do f=1,nface
            if (cbc(f, e, ifield).ne.'E  '.and.
     >          cbc(f, e, ifield).ne.'P  ') then ! cbc bndy
               do eq=1,toteq-1
                  call cmult(flux(1,f,e,eq),2.0,nxz)
               enddo
               eq=5
               if (cbc(f,e,ifield) .eq. 'I  ') then
                  call cmt_userflux(flux(1,f,e,eq)) ! replace this with userbc
               else ! v, O and W
                  call cmult(flux(1,f,e,eq),2.0,nxz)
               endif
            endif
         enddo
      enddo

      return
      end
