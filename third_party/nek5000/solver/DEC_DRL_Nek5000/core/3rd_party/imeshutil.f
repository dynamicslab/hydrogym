c random utilities to help with iMesh
c called by MYASSERT macro
#define NULLSTRIP(s) s(:index(s, '\0')-1)

c building without imesh for now

      subroutine imesh_err(ierr, imesh, file, line)

      implicit none
      integer ierr, line
      integer*8 imesh
      character*(*) file
      character*1024 errmsg
      integer iup
      iup = index(file, " ")
      print *, "ASSERT ERROR: ", ierr, 'in ', file(1:iup), 
     * ' line', line
      call iMesh_getDescription(%VAL(imesh), errmsg, ierr)
      print *, NULLSTRIP(errmsg)

#ifndef NDEBUG
      call exitt
#endif

      return
      end 


