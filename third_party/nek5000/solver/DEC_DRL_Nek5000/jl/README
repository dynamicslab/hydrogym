
A high-level view of the code in this directory is as follows. See each header
file listed for more documentation.

The following headers are fundamental to most of the code.

  name.h:    a given prefix is added to all external symbols;
             determines how FORTRAN routines are named
  types.h:   defines the integer types used everywhere (e.g., for array indices)
  mem.h:     memory-management wrappers;
             "array" type (generic dynamically sized array);
             "buffer" type ( = char array )
  comm.h:    wrappers for MPI calls (with alternative single proc versions)

The Gather/Scatter library top-level interface is defined in "gs.h".
The file "gs_defs.h" defines the datatypes and operations that it supports.

There are two coarse solvers (XXT and AMG), which are not currently very well
documented. The interface is given in "crs.h". 
 
"findpts" is documented in "findpts.c". The idea is that during a run of an
SEM code, we have a geometry map
  (processor, element, r, s, t) -> (x, y, z)
that defines our mesh. Within each element, the xyz coordinate is a
polynomial function of the parametric r,s,t coordinates.
"findpts" takes a distributed list of "(x,y,z)" points and computes the inverse
of the above map.
"findpts_eval" takes a list of "(proc,el,r,s,t)" coords, e.g., as returned by
  "findpts", and interpolates a given field at each point.


The "workhorses" of the implementations of much of the above are the
"sarray_sort" and "sarray_transfer" routines, documented in the respective
headers. The "array" type, defined in "mem.h", can be used to keep track of a
dynamically sized array of (arbitrary) structs.

  sarray_sort.h:     
    sort an array of structs (locally/sequentially) by one or two of its fields
  sarray_transfer.h:
    transfer each struct in array to the processor specified by a given field
    
These in turn, are implemented using the lower-level routines of
"sort.h", and "crystal_router.h".


The "findpts" algorithm makes use of a number of lower-level routines
possibly useful on their own.

  poly.h:     computation of quadrature nodes; fast polynomial interpolation
  lob_bnd.h:  (relatively) fast yet robust bounds for polynomials on [-1,1]^d
  obbox.h:    oriented as well as axis-aligned bounding boxes for spectral els
  tensor.h:   some tensor-product applications,
                with BLAS ops delegated to Nek, cblas, or a naive imp

All of the preprocessor macros that affect compilation are:
  name.h:  PREFIX="..."    prefix added to all C external symbols
          FPREFIX="..."    prefix added to all FORTRAN routines
    UPCASE, UNDERSCORE   determines FORTRAN naming convention
  types.h: USE_LONG, USE_LONG_LONG, GLOBAL_LONG, GLOBAL_LONG_LONG
           determine the integer types used by all code
  mem.h: PRINT_MALLOCS=1   (print all mem mngmt to stdout)
  comm.h: MPI  (use MPI when defined;
                otherwise, use a dummy single-proc implementation)
  tensor.h: USE_CBLAS, USE_NAIVE_BLAS
            (select BLAS implementation; default is Nek's mxm)
  fail.c: NO_NEK_EXITT    when defined, don't call Nek's exitt routine
  amg.c: AMG_BLOCK_ROWS   number of rows to read at a time (default=1200)
         GS_TIMING        record timings for the matrix multiplies
         GS_BARRIER       use a barrier to improve the quality of the timings
