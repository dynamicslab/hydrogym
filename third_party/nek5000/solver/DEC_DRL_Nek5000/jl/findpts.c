#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "poly.h"
#include "obbox.h"
#include "findpts_el.h"
#include "findpts_local.h"
#include "gs_defs.h"
#include "comm.h"
#include "crystal.h"
#include "sarray_transfer.h"
#include "sort.h"
#include "sarray_sort.h"
/*
#define DIAGNOSTICS
*/
#ifdef DIAGNOSTICS
#include <stdio.h>
#endif

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

struct ulong_range { ulong min, max; };
struct proc_index { uint proc, index; };

static slong lfloor(double x) { return floor(x); }
static slong lceil (double x) { return ceil (x); }

static ulong hash_index_aux(double low, double fac, ulong n, double x)
{
  const slong i = lfloor((x-low)*fac);
  return i<0 ? 0 : (n-1<(ulong)i ? n-1 : (ulong)i);
}

static void set_bit(unsigned char *const p, const uint i)
{
  const uint byte = i/CHAR_BIT;
  const unsigned bit = i%CHAR_BIT;
  p[byte] |= 1u<<bit;
}

static unsigned get_bit(const unsigned char *const p, const uint i)
{
  const uint byte = i/CHAR_BIT;
  const unsigned bit = i%CHAR_BIT;
  return p[byte]>>bit & 1u;
}

static unsigned byte_bits(const unsigned char x)
{
  unsigned bit, sum=0;
  for(bit=0;bit<CHAR_BIT;++bit) sum += x>>bit & 1u;
  return sum;
}

static uint count_bits(unsigned char *p, uint n)
{
  uint sum=0;
  for(;n;--n) sum+=byte_bits(*p++);
  return sum;
}

#define D 2
#define WHEN_3D(a)
#include "findpts_imp.h"
#undef WHEN_3D
#undef D

#define D 3
#define WHEN_3D(a) a
#include "findpts_imp.h"
#undef WHEN_3D
#undef D

/*--------------------------------------------------------------------------

  FORTRAN Interface

  --------------------------------------------------------------------------
  call findpts_setup(h, comm,np, ndim, xm,ym,zm, nr,ns,nt,nel,
                     mr,ms,mt, bbox_tol, loc_hash_size, gbl_hash_size,
                     npt_max, newt_tol)

    (zm,nt,mt all ignored when ndim==2)
                     
    h: (output) handle
    comm,np: MPI communicator and # of procs (checked against MPI_Comm_size)
    ndim: 2 or 3
    xm,ym,zm: element geometry (nodal x,y,z values)
    nr,ns,nt,nel: element dimensions --- e.g., xm(nr,ns,nt,nel)
  
    mr,ms,mt: finer mesh size for bounding box computation;
              must be larger than nr,ns,nt for correctness,
              recommend at least 2*nr,2*ns,2*nt
    bbox_tol: e.g., 0.01 - relative size to expand bounding boxes by;
              prevents points from falling through "cracks",
              and prevents "not found" failures for points just outside mesh
                (returning instead the closest point inside the mesh)

    loc_hash_size: e.g., nr*ns*nt*nel
                   maximum number of integers to use for local geom hash table;
                   minimum is nel+2 for the trivial table with one cell
                 
    gbl_hash_size: e.g., nr*ns*nt*nel
                   approx number of cells per proc for the distributed
                     global geometric hash table
                   NOTE: gbl_hash_size*np needs to fit in a "global" integer
                         (controlled by -DGLOBAL_LONG or -DGLOBAL_LONG_LONG;
                          see "types.h")
                   actual number of cells per proc will be greater by
                     ~ 3 gbl_hash_size^(2/3) / np^(1/3)
  
    npt_max: e.g., 256
             number of points to iterate on simultaneously
             enables dominant complexity to be matrix-matrix products
               (there is a sweet spot --- too high and the cache runs out)
             the memory allocation term dependent on npt_max is
               (12 + 2*(nr+ns+nt+nr*ns)) * npt_max     doubles
  
    newt_tol: e.g., 1024*DBL_EPSILON
              the iteration stops for a point when
                   the 1-norm of the step in (r,s,t) is smaller than newt_tol
                or the objective (dist^2) increases while the predicted (model)
                  decrease is smaller than newt_tol * (the objective)

  --------------------------------------------------------------------------
  call findpts_free(h)
  
  --------------------------------------------------------------------------
  call findpts(h, code_base,  code_stride,
                  proc_base,  proc_stride,
                    el_base,    el_stride,
                     r_base,     r_stride,
                 dist2_base, dist2_stride,
                     x_base,     x_stride,
                     y_base,     y_stride,
                     z_base,     z_stride, npt)

    (z_base, z_stride ignored when ndim==2)

    conceptually, locates npt points;
      data for each point is:
        ouput:
          code: 0 - inside an element
                1 - closest point on a border
                    (perhaps exactly, or maybe just near --- check dist2)
                2 - not found (bbox_tol controls cut-off between code 1 and 2)
          proc:    remote processor on which the point was found
          el:      element on remote processor in which the point was found
          r(ndim): parametric coordinates for point
          dist2: distance squared from found to sought point (in xyz space)
        input:
          x, y, z: coordinates of sought point
    
    the *_base arguments point to the data for the first point,
      each is advanced by the corresponding *_stride argument for the next point
    this allows fairly arbitrary data layout,
      but note the r,s,t coordinates for each point must be packed together
      (consequently, r_stride must be at least ndim)


  --------------------------------------------------------------------------
  call findpts_eval(h,  out_base,  out_stride,
                       code_base, code_stride,
                       proc_base, proc_stride,
                         el_base,   el_stride,
                          r_base,    r_stride, npt,
                    input_field)
  
    may be called immediately after findpts (or any other time)
    to evaluate input_field at the given points ---
      these specified by code,proc,el,r(ndim) and possibly remote
    --- storing the interpolated values in out
          [that is, at out_base(1+out_stride*(point_index-1)) ]
    
    for example, following a call to findpts, a call to findpts_eval with
      input_field = xm, will ideally result in out = x(1) for each point,
      or x(2) for ym, x(3) for zm


  --------------------------------------------------------------------------
  call findpts_eval_local(h,
                        out_base,  out_stride,
                         el_base,   el_stride,
                          r_base,    r_stride, npt,
                    input_field)

    just like findpts_eval, but does assumes all points are local,
    and does no communication. will use matrix-matrix products if
    points are grouped by element.

  --------------------------------------------------------------------------*/

#define ffindpts_setup      FORTRAN_NAME(findpts_setup     ,FINDPTS_SETUP     )
#define ffindpts_free       FORTRAN_NAME(findpts_free      ,FINDPTS_FREE      )
#define ffindpts            FORTRAN_NAME(findpts           ,FINDPTS           )
#define ffindpts_eval       FORTRAN_NAME(findpts_eval      ,FINDPTS_EVAL      )
#define ffindpts_eval_local FORTRAN_NAME(findpts_eval_local,FINDPTS_EVAL_LOCAL)

struct handle { void *data; unsigned ndim; };
static struct handle *handle_array = 0;
static int handle_max = 0;
static int handle_n = 0;

void ffindpts_setup(sint *const handle,
  const MPI_Fint *const comm, const sint *const np,
  const sint *ndim,
  const double *const xm, const double *const ym, const double *const zm,
  const sint *const nr, const sint *const ns, const sint *const nt,
  const sint *const nel,
  const sint *const mr, const sint *const ms, const sint *const mt,
  const double *const bbox_tol,
  const sint *const loc_hash_size, const sint *const gbl_hash_size,
  const sint *const npt_max,
  const double *const newt_tol)
{
  struct handle *h;
  if(handle_n==handle_max)
    handle_max+=handle_max/2+1,
    handle_array=trealloc(struct handle,handle_array,handle_max);
  h = &handle_array[handle_n];
  h->ndim = *ndim;
  if(h->ndim==2) {
    struct findpts_data_2 *const fd = tmalloc(struct findpts_data_2,1);
    const double *elx[2];
    uint n[2], m[2];
    elx[0]=xm,elx[1]=ym;
    n[0]=*nr,n[1]=*ns;
    m[0]=*mr,m[1]=*ms;
    h->data = fd;
    comm_init_check(&fd->cr.comm, *comm, *np);
    buffer_init(&fd->cr.data,1000);
    buffer_init(&fd->cr.work,1000);
    setup_aux_2(fd, elx,n,*nel,m,*bbox_tol,
                *loc_hash_size,*gbl_hash_size, *npt_max, *newt_tol);
  } else if(h->ndim==3) {
    struct findpts_data_3 *const fd = tmalloc(struct findpts_data_3,1);
    const double *elx[3];
    uint n[3], m[3];
    elx[0]=xm,elx[1]=ym,elx[2]=zm;
    n[0]=*nr,n[1]=*ns,n[2]=*nt;
    m[0]=*mr,m[1]=*ms,m[2]=*mt;
    h->data = fd;
    comm_init_check(&fd->cr.comm, *comm, *np);
    buffer_init(&fd->cr.data,1000);
    buffer_init(&fd->cr.work,1000);
    setup_aux_3(fd, elx,n,*nel,m,*bbox_tol,
                *loc_hash_size,*gbl_hash_size, *npt_max, *newt_tol);
  } else
    fail(1,__FILE__,__LINE__,
         "findpts_setup: ndim must be 2 or 3; given ndim=%u",(unsigned)h->ndim);
  *handle = handle_n++;
}

#define CHECK_HANDLE(func) \
  struct handle *h; \
  if(*handle<0 || *handle>=handle_n || !(h=&handle_array[*handle])->data) \
    fail(1,__FILE__,__LINE__,func ": invalid handle")

void ffindpts_free(const sint *const handle)
{
  CHECK_HANDLE("findpts_free");
  if(h->ndim==2)
    PREFIXED_NAME(findpts_free_2)(h->data);
  else
    PREFIXED_NAME(findpts_free_3)(h->data);
  h->data = 0;
}

void ffindpts(const sint *const handle,
          sint *const  code_base, const sint *const  code_stride,
          sint *const  proc_base, const sint *const  proc_stride,
          sint *const    el_base, const sint *const    el_stride,
        double *const     r_base, const sint *const     r_stride,
        double *const dist2_base, const sint *const dist2_stride,
  const double *const     x_base, const sint *const     x_stride,
  const double *const     y_base, const sint *const     y_stride,
  const double *const     z_base, const sint *const     z_stride,
  const sint *const npt)
{
  CHECK_HANDLE("findpts");
  if(h->ndim==2) {
    const double *xv_base[2];
    unsigned xv_stride[2];
    xv_base[0]=x_base, xv_base[1]=y_base;
    xv_stride[0] = *x_stride*sizeof(double),
    xv_stride[1] = *y_stride*sizeof(double);
    PREFIXED_NAME(findpts_2)(
      (uint*)code_base,(* code_stride)*sizeof(sint  ),
      (uint*)proc_base,(* proc_stride)*sizeof(sint  ),
      (uint*)  el_base,(*   el_stride)*sizeof(sint  ),
                r_base,(*    r_stride)*sizeof(double),
            dist2_base,(*dist2_stride)*sizeof(double),
               xv_base,     xv_stride,
      *npt, h->data);
  } else {
    const double *xv_base[3];
    unsigned xv_stride[3];
    xv_base[0]=x_base, xv_base[1]=y_base, xv_base[2]=z_base;
    xv_stride[0] = *x_stride*sizeof(double),
    xv_stride[1] = *y_stride*sizeof(double),
    xv_stride[2] = *z_stride*sizeof(double);
    PREFIXED_NAME(findpts_3)(
      (uint*)code_base,(* code_stride)*sizeof(sint  ),
      (uint*)proc_base,(* proc_stride)*sizeof(sint  ),
      (uint*)  el_base,(*   el_stride)*sizeof(sint  ),
                r_base,(*    r_stride)*sizeof(double),
            dist2_base,(*dist2_stride)*sizeof(double),
               xv_base,     xv_stride,
      *npt, h->data);
  }
}

void ffindpts_eval(const sint *const handle,
        double *const  out_base, const sint *const  out_stride,
  const   sint *const code_base, const sint *const code_stride,
  const   sint *const proc_base, const sint *const proc_stride,
  const   sint *const   el_base, const sint *const   el_stride,
  const double *const    r_base, const sint *const    r_stride,
  const sint *const npt, const double *const in)
{
  CHECK_HANDLE("findpts_eval");
  if(h->ndim==2)
    PREFIXED_NAME(findpts_eval_2)(
              out_base,(* out_stride)*sizeof(double),
      (uint*)code_base,(*code_stride)*sizeof(sint  ),
      (uint*)proc_base,(*proc_stride)*sizeof(sint  ),
      (uint*)  el_base,(*  el_stride)*sizeof(sint  ),
                r_base,(*   r_stride)*sizeof(double),
      *npt, in, h->data);
  else
    PREFIXED_NAME(findpts_eval_3)(
              out_base,(* out_stride)*sizeof(double),
      (uint*)code_base,(*code_stride)*sizeof(sint  ),
      (uint*)proc_base,(*proc_stride)*sizeof(sint  ),
      (uint*)  el_base,(*  el_stride)*sizeof(sint  ),
                r_base,(*   r_stride)*sizeof(double),
      *npt, in, h->data);
}

void ffindpts_eval_local(const sint *const handle,
        double *const  out_base, const sint *const  out_stride,
  const   sint *const   el_base, const sint *const   el_stride,
  const double *const    r_base, const sint *const    r_stride,
  const sint *const npt, const double *const in)
{
  CHECK_HANDLE("findpts_eval_local");
  if(h->ndim==2)
    findpts_local_eval_2(
              out_base,(* out_stride)*sizeof(double),
      (uint*)  el_base,(*  el_stride)*sizeof(sint  ),
                r_base,(*   r_stride)*sizeof(double),
      *npt, in, &((struct findpts_data_2 *)h->data)->local);
  else
    findpts_local_eval_3(
              out_base,(* out_stride)*sizeof(double),
      (uint*)  el_base,(*  el_stride)*sizeof(sint  ),
                r_base,(*   r_stride)*sizeof(double),
      *npt, in, &((struct findpts_data_3 *)h->data)->local);
}
