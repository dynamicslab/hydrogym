#include <stddef.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "../c99.h"
#include "../name.h"
#include "../fail.h"
#include "../mem.h"
#include "../types.h"
#include "../poly.h"
#include "../obbox.h"
#include "../findpts_el.h"
#include "../findpts_local.h"

#define findpts_local_setup   PREFIXED_NAME(findpts_local_setup)
#define findpts_local_free    PREFIXED_NAME(findpts_local_free )
#define findpts_local         PREFIXED_NAME(findpts_local      )
#define findpts_local_eval    PREFIXED_NAME(findpts_local_eval )

struct findpts_local_data {
  unsigned dim;    /* 2 or 3 */
  buffer buf;
  union {
    struct findpts_local_data_2 d2;
    struct findpts_local_data_3 d3;
  } fld;
};

struct findpts_local_data *findpts_local_setup(
  const unsigned dim,
  const double *const elx[], const unsigned n[], const uint nel,
  const unsigned m[], const double bbox_tol, const uint max_hash_size,
  const uint npt_max, const double newt_tol)
{
  struct findpts_local_data *p = tmalloc(struct findpts_local_data, 1);
  
  p->dim = dim;
  memset(&p->buf,0,sizeof(buffer));
  if(dim==2)
    findpts_local_setup_2(&p->fld.d2, elx,n,nel, m, bbox_tol,
                          max_hash_size, npt_max, newt_tol);
  else
    findpts_local_setup_3(&p->fld.d3, elx,n,nel, m, bbox_tol,
                          max_hash_size, npt_max, newt_tol);

  return p;
}

void findpts_local_free(struct findpts_local_data *p)
{
  buffer_free(&p->buf);
  if(p->dim==2) findpts_local_free_2(&p->fld.d2);
  else          findpts_local_free_3(&p->fld.d3);
  free(p);
}

void findpts_local(
        uint   *const  code_base  , const unsigned  code_stride  ,
        uint   *const    el_base  , const unsigned    el_stride  ,
        double *const     r_base  , const unsigned     r_stride  ,
        double *const dist2_base  , const unsigned dist2_stride  ,
  const double *const     x_base[], const unsigned     x_stride[],
  const uint npt, struct findpts_local_data *const p)
{
  if(p->dim==2) findpts_local_2(code_base, code_stride,
                                  el_base,   el_stride,
                                   r_base,    r_stride,
                               dist2_base,dist2_stride,
                                   x_base,    x_stride,
                                npt, &p->fld.d2, &p->buf);
  else          findpts_local_3(code_base, code_stride,
                                  el_base,   el_stride,
                                   r_base,    r_stride,
                               dist2_base,dist2_stride,
                                   x_base,    x_stride,
                                npt, &p->fld.d3, &p->buf);
}

void findpts_local_eval(
        double *const out_base, const unsigned out_stride,
  const uint   *const  el_base, const unsigned  el_stride,
  const double *const   r_base, const unsigned   r_stride,
  const uint npt,
  const double *const in, struct findpts_local_data *const p)
{
  if(p->dim==2) findpts_local_eval_2(out_base, out_stride,
                                      el_base,  el_stride,
                                       r_base,   r_stride,
                                     npt, in, &p->fld.d2);
  else          findpts_local_eval_3(out_base, out_stride,
                                      el_base,  el_stride,
                                       r_base,   r_stride,
                                     npt, in, &p->fld.d3);
}

