#ifndef NEK_FINDPTS_LOCAL_H
#define NEK_FINDPTS_LOCAL_H

#include "nek_config.h"

#define TOKEN_PASTE_(a,b) a##b
#define TOKEN_PASTE(a,b) TOKEN_PASTE_(a,b)

#ifdef NEK_FUN_PREFIX
#  define NEK_PREFIXED_NAME(x) TOKEN_PASTE(NEK_FUN_PREFIX,x)
#else
#  define NEK_PREFIXED_NAME(x) x
#endif


#define findpts_local_setup   NEK_PREFIXED_NAME(findpts_local_setup)
#define findpts_local_free    NEK_PREFIXED_NAME(findpts_local_free )
#define findpts_local         NEK_PREFIXED_NAME(findpts_local      )
#define findpts_local_eval    NEK_PREFIXED_NAME(findpts_local_eval )


struct findpts_local_data;

struct findpts_local_data *findpts_local_setup(
  const unsigned dim,
  const double *const elx[], const unsigned n[], const nek_uint nel,
  const unsigned m[], const double bbox_tol, const nek_uint max_hash_size,
  const nek_uint npt_max, const double newt_tol);

void findpts_local_free(struct findpts_local_data *p);

void findpts_local(
        nek_uint *const  code_base  , const unsigned  code_stride  ,
        nek_uint *const    el_base  , const unsigned    el_stride  ,
        double   *const     r_base  , const unsigned     r_stride  ,
        double   *const dist2_base  , const unsigned dist2_stride  ,
  const double   *const     x_base[], const unsigned     x_stride[],
  const nek_uint npt, struct findpts_local_data *const p);
  
void findpts_local_eval(
        double   *const out_base, const unsigned out_stride,
        nek_uint *const  el_base, const unsigned  el_stride,
  const double   *const   r_base, const unsigned   r_stride,
  const nek_uint npt,
  const double *const in, struct findpts_local_data *const p);

#endif
