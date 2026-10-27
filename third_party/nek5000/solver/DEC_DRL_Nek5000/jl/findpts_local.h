#ifndef FINDPTS_LOCAL_H
#define FINDPTS_LOCAL_H

#if !defined(MEM_H) || !defined(FINDPTS_EL_H) || !defined(OBBOX_H)
#warning "findpts_local.h" requires "mem.h", "findpts_el.h", "obbox.h"
#endif

#define findpts_local_setup_2   PREFIXED_NAME(findpts_local_setup_2)
#define findpts_local_free_2    PREFIXED_NAME(findpts_local_free_2 )
#define findpts_local_2         PREFIXED_NAME(findpts_local_2      )
#define findpts_local_eval_2    PREFIXED_NAME(findpts_local_eval_2 )

struct findpts_local_hash_data_2 {
  uint hash_n;
  struct dbl_range bnd[2];
  double fac[2];
  uint *offset;
  uint max;
};

struct findpts_local_data_2 {
  unsigned ntot;
  const double *elx[2];
  struct obbox_2 *obb;
  struct findpts_local_hash_data_2 hd;
  struct findpts_el_data_2 fed;
  double tol;
};

void findpts_local_setup_2(struct findpts_local_data_2 *const fd,
                           const double *const elx[2],
                           const unsigned n[2], const uint nel,
                           const unsigned m[2], const double bbox_tol,
                           const uint max_hash_size,
                           const unsigned npt_max, const double newt_tol);
void findpts_local_free_2(struct findpts_local_data_2 *const fd);
void findpts_local_2(
        uint   *const  code_base   , const unsigned  code_stride   ,
        uint   *const    el_base   , const unsigned    el_stride   ,
        double *const     r_base   , const unsigned     r_stride   ,
        double *const dist2_base   , const unsigned dist2_stride   ,
  const double *const     x_base[2], const unsigned     x_stride[2],
  const uint npt, struct findpts_local_data_2 *const fd,
  buffer *buf);
void findpts_local_eval_2(
        double *const out_base, const unsigned out_stride,
  const uint   *const  el_base, const unsigned  el_stride,
  const double *const   r_base, const unsigned   r_stride,
  const uint npt,
  const double *const in, struct findpts_local_data_2 *const fd);

#define findpts_local_setup_3   PREFIXED_NAME(findpts_local_setup_3)
#define findpts_local_free_3    PREFIXED_NAME(findpts_local_free_3 )
#define findpts_local_3         PREFIXED_NAME(findpts_local_3      )
#define findpts_local_eval_3    PREFIXED_NAME(findpts_local_eval_3 )

struct findpts_local_hash_data_3 {
  uint hash_n;
  struct dbl_range bnd[3];
  double fac[3];
  uint *offset;
  uint max;
};

struct findpts_local_data_3 {
  unsigned ntot;
  const double *elx[3];
  struct obbox_3 *obb;
  struct findpts_local_hash_data_3 hd;
  struct findpts_el_data_3 fed;
  double tol;
};

void findpts_local_setup_3(struct findpts_local_data_3 *const fd,
                           const double *const elx[3],
                           const unsigned n[3], const uint nel,
                           const unsigned m[3], const double bbox_tol,
                           const uint max_hash_size,
                           const unsigned npt_max, const double newt_tol);
void findpts_local_free_3(struct findpts_local_data_3 *const fd);
void findpts_local_3(
        uint   *const  code_base   , const unsigned  code_stride   ,
        uint   *const    el_base   , const unsigned    el_stride   ,
        double *const     r_base   , const unsigned     r_stride   ,
        double *const dist2_base   , const unsigned dist2_stride   ,
  const double *const     x_base[3], const unsigned     x_stride[3],
  const uint npt, struct findpts_local_data_3 *const fd,
  buffer *buf);
void findpts_local_eval_3(
        double *const out_base, const unsigned out_stride,
  const uint   *const  el_base, const unsigned  el_stride,
  const double *const   r_base, const unsigned   r_stride,
  const uint npt,
  const double *const in, struct findpts_local_data_3 *const fd);

#endif
