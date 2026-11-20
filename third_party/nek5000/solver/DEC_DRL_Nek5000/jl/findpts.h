#ifndef FINDPTS_H
#define FINDPTS_H

#if !defined(COMM_H)
#warning "findpts.h" requires "comm.h"
#endif

#define findpts_setup_2   PREFIXED_NAME(findpts_setup_2)
#define findpts_free_2    PREFIXED_NAME(findpts_free_2 )
#define findpts_2         PREFIXED_NAME(findpts_2      )
#define findpts_eval_2    PREFIXED_NAME(findpts_eval_2 )
#define findpts_setup_3   PREFIXED_NAME(findpts_setup_3)
#define findpts_free_3    PREFIXED_NAME(findpts_free_3 )
#define findpts_3         PREFIXED_NAME(findpts_3      )
#define findpts_eval_3    PREFIXED_NAME(findpts_eval_3 )

struct findpts_data_2;
struct findpts_data_3;

struct findpts_data_2 *findpts_setup_2(
  const struct comm *const comm,
  const double *const elx[2],
  const unsigned n[2], const uint nel,
  const unsigned m[2], const double bbox_tol,
  const uint local_hash_size, const uint global_hash_size,
  const unsigned npt_max, const double newt_tol);

struct findpts_data_3 *findpts_setup_3(
  const struct comm *const comm,
  const double *const elx[3],
  const unsigned n[3], const uint nel,
  const unsigned m[3], const double bbox_tol,
  const uint local_hash_size, const uint global_hash_size,
  const unsigned npt_max, const double newt_tol);

void findpts_free_2(struct findpts_data_2 *fd);
void findpts_free_3(struct findpts_data_3 *fd);

void findpts_2(    uint   *const  code_base   , const unsigned  code_stride   ,
                   uint   *const  proc_base   , const unsigned  proc_stride   ,
                   uint   *const    el_base   , const unsigned    el_stride   ,
                   double *const     r_base   , const unsigned     r_stride   ,
                   double *const dist2_base   , const unsigned dist2_stride   ,
             const double *const     x_base[2], const unsigned     x_stride[2],
             const uint npt, struct findpts_data_2 *const fd);

void findpts_3(    uint   *const  code_base   , const unsigned  code_stride   ,
                   uint   *const  proc_base   , const unsigned  proc_stride   ,
                   uint   *const    el_base   , const unsigned    el_stride   ,
                   double *const     r_base   , const unsigned     r_stride   ,
                   double *const dist2_base   , const unsigned dist2_stride   ,
             const double *const     x_base[3], const unsigned     x_stride[3],
             const uint npt, struct findpts_data_3 *const fd);

void findpts_eval_2(
        double *const  out_base, const unsigned  out_stride,
  const uint   *const code_base, const unsigned code_stride,
  const uint   *const proc_base, const unsigned proc_stride,
  const uint   *const   el_base, const unsigned   el_stride,
  const double *const    r_base, const unsigned    r_stride,
  const uint npt,
  const double *const in, struct findpts_data_2 *const fd);
 
void findpts_eval_3(
        double *const  out_base, const unsigned  out_stride,
  const uint   *const code_base, const unsigned code_stride,
  const uint   *const proc_base, const unsigned proc_stride,
  const uint   *const   el_base, const unsigned   el_stride,
  const double *const    r_base, const unsigned    r_stride,
  const uint npt,
  const double *const in, struct findpts_data_3 *const fd);

#endif
