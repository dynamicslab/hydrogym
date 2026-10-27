#ifndef FINDPTS_EL_H
#define FINDPTS_EL_H

#if !defined(NAME_H) || !defined(POLY_H)
#warning "findpts_el.h" requires "name.h", "poly.h"
#endif

#define findpts_el_setup_2   PREFIXED_NAME(findpts_el_setup_2)
#define findpts_el_free_2    PREFIXED_NAME(findpts_el_free_2 )
#define findpts_el_2         PREFIXED_NAME(findpts_el_2      )
#define findpts_el_eval_2    PREFIXED_NAME(findpts_el_eval_2 )

struct findpts_el_pt_2 {
  double x[2],r[2],oldr[2],dist2,dist2p,tr;
  unsigned index,flags;
};

struct findpts_el_gedge_2 { const double *x[2], *dxdn[2]; };
struct findpts_el_gpt_2   { double x[2], jac[4], hes[4]; };

struct findpts_el_data_2 {
  unsigned npt_max;
  struct findpts_el_pt_2 *p;

  unsigned n[2];
  double *z[2];
  lagrange_fun *lag[2];
  double *lag_data[2];
  double *wtend[2];
  
  const double *x[2];
  
  unsigned side_init;
  double *sides;
  struct findpts_el_gedge_2 edge[4]; /* R S=-1; R S=1; ... */
  struct findpts_el_gpt_2 pt[4];

  double *work;
};

void findpts_el_setup_2(struct findpts_el_data_2 *const fd,
                        const unsigned n[2],
                        const unsigned npt_max);
void findpts_el_free_2(struct findpts_el_data_2 *const fd);
void findpts_el_2(struct findpts_el_data_2 *fd, unsigned npt, const double tol);
void findpts_el_eval_2(
        double *const out_base, const unsigned out_stride,
  const double *const   r_base, const unsigned   r_stride, const unsigned pn,
  const double *const in, struct findpts_el_data_2 *const fd);

static void findpts_el_start_2(struct findpts_el_data_2 *const fd,
                               const double *const x[2])
{
  fd->side_init=0,fd->x[0]=x[0],fd->x[1]=x[1];
}

static struct findpts_el_pt_2 *findpts_el_points_2(
  struct findpts_el_data_2 *const fd)
{
  return fd->p;
}

#define findpts_el_setup_3   PREFIXED_NAME(findpts_el_setup_3)
#define findpts_el_free_3    PREFIXED_NAME(findpts_el_free_3 )
#define findpts_el_3         PREFIXED_NAME(findpts_el_3      )
#define findpts_el_eval_3    PREFIXED_NAME(findpts_el_eval_3 )

struct findpts_el_pt_3 {
  double x[3],r[3],oldr[3],dist2,dist2p,tr;
  unsigned index,flags;
};

struct findpts_el_gface_3 { const double *x[3], *dxdn[3]; };
struct findpts_el_gedge_3 { const double *x[3], *dxdn1[3], *dxdn2[3],
                                         *d2xdn1[3], *d2xdn2[3]; };
struct findpts_el_gpt_3   { double x[3], jac[9], hes[18]; };

struct findpts_el_data_3 {
  unsigned npt_max;
  struct findpts_el_pt_3 *p;

  unsigned n[3];
  double *z[3];
  lagrange_fun *lag[3];
  double *lag_data[3];
  double *wtend[3];
  
  const double *x[3];
  
  unsigned side_init;
  double *sides;
  struct findpts_el_gface_3 face[6]; /* ST R=-1,R=+1; TR S=-1,S=+1; ... */
  struct findpts_el_gedge_3 edge[12]; /* R S=-1,T=-1; R S=1,T=-1; ... */
  struct findpts_el_gpt_3 pt[8];

  double *work;
};

void findpts_el_setup_3(struct findpts_el_data_3 *const fd,
                        const unsigned n[3],
                        const unsigned npt_max);
void findpts_el_free_3(struct findpts_el_data_3 *const fd);
void findpts_el_3(struct findpts_el_data_3 *const fd, const unsigned npt,
                  const double tol);
void findpts_el_eval_3(
        double *const out_base, const unsigned out_stride,
  const double *const   r_base, const unsigned   r_stride, const unsigned pn,
  const double *const in, struct findpts_el_data_3 *const fd);

static void findpts_el_start_3(struct findpts_el_data_3 *const fd,
                               const double *const x[3])
{
  fd->side_init=0,fd->x[0]=x[0],fd->x[1]=x[1],fd->x[2]=x[2];
}

static struct findpts_el_pt_3 *findpts_el_points_3(
  struct findpts_el_data_3 *const fd)
{
  return fd->p;
}

#endif
