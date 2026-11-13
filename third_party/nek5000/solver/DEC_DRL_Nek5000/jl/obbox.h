#ifndef OBBOX_H
#define OBBOX_H

#if !defined(TYPES_H) || !defined(NAME_H)
#warning "obbox.h" requires "types.h" and "name.h"
#endif

#define obbox_calc_2  PREFIXED_NAME(obbox_calc_2)
#define obbox_calc_3  PREFIXED_NAME(obbox_calc_3)

/*--------------------------------------------------------------------------
   Oriented and axis-aligned bounding box computation for spectral elements
   
   Usage:
   
     double x[n][nt][ns][nr], y[n][nt][ns][nr], z[n][nt][ns][nr];
     obbox_3 ob[n];

     unsigned mr=4*nr, ms=4*ns, mt=4*nt;
     double tol = 1e-6;
     obbox_3_calc(ob, x,y,z, nr,ns,nt,n, mr,ms,mt, tol);
     
   The parameters mr,ms,mt specify number of points to use in computing
   bounds (see lob_bnd.h). It is expected that mr>nr, etc. For reasonable
   quality, a factor of at least 2 is recommended.
  
   tol is a relative amount by which to expand the bounding box.
   This would accommodate, e.g., rounding errors.
  
   The axis aligned bounds for a given element are
     ob[i].x.min <= x <= ob[i].x.max
     ob[i].y.min <= y <= ob[i].y.max
     ob[i].z.min <= z <= ob[i].z.max

   The oriented bounding box is given by
     (-1,-1,-1)^T <= ob[i].A * (x - ob[i].c0) <= (1,1,1)
   
   where the matrix is row-major format,
     dx = x - c0[0], dy = y - c0[1], dz = z - c0[2]
     -1 <= r[0] = A[0]*dx + A[1]*dy + A[2]*dz <= 1
     -1 <= r[1] = A[3]*dx + A[4]*dy + A[5]*dz <= 1
     -1 <= r[2] = A[6]*dx + A[7]*dy + A[8]*dz <= 1

   Also, ob[i].A * (x - ob[i].c0) should be a reasonable seed for Newton's.
    
  --------------------------------------------------------------------------*/

#ifndef LOB_BND_H
struct dbl_range { double min, max; };
#endif

struct obbox_2 { double c0[2], A[4];
                 struct dbl_range x[2]; };

struct obbox_3 { double c0[3], A[9];
                 struct dbl_range x[3]; };

void obbox_calc_2(struct obbox_2 *out,
                  const double *const elx[2],
                  const unsigned n[2], uint nel,
                  const unsigned m[2], const double tol);

void obbox_calc_3(struct obbox_3 *out,
                  const double *const elx[3],
                  const unsigned n[3], uint nel,
                  const unsigned m[3], const double tol);

/* positive when possibly inside */
static double obbox_axis_test_2(const struct obbox_2 *const b,
                                const double x[2])
{
  const double bx =  (x[0]-b->x[0].min)*(b->x[0].max-x[0]);
  return bx<0 ? bx : (x[1]-b->x[1].min)*(b->x[1].max-x[1]);
}

/* positive when possibly inside */
static double obbox_test_2(const struct obbox_2 *const b, const double x[2])
{
  const double bxy = obbox_axis_test_2(b,x);
  if(bxy<0) return bxy; else {
    const double dx = x[0]-b->c0[0], dy = x[1]-b->c0[1];
    const double r = b->A[0]*dx + b->A[1]*dy,
                 s = b->A[2]*dx + b->A[3]*dy;
    const double br = (r+1)*(1-r);
    return br<0 ? br : (s+1)*(1-s);
  }
}

/* positive when possibly inside */
static double obbox_axis_test_3(const struct obbox_3 *const b,
                                const double x[3])
{
  const double               bx = (x[0]-b->x[0].min)*(b->x[0].max-x[0]);
  const double               by = (x[1]-b->x[1].min)*(b->x[1].max-x[1]);
  return bx<0 ? bx : (by<0 ? by : (x[2]-b->x[2].min)*(b->x[2].max-x[2]));
}

/* positive when possibly inside */
static double obbox_test_3(const struct obbox_3 *const b, const double x[3])
{
  const double bxyz = obbox_axis_test_3(b,x);
  if(bxyz<0) return bxyz; else {
    const double dx = x[0]-b->c0[0], dy = x[1]-b->c0[1], dz = x[2]-b->c0[2];
    const double r = b->A[0]*dx + b->A[1]*dy + b->A[2]*dz,
                 s = b->A[3]*dx + b->A[4]*dy + b->A[5]*dz,
                 t = b->A[6]*dx + b->A[7]*dy + b->A[8]*dz;
    const double br = (r+1)*(1-r), bs = (s+1)*(1-s);
    return br<0 ? br : (bs<0 ? bs : (t+1)*(1-t));
  }
}

#endif

