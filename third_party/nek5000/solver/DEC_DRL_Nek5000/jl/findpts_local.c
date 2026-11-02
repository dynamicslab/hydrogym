#include <stdio.h>

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "c99.h"
#include "types.h"
#include "name.h"
#include "fail.h"
#include "mem.h"
#include "obbox.h"
#include "poly.h"
#include "sort.h"
#include "sarray_sort.h"
#include "findpts_el.h"

struct uint_range { uint min, max; };
struct index_el { uint index, el; };

static struct dbl_range dbl_range_merge(struct dbl_range a, struct dbl_range b)
{
  struct dbl_range m;
  m.min = b.min<a.min?b.min:a.min,
  m.max = a.max>b.max?a.max:b.max;
  return m;
}

static sint ifloor(double x) { return floor(x); }
static sint iceil (double x) { return ceil (x); }

static uint hash_index_aux(double low, double fac, uint n, double x)
{
  const sint i = ifloor((x-low)*fac);
  return i<0 ? 0 : (n-1<(uint)i ? n-1 : (uint)i);
}

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

#define D 2
#define WHEN_3D(a)
#include "findpts_local_imp.h"
#undef WHEN_3D
#undef D

#define D 3
#define WHEN_3D(a) a
#include "findpts_local_imp.h"
#undef WHEN_3D
#undef D
