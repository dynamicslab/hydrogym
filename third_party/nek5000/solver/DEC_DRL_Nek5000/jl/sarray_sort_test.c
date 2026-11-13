#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "sort.h"
#include "sarray_sort.h"

int main()
{
  struct rec { double d; slong l; sint i; float f; };
  buffer buf = {0,0,0};
  struct rec rec[500];
  uint i;
  
  for(i=0;i<500;++i) {
    sint num1 = rand() & 0xff;
    slong num2 = rand();
    num2<<=(CHAR_BIT)*sizeof(int)-1;
    num2|=rand();
    num2<<=(CHAR_BIT)*sizeof(int)-1;
    num2|=rand();
    num2= num2<0?-num2:num2;
    rec[i].d = num2;
    rec[i].f = num2;
    rec[i].l = num2;
    rec[i].i = num1;
  }
  sarray_sort_2(struct rec,rec,500, i,0, l,1, &buf);
  for(i=0;i<500;++i)
    printf("%g\t%g\t%ld\t%d\n",
      rec[i].d,rec[i].f,(long)rec[i].l,(int)rec[i].i);

  printf("\n");
  sarray_sort(struct rec,rec,500, l,1, &buf);
  for(i=0;i<500;++i)
    printf("%g\t%g\t%ld\t%d\n",
      rec[i].d,rec[i].f,(long)rec[i].l,(int)rec[i].i);
  buffer_free(&buf);
  return 0;
}

