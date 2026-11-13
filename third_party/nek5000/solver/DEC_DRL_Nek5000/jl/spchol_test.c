#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "sparse_cholesky.h"

int main()
{
#define x -1

  uint i,n=7;
  uint Aj [] = {0,2, 1,2,6, 0,1,2, 3,5,6, 4,5, 3,4,5, 1,3,6};
  double A[] = {2,x, 2,x,x, x,x,2, 2,x,x, 2,x, x,x,2, x,x,2};
#undef x
  uint Arp[] = {0,2,5,8,11,13,16,19};
  double x[7], b[7] = {0,0,0,0, 0,0,0};
  uint o[7] = {0,2,1,6,3,5,4};
/*
  uint i,n=10;
  uint Aj [] = {0,2,7, 1,4,9, 0,2,6, 3,8,9, 1,4,8,9, 5,6,7, 2,5,6, 0,5,7,8,9, 3,4,7,8, 1,3,4,7,9};
  real A  [] = {3,x,x, 2,x,x, x,2,x, 2,x,x, x,3,x,x, 2,x,x, x,x,2, x,x,4,x,x, x,x,x,3, x,x,x,x,4};
#undef x
  uint Arp[] = {0,     3,     6,     9,     12,      16,    19,    22,        27,      31,    36};
  real b[] = {1,2,3,4,5, 6,7,8,9,10};
*/
  struct sparse_cholesky data;
  buffer buf;
  buffer_init(&buf,4);
  sparse_cholesky_factor(n,Arp,Aj,A,&data,&buf);
  
  for(i=0;i<n;++i) { uint j;
    b[o[i]]=1;
    sparse_cholesky_solve(x,&data,b);
    for(j=0;j<n;++j) printf("\t%g",(double)x[o[j]]);
    printf("\n");
    b[o[i]]=0;
  }
  sparse_cholesky_free(&data);
  buffer_free(&buf);
  /*
  sparse_cholesky_solve(b,&data,b);
  sparse_cholesky_free(&data);
  for(i=0;i<n;++i) printf("%g\n", b[i]);
  */

  return 0;
}


