#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "sort.h"
#include "rdtsc.h"

#if 1

DEFINE_HW_COUNTER()

#define N (1<<20)

ulong A[N], out[N];
uint P[N];

int main()
{
  buffer buf = null_buffer;
  uint i;
  unsigned long long tic, toc;
  unsigned r;
  #define TIME(t, repeat, what) do { \
    for(r=repeat;r;--r) { what; } \
    tic = getticks(); \
    for(r=repeat;r;--r) { what; } \
    toc = getticks(); \
    t = toc-tic; \
  } while(0)

  for(i=0;i!=N;++i) {
    A[i]=rand();
    A[i]<<=CHAR_BIT*sizeof(int)-1;
    A[i]^=rand();
    A[i]<<=CHAR_BIT*sizeof(int)-1;
    A[i]^=rand();
    if(0) A[i]&=0x000ff00;
  }

  for(i=N;i;i>>=1) {
    unsigned long long t;
    TIME(t, (N/i), 
      sortv_long(out, A,i,sizeof(ulong), &buf));
    printf("sortv %d : %g cycles per item\n",
      (int)i, t/(double)(N/i)/(double)i);
  }

  for(i=N;i;i>>=1) {
    unsigned long long t;
    TIME(t, (N/i), 
      sortp_long(&buf,0, A,i,sizeof(ulong)));
    printf("sortp %d : %g cycles per item\n",
      (int)i, t/(double)(N/i)/(double)i);
  }

  buffer_free(&buf);
  return 0;
}

#else

int main()
{
  return 0;
}

#endif

