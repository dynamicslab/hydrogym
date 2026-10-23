#include <stdio.h>
#include <stdlib.h>
#include "name.h"

#define print_stack FORTRAN_UNPREFIXED(print_stack, PRINT_STACK)
#define sizeOfLongInt FORTRAN_UNPREFIXED(sizeoflongint, SIZEOFLONGINT)

#if defined __GLIBC__

#include <execinfo.h>
/* Obtain a backtrace and print it to stdout. */
void print_stack(void)
{
  void *bt[50];
  int i;
  int bt_size = backtrace(bt, 50);
  char **symbols = backtrace_symbols(bt, bt_size);
     
  printf ("backtrace(): obtained %d stack frames.\n", bt_size);
  for (i=0; i<bt_size; i++) printf("%s\n", symbols[i]);
  free (symbols);
}
#else
void print_stack(){};
#endif

int sizeOfLongInt()
{
  return sizeof(long int);
}
