#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <string.h>
#include "name.h"


#define cexit FORTRAN_UNPREFIXED(cexit, CEXIT)
#define print_stack FORTRAN_UNPREFIXED(print_stack, PRINT_STACK)
#define sizeOfLongInt FORTRAN_UNPREFIXED(sizeoflongint, SIZEOFLONGINT)
#define getmaxrss FORTRAN_UNPREFIXED(getmaxrss, GETMAXRSS)
#define set_stdout FORTRAN_UNPREFIXED(set_stdout, SET_STDOUT)

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

double getmaxrss()
{
  struct rusage r_usage;

  getrusage(RUSAGE_SELF,&r_usage);
#if defined(__APPLE__) && defined(__MACH__)
    return (double)r_usage.ru_maxrss;
#else
    return (double)(r_usage.ru_maxrss * 1024L);
#endif
}

int sizeOfLongInt()
{
  return sizeof(long int);
}

void set_stdout(char *f, int *sid, int flen)
{
  char *logfile = (char *) malloc((flen+2+5+1)*sizeof(char));
  strncpy(logfile, f, flen);
  int i;
  for (i=flen-1; i>=0; i--) if (logfile[i] != ' ') break;
  logfile[i+1] = '\0';

  int redirect = 0;
  char *envvar;

  if (logfile[0] != '\0') {
    redirect = 1;
  } 
  else if (envvar = getenv("NEK_LOGFILE")) {
    if (*sid >= 0) sprintf(logfile, "s%05d_", *sid);
    strcat(logfile + strlen(logfile), envvar);
    redirect = 1;
  }

  if (redirect) {
    printf("redirecting stdout to %s\n",logfile);
    freopen(logfile, "w+", stdout);
  }
}

void cexit(int *ierr)
{
  exit(*ierr);
}
