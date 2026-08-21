#ifndef ERRMEM_H
#define ERRMEM_H

#include <stdio.h>  /* printf */
#include <stdlib.h> /* malloc, calloc, realloc, free */
#include <stdarg.h> /* va_list, va_start, va_end */

/*--------------------------------------------------------------------------
   Error Reporting
   Memory Allocation Wrappers to Catch Out-of-memory
  --------------------------------------------------------------------------*/

static void fail(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(1);
}

static void failwith(const char *string)
{
  fail("%s\n",string);
}

static void *smalloc(size_t size, const char *file)
{
  void *res = malloc(size);
  if(!res && size) fail("%s: allocation of %d bytes failed\n",file,(int)size);
  return res;
}

static void *scalloc(size_t nmemb, size_t size, const char *file)
{
  void *res = calloc(nmemb, size);
  if(!res && nmemb)
    fail("%s: allocation of %d bytes failed\n",file,(int)size*nmemb);
  return res;
}

static void *srealloc(void *ptr, size_t size, const char *file)
{
  void *res = realloc(ptr, size);
  if(!res && size) fail("%s: allocation of %d bytes failed\n",file,(int)size);
  return res;
}

#define tmalloc(type, count)       smalloc((count)*sizeof(type),__FILE__)
#define tcalloc(type, count)       scalloc((count),sizeof(type),__FILE__)
#define trealloc(type, ptr, count) srealloc((ptr),(count)*sizeof(type),__FILE__)

#endif

