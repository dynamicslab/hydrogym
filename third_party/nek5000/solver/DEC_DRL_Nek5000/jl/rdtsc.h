#ifndef RDTSC_H
#define RDTSC_H

#define DEFINE_HW_COUNTER() \
static __inline__ unsigned long long getticks(void) \
{ \
   volatile unsigned low, high; \
   __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high)); \
   return ((unsigned long long)high)<<32 | low; \
}

#endif
