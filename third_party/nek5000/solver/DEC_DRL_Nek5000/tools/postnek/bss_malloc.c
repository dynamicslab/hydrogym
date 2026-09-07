/********************************bss_malloc.c**********************************
SPARSE GATHER-SCATTER PACKAGE: bss_malloc bss_malloc ivec error comm gs queue

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
11.21.97
*********************************bss_malloc.c*********************************/

/********************************bss_malloc.c**********************************
File Description:
-----------------

*********************************bss_malloc.c*********************************/
#include <stdio.h>
#include <stdlib.h>

#if   defined NXSRC
#ifndef DELTA
#include <nx.h>
#endif

#elif defined MPISRC
#include <mpi.h>

#endif

#include "const.h" 
#include "types.h"
#include "ivec.h" 
#include "error.h" 
#include "bss_malloc.h"


#ifdef NXLIB
#include <nxmalloc.h>
#endif

#if   defined NXSRC
#ifndef DELTA
#include <nx.h>
#endif
#include "comm.h"

#elif defined MPISRC
#include <mpi.h>
#include "comm.h"

#else
static int my_id=0;
static int num_nodes=1;


#endif


/* mission critical */
/* number of bytes given to malloc */
#ifdef MYMALLOC
#define PERM_MALLOC_BUF  4194304 /* 16777216 8388608 4194304 31072 16384 */
#define BSS_MALLOC_BUF   4194304 /* 524288  1048576 4194304 65536 */
#endif


void *malloc(size_t size);


/* malloc stats and space for bss and perm flavors */
static int    perm_req       = 0;
static int    num_perm_req   = 0;
static int    num_perm_frees = 0;
#ifdef MYMALLOC
static double perm_buf[PERM_MALLOC_BUF/sizeof(double)];
static double *perm_top = perm_buf;
#endif

static int    bss_req        = 0;
static int    num_bss_req    = 0;
static int    num_bss_frees  = 0;
#ifdef MYMALLOC
static double bss_buf[BSS_MALLOC_BUF/sizeof(double)];
static double *bss_top = bss_buf;
#endif



/********************************bss_malloc.c**********************************
Function: perm_init()

Input : 
Output: 
Return: 
Description: 

Add ability to pass in later ... for Fortran interface

Space to be passed later should be double aligned!!!
*********************************bss_malloc.c*********************************/
void 
perm_init(void)
{
  perm_req = 0;
  num_perm_req = 0;
  num_perm_frees = 0;

#ifdef MYMALLOC
  perm_top = perm_buf;
#endif
}



/********************************bss_malloc.c**********************************
Function: perm_malloc()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void *
perm_malloc(size_t size)
{
  void *tmp;
#ifdef MYMALLOC  
  double *space;
  int num_blocks;
#endif


  if (!size)
    {
#ifdef DEBUG
      error_msg_warning("perm_malloc() :: size=0"); 
#endif
      return(NULL);
    }

#if defined MYMALLOC
  if (size%sizeof(double))
    {num_blocks = size/sizeof(double) + 1;}
  else
    {num_blocks = size/sizeof(double);}
  
  if (num_blocks < (PERM_MALLOC_BUF/sizeof(double) - (perm_top - perm_buf)))
    {
      space = perm_top;
      perm_top += num_blocks;
      perm_req+=size;
      num_perm_req++;
      return(space);
    }

#else 
  if ((tmp = (void *) malloc(size)))
    {
      perm_req+=size;
      num_perm_req++;
      return(tmp);
    }
#endif

  error_msg_fatal("perm_malloc() :: can't satisfy %d byte request",size);
  return(NULL);
}



/********************************bss_malloc.c**********************************
Function: perm_free()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void 
perm_free(void *ptr)
{
  if (ptr)
    {
      num_perm_frees--;  

#ifdef MYMALLOC
      return;
#else
      free((void *) ptr);
#endif
    }
  else
    {error_msg_warning("perm_free() :: nothing to free!!!");}
}



/********************************bss_malloc.c**********************************
Function: perm_stats()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void 
perm_stats(void)
{
  int oprs[] = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD};
  int vals[sizeof(oprs)/sizeof(oprs[0])-1];
  int work[sizeof(oprs)/sizeof(oprs[0])-1];

  vals[0]=vals[1]=vals[2]=perm_req;
  giop(vals,work,sizeof(oprs)/sizeof(oprs[0])-1,oprs);
  vals[2]/=num_nodes;

  if (!my_id)
    {
      printf("%d :: perm_malloc stats:\n",my_id);
      printf("%d :: perm_req min = %d\n",my_id,vals[0]);
      printf("%d :: perm_req max = %d\n",my_id,vals[1]);
      printf("%d :: perm_req ave = %d\n",my_id,vals[2]);
      printf("%d :: perm_req me  = %d\n",my_id,perm_req);
    }

#ifdef DEBUG
  /* check to make sure that malloc and free calls are balanced */
  if (num_perm_frees+num_perm_req)
    {
      printf("%d :: perm # frees = %d\n",my_id,-1*num_perm_frees);
      printf("%d :: perm # calls = %d\n",my_id,num_perm_req);
    }

  fflush(stdout);
#endif
}



/********************************bss_malloc.c**********************************
Function: perm_frees()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
int
perm_frees(void)
{
  return(-num_perm_frees);
}



/********************************bss_malloc.c**********************************
Function: perm_calls()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
int
perm_calls(void)
{
  return(num_perm_req);
}



/********************************bss_malloc.c**********************************
Function: bss_init()

Input : 
Output: 
Return: 
Description: 

Add ability to pass in later ...

Space to be passed later should be double aligned!!!
*********************************bss_malloc.c*********************************/
void 
bss_init(void)
{
  bss_req = 0;
  num_bss_req = 0;
  num_bss_frees = 0;

#ifdef MYMALLOC
  bss_top = bss_buf;
#endif
}



/********************************bss_malloc.c**********************************
Function: bss_malloc()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void *
bss_malloc(size_t size)
{
  void *tmp;  
#ifdef MYMALLOC  
  double *space;
  int num_blocks;
#endif


  if (!size)
    {
#ifdef DEBUG
      error_msg_warning("bss_malloc() :: size=0!"); 
#endif
      return(NULL);
    }

#ifdef MYMALLOC
  if (size%sizeof(double))
    {num_blocks = size/sizeof(double) + 1;}
  else
    {num_blocks = size/sizeof(double);}

  if (num_blocks < (BSS_MALLOC_BUF/sizeof(double) - (bss_top - bss_buf)))
    {
      space = bss_top;
      bss_top += num_blocks;
      bss_req+=size;
      num_bss_req++;
      return(space);
    }

#else
  if (tmp = (void *) malloc(size))
    {
      bss_req+=size;
      num_bss_req++;
      return(tmp);
    }
#endif

  error_msg_fatal("bss_malloc() :: can't satisfy %d request",size);
  return(NULL);
}



/********************************bss_malloc.c**********************************
Function: bss_free()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void 
bss_free(void *ptr)
{
  if (ptr)
    {
      num_bss_frees--;

#ifdef MYMALLOC
      return;
#else
      free((void *) ptr);
#endif
    }
  else
    {error_msg_warning("bss_free() :: nothing to free!!!");}
}



/********************************bss_malloc.c**********************************
Function: bss_stats()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
void 
bss_stats(void)
{
  int oprs[] = {NON_UNIFORM,GL_MIN,GL_MAX,GL_ADD};
  int vals[sizeof(oprs)/sizeof(oprs[0])-1];
  int work[sizeof(oprs)/sizeof(oprs[0])-1];

  vals[0]=vals[1]=vals[2]=bss_req;
  giop(vals,work,sizeof(oprs)/sizeof(oprs[0])-1,oprs);
  vals[2]/=num_nodes;

  if (!my_id)
    {
      printf("%d :: bss_malloc stats:\n",my_id);
      printf("%d :: bss_req min  = %d\n",my_id,vals[0]);
      printf("%d :: bss_req max  = %d\n",my_id,vals[1]);
      printf("%d :: bss_req ave  = %d\n",my_id,vals[2]);
      printf("%d :: bss_req me   = %d\n",my_id,bss_req);
    }

#ifdef DEBUG
  /* check to make sure that malloc and free calls are balanced */
  if (num_bss_frees+num_bss_req)
    {
      printf("%d :: bss # frees  = %d\n",my_id,-1*num_bss_frees);
      printf("%d :: bss # calls  = %d\n",my_id,num_bss_req);
    }

  fflush(stdout);
#endif
}



/********************************bss_malloc.c**********************************
Function: bss_frees()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
int
bss_frees(void)
{
  return(-num_bss_frees);
}



/********************************bss_malloc.c**********************************
Function: bss_calls()

Input : 
Output: 
Return: 
Description: 
*********************************bss_malloc.c*********************************/
int
bss_calls(void)
{
  return(num_bss_req);
}


