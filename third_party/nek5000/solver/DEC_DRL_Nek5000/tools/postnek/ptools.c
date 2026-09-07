/**********************************ptools.c************************************
PARALLEL TOOLS PACKAGE: ptools

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
9.10.98
***********************************ptools.c***********************************/

/**********************************ptools.c************************************
File Description:
-----------------


NOTE:
-----
   o valid tuple element range [0,INT_MAX]!!!
***********************************ptools.c***********************************/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#if   defined NXSRC
#ifndef DELTA
#include <nx.h>
#endif

#elif defined MPISRC
#include <mpi.h>

#endif

#include "const.h"
#include "types.h"
#include "ptools.h"
#include "ivec.h"
#include "error.h"
#include "comm.h"


/* sorting args ivec.c ivec.c ... */
#define   SORT_STACK	50


/* allocate an address and size stack for sorter(s) */
static void *offset_stack[2*SORT_STACK];
static int   size_stack[SORT_STACK];


static void p_irank_mtuple(int *r, int *tl_in, int n, int m, int k, int sort);
static int  max_compression_mtuple(int *tl, int n, int m, int k, int sort);
static void compress_mtuple(int *tl, int k, int k_comp);
static void compress_mtuple_byte(int *tl, int k, int k_comp);
static void compress_mtuple_bit(int *tl, int k, int k_comp);
static void sort_mtuple_companion(int *ar, int *ar2, int size, int step);
static int  ext_gt(unsigned int *arg1, unsigned int *arg2, int n);
static int  ext_lt(unsigned int *arg1, unsigned int *arg2, int n);
static int  ext_eq(unsigned int *arg1, unsigned int *arg2, int n);
static int  int_ext_gt(int *arg1, int *arg2, int n);
static int  int_ext_lt(int *arg1, int *arg2, int n);
static int  int_ext_eq(int *arg1, int *arg2, int n);
static void reduced_rank_mtuple(int *tl, int *rank, int *index, int n, int k);


static int lb, ub;
static unsigned int n_items, n_bytes, n_bits, n_ints_bit, n_ints_byte;
static int method;


/******************************************************************************
Function: ivec_rank_mtuple().

Input :
Output: 
Return:

Fortran interface to routine!
******************************************************************************/
#ifdef UPCASE
void
P_RANK_MTUPLE  (int *r, void *tl_in, int *n, int *m, int *k, int *if_sort, 
                char *type)
#else
void
p_rank_mtuple_ (int *r, void *tl_in, int *n, int *m, int *k, int *if_sort,
                char *type)
#endif
{
   if ((type[0]=='i')||(type[0]=='I'))
   {
#ifdef DEBUG      
      printf("n=%d,m=%d,k=%d,s=%d,t=%c\n",*n,*m,*k,*if_sort,type[0]);
      fflush(stdout);
#endif
      p_rank_mtuple(r,tl_in,*n,*m,*k,*if_sort,INTEGER);
   }
   else
   {
#ifdef UPCASE
      error_msg_fatal("P_RANK_MTUPLE() :: invalid type %c\n",type[0]);
#else
      error_msg_fatal("p_rank_mtuple_() :: invalid type %c\n",type[0]);
#endif
   }
}


/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
void 
p_rank_mtuple(int *r, void *tl_in, int n, int m, int k, int sort, int type)
{
   if (type==INTEGER)
   {
      p_irank_mtuple(r,(int *)tl_in,n,m,k,sort);
   }
   else
   {
      error_msg_fatal("p_rank_mtuple() :: invalid type %d\n",type);
   }
}       



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void 
p_irank_mtuple(int *r, int *tl_in, int n, int m, int k, int sort)
{
   int i,j;
   int ntot, k_comp;
   int *tl, *iptr1, *iptr2, *iend;
   int *index;


#ifdef DEBUG
   if ((n<0)||(m<0)||(k<0)||(m<k)) 
   {
      error_msg_fatal("p_irank_mtuple() :: bad (n,m,k)=(%d,%d,%d)\n",m,n,k);
   }

   if (!r||!tl_in)
   {
      error_msg_fatal("p_irank_mtuple() :: bad lists\n");
   }
#endif


   /* how much can we compress? */
   k_comp = max_compression_mtuple(tl_in,n,m,k,sort);

   /* leave incoming tuple data unchanged upon exit */
   /* copy, truncate to ktuple, sort, compress      */
   /* extra room for sentinal                       */
   ntot = k_comp*n;
   tl   = (int *) bss_malloc((ntot+k)*INT_LEN);
   for (iptr1=tl_in,iptr2=tl,iend=tl+ntot;iptr2<iend;iptr1+=m,iptr2+=k_comp)
   {
      ivec_copy(iptr2,iptr1,k);
      if (sort) {ivec_sort(iptr2,k);}
      if (k!=k_comp) {compress_mtuple(iptr2,k,k_comp);}
   }
   ivec_copy(iptr2,tl,k);

   /* initialize index */
   index = (int *) bss_malloc(n*INT_LEN);
   ivec_c_index(index,n);

   sort_mtuple_companion(tl,index,n,k_comp);
   reduced_rank_mtuple(tl,r,index,n,k_comp);
   bss_free(tl);
   bss_free(index);
}       



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
max_compression_mtuple(int *tl, int n, int m, int k, int sort)
{
   unsigned int i,j;
   int vals[2], work[2], oprs[2]={GL_MIN,GL_MAX};

   ivec_lb_ub(tl,n*k,vals,vals+1);
   giop(vals,work,2,oprs);

   /* tuple elements must be semi-pd ==> [0,INT_MAX] */
   if (vals[0]&INT_MIN) 
   {error_msg_fatal("max_compression_mtuple() : lb=%d<0!\n",lb);}

   lb = vals[0]; ub = vals[1]; 

   n_items=j=(unsigned int)(vals[1]-vals[0]+1);

   /* how many bits for each one? */
   i=0;
   while(j) {j>>=1; i++;}
   n_bits=i;

   /* how many bytes for each one? */
   n_bytes=i/BYTE;
   if (i%BYTE) {n_bytes++;}

   /* can stuff them all into how many ints? */
   /* bit level compression */
   j=n_bits*k;
   n_ints_bit=j/(INT_LEN*BYTE);
   if (j%(INT_LEN*BYTE)) {n_ints_bit++;}

   /* can stuff them all into how many ints? */
   /* byte level compression */
   j=n_bytes*k;
   n_ints_byte=j/INT_LEN;
   if (j%INT_LEN) {n_ints_byte++;}


#ifdef DEBUG
   if (!my_id)
#else
   if ((n_ints_bit!=n_ints_byte)&&(!my_id))
#endif
   {
      printf("lb=%d,ub=%d,n_items=%d\n",vals[0],vals[1],n_items);
      printf("n_bits    =%4d,n_bytes    =%4d\n",n_bits,n_bytes);
      printf("n_ints_bit=%4d,n_ints_byte=%4d\n",n_ints_bit,n_ints_byte);
      fflush(stdout);
   }
   return(MIN(n_ints_bit,n_ints_byte));
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void
compress_mtuple(int *tl, int k, int k_comp)
{
   unsigned int i,j;
   int val,val2,tmp;
   int n;


#ifdef SAFE
   if (k==k_comp) {return;}
#endif

#ifdef DEBUG
   if ((k_comp<=0)||(k_comp>k))
   {error_msg_fatal("compress_mtuple() :: k_comp=%d, k=%d\n",k_comp,k);}
#endif

   if (n_ints_byte==n_ints_bit)
   {compress_mtuple_byte(tl,k,k_comp);}
   else if (n_ints_bit<n_ints_byte)
   {compress_mtuple_bit(tl,k,k_comp);}
   else
   {error_msg_fatal("compress_mtuple() :: %d,$d\n",n_ints_byte,n_ints_bit);}
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void
compress_mtuple_byte(int *tl, int k, int k_comp)
{
   unsigned int i,j;
   int val,val2,tmp;
   int n;

   n=0;
   for (i=0;i<k_comp;i++)
   {
      val=val2=0;
      for (j=0;j<INT_LEN/n_bytes;j++)
      {
         tmp=tl[n]-lb;
         tmp<<=(INT_LEN/n_bytes-j-1)*n_bytes*BYTE;
         if (tmp&val)
         {
            printf("i=%d\n",i);
            printf("j=%d\n",j);
            printf("tmp=%d\n",tmp);
            printf("val=%d\n",val);
            for (i=0;i<k;i++)
               {printf("tl[%d]=%d\n",i,tl[i]);}
            exit(0);
         }


         val2|=tmp;
         val+=tmp;
         if (val!=val2)
         {error_msg_warning("compress_mtuple() :: val=%d,val2=%d\n",val,val2);}
         n++;
         if (n==k) break;
      }
      tl[i]=val;
      if (n==k) break;
   }
}


/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void
compress_mtuple_bit(int *tl, int k, int k_comp)
{
   error_msg_fatal("compress_mtuple_bit() :: not written yet!\n");
}





/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void
reduced_rank_mtuple(register int *tl, register int *rank, register int *index, 
                    register int n, register int k)
{
   register int i=1;

   while (n--)
   {
      rank[*index++]=i;
#ifdef DEBUG
      if (!n) {printf(" # vertices (unique) = %4d\n",i);}
#endif
      if (!ext_eq(tl,tl+k,k)) {i++;}
      tl+=k;
   }
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
void
sort_mtuple_companion(register int *ar, register int *ar2, register int size,
                      register int step)
{
  register int *pi, *pj, temp, temp2;
  register int **top_a = (int **)offset_stack;
  register int *top_s = size_stack, *bottom_s = size_stack; 
  register int *pi2, *pj2;
  register int mid;


#ifdef DEBUG  
  if (step<1) {error_msg_fatal("sort_mtuple_companion() :: step=%d\n",step);}
#endif

  /* we're really interested in the offset of the last element */
  /* ==> length of the list is now size + 1                    */
  size--;

  /* do until we're done ... return when stack is exhausted */
  for (;;)
  {
     /* if list is large enough use quicksort partition exchange code */
     if (size>3)
     {	
        /* start up pointer at element 1 and down at size     */  
        mid = size>>1;
        pi2 = ar2+1;
        pj2 = ar2+mid;

        mid *= step;
        pi = ar+step;
        pj = ar+mid;

        /* find middle element in list and swap w/ element 1 */
        for (mid=0;mid<step;mid++) {SWAP(*(pi+mid),*(pj+mid))}
        SWAP(*pi2,*pj2)

        /* order element 0,1,size-1 st {M,L,...,U} w/L<=M<=U */
        /* note ==> pivot_value in index 0                   */
        pj = ar+size*step;
        pj2 = ar2+size;
        if (ext_gt(pi,pj,step))
        {
           for (mid=0;mid<step;mid++) {SWAP(*(pi+mid),*(pj+mid))}
           SWAP(*pi2,*pj2)
        }          
        
        if (ext_gt(ar,pj,step))
        {
           for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(pj+mid))}
           SWAP(*ar2,*pj2)
        }
        else if (ext_gt(pi,ar,step))
        {
           for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+step+mid))}
           SWAP(*(ar2),*(ar2+1))
        }

        /* partition about pivot_value ...  	                    */
        /* note lists of length 2 are not guaranteed to be sorted */
        for(;;)
        {
           /* walk up ... and down ... swap if equal to pivot! */
           do {pi+=step; pi2++;} while (ext_lt(pi,ar,step));
           do {pj-=step; pj2--;} while (ext_gt(pj,ar,step));

           /* if we've crossed we're done */
           if (pj<pi) break;
            
           /* else swap */
           for (mid=0;mid<step;mid++) {SWAP(*(pi+mid),*(pj+mid))}
           SWAP(*pi2,*pj2)
        }

        /* place pivot_value in it's correct location */
        for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(pj+mid))}
        SWAP(*ar2,*pj2)
        
        /* test stack_size to see if we've exhausted our stack */
        if (top_s-bottom_s >= SORT_STACK)
        {error_msg_fatal("sort__mtuple_companion() :: STACK EXHAUSTED!!!");}
        
        /* push right hand child iff length > 1 */
        if (*top_s = size-((int) (pi2-ar2)))
        {
           *(top_a++) = pi;
           *(top_a++) = pi2;
           size -= *top_s+2;  
           top_s++;
        }
        /* set up for next loop iff there is something to do */
        else if (size -= *top_s+2) 
        {;}
        /* might as well pop - note NR_OPT >=2 ==> we're ok! */
        else
        {
           ar2 = *(--top_a);
           ar  = *(--top_a);
           size = *(--top_s);
        }
     }
     /* else sort small list directly then pop another off stack */
     else
     {
        switch (size) {
         case 3: /* lists of length four */
           /* a1<a3 and a2<a4 */
           if (ext_gt(ar,ar+2*step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+2*step+mid))}
              SWAP(*(ar2),*(ar2+2))
           }
           if (ext_gt(ar+step,ar+3*step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+step+mid),*(ar+3*step+mid))}
              SWAP(*(ar2+1),*(ar2+3))
           }
            
           /* reduce to three problem w/a2<a3 */
           if (ext_gt(ar+2*step,ar+3*step,step))   
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+2*step+mid),*(ar+3*step+mid))}
              SWAP(*(ar2+2),*(ar2+3))
              if (ext_gt(ar,ar+step,step))
              {
                 for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+step+mid))}
                 SWAP(*(ar2),*(ar2+1))
              }
              if (ext_gt(ar+step,ar+2*step,step))
              {
                 for (mid=0;mid<step;mid++) {SWAP(*(ar+step+mid),*(ar+2*step+mid))}
                 SWAP(*(ar2+1),*(ar2+2))
              }
              break;
           }
           /* reduced to standard three problem w/a1<a3 */
           else
           {
              if (ext_gt(ar+step,ar+2*step,step))
              {
                 for (mid=0;mid<step;mid++) {SWAP(*(ar+step+mid),*(ar+2*step+mid))}
                 SWAP(*(ar2+1),*(ar2+2))
              }
              if (ext_gt(ar,ar+step,step))
              {
                 for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+step+mid))}
                 SWAP(*(ar2),*(ar2+1))
              }
              break;
           }
           break;
           
         case 2: /*lists of length three */
           /* a1<a3 */
           if (ext_gt(ar,ar+2*step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+2*step+mid))}
              SWAP(*(ar2),*(ar2+2))
           }
           
           /* a2 belongs in last position */
           if (ext_gt(ar+step,ar+2*step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+step+mid),*(ar+2*step+mid))}
              SWAP(*(ar2+1),*(ar2+2))
              break;
           }
           
           /* a2 belongs in first position */
           if (ext_gt(ar,ar+step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+step+mid))}
              SWAP(*(ar2),*(ar2+1))
              break;
           }
            
           /* a2 already in place */
           break;

         case 1: /*lists of length two */
           if (ext_gt(ar,ar+step,step))
           {
              for (mid=0;mid<step;mid++) {SWAP(*(ar+mid),*(ar+step+mid))}
              SWAP(*(ar2),*(ar2+1))
              break;
           }
           break;
           /*note - lists of length one are sorted!! */
        }


        /* check to see if stack is exhausted ==> DONE */
        if (top_s==bottom_s) return;
        
        /* else pop another list from the stack */
        ar2 = *(--top_a);
        ar  = *(--top_a);
        size = *(--top_s);
     }
  }
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
ext_gt(unsigned int *arg1, unsigned int *arg2, int n)
{
   while (n--) 
   {
      if (*arg1<*arg2) return(FALSE);
      if (*arg1++>*arg2++) return(TRUE);
   }
   return(FALSE);
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
ext_lt(unsigned int *arg1, unsigned int *arg2, int n)
{
   while (n--) 
   {
      if (*arg1>*arg2) return(FALSE);
      if (*arg1++<*arg2++) return(TRUE);
   }
   return(FALSE);
}

   

/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
ext_eq(unsigned int *arg1, unsigned int *arg2, int n)
{
   while (n--) {if (*arg1++!=*arg2++) return(FALSE);}
   return(TRUE);
}


/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
int_ext_gt(int *arg1, int *arg2, int n)
{
   while (n--) 
   {
      if (*arg1<*arg2) return(FALSE);
      if (*arg1++>*arg2++) return(TRUE);
   }
   return(FALSE);
}



/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
int_ext_lt(int *arg1, int *arg2, int n)
{
   while (n--) 
   {
      if (*arg1>*arg2) return(FALSE);
      if (*arg1++<*arg2++) return(TRUE);
   }
   return(FALSE);
}

   

/**********************************ptools.c************************************
Function 

Input : 
Output: 
Return: 
Description: 
***********************************ptools.c***********************************/
static
int
int_ext_eq(int *arg1, int *arg2, int n)
{
   while (n--) {if (*arg1++!=*arg2++) return(FALSE);}
   return(TRUE);
}
