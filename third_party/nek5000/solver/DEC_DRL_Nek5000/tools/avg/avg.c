/************************************byte.c************************************
Module Name: byte
Module Info:

Author:  Henry M. Tufo III

e-mail:  hmt@cs.brown.edu

sn-mail: Division of Applied Mathematics, 
n         Brown University,
         Box F
         Providence, RI 02912

Tel:     (401) 863-7666


Last Modification: 11.24.97
*************************************byte.c***********************************/


/************************************byte.c************************************
NOTES ON USAGE: 

*************************************byte.c***********************************/

/************************************byte.c************************************
FILE FORMAT: 
------------------------------ Begin File -------------------------------------

------------------------------ End   File -------------------------------------

Note: cc -o fld_avg fld_avg.c -lm
*************************************byte.c***********************************/

/* C modules for I/O etc. */
/* sampson: cc -ansi -o cat_interp3 cat_interp3.c -lm */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <float.h>
#include <time.h>
#include <unistd.h>


#define MAX_BUF 512
#define TRUE  1
#define FALSE 0
#define TAG_VAL  6.54321
#define EPS      0.000001
#define MAX_NAME 80

#define SWAP(a,b)       temp=(a); (a)=(b); (b)=temp;
#define MAX(x,y)        ((x)>(y)) ? (x) : (y)
#define MIN(x,y)        ((x)<(y)) ? (x) : (y)


/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_reverse(float *buf, int n)
{
  char temp, *ptr;


#ifdef SAFE
  if (n<0)
    {printf("byte_reverse() :: n must be positive\n"); exit(1);}
#endif
  
  for (ptr=(char *)buf; n--; ptr+=4)
    {
      SWAP(ptr[0],ptr[3])
      SWAP(ptr[1],ptr[2])
     }
}


/********************************ivec.c**************************************
Function rvec_axpy()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void
rvec_axpy(register float *arg1, register float *arg2, register float scale, 
	  register int n)
{
  while (n--)  {*arg1++ += scale * *arg2++;}
}


/********************************ivec.c**************************************
Function rvec_zero()

Input : 
Output: 
Return: 
Description: 
*********************************ivec.c*************************************/
void 
rvec_zero(register float *arg1, register int n)
{
  while (n--)  {*arg1++ = 0.0;}
}


/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
main(argc, argv)
int argc;
char **argv;
{
  int i, npts, min, nr, nn=0, done=FALSE, rev=FALSE;
  float *wts, *tags, *out_buf, *in_buf;
  float sum, max_tag=-FLT_MAX, min_tag=FLT_MAX;
  char fld_name[MAX_NAME+1];
  FILE *nfp, **ifps, *ofp;


  /* check number of args */
  if (argc != 2)
    {printf("usage: exec file(fld time)\n"); exit(1);}

  /* how many fld files? */
  if (!(nfp = fopen(argv[1],"r")))
    {printf("can't open name file!\n"); exit(1);}
  while (fscanf(nfp,"%s\t%f",fld_name,&sum)==2) {nn++;}
  if (!nn) {printf("name file empty?\n"); exit(1);}
  rewind(nfp);

  /* get space for read/write buffers, etc. */
  if (!(out_buf=(float *)malloc(MAX_BUF*sizeof(float))))
    {printf("can't malloc space for out_buf!\n"); exit(1);}
  if (!( in_buf=(float *)malloc(MAX_BUF*sizeof(float))))
    {printf("can't malloc space for  in_buf!\n"); exit(1);}
  if (!(wts=(float *)malloc(nn*sizeof(float))))
    {printf("can't malloc space for weights!\n"); exit(1);}
  if (!(tags=(float *)malloc(nn*sizeof(float))))
    {printf("can't malloc space for tags!\n"); exit(1);}
  if (!(ifps=(FILE **)malloc(nn*sizeof(FILE *))))
    {printf("can't malloc space for file pointers!\n"); exit(1);}

  /* open fld files, get times and tags */
  for (sum=0.0,i=0;i<nn;i++)
    {
      if (fscanf(nfp,"%s\t%f",fld_name,wts+i)!=2)
	{printf("can't read names ... pass 2!\n"); exit(1);}
      if (!(ifps[i] = fopen(fld_name,"r")))
	{printf("can't open fld file %s (line %d)!\n",fld_name,i+1); exit(1);}
      if (fread(out_buf,1,80,ifps[i])!=80)
	{printf("can't read header info for fld file #%d!\n",i+1); exit(1);}
      if (fread(tags+i,sizeof(float),1,ifps[i])!=1)
	{printf("can't read tag for fld file #%d!\n",i+1); exit(1);}
      sum+=wts[i];
    }
  if (fclose(nfp))
    {printf("couldn't close name file!\n"); exit(1);}

  /* byte reverse? and determine weights */
  for (i=0;i<nn;i++) 
    {
      max_tag = MAX(max_tag,tags[i]);
      min_tag = MIN(min_tag,tags[i]);
      wts[i] /= sum;
    }
  if (max_tag!=min_tag)
    {printf("tags don't match!\n"); exit(1);}
/*else if (fabs(max_tag-TAG_VAL)>EPS)*/
  else if ( abs(max_tag-TAG_VAL)>EPS)
    {rev=TRUE;}
  else if (nn==1)
    {printf("only one fld file and tag ok ... done!\n"); exit(0);}

  /* time to make the donuts */
  printf("output file name? : ");
  scanf("%s",fld_name);
  if (!(ofp = fopen(fld_name,"w")))
    {printf("can't open output file!\n"); exit(1);}

  /* copy last input file's header info into outfiles */
  fwrite(out_buf,1,80,ofp);

  /* put in tag value */
  out_buf[0]=TAG_VAL;
  fwrite(out_buf,sizeof(float),1,ofp);
  while (!done)
    {
      rvec_zero(out_buf,MAX_BUF);
      for (i=0;i<nn;i++) 
	{
	  if ((nr=fread(in_buf,sizeof(float),MAX_BUF,ifps[i]))!=MAX_BUF)
	    {done=TRUE;}
	  if (rev) {byte_reverse(in_buf,nr);}
	  rvec_axpy(out_buf,in_buf,wts[i],nr);
	}
      npts+=nr;
      fwrite(out_buf,sizeof(float),nr,ofp);
    }

  for (i=0;i<nn;i++) 
    {
      if (fread(in_buf,sizeof(float),1,ifps[i]))
	{printf("poss. error ... something left in file %d!\n",i+1);}
      if (fclose(ifps[i]))
	{printf("couldn't close file %d!\n",i+1);}
    }
  if (fclose(ofp))
    {printf("couldn't close output file %d!\n",i+1);}

  printf("success ... found %d points.\n",npts);
  exit(0);
}

