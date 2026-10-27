/************************************byte.c************************************
Module Name: byte
Module Info:

Author:  Henry M. Tufo III

e-mail:  hmt@cs.brown.edu

sn-mail: Division of Applied Mathematics, 
         Brown University,
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

Note: 
*************************************byte.c***********************************/

/* C modules for I/O etc. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <unistd.h>


#define READ     1
#define WRITE    2
#define MAX_NAME 80


#define SWAP(a,b)       temp=(a); (a)=(b); (b)=temp;


static FILE *fp=NULL;
static int  flag=0;
static char name[MAX_NAME+1];


/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_open_(char *n)
{
  int len;


  len = strlen(n);
  
  if (len<0)
    {printf("byte_open() :: file name has negative length!\n"); abort();}

  if (len>MAX_NAME)
    {printf("byte_open() :: file name too long!\n"); abort();}

  strcpy(name,n);
}



/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_close_()
{
  if (!fp)
    {return;}

  if (fclose(fp))
    {printf("byte_close() :: couldn't close file!\n"); abort();}

  fp=NULL;
}



/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_rewind_()
{
  if (!fp)
    {return;}

  rewind(fp);
}



/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_write_(float *buf, int *n)
{
#ifdef SAFE
  if (*n<0)
    {printf("byte_reverse() :: n must be positive\n"); abort();}
#endif

  if (!fp)
    {
      if (!(fp=fopen(name,"wb")))
	{printf("byte_write() :: fopen failure!\n"); abort();}
      flag=WRITE;
    }

  if (flag==WRITE)
    {fwrite(buf,sizeof(float),*n,fp);}
  else
    {printf("byte_write() :: can't write after reading!\n"); abort();}

}



/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_read_(float *buf, int *n)
{
#ifdef SAFE
  if (*n<0)
    {printf("byte_reverse() :: n must be positive\n"); abort();}
#endif

  if (!fp)
    {
      if (!(fp=fopen(name,"rb")))
	{printf("byte_read() :: fopen failure!\n"); abort();}
      flag=READ;
    }

  if (flag==READ)
    {fread(buf,sizeof(float),*n,fp);}
  else
    {printf("byte_read() :: can't read after writing!\n"); abort();}


}



/************************************byte.c************************************
Function: 

Input : 
Output: 
Return: 
Description:  
*************************************byte.c***********************************/
void
byte_reverse8_(float *buf, int *nn)
{
  int n;
  char temp, *ptr;


#ifdef SAFE
  if (*n<0)
    {printf("byte_reverse() :: n must be positive\n"); abort();}
#endif
  if (*nn % 2 != 0)
    {printf("byte_reverse() :: n must be multiple of 2\n"); abort();}
  
  for (ptr=(char *)buf,n=*nn,n=n+2; n-=2; ptr+=8)
    {
     SWAP(ptr[0],ptr[7])
     SWAP(ptr[1],ptr[6])
     SWAP(ptr[2],ptr[5])
     SWAP(ptr[3],ptr[4])
     }
}
/************************************byte.c***********************************/
void
byte_reverse_(float *buf, int *nn)
{
  int n;
  char temp, *ptr;


#ifdef SAFE
  if (*n<0)
    {printf("byte_reverse() :: n must be positive\n"); abort();}
#endif
  
  for (ptr=(char *)buf,n=*nn; n--; ptr+=4)
    {
      SWAP(ptr[0],ptr[3])
      SWAP(ptr[1],ptr[2])
     }
}






#ifdef NOT
main(int argc, char **argv)
{
  int i,j,k;
  int inplace;
  int input_ch;
  char ch, *name;
  FILE *ifp,*ofp;
  float rf[]={1.0,2.0,3.0,4.0};

  if (argc==2)
    {
      inplace=TRUE;
      printf("warning: original file will be destroyed!!!\n"); abort();
      printf("continue? (y/n) : ");
      input_ch = getchar();
      ch=toupper(input_ch);
      if (ch!='Y')
	{exit(0);}
    }
  else if (argc==3)
    {
      inplace=FALSE;
      printf("warning: new file will be created!!!\n");
      printf("continue? (y/n) : ");
      input_ch = getchar();
      ch=toupper(input_ch);
      if (ch!='Y')
	{exit(0);}
    }
  else
    {
      printf("usage: rbyte i/o_file\n");
      printf("or\n");
      printf("usage: rbyte i_file o_file\n");
      exit(0);
    }
  printf("continuing ...\n");

  if (inplace)
    {
      if ((name=tmpnam(NULL))==NULL)
	{printf("tmpname() failure!\n"); exit(0);}
    }
  else
    {
      name = *(argv+2);
    }

  printf("%s,%s\n",*(argv+1),*(argv+2));

  ifp = fopen(*(argv+1),"rb");
  ofp = fopen(name,"wb");

  if (!ifp||!ofp)
    {printf("fopen() failure!\n"); exit(0);}

  fwrite(&rf,sizeof(float),4,ofp);
  fclose(ifp);
  fclose(ofp);

  if (inplace)
    {
      if (remove(*(argv+1))||rename(name,*(argv+1)))
	{printf("remove() or rename() failure!\n"); exit(0);}
    }

  exit(0);
}
  
#endif
