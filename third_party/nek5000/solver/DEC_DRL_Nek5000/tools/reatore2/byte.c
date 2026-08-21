#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef FNAME_H
#define FNAME_H

/*
   FORTRAN naming convention
     default      cpgs_setup, etc.
     -DUPCASE     CPGS_SETUP, etc.
     -DUNDERSCORE cpgs_setup_, etc.
*/

#ifdef UPCASE
#  define FORTRAN_NAME(low,up) up
#else
#ifdef UNDERSCORE
#  define FORTRAN_NAME(low,up) low##_
#else
#  define FORTRAN_NAME(low,up) low
#endif
#endif

#endif

#define byte_reverse FORTRAN_NAME(byte_reverse, BYTE_REVERSE)
#define byte_open    FORTRAN_NAME(byte_open,    BYTE_OPEN   )
#define byte_close   FORTRAN_NAME(byte_close,   BYTE_CLOSE  )
#define byte_rewind  FORTRAN_NAME(byte_rewind,  BYTE_REWIND )
#define byte_read    FORTRAN_NAME(byte_read,    BYTE_READ   )
#define byte_write   FORTRAN_NAME(byte_write,   BYTE_WRITE  )
#define set_bytesw_write FORTRAN_NAME(set_bytesw_write,SET_BYTESW_WRITE)
#define set_bytesw_read  FORTRAN_NAME(set_bytesw_read ,SET_BYTESW_READ )
#define get_bytesw_write FORTRAN_NAME(get_bytesw_write,GET_BYTESW_WRITE)
#define get_bytesw_read  FORTRAN_NAME(get_bytesw_read ,GET_BYTESW_READ )

#define READ     1
#define WRITE    2
#define MAX_NAME 132

#define SWAP(a,b)       temp=(a); (a)=(b); (b)=temp;

static FILE *fp=NULL;
static int  flag=0;
static char name[MAX_NAME+1];

int bytesw_write=0;
int bytesw_read=0;

/*************************************byte.c***********************************/

#ifdef UNDERSCORE
  void exitt_();
#else
  void exitt();
#endif

void cexitt()
{
#ifdef UNDERSCORE
  exitt_();
#else
  exitt();
#endif
}

void byte_reverse(float *buf, int *nn)
{
  int n;
  char temp, *ptr;

  if (*nn<0)
  {
    printf("byte_reverse() :: n must be positive\n"); 
    cexitt();
  }
  
  for (ptr=(char *)buf,n=*nn; n--; ptr+=4)
  {
     SWAP(ptr[0],ptr[3])
     SWAP(ptr[1],ptr[2])
  }
}


void byte_open(char *n)
{
  int  i,len,istat;
  char slash;
  char dirname[MAX_NAME+1];

  len = strlen(n);
  
  if (len<0)
  {
    printf("byte_open() :: file name has negative length!\n"); 
    cexitt();
  }

  if (len>MAX_NAME)
  {
    printf("byte_open() :: file name too long!\n"); 
    cexitt();
  }

  strcpy(name,n);
  strcpy(dirname,n);

  for (i=1;dirname[i]!='\0';i++)
  {
     if (i>0 && dirname[i]=='/')
     {
       slash = name[i];
       dirname[i] = '\0';
       istat = mkdir(dirname,0755);
     }
  }
}

void byte_close()
{
  if (!fp) return;

  if (fclose(fp))
  {
    printf("byte_close() :: couldn't fclose file!\n");
    cexitt();
  }

  fp=NULL;
}

void byte_rewind()
{
  if (!fp) return;

  rewind(fp);
}


void byte_write(float *buf, int *n)
{
  int flags;
  mode_t mode;

  if (*n<0)
  {
    printf("byte_write() :: n must be positive\n"); 
    cexitt();
  }

  if (!fp)
  {
    if (!(fp=fopen(name,"wb")))
    {
      printf("byte_write() :: fopen failure!\n"); 
      cexitt();
    }
    flag=WRITE;
  }

  if (flag==WRITE)
    {
      if (bytesw_write == 1)
        byte_reverse (buf,n);
      fwrite(buf,sizeof(float),*n,fp);
    }
  else
  {
      printf("byte_write() :: can't fwrite after freading!\n"); 
      cexitt();
  }
}


void byte_read(float *buf, int *n)
{
  int flags;
  mode_t mode;

  if (*n<0)
    {printf("byte_read() :: n must be positive\n"); cexitt();}

  if (!fp)
  {
     if (!(fp=fopen(name,"rb")))
     {
        printf("%s\n",name);
        printf("byte_read() :: fopen failure2!\n"); cexitt();
     }
     flag=READ;
  }

  if (flag==READ)
  {
     if (bytesw_read == 1)
        byte_reverse (buf,n);
     fread(buf,sizeof(float),*n,fp);
     if (ferror(fp))
     {
       printf("ABORT: Error reading %s\n",name);
       cexitt();
     }
     else if (feof(fp))
     {
       printf("ABORT: EOF found while reading %s\n",name);
       cexitt();
     }

  }
  else
  {
     printf("byte_read() :: can't fread after fwriting!\n"); 
     cexitt();
  }
}

void set_bytesw_write (int *pa)
{
    if (*pa != 0)
       bytesw_write = 1;
    else
       bytesw_write = 0;
}

void set_bytesw_read (int *pa)
{
    if (*pa != 0)
       bytesw_read = 1;
    else
       bytesw_read = 0;
}

void get_bytesw_write (int *pa)
{
    *pa = bytesw_write;
}

void get_bytesw_read (int *pa)
{
    *pa = bytesw_read;
}
