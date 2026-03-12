/*==========================================================================

  File I/O for AMG
  small wrappers to read/write files consisting of doubles

  ==========================================================================*/

#ifndef MPI
#undef USEMPIIO
#endif

#ifdef NOMPIIO
#undef USEMPIIO
#endif

/* reverse the byte order of the given double
   portable up to 129-byte doubles, standards conforming */
#define N sizeof(double)
static double byteswap(double x)
{
  char t, buf[N];
  memcpy(buf,&x,N);
#define SWAP1(i) if(N>2*(i)+1) t=buf[i],buf[i]=buf[N-1-(i)],buf[N-1-(i)]=t
#define SWAP2(i) SWAP1(i); SWAP1((i)+0x01); SWAP1((i)+0x02); SWAP1((i)+0x03)
#define SWAP3(i) SWAP2(i); SWAP2((i)+0x04); SWAP2((i)+0x08); SWAP2((i)+0x0c)
#define SWAP4(i) SWAP3(i); SWAP3((i)+0x10); SWAP3((i)+0x20); SWAP3((i)+0x30)
  SWAP4(0);
#undef SWAP1
#undef SWAP2
#undef SWAP3
#undef SWAP4
  memcpy(&x,buf,N);
  return x;
}
#undef N

struct file {
  FILE *fptr;
  int swap;
};

#ifdef USEMPIIO
struct sfile {
  MPI_File fh;
  int swap;
};
#endif

static int rfread(void *ptr, size_t n, FILE *const fptr)
{
  size_t na; char *p=ptr;
  while(n && (na=fread (p,1,n,fptr))) n-=na, p+=na;
  return n!=0;
}

static int rfwrite(FILE *const fptr, const void *ptr, size_t n)
{
  size_t na; const char *p=ptr;
  while(n && (na=fwrite(p,1,n,fptr))) n-=na, p+=na;
  return n!=0;
}


static struct file dopen(const char *filename, const char mode,int *code)
{
  const double magic = 3.14159;
  double t;
  struct file f;
  f.fptr = fopen(filename,mode=='r'?"r":"w");
  f.swap = 0;
  if(f.fptr==0) {
   diagnostic("ERROR ",__FILE__,__LINE__,
              "AMG: could not open %s for %s",
              filename,mode=='r'?"reading":"writing");
   *code=1;
   return f;
  }
  if(mode=='r') {
    if(rfread(&t,sizeof(double), f.fptr)){
      diagnostic("ERROR ",__FILE__,__LINE__,
                 "AMG: could not read from %s",filename);
      *code=1;
      return f;
    }
    if(fabs(t-magic)>0.000001) {
      t=byteswap(t);
      if(fabs(t-magic)>0.000001) {
         diagnostic("ERROR ",__FILE__,__LINE__,
                    "AMG: magic number for endian test not found in %s",
                     filename);
         *code=1;
         return f;
      }
      f.swap = 1;
    }
  } else {
    if(rfwrite(f.fptr, &magic,sizeof(double))){
      diagnostic("ERROR ",__FILE__,__LINE__,
           "AMG: could not write to %s",filename);
      *code=1;
      return f;
    }
  }
  return f;
}

static void dread(double *p, size_t n, const struct file f,int *code)
{
  if(rfread(p,n*sizeof(double),f.fptr)) {
     diagnostic("ERROR ",__FILE__,__LINE__,
                "AMG: failed reading %u doubles from disk",(unsigned)n);
     *code=1;
     return;
  }
  if(f.swap) while(n) *p=byteswap(*p), ++p, --n;
}

static void dwrite(const struct file f, const double *p, size_t n,int *code)
{
  if(rfwrite(f.fptr, p,n*sizeof(double))) {
     diagnostic("ERROR ",__FILE__,__LINE__,
                "AMG: failed writing %u doubles to disk",(unsigned)n);
     *code=1;
  }
}

static void dclose(const struct file f)
{
  fclose(f.fptr);
}

#ifdef USEMPIIO
static void dread_mpi(double *buf, uint n, const struct sfile f,int *code)
{
  MPI_File fh=f.fh;
  MPI_Status status;
  int count=n;

  if(MPI_File_read_all(fh,buf,count,MPI_DOUBLE,&status)) {
     diagnostic("ERROR ",__FILE__,__LINE__,
                "AMG: failed reading %u doubles from disk",(unsigned)count);
     *code=1;
     return;
  }
  if(f.swap) while(count) *buf=byteswap(*buf), ++buf, --count;
}

static struct sfile dopen_mpi(const char *filename, const char mode,
                                 const struct comm *c,int *code)
{
  const double magic = 3.14159;
  double t;
  struct sfile f={0,0};
  MPI_File fh;

  int amode = mode=='r' ? MPI_MODE_RDONLY : MPI_MODE_CREATE|MPI_MODE_WRONLY;
  MPI_File_open(c->c,filename,amode,MPI_INFO_NULL,&fh);

  f.fh = fh; 

  if(f.fh==0) {
   diagnostic("ERROR ",__FILE__,__LINE__,
              "AMG: could not open %s for %s",
              filename,mode=='r'?"reading":"writing");
   *code=1;
   return f;
  }

  if(mode=='r') {
    dread_mpi(&t, 1, f, code);
    if(*code!=0) {
      diagnostic("ERROR ",__FILE__,__LINE__,
                 "AMG: could not read from %s",filename);
      return f;
    }
    if(fabs(t-magic)>0.000001) {
      t=byteswap(t);
      if(fabs(t-magic)>0.000001) {
         diagnostic("ERROR ",__FILE__,__LINE__,
                    "AMG: magic number for endian test not found in %s",
                     filename);
         *code=1;
         return f;
      }
      f.swap = 1;
    }
  } else {
    /* dwrite_mpi(&t, 1, f.fh, code); */
    if(*code!=0) {
      diagnostic("ERROR ",__FILE__,__LINE__,
                 "AMG: could not write to %s",filename);
      return f;
    }
  }
  return f;
}

static void dview_mpi(ulong n, const struct sfile f, int *code)
{
  MPI_File fh=f.fh;
  MPI_Info info;
  MPI_Offset disp=n*sizeof(double);

  if(MPI_File_set_view(fh,disp,MPI_DOUBLE,MPI_DOUBLE,"native",MPI_INFO_NULL)) {
     diagnostic("ERROR ",__FILE__,__LINE__,
                "AMG: failed chaning view to %u doubles",(unsigned)n);
     *code=1;
     return;
  }
}

static void dclose_mpi(const struct sfile f)
{
  MPI_File fh=f.fh;
  MPI_File_close(&fh);
}
#endif

