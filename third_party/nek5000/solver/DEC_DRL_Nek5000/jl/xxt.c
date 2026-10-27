#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "tensor.h"
#include "gs_defs.h"
#include "comm.h"
#include "mem.h"
#include "sort.h"
#include "sarray_sort.h"
#include "sparse_cholesky.h"
#include "gs.h"

#define crs_setup PREFIXED_NAME(crs_setup)
#define crs_solve PREFIXED_NAME(crs_solve)
#define crs_stats PREFIXED_NAME(crs_stats)
#define crs_free  PREFIXED_NAME(crs_free )

/*
  portable log base 2
  
  does a binary search to find leading order bit
  
  UINT_BITS = number of bits in a uint
  BITS(0) = UINT_BITS
  BITS(i) = half of BITS(i-1), rounded up
  MASK(i) = bitmask with BITS(i) 1's followed by BITS(i) 0's
*/

static unsigned lg(uint v)
{
  unsigned r = 0;
#define UINT_BITS (sizeof(uint)*CHAR_BIT)
#define BITS(i) ((UINT_BITS+(1<<(i))-1)>>(i))
#define MASK(i) ((((uint)1<<BITS(i)) - 1) << BITS(i))
#define CHECK(i) if((BITS(i)!=1) && (v&MASK(i))) v>>=BITS(i), r+=BITS(i)  
  CHECK(1); CHECK(2); CHECK(3); CHECK(4); CHECK(5); CHECK(6); CHECK(7);
  CHECK(8); CHECK(9); /* this covers up to 1024-bit uints */
  if(v&2) ++r;
  return r;
#undef UINT_BITS
#undef BITS
#undef MASK
#undef CHECK
}

struct csr_mat {
  uint n, *Arp, *Aj; double *A;
};

struct xxt {

  /* communication */

  struct comm comm;
  uint pcoord;   /* coordinate in communication tree */ 
  unsigned plevels; /* # of stages of communication */
  sint *pother;     /* let p = pother[i], then during stage i of fan-in,
                           if p>=0, receive from p
                           if p< 0, send to (-p-1)
                       fan-out is just the reverse ...
                       on proc 0, pother is never negative
                       on others, pother is negative for the last stage only */
  comm_req *req;
  
  /* separators */

  unsigned nsep;  /* number of separators */
  uint *sep_size; /* # of dofs on each separator,
                     ordered from the bottom to the top of the tree:
                     separator 0 is the bottom-most one (dofs not shared)
                     separator nsep-1 is the root of the tree */

  unsigned null_space;
  double *share_weight;

  /* vector sizes */

  uint un;        /* user's vector size */
  
  /* xxt_solve works with "condensed" vectors;
     same dofs as in user's vectors, but no duplicates and no Dirichlet nodes,
     and also ordered topologically (children before parents) according to the
     separator tree */
  
  uint cn;        /* size of condensed vectors */
  sint *perm_u2c; /* permutation from user vector to condensed vector,
                     p=perm_u2c[i]; xu[i] = p=-1 ? 0 : xc[p];          */
  uint ln, sn;    /* xc[0 ... ln-1] are not shared   (ln=sep_size[0])
                     xc[ln ... ln+sn-1] are shared
                     ln+sn = cn                    */
  
  uint xn;        /* # of columns of x = sum_i(sep_size[i]) - sep_size[0] */

  /* data */
  struct sparse_cholesky fac_A_ll;
  struct csr_mat             A_sl;
  uint *Xp; double *X;   /* column i of X starts at X[Xp[i]] */
  
  /* execution buffers */
  double *vl, *vc, *vx, *combuf;
};


/*
  for the binary communication tree, the procs are divided in half
  at each level, with the second half always the larger
  
  e.g., for np = 13:

       +------13-------+
       |               |
   +---6---+       +---7---+
   |       |       |       |
 +-3-+   +-3-+   +-3-+   +-4-+
 1   2   1   2   1   2   2   2
    1^1     1^1     1^1 1^1 1^1

  plevels is the number of levels in the tree
    = np==1 ? 1 : ( lg(np-1)+2 )

  labelling the nodes with proc id's gives this communication tree:

       +-------0-------+
       |               |
   +---0---+       +---6---+
   |       |       |       |
 +-0-+   +-3-+   +-6-+   +-9-+
 0   1   3   4   6   7   9   b
    1^2     4^5     7^8 9^a b^c

  consider proc 7 (pid = 7);
  pcoord gives the position of the leaf labelled 7:
    Root Right Left Right Left -> RRLRL -> 11010
    so pcoord = 11010 binary
  note the parent coordinate can be found by bit shifting right
    (i.e. dividing by 2)
*/

/* sets: pcoord, nsep, plevels, pother, req */
static void locate_proc(struct xxt *data)
{
  const uint id = data->comm.id;
  uint n = data->comm.np, c=1, odd=0, base=0;
  unsigned level=0;
  while(n>1) {
    ++level;
    odd=(odd<<1)|(n&1);
    c<<=1, n>>=1;
    if(id>=base+n) c|=1, base+=n, n+=(odd&1);
  }
  data->pcoord=c;
  data->nsep = level+1;
  data->plevels = data->nsep-1;
  data->pother = tmalloc(sint,data->plevels);
  data->req = tmalloc(comm_req,data->plevels);
  for(level=0;level<data->plevels;++level) {
    if((c&1)==1) {
      uint targ = id - (n-(odd&1));
      data->pother[level]=-(sint)(targ+1);
      data->plevels = level+1;
      break;
    } else {
      data->pother[level]=id+n;
      c>>=1, n=(n<<1)+(odd&1), odd>>=1;
    }
  }
}

/* the tuple list describing the condensed dofs:
   [(separator level, share count, global id)] */
struct dof { ulong id; uint level, count; };

/* determine the size of each separator;
   sums the separator sizes following the fan-in, fan-out comm. pattern
   uses the share-counts to avoid counting dofs more than once */
/* sets: xn, sep_size, ln, sn */
static void discover_sep_sizes(struct xxt *data,
                               struct array *dofa, buffer *buf)
{
  const unsigned ns=data->nsep, nl=data->plevels;
  const uint n = dofa->n;
  float *v, *recv;
  unsigned i,lvl; uint j;
  const struct dof *dof = dofa->ptr;

  buffer_reserve(buf,2*ns*sizeof(float));
  v=buf->ptr, recv=v+ns;
    
  for(i=0;i<ns;++i) v[i]=0;
  for(j=0;j<n;++j) v[dof[j].level]+=1/(float)dof[j].count;

  /* fan-in */
  for(lvl=0;lvl<nl;++lvl) {
    sint other = data->pother[lvl];
    unsigned s = ns-(lvl+1);
    if(other<0) {
      comm_send(&data->comm,v   +lvl+1,s*sizeof(float),-other-1,s);
    } else {
      comm_recv(&data->comm,recv+lvl+1,s*sizeof(float),other,s);
      for(i=lvl+1;i<ns;++i) v[i]+=recv[i];
    }
  }
  /* fan-out */
  for(;lvl;) {
    sint other = data->pother[--lvl];
    unsigned s = ns-(lvl+1);
    if(other<0)
      comm_recv(&data->comm,v+lvl+1,s*sizeof(float),-other-1,s);
    else
      comm_send(&data->comm,v+lvl+1,s*sizeof(float),other,s);
  }

  data->xn=0;    
  data->sep_size = tmalloc(uint,ns);
  for(i=0;i<ns;++i) { 
    uint s=v[i]+.1f;
    data->sep_size[i]=s;
    data->xn+=s;
  }
  data->ln=data->sep_size[0];
  data->sn=data->cn-data->ln;
  data->xn-=data->ln;
}

/* assuming [A,Aend) is sorted,
   removes 0's and any duplicate entries,
   returns new end */
static ulong *unique_nonzero(ulong *A, ulong *Aend)
{
  if(Aend==A) return A;
  else {
    ulong *end = Aend-1, last=*end, *p=A,*q=A,v=0;
    *end = 1;
    while(*q==0) ++q;   /*  *q==0 => q!=end since *end==0 */
    *end = 0;
    while(q!=end) {
      v=*q++, *p++=v; 
      while(*q==v) ++q; /*  *q==v => q!=end since *end==0 */
    }
    if(last!=v) *p++=last;
    return p;
  }
}

static void merge_sep_ids(struct xxt *data, ulong *sep_id, ulong *other,
                          ulong *work, unsigned s0, buffer *buf)
{
  const unsigned ns = data->nsep;
  unsigned s;
  ulong *p=sep_id, *q=other;
  for(s=s0;s<ns;++s) {
    ulong *end;
    uint size = data->sep_size[s];
    memcpy(work     ,p,size*sizeof(ulong));
    memcpy(work+size,q,size*sizeof(ulong));
    sortv_long(work, work,2*size,sizeof(ulong), buf);
    end = unique_nonzero(work,work+2*size);
    memcpy(p,work,(end-work)*sizeof(ulong));
    p+=size, q+=size;
  }
}

static void init_sep_ids(struct xxt *data, struct array *dofa, ulong *xid)
{
  const unsigned ns=data->nsep;
  const uint n=data->cn, *sep_size=data->sep_size;
  unsigned s=1;
  uint i, size;
  const struct dof *dof = dofa->ptr;
  if(ns==1) return;
  size=sep_size[s];
  for(i=data->ln;i<n;++i) {
    unsigned si = dof[i].level;
    while(s!=si) {
      memset(xid,0,size*sizeof(ulong));
      xid+=size;
      if(++s != ns) size=data->sep_size[s];
    }
    *xid++ = dof[i].id, --size;
  }
  while(s!=ns) {
    memset(xid,0,size*sizeof(ulong));
    xid+=size;
    if(++s != ns) size=data->sep_size[s];
  }
}

static void find_perm_x2c(uint ln, uint cn, const struct array *dofc,
                          uint xn, const ulong *xid, sint *perm)
{
  const struct dof *dof = dofc->ptr, *dof_end = dof+cn;
  const ulong *xid_end = xid+xn; uint i=ln;
  dof+=ln;
  while(dof!=dof_end) {
    ulong v=dof->id;
    while(*xid!=v) ++xid, *perm++ = -1;
    *perm++ = i++, ++dof, ++xid;
  }
  while(xid!=xid_end) ++xid, *perm++ = -1;
}

/* sets: perm_x2c */
static sint *discover_sep_ids(struct xxt *data, struct array *dofa, buffer *buf)
{
  const unsigned ns=data->nsep, nl=data->plevels;
  const uint xn=data->xn, *sep_size=data->sep_size;
  ulong *xid, *recv, *work, *p;
  unsigned lvl;
  uint size,ss;
  sint *perm_x2c;
  
  size=0; for(lvl=1;lvl<ns;++lvl) if(sep_size[lvl]>size) size=sep_size[lvl];
  xid=tmalloc(ulong,2*xn+2*size), recv=xid+xn, work=recv+xn;
  
  init_sep_ids(data,dofa,xid);

  if(nl) {
    /* fan-in */
    p=xid, size=xn;
    for(lvl=0;lvl<nl;++lvl) {
      sint other = data->pother[lvl];
      if(other<0) {
        comm_send(&data->comm,p   ,size*sizeof(ulong),-other-1,size);
      } else {
        comm_recv(&data->comm,recv,size*sizeof(ulong),other,size);
        merge_sep_ids(data,p,recv,work,lvl+1,buf);
      }
      ss=data->sep_size[lvl+1];
      if(ss>=size || lvl==nl-1) break;
      p+=ss, size-=ss;
    }
    /* fan-out */
    for(;;) {
      sint other = data->pother[lvl];
      if(other<0)
        comm_recv(&data->comm,p,size*sizeof(ulong),-other-1,size);
      else
        comm_send(&data->comm,p,size*sizeof(ulong),other,size);
      if(lvl==0) break;
      ss=data->sep_size[lvl];
      p-=ss, size+=ss, --lvl;
    }
  }
 
  perm_x2c=tmalloc(sint,xn);
  find_perm_x2c(data->ln,data->cn,dofa, xn,xid, perm_x2c);
  free(xid);
  
  return perm_x2c;
}

static void apply_QQt(struct xxt *data, double *v, uint n, uint tag)
{
  const unsigned nl=data->plevels;
  double *p=v, *recv=data->combuf;
  unsigned lvl, nsend=0;
  uint size=n, ss;
  
  if(n==0 || nl==0) return;

  tag=tag*2+0;
  /* fan-in */
  for(lvl=0;lvl<nl;++lvl) {
    sint other = data->pother[lvl];
    if(other<0) {
      comm_send(&data->comm,p   ,size*sizeof(double),-other-1,tag);
    } else {
      uint i;
      comm_recv(&data->comm,recv,size*sizeof(double),other   ,tag);
      for(i=0;i<size;++i) p[i]+=recv[i];
    }
    ss=data->sep_size[lvl+1];
    if(ss>=size || lvl==nl-1) break;
    p+=ss, size-=ss;
  }
  /* fan-out */
  for(;;) {
    sint other = data->pother[lvl];
    if(other<0) {
      comm_recv (&data->comm,p,size*sizeof(double),-other-1,tag);
    } else {
      comm_isend(&data->req[nsend++],&data->comm,
                             p,size*sizeof(double),other   ,tag);
    }
    if(lvl==0) break;
    ss=data->sep_size[lvl];
    p-=ss, size+=ss, --lvl;
  }
  if(nsend) comm_wait(data->req,nsend);
}

static double sum(struct xxt *data, double v, uint n, uint tag)
{
  const unsigned nl=data->plevels;
  double r;
  unsigned lvl,nsend=0;
  uint size=n, ss;

  tag=tag*2+1;
  if(n==0 || nl==0) return v;
  /* fan-in */
  for(lvl=0;lvl<nl;++lvl) {
    sint other = data->pother[lvl];
    if(other<0) {
      comm_send(&data->comm,&v,sizeof(double),-other-1,tag);
    } else {
      comm_recv(&data->comm,&r,sizeof(double),other   ,tag);
      v+=r;
    }
    ss=data->sep_size[lvl+1];
    if(ss>=size || lvl==nl-1) break;
    size-=ss;
  }
  /* fan-out */
  for(;;) {
    sint other = data->pother[lvl];
    if(other<0) {
      comm_recv (&data->comm,&v,sizeof(double),-other-1,tag);
    } else {
      comm_isend(&data->req[nsend++],&data->comm,
                             &v,sizeof(double),other   ,tag);
    }
    if(lvl==0) break;
    ss=data->sep_size[lvl];
    size+=ss, --lvl;
  }
  if(nsend) comm_wait(data->req,nsend);
  return v;
}

/* sorts an array of ids, removes 0's and duplicates;
   just returns the permutation */
static uint unique_ids(uint n, const ulong *id, sint *perm, buffer *buf)
{
  uint *p, i, un=0; ulong last=0;
  p = sortp_long(buf,0, id,n,sizeof(ulong));
  for(i=0;i<n;++i) {
    uint j = p[i]; ulong v = id[j];
    if(v==0) perm[j]=-1;
    else {
      if(v!=last) last=v, ++un;
      perm[j]=un-1;
    }
  }
  buf->n=0;
  return un;
}

/* given user's list of dofs (as id's)
   uses gather-scatter to find share-count and separator # for each
   outputs as a list, sorted topologically (children before parents)
                      according to the sep. tree (and without duplicates),
           as well as the permutation to get there from the user's list */
/* sets: un, cn, perm_u2c */
static void discover_dofs(
  struct xxt *data, uint n, const ulong *id,
  struct array *dofa, buffer *buf, const struct comm *comm)
{
  const uint pcoord = data->pcoord, ns=data->nsep;
  sint *perm;
  uint i, cn, *p, *pi;
  ulong *bid;
  struct gs_data *gsh; sint *v;
  struct dof *dof;
  
  data->un = n;
  data->perm_u2c = perm = tmalloc(sint,n);
  data->cn = cn = unique_ids(n,id,perm,buf);
  array_init(struct dof,dofa,cn), dofa->n=cn, dof=dofa->ptr;
  buffer_reserve(buf,cn*sizeof(ulong)), bid=buf->ptr;
  for(i=0;i<n;++i) if(perm[i]>=0) bid[perm[i]]=dof[perm[i]].id=id[i];

  gsh = gs_setup((const slong*)bid,cn,comm,0,gs_crystal_router,0);
  v = tmalloc(sint,cn);

  for(i=0;i<cn;++i) v[i]=pcoord;
  gs(v,gs_sint,gs_bpr,0,gsh,buf);
  for(i=0;i<cn;++i) dof[i].level=ns-1-lg((uint)v[i]);
  
  for(i=0;i<cn;++i) v[i]=1;
  gs(v,gs_sint,gs_add,0,gsh,buf);
  for(i=0;i<cn;++i) dof[i].count=v[i];
  
  free(v);
  gs_free(gsh);

  if(!cn) return;
  buffer_reserve(buf,2*cn*sizeof(uint));
  p = sortp(buf,0, &dof[0].level,cn,sizeof(struct dof));
  pi = p+cn; for(i=0;i<cn;++i) pi[p[i]]=i;
  for(i=0;i<n;++i) if(perm[i]>=0) perm[i]=pi[perm[i]];
  sarray_permute_buf(struct dof,dof,cn, buf);
}

/* vl += A_ls * vs */
static void apply_p_Als(double *vl, struct xxt *data, const double *vs, uint ns)
{
  const uint *Arp = data->A_sl.Arp,
             *Aj  = data->A_sl.Aj;
  const double *A = data->A_sl.A;
  uint i,p,pe;
  for(i=0;i<ns;++i)
    for(p=Arp[i],pe=Arp[i+1];p!=pe;++p)
      vl[Aj[p]]+=A[p]*vs[i];
}

/* vs -= A_sl * vl */
static void apply_m_Asl(double *vs, uint ns, struct xxt *data, const double *vl)
{
  const uint *Arp = data->A_sl.Arp,
             *Aj  = data->A_sl.Aj;
  const double *A = data->A_sl.A;
  uint i,p,pe;
  for(i=0;i<ns;++i)
    for(p=Arp[i],pe=Arp[i+1];p!=pe;++p)
      vs[i]-=A[p]*vl[Aj[p]];
}

/* returns a column of S : vs = -S(0:ei-1,ei) */
static void apply_S_col(double *vs, struct xxt *data, 
                        struct csr_mat *A_ss, uint ei, double *vl)
{
  const uint ln=data->ln;
  const uint *Asl_rp = data->A_sl.Arp, *Ass_rp = A_ss->Arp,
             *Asl_j  = data->A_sl.Aj,  *Ass_j  = A_ss->Aj;
  const double *Asl  = data->A_sl.A,   *Ass    = A_ss->A;
  uint i,p,pe;
  for(i=0;i<ei;++i) vs[i]=0;
  for(p=Ass_rp[ei],pe=Ass_rp[ei+1];p!=pe;++p) {
    uint j=Ass_j[p];
    if(j>=ei) break;
    vs[j]=-Ass[p];
  }
  for(i=0;i<ln;++i) vl[i]=0;
  for(p=Asl_rp[ei],pe=Asl_rp[ei+1];p!=pe;++p) vl[Asl_j[p]]=-Asl[p];
  sparse_cholesky_solve(vl,&data->fac_A_ll,vl);
  apply_m_Asl(vs,ei,data,vl);
}

static void apply_S(double *Svs, uint ns, struct xxt *data, 
                    struct csr_mat *A_ss, const double *vs, double *vl)
{
  const uint ln=data->ln;
  const uint *Ass_rp = A_ss->Arp,
             *Ass_j  = A_ss->Aj;
  const double *Ass  = A_ss->A;
  uint i, p,pe;
  for(i=0;i<ns;++i) {
    double sum=0;
    for(p=Ass_rp[i],pe=Ass_rp[i+1];p!=pe;++p) {
      uint j=Ass_j[p];
      if(j>=ns) break;
      sum+=Ass[p]*vs[j];
    }
    Svs[i]=sum;
  }
  for(i=0;i<ln;++i) vl[i]=0;
  apply_p_Als(vl,data,vs,ns);
  sparse_cholesky_solve(vl,&data->fac_A_ll,vl);
  apply_m_Asl(Svs,ns,data,vl);
}

/* vx = X' * vs */
static void apply_Xt(double *vx, uint nx, const struct xxt *data,
                     const double *vs)
{
  const double *X = data->X; const uint *Xp = data->Xp;
  uint i; for(i=0;i<nx;++i) vx[i]=tensor_dot(vs,X+Xp[i],Xp[i+1]-Xp[i]);
}

/* vs = X * vx */
static void apply_X(double *vs, uint ns, const struct xxt *data,
                    const double *vx, uint nx)
{
  const double *X = data->X; const uint *Xp = data->Xp;
  uint i,j;
  for(i=0;i<ns;++i) vs[i]=0;
  for(i=0;i<nx;++i) {
    const double v = vx[i];
    const double *x = X+Xp[i]; uint n=Xp[i+1]-Xp[i];
    for(j=0;j<n;++j) vs[j]+=x[j]*v;
  }
}

static void allocate_X(struct xxt *data, sint *perm_x2c)
{
  uint xn=data->xn;
  uint i,h=0;
  if(data->null_space && xn) --xn;
  data->Xp = tmalloc(uint,xn+1);
  data->Xp[0]=0;
  for(i=0;i<xn;++i) {
    if(perm_x2c[i]!=-1) ++h;
    data->Xp[i+1]=data->Xp[i]+h;
  }
  data->X = tmalloc(double,data->Xp[xn]);
}

static void orthogonalize(struct xxt *data, struct csr_mat *A_ss,
                          sint *perm_x2c, buffer *buf)
{
  uint ln=data->ln, sn=data->sn, xn=data->xn;
  double *vl, *vs, *vx, *Svs;
  uint i,j;

  allocate_X(data,perm_x2c);
  
  buffer_reserve(buf,(ln+2*sn+xn)*sizeof(double));
  vl=buf->ptr, vs=vl+ln, Svs=vs+sn, vx=Svs+sn;

  if(data->null_space && xn) --xn;
  for(i=0;i<xn;++i) {
    uint ns=data->Xp[i+1]-data->Xp[i];
    sint ui = perm_x2c[i];
    double ytsy, *x;

    if(ui == -1) {
      for(j=0;j<i;++j) vx[j]=0;
    } else {
      ui-=ln;
      apply_S_col(vs, data,A_ss, ui, vl);
      apply_Xt(vx,i, data, vs);
    }
    apply_QQt(data,vx,i,xn-i);
    apply_X(vs,ns, data, vx,i);
    if(ui!=-1) vs[ui]=1;
    apply_S(Svs,ns, data,A_ss, vs, vl);
    ytsy = tensor_dot(vs,Svs,ns);
    ytsy = sum(data,ytsy,i+1,xn-(i+1));
    if(ytsy<DBL_EPSILON/128) ytsy=0; else ytsy = 1/sqrt(ytsy);
    x=&data->X[data->Xp[i]];
    for(j=0;j<ns;++j) x[j]=ytsy*vs[j];
  }
}

struct yale_mat { uint i,j; double v; };

/* produces CSR matrix from Yale-like format, summing duplicates */
static void condense_matrix(struct array *mat, uint nr,
                            struct csr_mat *out, buffer *buf)
{
  uint k, nz=mat->n;
  struct yale_mat *p, *q;
  sarray_sort_2(struct yale_mat,mat->ptr,mat->n, i,0, j,0, buf);
  
  p = mat->ptr;
  for(k=0;k+1<nz;++k,++p) if(p[0].i==p[1].i && p[0].j==p[1].j) break;
  if(++k<nz) {
    uint i=p->i,j=p->j;
    q = p+1;
    for(;k<nz;++k,++q) {
      if(i==q->i&&j==q->j) p->v += q->v, --mat->n;
      else ++p, p->i=i=q->i,p->j=j=q->j, p->v=q->v;
    }
  }
  
  nz=mat->n;
  out->n=nr;
  out->Arp = tmalloc(uint,nr+1+mat->n);
  out->Aj = out->Arp+nr+1;
  out->A = tmalloc(double,mat->n);
  for(k=0;k<nr;++k) out->Arp[k]=0;
  for(p=mat->ptr,k=0;k<nz;++k,++p)
    out->Arp[p->i]++, out->Aj[k]=p->j, out->A[k]=p->v;
  nz=0; for(k=0;k<=nr;++k) { uint t=out->Arp[k]; out->Arp[k]=nz, nz+=t; }
}

static void separate_matrix(
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  const sint *perm, uint ln, uint sn,
  struct csr_mat *out_ll, struct csr_mat *out_sl, struct csr_mat *out_ss,
  buffer *buf
)
{
  uint k,n;
  struct array mat_ll, mat_sl, mat_ss;
  struct yale_mat *mll, *msl, *mss;
  array_init(struct yale_mat,&mat_ll,2*nz), mll=mat_ll.ptr;
  array_init(struct yale_mat,&mat_sl,2*nz), msl=mat_sl.ptr;
  array_init(struct yale_mat,&mat_ss,2*nz), mss=mat_ss.ptr;
  for(k=0;k<nz;++k) {
    sint i=perm[Ai[k]], j=perm[Aj[k]];
    if(i<0 || j<0 || A[k]==0) continue;
    if((uint)i<ln) {
      if((uint)j<ln)
        n=mat_ll.n++,mll[n].i=i,mll[n].j=j,mll[n].v=A[k];
    } else {
      if((uint)j<ln)
        n=mat_sl.n++,msl[n].i=i-ln,msl[n].j=j,msl[n].v=A[k];
      else
        n=mat_ss.n++,mss[n].i=i-ln,mss[n].j=j-ln,mss[n].v=A[k];
    }
  }
  condense_matrix(&mat_ll,ln,out_ll,buf);
  condense_matrix(&mat_sl,sn,out_sl,buf);
  condense_matrix(&mat_ss,sn,out_ss,buf);
  array_free(&mat_ll);
  array_free(&mat_sl);
  array_free(&mat_ss);
}

struct xxt *crs_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  uint null_space, const struct comm *comm)
{
  struct xxt *data = tmalloc(struct xxt,1);
  sint *perm_x2c;
  struct array dofa;
  struct csr_mat A_ll, A_ss;
  buffer buf;

  comm_dup(&data->comm,comm);

  locate_proc(data);

  data->null_space=null_space;

  buffer_init(&buf,1024);

  discover_dofs(data,n,id,&dofa,&buf,&data->comm);
  discover_sep_sizes(data,&dofa,&buf);

  perm_x2c = discover_sep_ids(data,&dofa,&buf);
  if(data->null_space) {
    uint i; double count = 0; struct dof *dof = dofa.ptr;
    for(i=0;i<data->cn;++i) count+=1/(double)dof[i].count;
    count=1/sum(data,count,data->xn,0);
    data->share_weight=tmalloc(double,data->cn);
    for(i=0;i<data->cn;++i)
      data->share_weight[i]=count/dof[i].count;
  }
  array_free(&dofa);

  if(!data->null_space || data->xn!=0) {
    separate_matrix(nz,Ai,Aj,A,data->perm_u2c,
                    data->ln,data->sn,
                    &A_ll,&data->A_sl,&A_ss,
                    &buf);
  } else {
    separate_matrix(nz,Ai,Aj,A,data->perm_u2c,
                    data->ln-1,1,
                    &A_ll,&data->A_sl,&A_ss,
                    &buf);
  }                

  sparse_cholesky_factor(A_ll.n,A_ll.Arp,A_ll.Aj,A_ll.A,
                         &data->fac_A_ll, &buf);
  free(A_ll.Arp); free(A_ll.A);

  data->vl = tmalloc(double,data->ln+data->cn+2*data->xn);
  data->vc = data->vl+data->ln;
  data->vx = data->vc+data->cn;
  data->combuf = data->vx+data->xn;

  orthogonalize(data,&A_ss,perm_x2c,&buf);
  free(A_ss.Arp); free(A_ss.A);
  free(perm_x2c);
  buffer_free(&buf);

  return data;
}

void crs_solve(double *x, struct xxt *data, const double *b)
{
  uint cn=data->cn, un=data->un, ln=data->ln, sn=data->sn, xn=data->xn;
  double *vl=data->vl, *vc=data->vc, *vx=data->vx;
  uint i;
  for(i=0;i<cn;++i) vc[i]=0;
  for(i=0;i<un;++i) {
    sint p=data->perm_u2c[i];
    if(p>=0) vc[p]+=b[i];
  }
  if(xn>0 && (!data->null_space || xn>1)) {
    if(data->null_space) --xn;
    sparse_cholesky_solve(vc,&data->fac_A_ll,vc);
    apply_m_Asl(vc+ln,sn, data, vc);
    apply_Xt(vx,xn, data, vc+ln);
    apply_QQt(data,vx,xn,0);
    apply_X(vc+ln,sn, data, vx,xn);
    for(i=0;i<ln;++i) vl[i]=0;
    apply_p_Als(vl, data, vc+ln,sn);
    sparse_cholesky_solve(vl,&data->fac_A_ll,vl);
    for(i=0;i<ln;++i) vc[i]-=vl[i];
  } else {
    sparse_cholesky_solve(vc,&data->fac_A_ll,vc);
    if(data->null_space) {
      if(xn==0) vc[ln-1]=0;
      else if(sn==1) vc[ln]=0;
    }
  }
  if(data->null_space) {
    double s=0;
    for(i=0;i<cn;++i) s+=data->share_weight[i]*vc[i];
    s = sum(data,s,data->xn,0);
    for(i=0;i<cn;++i) vc[i]-=s;
  }
  for(i=0;i<un;++i) {
    sint p=data->perm_u2c[i];
    x[i] = p>=0 ? vc[p] : 0;
  }
}

void crs_stats(struct xxt *data)
{
  int a,b; uint xcol;
  if(data->comm.id==0) {
    unsigned s;
    printf("xxt: separator sizes on %d =",(int)data->comm.id);
    for(s=0;s<data->nsep;++s) printf(" %d",(int)data->sep_size[s]);
    printf("\n");
    printf("xxt: shared dofs on %d = %d\n",(int)data->comm.id,(int)data->sn);
  }
  a=data->ln;
  comm_allreduce(&data->comm,gs_int,gs_max, &a,1, &b);
  if(data->comm.id==0) printf("xxt: max non-shared dofs = %d\n",a);
  a=data->sn;
  comm_allreduce(&data->comm,gs_int,gs_max, &a,1, &b);
  if(data->comm.id==0) printf("xxt: max shared dofs = %d\n",a);
  xcol=data->xn; if(xcol&&data->null_space) --xcol;
  a=xcol;
  comm_allreduce(&data->comm,gs_int,gs_max, &a,1, &b);
  if(data->comm.id==0) printf("xxt: max X cols = %d\n",a);
  a=data->Xp[xcol]*sizeof(double);
  comm_allreduce(&data->comm,gs_int,gs_max, &a,1, &b);
  if(data->comm.id==0) printf("xxt: max X size = %d bytes\n",a);
}

void crs_free(struct xxt *data)
{
  comm_free(&data->comm);
  free(data->pother);
  free(data->req);
  free(data->sep_size);
  free(data->perm_u2c);
  if(data->null_space) free(data->share_weight);
  sparse_cholesky_free(&data->fac_A_ll);
  free(data->A_sl.Arp); free(data->A_sl.A);
  free(data->Xp); free(data->X);
  free(data->vl);
  free(data);
}

