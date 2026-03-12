
#define obbox           TOKEN_PASTE(obbox_,D)
#define local_hash_data TOKEN_PASTE(findpts_local_hash_data_,D)
#define hash_data       TOKEN_PASTE(findpts_hash_data_,D)
#define hash_index      TOKEN_PASTE(hash_index_       ,D)
#define hash_setfac     TOKEN_PASTE(hash_setfac_      ,D)
#define hash_range      TOKEN_PASTE(hash_range_       ,D)
#define hash_bb         TOKEN_PASTE(hash_bb_          ,D)
#define set_local_mask  TOKEN_PASTE(set_local_mask_   ,D)
#define fill_hash       TOKEN_PASTE(fill_hash_        ,D)
#define table_from_hash TOKEN_PASTE(table_from_hash_  ,D)
#define hash_build      TOKEN_PASTE(hash_build_       ,D)
#define hash_free       TOKEN_PASTE(hash_free_        ,D)

#define findpts_local_data  TOKEN_PASTE(findpts_local_data_,D)
#define findpts_local_setup TOKEN_PASTE(PREFIXED_NAME(findpts_local_setup_),D)
#define findpts_local_free  TOKEN_PASTE(PREFIXED_NAME(findpts_local_free_ ),D)
#define findpts_local       TOKEN_PASTE(PREFIXED_NAME(findpts_local_      ),D)
#define findpts_local_eval  TOKEN_PASTE(PREFIXED_NAME(findpts_local_eval_ ),D)
#define findpts_data        TOKEN_PASTE(findpts_data_,D)
#define src_pt              TOKEN_PASTE(src_pt_      ,D)
#define out_pt              TOKEN_PASTE(out_pt_      ,D)
#define eval_src_pt         TOKEN_PASTE(eval_src_pt_ ,D)
#define eval_out_pt         TOKEN_PASTE(eval_out_pt_ ,D)
#define setup_aux           TOKEN_PASTE(setup_aux_,D)
#define findpts_setup       TOKEN_PASTE(PREFIXED_NAME(findpts_setup_),D)
#define findpts_free        TOKEN_PASTE(PREFIXED_NAME(findpts_free_ ),D)
#define findpts             TOKEN_PASTE(PREFIXED_NAME(findpts_      ),D)
#define findpts_eval        TOKEN_PASTE(PREFIXED_NAME(findpts_eval_ ),D)

struct hash_data {
  ulong hash_n;
  struct dbl_range bnd[D];
  double fac[D];
  uint *offset;
};

static ulong hash_index(const struct hash_data *p, const double x[D])
{
  const ulong n = p->hash_n;
  return ( WHEN_3D( hash_index_aux(p->bnd[2].min,p->fac[2],n,x[2])  *n )
                   +hash_index_aux(p->bnd[1].min,p->fac[1],n,x[1]) )*n
                   +hash_index_aux(p->bnd[0].min,p->fac[0],n,x[0]);
}

static void hash_setfac(struct hash_data *p, const ulong n)
{
  unsigned d;
  p->hash_n = n;
  for(d=0;d<D;++d) p->fac[d] = n/(p->bnd[d].max-p->bnd[d].min);
}

static struct ulong_range hash_range(const struct hash_data *p, unsigned d,
                                     const struct dbl_range r)
{
  struct ulong_range ir;
  const slong i0 = lfloor( (r.min - p->bnd[d].min) * p->fac[d] );
  const ulong i1 = lceil ( (r.max - p->bnd[d].min) * p->fac[d] );
  ir.min = i0<0 ? 0 : i0;
  ir.max = i1<p->hash_n ? i1 : p->hash_n;
  if(ir.max==ir.min) ++ir.max;
  return ir;
}

static void hash_bb(struct hash_data *p, const struct local_hash_data *lp,
                    const struct comm *comm, uint hash_size)
{
  double x[D], buf[D], ghs;
  unsigned d;
  for(d=0;d<D;++d) x[d]=lp->bnd[d].min;
  comm_allreduce(comm,gs_double,gs_min,x,D,buf);
  for(d=0;d<D;++d) p->bnd[d].min=x[d];

  for(d=0;d<D;++d) x[d]=lp->bnd[d].max;
  comm_allreduce(comm,gs_double,gs_max,x,D,buf);
  for(d=0;d<D;++d) p->bnd[d].max=x[d];

  ghs = hash_size; comm_allreduce(comm,gs_double,gs_add,&ghs,1,buf);
  hash_setfac(p,lceil(pow(ghs,1./D)));
  
  #ifdef DIAGNOSTICS
  if(comm->id==0) {
    printf("global bounding box (%g^%u):\n",(double)p->hash_n,D);
    for(d=0;d<D;++d) printf("  [%.17g, %.17g]\n",p->bnd[d].min,p->bnd[d].max);
  }
  #endif
}

static void set_local_mask(unsigned char *const local_mask,
                           const ulong local_base[D], const uint local_n[D],
                           const struct hash_data *const p,
                           const struct obbox *const obb, const uint nel
                          )
{
  uint el;
  for(el=0;el<nel;++el) {
    struct ulong_range ir[D]; unsigned d;
    for(d=0;d<D;++d) ir[d]=hash_range(p,d,obb[el].x[d]);
    #define FOR_LOOP() do { ulong i,j; WHEN_3D(ulong k;) \
      WHEN_3D(for(k=ir[2].min;k<ir[2].max;++k)) \
              for(j=ir[1].min;j<ir[1].max;++j) \
              for(i=ir[0].min;i<ir[0].max;++i) \
                set_bit(local_mask, (WHEN_3D((k-local_base[2]) *local_n[1]) \
                                            +(j-local_base[1]))*local_n[0] \
                                            +(i-local_base[0]) \
                       ); \
    } while(0)
    FOR_LOOP();
    #undef FOR_LOOP
  }
}

static void fill_hash(struct array *const hash,
                      const unsigned char *const local_mask,
                      const ulong local_base[D], const uint local_n[D],
                      const ulong hn, const uint np)
{
  struct proc_index *hp = hash->ptr;
  #define FOR_LOOP() do { uint bit=0,i,j; WHEN_3D(uint k;) \
    WHEN_3D(for(k=0;k<local_n[2];++k)) \
            for(j=0;j<local_n[1];++j) \
            for(i=0;i<local_n[0];++i) { ulong hi; \
              if(get_bit(local_mask,bit++)==0) continue; \
              hi = (WHEN_3D( (local_base[2]+k) *hn ) \
                            +(local_base[1]+j))*hn \
                            +(local_base[0]+i); \
              hp->proc = hi%np, hp->index = hi/np; \
              ++hp; \
            } \
  } while(0)
  FOR_LOOP();
  #undef FOR_LOOP
}

static void table_from_hash(struct hash_data *const p,
                            struct array *const hash,
                            const uint np, buffer *buf)
{
  const ulong hn = p->hash_n;
  ulong hnd;
  uint ncell, *offset, i, next_cell;
  const struct proc_index *const hp = hash->ptr;
  const uint n = hash->n;
  hnd = hn*hn; WHEN_3D(hnd*=hn);
  ncell = (hnd-1)/np+1;
  p->offset = offset = tmalloc(uint,ncell+1+n);
  sarray_sort(struct proc_index,hash->ptr,n, index,0, buf);
  next_cell = 0;
  for(i=0;i<n;++i) {
    const uint cell = hp[i].index;
    const uint off = ncell+1+i;
    offset[off]=hp[i].proc;
    while(next_cell<=cell ) offset[next_cell++]=off;
  }
  { const uint off = ncell+1+i;
    while(next_cell<=ncell) offset[next_cell++]=off;
  }
}

static void hash_build(struct hash_data *const p,
                       const struct local_hash_data *const lp,
                       const struct obbox *const obb, const uint nel,
                       const uint hash_size,
                       struct crystal *cr)
{
  ulong local_base[D]; uint local_n[D], local_ntot=1;
  unsigned char *local_mask;
  struct array hash; uint nc;
  unsigned d;
  hash_bb(p,lp,&cr->comm,hash_size);
  for(d=0;d<D;++d) {
    struct ulong_range rng=hash_range(p,d,lp->bnd[d]);
    local_base[d]=rng.min;
    local_n[d]=rng.max-rng.min;
    local_ntot*=local_n[d];
    #ifdef DIAGNOSTICS
    if(cr->comm.id==0) {
      printf("local_range %u: %lu to %lu\n",
             d,(unsigned long)rng.min,(unsigned long)rng.max);
    }
    #endif
  }
  local_mask = tcalloc(unsigned char, (local_ntot+CHAR_BIT-1)/CHAR_BIT);
  set_local_mask(local_mask,local_base,local_n,p,obb,nel);
  nc=count_bits(local_mask,(local_ntot+CHAR_BIT-1)/CHAR_BIT);
  #ifdef DIAGNOSTICS
  printf("findpts_hash(%u): local cells : %u / %u\n",cr->comm.id,nc,local_ntot);
  #endif
  array_init(struct proc_index,&hash,nc), hash.n=nc;
  fill_hash(&hash,local_mask,local_base,local_n,p->hash_n,cr->comm.np);
  free(local_mask);
  sarray_transfer(struct proc_index,&hash,proc,1,cr);
  table_from_hash(p,&hash,cr->comm.np,&cr->data);
  array_free(&hash);
}

static void hash_free(struct hash_data *p) { free(p->offset); }

struct findpts_data {
  struct crystal cr;
  struct findpts_local_data local;
  struct hash_data hash;
};

static void setup_aux(
  struct findpts_data *const fd,
  const double *const elx[D],
  const unsigned n[D], const uint nel,
  const unsigned m[D], const double bbox_tol,
  const uint local_hash_size, const uint global_hash_size,
  const unsigned npt_max, const double newt_tol)
{
  findpts_local_setup(&fd->local,elx,n,nel,m,bbox_tol,local_hash_size,
                      npt_max, newt_tol);
  hash_build(&fd->hash,&fd->local.hd,fd->local.obb,nel,
             global_hash_size,&fd->cr);
}

struct findpts_data *findpts_setup(
  const struct comm *const comm,
  const double *const elx[D],
  const unsigned n[D], const uint nel,
  const unsigned m[D], const double bbox_tol,
  const uint local_hash_size, const uint global_hash_size,
  const unsigned npt_max, const double newt_tol)
{
  struct findpts_data *const fd = tmalloc(struct findpts_data, 1);
  crystal_init(&fd->cr,comm);
  setup_aux(fd,elx,n,nel,m,bbox_tol,
            local_hash_size,global_hash_size,npt_max,newt_tol);
  return fd;
}

void findpts_free(struct findpts_data *fd)
{
  hash_free(&fd->hash);
  findpts_local_free(&fd->local);
  crystal_free(&fd->cr);
  free(fd);
}

struct src_pt { double x[D]; uint index, proc; };
struct out_pt { double r[D], dist2; uint index, code, el, proc; };

void findpts(      uint   *const  code_base   , const unsigned  code_stride   ,
                   uint   *const  proc_base   , const unsigned  proc_stride   ,
                   uint   *const    el_base   , const unsigned    el_stride   ,
                   double *const     r_base   , const unsigned     r_stride   ,
                   double *const dist2_base   , const unsigned dist2_stride   ,
             const double *const     x_base[D], const unsigned     x_stride[D],
             const uint npt, struct findpts_data *const fd)
{
  const uint np = fd->cr.comm.np, id=fd->cr.comm.id;
  struct array hash_pt, src_pt, out_pt;
  /* look locally first */
  if(npt) findpts_local( code_base, code_stride,
                           el_base,   el_stride,
                            r_base,    r_stride,
                        dist2_base,dist2_stride,
                            x_base,    x_stride,
                        npt,&fd->local,&fd->cr.data);
  /* send unfound and border points to global hash cells */
  {
    uint index;
    uint *code=code_base, *proc=proc_base;
    const double *xp[D];
    struct src_pt *pt;
    unsigned d; for(d=0;d<D;++d) xp[d]=x_base[d];
    array_init(struct src_pt, &hash_pt, npt), pt=hash_pt.ptr;
    for(index=0;index<npt;++index) {
      double x[D]; for(d=0;d<D;++d) x[d]=*xp[d];
      *proc = id;
      if(*code!=CODE_INTERNAL) {
        const uint hi = hash_index(&fd->hash,x);
        unsigned d;
        for(d=0;d<D;++d) pt->x[d]=x[d];
        pt->index=index;
        pt->proc=hi%np;
        ++pt;
      }
      for(d=0;d<D;++d)
      xp[d] = (const double*)((const char*)xp[d]+   x_stride[d]);
      code  =         (uint*)(      (char*)code +code_stride   );
      proc  =         (uint*)(      (char*)proc +proc_stride   );
    }
    hash_pt.n = pt - (struct src_pt*)hash_pt.ptr;
    sarray_transfer(struct src_pt,&hash_pt,proc,1,&fd->cr);
  }
  /* look up points in hash cells, route to possible procs */
  {
    const uint *const hash_offset = fd->hash.offset;
    uint count=0, *proc, *proc_p;
    const struct src_pt *p = hash_pt.ptr, *const pe = p+hash_pt.n;
    struct src_pt *q;
    for(;p!=pe;++p) {
      const uint hi = hash_index(&fd->hash,p->x)/np;
      const uint i = hash_offset[hi], ie = hash_offset[hi+1];
      count += ie-i;
    }
    proc_p = proc = tmalloc(uint,count);
    array_init(struct src_pt,&src_pt,count), q=src_pt.ptr;
    for(p=hash_pt.ptr;p!=pe;++p) {
      const uint hi = hash_index(&fd->hash,p->x)/np;
      uint i = hash_offset[hi]; const uint ie = hash_offset[hi+1];
      for(;i!=ie;++i) {
        const uint pp = hash_offset[i];
        if(pp==p->proc) continue; /* don't send back to source proc */
        *proc_p++ = pp;
        *q++ = *p;
      }
    }
    array_free(&hash_pt);
    src_pt.n = proc_p-proc;
    #ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n",id,(unsigned)src_pt.n,count);
    #endif
    sarray_transfer_ext(struct src_pt,&src_pt,proc,sizeof(uint),&fd->cr);
    free(proc);
  }
  /* look for other procs' points, send back */
  {
    uint n=src_pt.n;
    const struct src_pt *spt;
    struct out_pt *opt;
    array_init(struct out_pt,&out_pt,n), out_pt.n=n;
    spt=src_pt.ptr, opt=out_pt.ptr;
    for(;n;--n,++spt,++opt) opt->index=spt->index,opt->proc=spt->proc;
    spt=src_pt.ptr, opt=out_pt.ptr;
    if(src_pt.n) {
      const double *spt_x_base[D]; unsigned spt_x_stride[D];
      unsigned d; for(d=0;d<D;++d) spt_x_base[d] = spt[0].x+d,
                                   spt_x_stride[d] = sizeof(struct src_pt);
      findpts_local(&opt[0].code ,sizeof(struct out_pt),
                    &opt[0].el   ,sizeof(struct out_pt),
                     opt[0].r    ,sizeof(struct out_pt),
                    &opt[0].dist2,sizeof(struct out_pt),
                     spt_x_base  ,spt_x_stride,
                    src_pt.n,&fd->local,&fd->cr.data);
    }
    array_free(&src_pt);
    /* group by code to eliminate unfound points */
    sarray_sort(struct out_pt,opt,out_pt.n, code,0, &fd->cr.data);
    n=out_pt.n; while(n && opt[n-1].code==CODE_NOT_FOUND) --n;
    out_pt.n=n;
    #ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n",id,(unsigned)out_pt.n);
    #endif
    sarray_transfer(struct out_pt,&out_pt,proc,1,&fd->cr);
  }
  /* merge remote results with user data */
  {
    #define  AT(T,var,i) (T*)((char*)var##_base+(i)*var##_stride)
    uint n=out_pt.n;
    struct out_pt *opt;
    for(opt=out_pt.ptr;n;--n,++opt) {
      const uint index = opt->index;
      uint *code = AT(uint,code,index);
      double *dist2 = AT(double,dist2,index);
      if(*code==CODE_INTERNAL) continue;
      if(*code==CODE_NOT_FOUND
         || opt->code==CODE_INTERNAL
         || opt->dist2<*dist2) {
        double *r = AT(double,r,index);
        uint  *el = AT(uint,el,index), *proc = AT(uint,proc,index);
        unsigned d; for(d=0;d<D;++d) r[d]=opt->r[d];
        *dist2 = opt->dist2;
        *proc = opt->proc;
        *el = opt->el;
        *code = opt->code;
      }
    }
    array_free(&out_pt);
    #undef AT
  }
}

struct eval_src_pt { double r[D]; uint index, proc, el; };
struct eval_out_pt { double out; uint index, proc; };

void findpts_eval(
        double *const  out_base, const unsigned  out_stride,
  const uint   *const code_base, const unsigned code_stride,
  const uint   *const proc_base, const unsigned proc_stride,
  const uint   *const   el_base, const unsigned   el_stride,
  const double *const    r_base, const unsigned    r_stride,
  const uint npt,
  const double *const in, struct findpts_data *const fd)
{
  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  {
    uint index;
    const uint *code=code_base, *proc=proc_base, *el=el_base;
    const double *r=r_base;
    struct eval_src_pt *pt;
    array_init(struct eval_src_pt, &src, npt), pt=src.ptr;
    for(index=0;index<npt;++index) {
      if(*code!=CODE_NOT_FOUND) {
        unsigned d;
        for(d=0;d<D;++d) pt->r[d]=r[d];
        pt->index=index;
        pt->proc=*proc;
        pt->el=*el;
        ++pt;
      }
      r    = (const double*)((const char*)r   +   r_stride);
      code = (const   uint*)((const char*)code+code_stride);
      proc = (const   uint*)((const char*)proc+proc_stride);
      el   = (const   uint*)((const char*)el  +  el_stride);
    }
    src.n = pt - (struct eval_src_pt*)src.ptr;
    sarray_transfer(struct eval_src_pt,&src,proc,1,&fd->cr);
  }
  /* evaluate points, send back */
  {
    uint n=src.n;
    const struct eval_src_pt *spt;
    struct eval_out_pt *opt;
    /* group points by element */
    sarray_sort(struct eval_src_pt,src.ptr,n, el,0, &fd->cr.data);
    array_init(struct eval_out_pt,&outpt,n), outpt.n=n;
    spt=src.ptr, opt=outpt.ptr;
    for(;n;--n,++spt,++opt) opt->index=spt->index,opt->proc=spt->proc;
    spt=src.ptr, opt=outpt.ptr;
    findpts_local_eval(&opt->out ,sizeof(struct eval_out_pt),
                       &spt->el  ,sizeof(struct eval_src_pt),
                        spt->r   ,sizeof(struct eval_src_pt),
                       src.n, in,&fd->local);
    array_free(&src);
    sarray_transfer(struct eval_out_pt,&outpt,proc,1,&fd->cr);
  }
  /* copy results to user data */
  {
    #define  AT(T,var,i) (T*)((char*)var##_base+(i)*var##_stride)
    uint n=outpt.n;
    struct eval_out_pt *opt;
    for(opt=outpt.ptr;n;--n,++opt) *AT(double,out,opt->index)=opt->out;
    array_free(&outpt);
    #undef AT
  }
}

#undef findpts_eval
#undef findpts
#undef findpts_free
#undef findpts_setup
#undef setup_aux
#undef eval_out_pt
#undef eval_src_pt
#undef out_pt
#undef src_pt
#undef findpts_data
#undef findpts_local_eval
#undef findpts_local
#undef findpts_local_free
#undef findpts_local_setup
#undef findpts_local_data

#undef hash_free
#undef hash_build
#undef table_from_hash
#undef fill_hash
#undef set_local_mask
#undef hash_bb
#undef hash_range
#undef hash_setfac
#undef hash_index
#undef hash_data
#undef local_hash_data
#undef obbox
