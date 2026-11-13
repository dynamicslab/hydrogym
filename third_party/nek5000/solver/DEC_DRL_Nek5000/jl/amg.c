#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "sort.h"
#include "sarray_sort.h"
#include "gs_defs.h"
#include "comm.h"
#include "crystal.h"
#include "sarray_transfer.h"
#include "gs.h"

#define crs_setup PREFIXED_NAME(crs_setup)
#define crs_solve PREFIXED_NAME(crs_solve)
#define crs_stats PREFIXED_NAME(crs_stats)
#define crs_free  PREFIXED_NAME(crs_free )

#ifndef AMG_BLOCK_ROWS
#  define AMG_BLOCK_ROWS 1200
#endif

static double get_time(void)
{
#ifdef GS_TIMING
  return comm_time();
#else
  return 0;
#endif
}

static void barrier(const struct comm *c)
{
#ifdef GS_BARRIER
  comm_barrier(c);
#endif
}

/* sparse matrix, condensed sparse row */
struct csr_mat {
  uint rn, cn, *row_off, *col;
  double *a;
};

/* z = alpha y + beta M x */
static double apply_M(
  double *z, const double alpha, const double *y,
  const double beta, const struct csr_mat *const M, const double *x)
{
  uint i; const uint rn=M->rn;
  const uint *const row_off = M->row_off, *col = M->col;
  const double *a = M->a;
  const double t0 = get_time();
  for(i=0;i<rn;++i) {
    uint j; const uint je=row_off[i+1]; double t = 0;
    for(j=row_off[i]; j<je; ++j) t += (*a++) * x[*col++];
    *z++ = alpha*(*y++) + beta*t;
  }
  return get_time()-t0;
}

/* z = M^t x */
static double apply_Mt(
  double *const z, const struct csr_mat *const M, const double *x)
{
  uint i; const uint rn=M->rn,cn=M->cn;
  const uint *const row_off = M->row_off, *col = M->col;
  const double *a = M->a;
  const double t0 = get_time();
  for(i=0;i<cn;++i) z[i]=0;
  for(i=0;i<rn;++i) {
    uint j; const uint je=row_off[i+1]; const double xi = *x++;
    for(j=row_off[i]; j<je; ++j) z[*col++] += (*a++) * xi;
  }
  return get_time()-t0;
}

struct Q { uint nloc; struct gs_data *gsh; };

/* ve = Q v */
static double apply_Q(
  double *const ve, const struct Q *const Q, const double *const v,
  const struct comm *comm)
{
  double t0;
  memcpy(ve,v,Q->nloc*sizeof(double));
  barrier(comm); t0=get_time();
  gs(ve,gs_double,gs_add,0,Q->gsh,0);
  return get_time()-t0;
}

/* z := alpha y + beta Q^t x
   (x := Q Q^t x   as a side effect)
 */
static double apply_Qt(
  double *z, const double alpha, const double *y,
  const double beta, const struct Q *Q, double *x,
  const struct comm *comm)
{
  double t0,t1;
  uint i; const uint nloc=Q->nloc;
  barrier(comm); t0=get_time();
  gs(x,gs_double,gs_add,1,Q->gsh,0);
  t1 = get_time() - t0;
  for(i=0;i<nloc;++i) *z++ = alpha*(*y++) + beta*(*x++);
  return t1;
}

struct crs_data {
  struct comm comm;
  struct gs_data *gs_top;
  uint un, *umap; /* number of unique id's on this proc, map to user ids */
  double tni; /* 1 / (total number of unique ids)  ... for computing mean */
  int null_space;
  unsigned levels;
  unsigned *cheb_m; /* cheb_m  [levels-1] : smoothing steps */
  double *cheb_rho; /* cheb_rho[levels-1] : spectral radius of smoother */
  uint *lvl_offset;
  double *Dff;      /* diagonal smoother, F-relaxation */
  struct Q *Q_W, *Q_AfP, *Q_Aff;
  struct csr_mat *W, *AfP, *Aff;
  double *b, *x, *c, *c_old, *r, *buf;
  double *timing; uint timing_n;
};

static void amg_exec(struct crs_data *const amg)
{
  unsigned lvl; const unsigned levels=amg->levels;
  const uint *const off = amg->lvl_offset;
  double *const b = amg->b, *const x = amg->x;
  double *c = amg->c, *c_old = amg->c_old, *r = amg->r;
  double *timing = amg->timing;
  /* restrict down all levels */
  for(lvl=0;lvl<levels-1;++lvl,timing+=6) {
    double *const b_l = b+off[lvl], *const b_lp1 = b+off[lvl+1];
    /* b_{l+1} += W^t b_l */
    timing[1]+=apply_Mt(amg->buf, &amg->W[lvl],b_l);
    timing[0]+=apply_Qt(b_lp1, 1,b_lp1, 1,&amg->Q_W[lvl],amg->buf, &amg->comm);
  }
  /* solve bottom equation (1 dof) */
  { const uint i=off[levels-1];
    if(off[levels]-i) x[i]=amg->Dff[i]*b[i]; }
  for(lvl=levels-1;lvl--;) {
    double *const b_l = b+off[lvl];
    double *const x_l = x+off[lvl], *const x_lp1 = x+off[lvl+1];
    const double *const d_l = amg->Dff+off[lvl];
    const uint n = off[lvl+1]-off[lvl]; uint i;
    const unsigned m = amg->cheb_m[lvl]; unsigned ci;
    double alpha, beta, gamma;
    timing = amg->timing + lvl*6;
    /* buf = Q x_{l+1} */
    timing[2]+=apply_Q(amg->buf, &amg->Q_AfP[lvl],x_lp1, &amg->comm);
    /* x_l = W x_{l+1} */
    timing[3]+=apply_M(x_l, 0,b_l,  1,&amg->W  [lvl], amg->buf);
    /* b_l -= AfP x_{l+1} */
    timing[3]+=apply_M(b_l, 1,b_l, -1,&amg->AfP[lvl], amg->buf);
    /* c_1 = Dff b_l */
    for(i=0;i<n;++i) c[i]=d_l[i]*b_l[i];
    if(m>1) {
      alpha = amg->cheb_rho[lvl]/2, alpha*=alpha;
      gamma = 2*alpha/(1-2*alpha), beta = 1 + gamma;
      /* r_1 = b_l - Aff c_1 */
      timing[4]+=apply_Q(amg->buf, &amg->Q_Aff[lvl],c, &amg->comm);
      timing[5]+=apply_M(r, 1,b_l, -1,&amg->Aff[lvl],amg->buf);
      /* c_2 = (1+gamma)(c_1 + Dff r_1) */
      { double *const temp = c; c=c_old,c_old=temp; }
      for(i=0;i<n;++i) c[i] = beta*(c_old[i]+d_l[i]*r[i]);
    }
    for(ci=3;ci<=m;++ci) {
      gamma = alpha*beta, gamma = gamma/(1-gamma), beta = 1 + gamma;
      /* r_i = b_l - Aff c_i */
      timing[4]+=apply_Q(amg->buf, &amg->Q_Aff[lvl],c, &amg->comm);
      timing[5]+=apply_M(r, 1,b_l, -1,&amg->Aff[lvl],amg->buf);
      /* c_{i+1} = (1+gamma)*(c_i+D r_i) - gamma c_{i-1} */
      { double *const temp = c; c=c_old,c_old=temp; }
      for(i=0;i<n;++i) c[i] = beta*(c_old[i]+d_l[i]*r[i]) - gamma*c[i];
    }
    for(i=0;i<n;++i) x_l[i]+=c[i];
  }
  amg->timing_n++;
}

void crs_solve(double *x, struct crs_data *data, double *b)
{
  uint i; const uint un = data->un; const uint *const umap = data->umap;
  double *const ub = data->b, *const ux = data->x;
  
  gs(b, gs_double,gs_add, 1, data->gs_top, 0);
  for(i=0;i<un;++i) ub[i]=b[umap[i]];
  
  amg_exec(data);
  
  if(data->null_space) {
    const double avg = data->tni*comm_reduce_double(&data->comm, gs_add, ux,un);
    for(i=0;i<un;++i) ux[i] -= avg;
  }

  for(i=0;i<un;++i) x[umap[i]]=ux[i];
  gs(x, gs_double,gs_add, 0, data->gs_top, 0);
}

void crs_stats(const struct crs_data *const data)
{
  const unsigned lm1 = data->levels-1;
  double *avg = tmalloc(double, 2*6*lm1);
  double ni = 1/((double)data->timing_n * data->comm.np);
  uint i;
  for(i=0;i<6*lm1;++i) avg[i] = ni*data->timing[i];
  comm_allreduce(&data->comm,gs_double,gs_add, avg,6*lm1, avg+6*lm1);
  if(data->comm.id==0) {
    double *t = avg; unsigned lvl;
    printf("AMG stats:\n");
    for(lvl=0;lvl<lm1;++lvl,t+=6)
      printf("  [lvl %02u] comm: Wt=%0.3e W,AfP=%0.3e Aff=%0.3e\n"
             "  [lvl %02u] work: Wt=%0.3e W,AfP=%0.3e Aff=%0.3e\n",
             lvl, t[0],t[2],t[4],   lvl, t[1],t[3],t[5]);
  }
  free(avg);
}

/*==========================================================================

  Setup

  ==========================================================================*/


/* remote id - designates the i-th uknown owned by proc p */
struct rid { uint p,i; };

/* the data for each id to be read from amg.dat */
struct id_data {
  double D;
  ulong id;
  unsigned level;
};

/* global non-zero (matrix entry) */
struct gnz { ulong i,j; double a; };

/*
  The user ids   id[n]   are assigned to unique procs.
  
  Output
    uid --- ulong array; uid[0 ... uid.n-1]  are the ids owned by this proc
    rid_map --- id[i] is uid[ rid_map[i].i ] on proc rid_map[i].p
    map = assign_dofs(...) --- uid[i] is id[map[i]]

    map is returned only when rid_map is null, and vice versa
*/
static uint *assign_dofs(struct array *const uid,
                         struct rid *const rid_map,
                         const ulong *const id, const uint n,
                         const uint p,
                         struct gs_data *gs_top, buffer *const buf)
{
  uint i,count, *map=0;
  struct rid *const rid = rid_map?rid_map:tmalloc(struct rid,n);
  for(i=0;i<n;++i) rid[i].p=p, rid[i].i=i;
  gs_vec(n?&rid[0].p:0,2, gs_sint,gs_add,0, gs_top,buf);
  for(count=0,i=0;i<n;++i) {
    if(rid[i].i==i && id[i]!=0 && rid[i].p==p) ++count; }
  array_init(ulong,uid,count); uid->n=count;
  if(rid_map==0) {
    map=tmalloc(uint,count); 
    for(count=0,i=0;i<n;++i) {
      if(rid[i].i==i && id[i]!=0 && rid[i].p==p) map[count++]=i; }
  }
  { ulong *const uid_p = uid->ptr;
    for(count=0,i=0;i<n;++i) if(rid[i].i==i && id[i]!=0 && rid[i].p==p)
        uid_p[count]=id[i], rid[i].i=count, ++count;
  }
  if(rid_map) gs_vec(n?&rid[0].p:0,2, gs_sint,gs_add,0, gs_top,buf);
  else free(rid);
  return map;
}

static void localize_rows(struct gnz *p, uint nz,
  const ulong *const uid, const uint *const id_perm)
{
  uint i=0; while(nz--) {
    const ulong id=p->i;
    while(uid[i]<id) ++i;
    (p++)->i=id_perm[i];
  }
}

static void find_mat_offs(uint *const off, const int levels,
  const struct gnz *const p, const uint nz,
  const struct id_data *const id)
{
  int lvl=-1; uint i; for(i=0;i<nz;++i) {
    const int lvl_i = id[p[i].i].level;
    while(lvl<lvl_i) off[++lvl]=i;
  }
  while(lvl<levels-1) off[++lvl]=i;
}

static void compress_mat(
  uint *const row_off, const int rn, uint *const col, double *const a,
  const struct gnz *const p, const uint nz)
{
  int r = -1; uint k;
  for(r=-1, k=0;k<nz;++k) {
    const int row = p[k].i;
    while(r<row) row_off[++r]=k;
    col[k] = p[k].j;
    a[k] = p[k].a;
  }
  while(r<rn) row_off[++r]=k;
}

static void amg_setup_mat(
  struct array *ids, const uint nloc, struct csr_mat *const mat,
  const ulong *uid, const uint uid_n, const uint *id_perm,
  const uint i0, const uint j0, struct gnz *const p, const uint nz,
  buffer *buf)
{
  slong *id = ids->ptr; ulong last_id = -(ulong)1;
  const uint ne = ids->n;
  uint k,i=0,ie=nloc;
  sarray_sort(struct gnz,p,nz, j,1, buf); /* sort by col */
  for(k=0;k<nz;++k) {
    const ulong j_id = p[k].j;
    p[k].i = p[k].i - i0;
    if(j_id == last_id) { p[k].j = p[k-1].j; continue; }
    last_id = j_id;
    /* check local ids */
    while(i<uid_n && uid[i]<j_id) ++i;
    if(i<uid_n && uid[i]==j_id) { p[k].j=id_perm[i]-j0; continue; }
    /* check old group of remote ids */
    while(ie<ne && (ulong)(-id[ie])<j_id) ++ie;
    if(ie<ne && (ulong)(-id[ie])==j_id) { p[k].j=ie; continue; }
    /* not found, add to new group of remote ids */
    id = array_reserve(slong, ids, ids->n+1);
    id[ids->n] = -(slong)j_id, p[k].j = ids->n++;
  }
  sarray_sort_2(struct gnz,p,nz, i,1, j,1, buf);

  mat->cn = ids->n;
  compress_mat(mat->row_off,mat->rn, mat->col,mat->a, p,nz);
}

static uint amg_setup_mats(
  struct crs_data *const data,
  const ulong *const uid, const uint uid_n,
  const uint *const id_perm,
  const struct id_data *const id,
  struct array *mat,
  buffer *buf)
{
  const uint *const off = data->lvl_offset;
  struct array ide = null_array; uint max_e=0;
  const unsigned levels=data->levels;
  unsigned lvl, m;
  uint *mat_off[3]; struct csr_mat *csr_mat[3];
  data->Q_W = tmalloc(struct Q, (levels-1)*3);
  data->Q_AfP = data->Q_W + (levels-1);
  data->Q_Aff = data->Q_AfP + (levels-1);
  csr_mat[0] = data->W = tmalloc(struct csr_mat, (levels-1)*3);
  csr_mat[1] = data->AfP = data->W + (levels-1);
  csr_mat[2] = data->Aff = data->AfP + (levels-1);
  mat_off[0] = tmalloc(uint, levels*3);
  mat_off[1] = mat_off[0]+levels;
  mat_off[2] = mat_off[1]+levels;
  for(m=0;m<3;++m) {
    /* change row from uid to local index */
    localize_rows(mat[m].ptr,mat[m].n, uid,id_perm);
    /* sort by row */
    sarray_sort(struct gnz,mat[m].ptr,mat[m].n, i,1, buf);
    /* find offsets of each level */
    find_mat_offs(mat_off[m],levels, mat[m].ptr,mat[m].n, id);
  }
  /* allocate CSR arrays */
  if(levels>1) {
    uint *ui = tmalloc(uint, 3*((off[levels-1]-off[0])+(levels-1))
                             +mat[0].n+mat[1].n+mat[2].n);
    double *a = tmalloc(double, mat[0].n+mat[1].n+mat[2].n);
    for(m=0;m<3;++m) for(lvl=0;lvl<levels-1;++lvl) {
      const uint rn = off[lvl+1]-off[lvl];
      const uint nz = mat_off[m][lvl+1]-mat_off[m][lvl];
      csr_mat[m][lvl].rn = rn;
      csr_mat[m][lvl].row_off = ui, ui += rn+1;
      csr_mat[m][lvl].col = ui, ui += nz;
      csr_mat[m][lvl].a = a, a += nz;
    }
  }
  for(lvl=0;lvl<levels-1;++lvl) {
    uint i; const uint j0=off[lvl+1], nloc = off[levels]-j0;
    slong *p = array_reserve(slong, &ide, nloc);
    for(i=0;i<nloc;++i) p[i] = id[i+j0].id;
    data->Q_W[lvl].nloc = data->Q_AfP[lvl].nloc = ide.n = nloc;
    amg_setup_mat(&ide,nloc, &data->W[lvl], 
                  uid, uid_n,  id_perm, off[lvl],j0,
                  (struct gnz*)mat[0].ptr + mat_off[0][lvl],
                  mat_off[0][lvl+1]-mat_off[0][lvl], buf);
    data->Q_W[lvl].gsh = gs_setup(ide.ptr,ide.n, &data->comm, 0,gs_auto,1);
    amg_setup_mat(&ide,nloc, &data->AfP[lvl], 
                  uid, uid_n,  id_perm, off[lvl],j0,
                  (struct gnz*)mat[1].ptr + mat_off[1][lvl],
                  mat_off[1][lvl+1]-mat_off[1][lvl], buf);
    data->Q_AfP[lvl].gsh = gs_setup(ide.ptr,ide.n, &data->comm, 0,gs_auto,1);
    if(ide.n>max_e) max_e=ide.n;
  }
  for(lvl=0;lvl<levels-1;++lvl) {
    uint i; const uint j0=off[lvl], nloc = off[lvl+1]-j0;
    slong *p = array_reserve(slong, &ide, nloc);
    for(i=0;i<nloc;++i) p[i] = id[i+j0].id;
    data->Q_Aff[lvl].nloc = ide.n = nloc;
    amg_setup_mat(&ide,nloc, &data->Aff[lvl], 
                  uid, uid_n,  id_perm, j0,j0,
                  (struct gnz*)mat[2].ptr + mat_off[2][lvl],
                  mat_off[2][lvl+1]-mat_off[2][lvl], buf);
    data->Q_Aff[lvl].gsh = gs_setup(ide.ptr,ide.n, &data->comm, 0,gs_auto,1);
    if(ide.n>max_e) max_e=ide.n;
  }
  free(mat_off[0]);
  array_free(&ide);
  return max_e;
}

static uint *compute_offsets(
  const struct id_data *const id, const uint n,
  const int levels)
{
  uint i, *off = tmalloc(uint, levels+1);
  int lvl = -1;
  for(i=0;i<n;++i) {
    const int lvl_i = id[i].level;
    while(lvl<lvl_i) off[++lvl]=i;
  }
  while(lvl<levels) off[++lvl]=i;
  return off;
}

static void read_data(
  struct crs_data *const data,
  struct array *ids, struct array mats[3],
  struct crystal *const cr,
  const ulong *uid, const uint uid_n);

static void amg_setup_aux(struct crs_data *data,  uint n, const ulong *id)
{
  struct crystal cr;
  struct array uid; uint *id_perm;
  struct array ids, mat[3];
  uint max_e; ulong temp_long;

  crystal_init(&cr, &data->comm);
  data->umap = assign_dofs(&uid,0, id,n,data->comm.id,data->gs_top,&cr.data);
  data->un = uid.n;
  
  sortp_long(&cr.data,0, uid.ptr,uid.n,sizeof(ulong));
  sarray_permute(ulong,uid.ptr   ,uid.n, cr.data.ptr, &temp_long);
  sarray_permute(uint ,data->umap,uid.n, cr.data.ptr, &max_e);

  read_data(data, &ids, mat, &cr, uid.ptr,uid.n);
  
  /* we should have data for every uid;
     if not, then the data is for a smaller problem than we were given */
  { int not_happy = ids.n==uid.n ? 0 : 1;
    if(comm_reduce_int(&data->comm, gs_max, &not_happy,1)) {
      comm_barrier(&data->comm);
      if(data->comm.id==0)
        fail(1,__FILE__,__LINE__,"AMG: missing data for some rows");
      else die(1);
    }
  }
  
  sarray_sort(struct id_data,ids.ptr,ids.n, id,1, &cr.data);
  sarray_sort(struct id_data,ids.ptr,ids.n, level,0, &cr.data);
  sarray_permute(uint,data->umap,ids.n, cr.data.ptr, &max_e);
  id_perm = tmalloc(uint, uid.n);
  sarray_perm_invert(id_perm, cr.data.ptr, uid.n);
  /* the global id   uid[i]   will have local index    id_perm[i]
     (the local storage is sorted by level) */

  data->lvl_offset = compute_offsets(ids.ptr,ids.n, data->levels);

  max_e = amg_setup_mats(data,uid.ptr,uid.n,id_perm,ids.ptr,mat, &cr.data);

  { const unsigned levels=data->levels;
    const uint *const off = data->lvl_offset;
    struct id_data *const id = ids.ptr; const uint n = ids.n;
    double *d;
    uint i,max_f=0;
    for(i=0;i<levels-1;++i) {
      const uint nf = off[i+1]-off[i];
      if(nf>max_f) max_f=nf;
    }
    d = data->Dff = tmalloc(double, 3*n + 3*max_f + max_e + 6*(levels-1));
    data->x = data->Dff + n;
    data->b = data->x + n;
    data->c = data->b + n;
    data->c_old = data->c + max_f;
    data->r = data->c_old + max_f;
    data->buf = data->r + max_f;
    data->timing = data->buf + max_e;
    for(i=0;i<n;++i) d[i]=id[i].D;
    for(i=0;i<6*(levels-1);++i) data->timing[i]=0;
    data->timing_n=0;
  }

  free(id_perm);
  array_free(&ids);
  array_free(&mat[0]);
  array_free(&mat[1]);
  array_free(&mat[2]);
  array_free(&uid);
  crystal_free(&cr);
}

static void amg_dump(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  struct crs_data *data);

struct crs_data *crs_setup(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  uint null_space, const struct comm *comm)
{
  struct crs_data *data = tmalloc(struct crs_data,1);
  
#ifdef AMG_DUMP
  int dump=1;
#else
  int dump=0;
#endif
  
  comm_dup(&data->comm,comm);

  data->gs_top = gs_setup((const slong*)id,n, &data->comm, 1,
    dump?gs_crystal_router:gs_auto, !dump);

  if(dump) {
    amg_dump(n,id,nz,Ai,Aj,A,data);
    gs_free(data->gs_top);

    if(data->comm.id==0) printf("AMG dump successful\n"), fflush(stdout);
    comm_barrier(&data->comm);
    comm_free(&data->comm);
    free(data);
    die(0);
  } else {
    data->null_space = null_space;
    amg_setup_aux(data, n,id);
  }
  return data;
}

void crs_free(struct crs_data *data)
{
  const unsigned levels = data->levels;

  free(data->Dff);

  if(levels>1) {
    unsigned lvl;
    for(lvl=0;lvl<levels-1;++lvl)
      gs_free(data->Q_Aff[lvl].gsh),
      gs_free(data->Q_AfP[lvl].gsh),
      gs_free(data->Q_W[lvl].gsh);
    free(data->W[0].a);
    free(data->W[0].row_off);
  }
  
  free(data->W);
  free(data->Q_W);

  free(data->lvl_offset);

  free(data->cheb_m);
  free(data->cheb_rho);
  
  free(data->umap);
  gs_free(data->gs_top);
  comm_free(&data->comm);
  
  free(data);
}

/*==========================================================================

  Find ID
  
  As proc 0 reads in the files, it needs to know where to send the data.
  The find_id function below takes a sorted list of id's (with no repeats)
  and outputs the corresponding list of owning procs.

  ==========================================================================*/

struct find_id_map { ulong id; uint p; };

struct find_id_data {
  struct array map;
  struct array work;
  struct crystal *cr;
};

static void find_id_setup(
  struct find_id_data *const data,
  const ulong *id, uint n,
  struct crystal *const cr)
{
  const uint np = cr->comm.np;
  uint i; struct find_id_map *q;

  data->cr = cr;
  data->work.ptr=0, data->work.n=0, data->work.max=0;
  array_init(struct find_id_map, &data->map, n);
  data->map.n=n;
  for(q=data->map.ptr,i=0;i<n;++i)
    q[i].id = id[i], q[i].p = id[i] % np;
  sarray_transfer(struct find_id_map,&data->map,p,1,cr);
  sarray_sort(struct find_id_map,data->map.ptr,data->map.n, id,1, &cr->data);
}

static void find_id_free(struct find_id_data *const data)
{
  array_free(&data->map);
  array_free(&data->work);
}

struct find_id_work { ulong id; uint p; uint wp; };

static int find_id(
  uint *p_out, const unsigned p_stride,
  struct find_id_data *const data,
  const ulong *id, const unsigned id_stride, const uint n)
{
  struct find_id_work *p, *p_end;
  const struct find_id_map *q=data->map.ptr, *const q_end = q+data->map.n;
  const uint np = data->cr->comm.np;
  int not_found = 0;

  uint nn;
  p = array_reserve(struct find_id_work, &data->work, n);
  for(nn=n;nn;--nn,id=(const ulong*)((const char*)id+id_stride))
    p->id=*id, p->p=0, p->wp = (*id)%np, ++p;
  data->work.n=n;
  
  /* send to work proc */
  sarray_transfer(struct find_id_work,&data->work,wp,1,data->cr);

  /* match id's with rid's */
  sarray_sort(struct find_id_work,data->work.ptr,data->work.n, id,1,
    &data->cr->data);
  for(p=data->work.ptr,p_end=p+data->work.n;p!=p_end;++p) {
    while(q!=q_end && q->id<p->id) ++q;
    if(q==q_end) break;
    if(q->id!=p->id) not_found=1,p->p=-(uint)1;
    else p->p=q->p;
  }
  for(;p!=p_end;++p) not_found=1,p->p=-(uint)1;
  
  /* send back */
  sarray_transfer(struct find_id_work,&data->work,wp,0,data->cr);

  /* map back to user data */
  sarray_sort(struct find_id_work,data->work.ptr,data->work.n, id,1,
    &data->cr->data);
  p = data->work.ptr;
  for(nn=n;nn;--nn,p_out=(uint*)((char*)p_out+p_stride))
    *p_out = p->p, ++p;
  
  return comm_reduce_int(&data->cr->comm, gs_max, &not_found,1);
}


/*==========================================================================

  File I/O
  
  small wrappers to read/write files consisting of doubles
  first double is 3.14159 --- a marker to test for endianness

  ==========================================================================*/

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
      fail("ERROR ",__FILE__,__LINE__,
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

/*==========================================================================

  Read amg.dat, amg_*.dat
  
  The function read_data is responsible for reading these files,
  and creating arrays
    ids     of struct id_data
    mat[3]  of struct gnz      (W, AfP, Aff)
  distributed to the appropriate procs
  
  ==========================================================================*/

static ulong read_level_data(struct crs_data *data, struct file f)
{
  unsigned i,n; ulong tn; double *buf;
  int updt=1;
  int code=0;
  if(data->comm.id==0) {
//Check header for newest amg_matlab code.
    double hdr; dread(&hdr,1,f,&code);
    if(code==0) {
      printf("AMG version %3.2f\n",hdr);
      if(hdr!=2.01) {
        printf("Update amg_matlab tool and create new .dat files before re-running\n"); 
        updt=0;
     }
      double t; dread(&t,1,f,&code);
      data->levels = t;
      printf("AMG: %u levels\n", data->levels);
    }
  }
  comm_bcast(&data->comm, &data->levels,sizeof(unsigned), 0);
  comm_bcast(&data->comm, &updt,sizeof(int), 0);
  comm_bcast(&data->comm, &code,sizeof(int), 0);
  if(!updt||code!=0) die(1);

  n = data->levels-1;
  data->cheb_m   = tmalloc(unsigned,n);
  data->cheb_rho = tmalloc(double  ,n);
  
  buf = tmalloc(double, 2*n+1);
  if(data->comm.id==0) dread(buf,2*n+1,f,&code);
  comm_bcast(&data->comm, &code,sizeof(int), 0);
  if(code!=0) die(1);

  comm_bcast(&data->comm, buf,(2*n+1)*sizeof(double), 0);
  for(i=0;i<n;++i) data->cheb_m[i]   = buf[i];
  for(i=0;i<n;++i) data->cheb_rho[i] = buf[n+i];
  tn = buf[2*n];
  data->tni = 1/(double)tn;
  free(buf);
  
  if(data->comm.id==0) {
    printf("AMG Chebyshev smoother data:\n");
    for(i=0;i<n;++i)
        printf("AMG  level %u: %u iterations with rho = %g\n",
          i+1, data->cheb_m[i], (double)data->cheb_rho[i]);
    printf("AMG: %lu rows\n", (unsigned long)tn);
  }
  
  return tn;
}

static void read_data(
  struct crs_data *const data,
  struct array *ids, struct array mat[3],
  struct crystal *const cr,
  const ulong *uid, const uint uid_n)
{
  int code;
  struct find_id_data fid;
  const uint pid = data->comm.id;
  ulong r,tn;
  struct array read_buffer=null_array;
  struct array id_buffer = null_array, mat_buffer = null_array;
  uint *row_lens=0, *id_proc=0, *id_perm=0;
  struct array mat_proc = null_array;
  struct file f={0,0}, fm[3]={{0,0},{0,0},{0,0}};
  unsigned m;
  ids->ptr=0, ids->n=0, ids->max=0;
  for(m=0;m<3;++m) mat[m].ptr=0,mat[m].n=0,mat[m].max=0;
  find_id_setup(&fid, uid,uid_n, cr);
  code=0;
  if(pid==0) {
    f = dopen("amg.dat",'r',&code);
    fm[0] = dopen("amg_W.dat",'r',&code);
    fm[1] = dopen("amg_AfP.dat",'r',&code);
    fm[2] = dopen("amg_Aff.dat",'r',&code);
    if(code==0) {
     array_init(double, &read_buffer, 6*AMG_BLOCK_ROWS);
     row_lens = tmalloc(uint, 5*AMG_BLOCK_ROWS);
     id_proc = row_lens + 3*AMG_BLOCK_ROWS;
     id_perm = id_proc + AMG_BLOCK_ROWS;
     array_reserve(struct id_data, &id_buffer, AMG_BLOCK_ROWS);
    }
  }
  comm_bcast(&data->comm,&code,sizeof(int),0);
  if(code!=0) die(1);

  tn = read_level_data(data,f);
  for(r=0;r<tn;r+=AMG_BLOCK_ROWS) {
    unsigned nr = tn-r>AMG_BLOCK_ROWS ? AMG_BLOCK_ROWS : (unsigned)(tn-r);
    uint mat_size[3]={0,0,0};
    
    /* read id data */
    if(pid==0) {
      unsigned i;
      struct id_data *idp = id_buffer.ptr;
      double *b = read_buffer.ptr;
      dread(b,nr*6, f,&code);
      if(code==0) {
        printf("AMG: reading through row %lu, pass %u/%u\n",
        (unsigned long)b[(nr-1)*6],
        (unsigned)(r/AMG_BLOCK_ROWS+1),
        (unsigned)((tn+AMG_BLOCK_ROWS-1)/AMG_BLOCK_ROWS));
        for(i=0;i<nr;++i) {
          idp[i].id = *b++;
          idp[i].level = (uint)(*b++) - 1;
          mat_size[0] += row_lens[3*i+0] = *b++; /* W   row length */
          mat_size[1] += row_lens[3*i+1] = *b++; /* AfP row length */
          mat_size[2] += row_lens[3*i+2] = *b++; /* Aff row length */
          idp[i].D = *b++;
        }
        id_buffer.n=nr;
      
      /* sort id_buffer, remember how to undo */
        sarray_sort(struct id_data,idp,nr, id,1, &cr->data);
        sarray_perm_invert(id_perm, cr->data.ptr, nr);
      }
    } else
      id_buffer.n=0;

    comm_bcast(&data->comm,&code,sizeof(int),0);
    if(code!=0)die(1);

    /* find who owns each row */
    if(find_id(id_proc,sizeof(uint), &fid,
       (const ulong*)((const char*)id_buffer.ptr+offsetof(struct id_data,id)),
       sizeof(struct id_data), id_buffer.n)) {
      if(pid==0)
        fail(1,__FILE__,__LINE__,"AMG: data has more rows than given problem");
      else die(1);
    }
    if(pid==0) { /* undo sorting of id_buffer */
      buffer_reserve(&cr->data,sizeof(struct id_data)+sizeof(uint));
      sarray_permute(struct id_data,id_buffer.ptr,nr, id_perm, cr->data.ptr);
      sarray_permute(uint,          id_proc,      nr, id_perm, cr->data.ptr);
    }
    /* read matrix data */
    for(m=0;m<3;++m) {
      if(pid==0) {
        const struct id_data *const idp = id_buffer.ptr;
        unsigned i; uint j,rl;
        struct gnz *p = array_reserve(struct gnz, &mat_buffer, mat_size[m]);
        double *b = array_reserve(double, &read_buffer, 2*mat_size[m]);
        uint *proc = array_reserve(uint, &mat_proc, mat_size[m]);
        dread(b,2*mat_size[m], fm[m],&code);
        if(code==0) {
          for(i=0;i<nr;++i) {
            const ulong i_id = idp[i].id; 
            const uint i_p = id_proc[i];
            for(rl=row_lens[3*i+m],j=0;j<rl;++j)
              p->i = i_id, p->j = *b++, p->a = *b++, *proc++ = i_p, ++p;
          }
          mat_buffer.n = mat_size[m];
        }
      } else
        mat_buffer.n = 0;
      comm_bcast(&data->comm,&code,sizeof(int),0);
      if(code!=0)die(1);
      sarray_transfer_ext(struct gnz,&mat_buffer,mat_proc.ptr,sizeof(uint),cr);
      array_cat(struct gnz,&mat[m],mat_buffer.ptr,mat_buffer.n);
    }
    /* send id_data to owner */
    sarray_transfer_ext(struct id_data,&id_buffer,id_proc,sizeof(uint),cr);
    array_cat(struct id_data,ids,id_buffer.ptr,id_buffer.n);
  }
  array_free(&id_buffer);
  array_free(&mat_buffer);
  if(pid==0) {
    array_free(&read_buffer);
    free(row_lens);
    array_free(&mat_proc);
    dclose(fm[2]);
    dclose(fm[1]);
    dclose(fm[0]);
    dclose(f);
  }
  find_id_free(&fid);
}

/*==========================================================================

  Write amgdmp_*.dat
  
  The user's matrix is first assembled, then written out.
  
  ==========================================================================*/


enum mat_order { row_major, col_major };
enum distr { row_distr, col_distr };

#define rid_equal(a,b) ((a).p==(b).p && (a).i==(b).i)

/* rnz is a mnemonic for remote non zero */
struct rnz {
  double v; struct rid i,j;
};
#define nz_pos_equal(a,b) \
  (rid_equal((a).i,(b).i) && rid_equal((a).j,(b).j))

static void mat_sort(struct array *const mat,
                     const enum mat_order order, buffer *const buf)
{
  switch(order) {
  case col_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                j.p,0,j.i,0, i.p,0,i.i,0, buf); break;
  case row_major: sarray_sort_4(struct rnz,mat->ptr,mat->n,
                                i.p,0,i.i,0, j.p,0,j.i,0, buf); break;
  }
}

/* assumes matrix is already sorted */
static void mat_condense_sorted(struct array *const mat)
{
  struct rnz *dst,*src, *const end=(struct rnz*)mat->ptr+mat->n;
  if(mat->n<=1) return;
  for(dst=mat->ptr;;++dst) {
    if(dst+1==end) return;
    if(nz_pos_equal(dst[0],dst[1])) break;
  }
  for(src=dst+1;src!=end;++src) {
    if(nz_pos_equal(*dst,*src))
      dst->v += src->v;
    else
      *(++dst) = *src;
  }
  mat->n = (dst+1)-(struct rnz*)mat->ptr;
}

static void mat_condense(
  struct array *const mat, const enum mat_order order, buffer *const buf)
{
  mat_sort(mat,order,buf); mat_condense_sorted(mat);
}

static void mat_distribute(
  struct array *const mat, const enum distr d, const enum mat_order o,
  struct crystal *const cr)
{
  switch(d) {
  case row_distr: mat_condense(mat,row_major,&cr->data);
                  sarray_transfer(struct rnz,mat, i.p,0, cr); break;
  case col_distr: mat_condense(mat,col_major,&cr->data);
                  sarray_transfer(struct rnz,mat, j.p,0, cr); break;
  }
  mat_condense(mat,o,&cr->data);
}

struct labelled_rid {
  struct rid rid; ulong id;
};

static void mat_list_nonlocal_sorted(
  struct array *const nonlocal_id,
  const struct array *const mat, const enum distr d,
  const ulong *uid, struct crystal *const cr)
{
  const uint pid = cr->comm.id;
  struct rid last_rid;
  const struct rnz *p, *const e=(const struct rnz*)mat->ptr+mat->n;
  uint count; struct labelled_rid *out, *end;
  #define BUILD_LIST(k) do { \
    last_rid.p=-(uint)1,last_rid.i=-(uint)1; \
    for(count=0,p=mat->ptr;p!=e;++p) { \
      if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
      last_rid=p->k; ++count; \
    } \
    array_init(struct labelled_rid, nonlocal_id, count); \
    nonlocal_id->n=count; out=nonlocal_id->ptr; \
    last_rid.p=-(uint)1,last_rid.i=-(uint)1; \
    for(p=mat->ptr;p!=e;++p) { \
      if(p->k.p==pid || rid_equal(last_rid,p->k)) continue; \
      (out++)->rid=last_rid=p->k; \
    } \
  } while(0)
  switch(d) {
    case row_distr: BUILD_LIST(j); break;
    case col_distr: BUILD_LIST(i); break;
  }
  #undef BUILD_LIST
  sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
  for(out=nonlocal_id->ptr,end=out+nonlocal_id->n;out!=end;++out)
    out->id=uid[out->rid.i];
  sarray_transfer(struct labelled_rid,nonlocal_id,rid.p,1,cr);
  sarray_sort_2(struct labelled_rid,nonlocal_id->ptr,nonlocal_id->n,
                rid.p,0, rid.i,0, &cr->data);
}

static void mat_list_nonlocal(
  struct array *const nonlocal_id,
  struct array *const mat, const enum distr d,
  const ulong *uid, struct crystal *const cr)
{
  switch(d) {
    case row_distr: mat_sort(mat,col_major,&cr->data); break;
    case col_distr: mat_sort(mat,row_major,&cr->data); break;
  }
  mat_list_nonlocal_sorted(nonlocal_id,mat,d,uid,cr);
}

static uint dump_matrix_setdata(
  buffer *const buf, /* output; ok if this is one of cr's buffers */
  struct array *const mat, const ulong *const uid,
  struct crystal *const cr)
{
  const uint pid = cr->comm.id;
  struct array nonlocal_id;
  double *vi, *vj, *va; uint n;
  const struct rnz *nz, *enz;
  const struct labelled_rid *rlbl;
  
  mat_distribute(mat,row_distr,col_major,cr);
  n = mat->n;

  mat_list_nonlocal_sorted(&nonlocal_id,mat,row_distr,uid,cr);
  
  buffer_reserve(buf,3*n*sizeof(double));
  vi=buf->ptr, vj=vi+n, va=vj+n;
  rlbl = nonlocal_id.ptr;
  for(nz=mat->ptr,enz=nz+n;nz!=enz;++nz) {
    *vi++ = uid[nz->i.i];
    *va++ = nz->v;
    if(nz->j.p==pid)
      *vj++ = uid[nz->j.i];
    else {
      const uint jp = nz->j.p, ji = nz->j.i;
      while(rlbl->rid.p<jp) ++rlbl;
      if(rlbl->rid.p!=jp) printf("dump_matrix: FAIL!!!\n");
      while(rlbl->rid.i<ji) ++rlbl;
      if(rlbl->rid.i!=ji) printf("dump_matrix: FAIL!!!\n");
      *vj++ = rlbl->id;
    }
  }
  array_free(&nonlocal_id);
  return n;
}

static void dump_matrix(
  struct array *const mat, const ulong *const uid,
  struct crystal *const cr)
{
  const struct comm *comm = &cr->comm;
  const uint pid = comm->id, np = comm->np;
  buffer *const buf = &cr->data;
  struct file fi={0,0}, fj={0,0}, fp={0,0};
  uint i,n;
  int code;
  
  n = dump_matrix_setdata(buf, mat,uid,cr);

  code = 0;
  if(pid==0) {
    fi=dopen("amgdmp_i.dat",'w',&code);
    fj=dopen("amgdmp_j.dat",'w',&code);
    fp=dopen("amgdmp_p.dat",'w',&code);
  }
  comm_bcast(comm,&code,sizeof(int),0);
  if(code!=0) die(1);
   
  for(i=0;i<np;++i) {
    comm_barrier(comm);
    if(pid!=0 && i==pid)
      comm_send(comm, &n,sizeof(uint), 0,i),
      comm_send(comm, buf->ptr,3*n*sizeof(double), 0,np+i);
    else if(pid==0) {
      double *v;
      printf("AMG writing data from proc %u\n",(unsigned)i),fflush(stdout);
      if(i!=0) {
        comm_recv(comm, &n,sizeof(uint), i,i);
        buffer_reserve(buf,3*n*sizeof(double));
        comm_recv(comm, buf->ptr,3*n*sizeof(double), i,np+i);
      }
      v = buf->ptr;
      if(code==0) {
        dwrite(fi, v    , n,&code);
        dwrite(fj, v+  n, n,&code);
        dwrite(fp, v+2*n, n,&code);
      }
    }
  }
  if(pid==0) dclose(fi),dclose(fj),dclose(fp);
  comm_bcast(comm,&code,sizeof(int),0);
  if(code!=0) die(1);
  comm_barrier(comm);
}

/* assumes data->comm and data->gs_top are set */
static void amg_dump(
  uint n, const ulong *id,
  uint nz, const uint *Ai, const uint *Aj, const double *A,
  struct crs_data *data)
{
  struct crystal cr;
  struct array uid; struct rid *rid_map = tmalloc(struct rid,n);

  struct array mat;
  uint k; struct rnz *out;

  crystal_init(&cr, &data->comm);
  assign_dofs(&uid,rid_map, id,n,data->comm.id,data->gs_top,&cr.data);

  array_init(struct rnz,&mat,nz);
  for(out=mat.ptr,k=0;k<nz;++k) {
    uint i=Ai[k], j=Aj[k]; double a=A[k];
    if(id[i]==0 || id[j]==0 || fabs(a)==0) continue;
    out->v = a, out->i=rid_map[i], out->j=rid_map[j];
    ++out;
  }
  mat.n = out-(struct rnz*)mat.ptr;
  free(rid_map);

  dump_matrix(&mat,uid.ptr,&cr);
  
  array_free(&uid);
  crystal_free(&cr);
}
