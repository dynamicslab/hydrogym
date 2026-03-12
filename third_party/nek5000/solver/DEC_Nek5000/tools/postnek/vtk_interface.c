/**********************************ivec.c**************************************
INTERFACE TO VTK TOOLKIT

Author: Henry M. Tufo III

e-mail: tufo@mcs.anl.gov

snail-mail:

Last Modification: 
11.01.99
***********************************ivec.c*************************************/

/**********************************ivec.c**************************************
File Description:
-----------------

***********************************ivec.c*************************************/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#ifdef MPISRC
#include <mpi.h>
#endif

#include "const.h"
#include "types.h"
#include "bss_malloc.h"
#include "error.h"
#include "vtk_interface.h"

/* hack ... hold data for vtk interface as globals */
char *file_header;      /* file prefix                                      */
int tag;                /* file number (in sequence)                        */
int npts;               /* number of (x,y,z) points                         */
float *pt_coords;       /* (x,y,z) triplets (3*npts)                        */
int nhexes;             /* number of VTK_HEXAHEDRONS (12)                   */
int *hex_cells;         /* (v0,...,v7)-tuples (8*nhexes), v is index in pts */
float *contour_scalar;  /* scalar to contour on                             */
float *passive_scalar;  /* scalar to map to isosurface                      */
int nsrfs;              /* number of surfaces to generate                   */
float *srf_thresholds;  /* surface thresholds                               */

main()
{
  mainf_();
}


void
clear_vtk_all_(void)
{
  tag=-1;
  bss_free(file_header);
  file_header=NULL;
  npts=-1;
  bss_free(pt_coords);
  pt_coords=NULL;
  nhexes=-1;
  bss_free(hex_cells);
  hex_cells=NULL;
  bss_free(contour_scalar);
  contour_scalar=NULL;
  bss_free(passive_scalar);
  passive_scalar=NULL;
  nsrfs=-1;
  bss_free(srf_thresholds);
  srf_thresholds=NULL;
}

void
get_srfs(void)
{
  int i;
  char buffer[81];
  FILE *fp;

  
  while ((fp=fopen("thresholds","r"))==NULL) 
    {
      error_msg_warning("hey ... can't open \"thresholds\" file!!!\n");
      system ("sleep(2)");
    }
  
  i=0;
  while (fgets(buffer,80,fp))
    {i++;}
  nsrfs=i;
  srf_thresholds = (float *) bss_malloc(nsrfs*sizeof(float));

  rewind(fp);
  for (i=0;i<nsrfs;i++)
    {
      fgets(buffer,80,fp);
      srf_thresholds[i] = atof(buffer);
    }
}


void
make_vtk_srf_(int *flag)
{
  int error;

  if (tag==-1) {error_msg_fatal("bad tag!!!\n");}
  if (!file_header) {error_msg_fatal("bad file header!!!\n");}
  if (npts==-1) {error_msg_fatal("bad npts!!!\n");}
  if (!pt_coords) {error_msg_fatal("bad pt_coords!!!\n");}
  if (nhexes==-1) {error_msg_fatal("bad nhexes!!!\n");}
  if (!hex_cells) {error_msg_fatal("bad hex_cells!!!\n");}
  if (!contour_scalar) {error_msg_fatal("bad contour scalar!!!\n");}
  if (!passive_scalar) {error_msg_fatal("bad contour scalar!!!\n");}

  if (*flag) {get_srfs();}

  if ((nsrfs==-1)||(nsrfs<1)) {error_msg_fatal("bad nsrfs!!!\n");}
  if (!srf_thresholds) {error_msg_fatal("bad srf_thresholds!!!\n");}
  
  printf("make_vtk_interface %d %s %d %f %d %d %f %f %d %f %d\n",
	 tag,file_header,npts,*pt_coords,nhexes,*hex_cells,*contour_scalar,
	 *passive_scalar,nsrfs,*srf_thresholds,*flag);

  fflush(NULL);

  error = createsurface(tag,file_header,npts,pt_coords,nhexes,hex_cells,
			 contour_scalar,passive_scalar,nsrfs,srf_thresholds,
			 *flag);

  if (error!=1)
    {error_msg_fatal("create_surface failed ... stop!!!\n");}
}

void
set_vtk_file_tag_(int *n)
{
  tag = *n;
}


void
set_vtk_file_name_(char *fn)
{
  int i;

  comm_init();
  bss_init();
  perm_init();


  file_header = (char *) bss_malloc(81*sizeof(char));
  for (i=0;i<80;i++)
    {
      if (fn[i]) 
	{file_header[i]=fn[i];}
      else
	{break;}
    }
  file_header[i]='\0';
}


void
set_vtk_pts_(float *x, float *y, float *z, int *n, int *reflect)
{
  int i;
  float *ptr, *rptr;

  /* are we reflecting around symm. plane? */
  if (*reflect==TRUE)
    {npts=*n<<1;}
  else
    {npts=*n;}

  if (pt_coords) {bss_free(pt_coords);}
  ptr = pt_coords = (float *) bss_malloc(npts*3*sizeof(float));

  if (*reflect==TRUE)
    {
      rptr = ptr+*n*3;
      for (i=0;i<*n;i++)
	{*ptr++=*x;*ptr++=*y;*ptr++=*z;*rptr++=*x++;*rptr++=-*y++;*rptr++=*z++;}
    }
  else
    {
      for (i=0;i<*n;i++)
	{*ptr++ = *x++; *ptr++ = *y++; *ptr++ = *z++;}
    }
}


void
set_vtk_cells_(int *cells, int *n, int *reflect)
{
  int i;
  int *ptr, *rptr;

  /* are we reflecting around symm. plane? */
  if (*reflect==TRUE)
    {nhexes=*n<<1;}
  else
    {nhexes=*n;}

  if (hex_cells) {bss_free(hex_cells);}
  ptr = hex_cells = (int *) bss_malloc(nhexes*8*sizeof(int));

  if (*reflect==TRUE) 
    {
      rptr = ptr+*n*8;
      for (i=0;i<*n*8;i++)
	{*ptr++ = *cells; *rptr++ = *cells++ + *n;}
    }
  else
    {
      for (i=0;i<*n*8;i++)
	{*ptr++ = *cells++;}
    }
}


void
set_vtk_cscalar_(float *vals, int *n, int *reflect)
{
  int i;
  float *ptr, *rptr;

  /* check to see if we match number of points */
  if (*reflect==TRUE)
    {
      if (npts-(*n<<1)) 
	{error_msg_fatal("nsclrs doesn't match npts %d %d %d!!!\n",npts,*n,*reflect);}
    }
  else
    {
      if (npts-*n) 
	{error_msg_fatal("nsclrs doesn't match npts %d %d %d!!!\n",npts,*n,*reflect);}
    }

  /* has the space already been allocated? */
  if (!contour_scalar)
    {contour_scalar = (float *) bss_malloc(npts*sizeof(float));}
  
  ptr = contour_scalar;
  if (*reflect==TRUE) 
    {
      rptr = ptr+*n;
      for (i=0;i<*n;i++)
	{*ptr++ = *vals; *rptr++ = *vals++;}
    }
  else
    {
      for (i=0;i<*n;i++)
	{*ptr++ = *vals++;}
    }
}


void
set_vtk_pscalar_(float *vals, int *n, int *reflect)
{
  int i;
  float *ptr, *rptr;

  /* check to see if we match number of points */
  if (*reflect==TRUE)
    {
      if (npts-(*n<<1)) 
	{error_msg_fatal("nsclrs doesn't match npts %d %d %d!!!\n",npts,*n,*reflect);}
    }
  else
    {
      if (npts-*n) 
	{error_msg_fatal("nsclrs doesn't match npts %d %d %d!!!\n",npts,*n,*reflect);}
    }

  /* has the space already been allocated? */
  if (!passive_scalar)
    {passive_scalar = (float *) bss_malloc(npts*sizeof(float));}
  
  ptr = passive_scalar;
  if (*reflect==TRUE) 
    {
      rptr = ptr+*n;
      for (i=0;i<*n;i++)
	{*ptr++ = *vals; *rptr++ = *vals++;}
    }
  else
    {
      for (i=0;i<*n;i++)
	{*ptr++ = *vals++;}
    }
}
