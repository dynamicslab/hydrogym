#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/Xresource.h>
#include "icon"
#include <assert.h>
#include "errmem.h"

/* number of colors (this includes ed's 16)  */
#define NUM_COLOR_CELLS  15
#define MAX_WINDOWH  8960          /*  800  896 1412 1694   */
#define MAX_WINDOWW  9520          /*  850  952 1500 1800   */
#define SAFTY  0

/* macros and what not */
#define MAX(x,y)        ((x)>(y)) ? (x) : (y)
#define MIN(x,y)        ((x)<(y)) ? (x) : (y)
#define FALSE		0
#define TRUE		1

/* checkerboard used in dimBorder */
#define check_width 16
#define check_height 16
static char check_bits[] = {
   0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa,
   0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa,
   0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa };
#define MOUSE 0
#define LINE  1

/* mouse event place holder */
struct mse_event_rec
{
   int xclicked;
   int yclicked;
   int buttonNumber;
};
static struct mse_event_rec mouseRecord;

/* display window and drawables */
static Display *dpy=NULL;
static Window theWindow, iconWindow;
static Pixmap thePixmap, IconPix, dimBorder;

/* FASLE means that we draw on the pixmap whenever we draw on the window */
static int hold_Pixmap=FALSE;
static int hold_Window=FALSE;

static int theScreen;
static int theDepth;

/* graphics contexts and what not */
static Cursor cursorPen;
static XFontStruct *textFontInfo=NULL;
static GC colorPenGC, fillGC, fountainPenGC, calligPenGC;
static XGCValues gcv;

/* window size in pixels and offset w.r.t. root */
static int windowh = MAX_WINDOWH, windoww = MAX_WINDOWW;
static int vfx = 0, vfy = 0;

/* pen, point, pixel and color map tables */
static XPoint colorPenPosition, fountainPenPosition;
static Colormap cmap;
static XColor colorCells[NUM_COLOR_CELLS];
static unsigned long pixel_map[NUM_COLOR_CELLS+SAFTY];
static unsigned char     r_map[NUM_COLOR_CELLS+SAFTY];
static unsigned char     g_map[NUM_COLOR_CELLS+SAFTY];
static unsigned char     b_map[NUM_COLOR_CELLS+SAFTY];
static int n_pixels=0;
static int ForeColor, BackColor;

/* not mine but obv. i/o information holders */
static int char_spacing = 6;
static int line_spacing = 15;
static int text_screenh = 150;
static int left_margin = 000;
static char Line[10][90];
static int active_line = 0;
static int top_line = 0;
static int num_lines = 10;
static int lineEntered=0;
static int scroll=0;
static char input_line[81];
static int input_line_idx=0;


/* function prototypes */
static void showChar(char c);
static void backSpace();
static void TypeChar(char c);
static void error (char *mesg);
static int  Quit();
static void error();
static void Random_Seed(unsigned seed);
static int  Random_Integer(int lb, int ub);
static int  lvec_linear_search(unsigned long item,unsigned long *list, int n);
static void close_env (void);
static void SetUpEnv (void);
static void xmainloop (int code);
static void advanceLine();
static void resetXPRINT();
#ifdef CRAY
static void SYSTEM(char *string);
static void RENAME(char *from, *to);
#endif



/**********************************xdriver.c***********************************
*******************************************************************************
EXTERNALLY VISIBLE FUNCTIONS!!!
*******************************************************************************
***********************************xdriver.c**********************************/

/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!
***********************************xdriver.c**********************************/
void   
hmt_refresh_ (void)
{
   XCopyArea(dpy,thePixmap,theWindow,colorPenGC,0,0,windoww,windowh,0,0);
}

/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
void 
hpm_t_ (void)
{
   hold_Pixmap = TRUE;
}

void 
htw_t_ (void)
{
   hold_Window = TRUE;
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
void 
hpm_f_ (void)
{
   hold_Pixmap = FALSE;
}

void 
htw_f_ (void)
{
  hold_Window = FALSE;
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
void grab_window_raw_(float *x1, float *y1, float *x2, float *y2, char *filename)
{
  FILE *fp;
  XWindowAttributes xwa; XVisualInfo vi_temp, *vi; int nvi;
  XColor *colors=0;
  unsigned mask[3], shift[3];
  unsigned char *map[3]={0,0,0};
  char catstring[120];

  Window root;
  int dx,dy;
  unsigned w,h,border,depth, x,y, tw,th, xb, xe, yb, ye;

  XImage *ximage;
  unsigned char *buf;

  sprintf(catstring,"%s.rgb",filename);

  if (!(fp=fopen(catstring,"wb")))
  {error("grab_window() :: raw rgb file fopen error!\n");}

  XGetWindowAttributes(dpy,theWindow,&xwa);
  vi_temp.screen = XScreenNumberOfScreen(xwa.screen);
  vi_temp.visualid = XVisualIDFromVisual(xwa.visual);
  vi = XGetVisualInfo(dpy, VisualScreenMask|VisualIDMask, &vi_temp, &nvi);

  if(vi->class==PseudoColor || vi->class==GrayScale) {
    unsigned i, n = vi->colormap_size;
    colors = tcalloc(XColor,n+1);
    for(i=0;i<n;++i) colors[i].pixel=i;
    XQueryColors(dpy,xwa.colormap,colors,n);
  }

  XGetGeometry(dpy,thePixmap,&root,&dx,&dy,&w,&h,&border,&depth);
         
  ximage = XGetImage(dpy,thePixmap,0,0,w,h,~0ul,ZPixmap); 

  if(!colors) {
    unsigned c;
    mask[0]=vi->red_mask;
    mask[1]=vi->green_mask;
    mask[2]=vi->blue_mask;
    for(c=0;c<3;++c) {
      unsigned m = mask[c], s=0;
      if(!m) failwith("color mask is 0");
      while((m&1)==0) m>>=1, ++s;
      shift[c]=s;
      map[c]=tmalloc(unsigned char,m+1);
      for(s=0;s<=m;++s) map[c][s]=(unsigned char)(.5+255.*s/(double)m);
    }
  }

  XFree(vi);

  tw=w; 
  th=h; 

  buf = tmalloc(unsigned char,tw*th*3);

  xb = (int)(MIN(*x1,*x2));
  xe = (int)(MAX(*x1,*x2));
  yb = (int)(MIN(*y1,*y2));
  ye = (int)(MAX(*y1,*y2));

  printf( "%d %d \n", xb, xe );
  printf( "%d %d \n", yb, ye );

  if(colors) {
    unsigned char *line, *pix;

    for(y=0,line=buf;y<h;++y,line+=tw*3) {
      for(x=0,pix=line;x<w;++x,pix+=3) {
	      /*
    for(y=yb,line=buf;y<ye;++y,line+=tw*3) {
      for(x=xb,pix=line;x<xe;++x,pix+=3) {
      */
        unsigned long pixel = ximage->f.get_pixel(ximage,x,y);
        pix[0] = colors[pixel].red;
        pix[1] = colors[pixel].green;
        pix[2] = colors[pixel].blue;
      }
    }
  } else {
    unsigned char *line, *pix;
    line = buf;
    for(y=yb ;y < ye;++y) {
      pix = line;
      for(x=xb;x < xe;++x) {
        unsigned long pixel = ximage->f.get_pixel(ximage,x,y);
        pix[0] = map[0][(pixel&mask[0])>>shift[0]];
        pix[1] = map[1][(pixel&mask[1])>>shift[1]];
        pix[2] = map[2][(pixel&mask[2])>>shift[2]];
	pix += 3;
      }
      line+= 3*(xe-xb);
    }
  }
  
  if(colors) 
	  free(colors); 
  else 
	  free(map[0]),free(map[1]),free(map[2]);

  XDestroyImage(ximage);
  
/* fwrite(buf,sizeof(unsigned char),3*(x2-x1)*(y2-y1),fp); */
  printf( "Image Size : %d %d \n", (xe-xb), (ye-yb) );
  fwrite(buf,sizeof(unsigned char),3*(xe-xb)*(ye-yb),fp);

  free(buf);
}


#ifdef OLDFILE
void 
grab_window_raw_(float *x1, float *y1, float *x2, float *y2, char *filename)
{       

   register int i, j, index1;
   register int max_i, max_j;
   register int index;
   register unsigned char *buf, *off;
   register unsigned long pixel, pixel_old;
   FILE *fp;
   XImage *xim;
   XColor rgb_ans;
   char catstring[120];
   FILE  *fdebug;

   /* append rgb suffix to file name */
   sprintf(catstring,"%s.rgb",filename);

#ifdef DEBUG
   printf("raw rgb file name %s\n",catstring);
#endif

   if (!(fp=fopen(catstring,"wb")))
   {error("grab_window() :: raw rgb file fopen error!\n");}

   /* create image and copy pixmap into image */
/*   xim = XGetImage(dpy,thePixmap,0,0,windoww,windowh,AllPlanes,ZPixmap); */

   hold_Window=1;
   xim = XGetImage(dpy,theWindow,0,0,windoww,windowh,AllPlanes,ZPixmap);
   /*
   i=windoww*windowh;
   printf( "RGB data size %d %d \n", windoww, windowh );
   assert( fwrite(xim->data,sizeof(unsigned char),i,fp) == i);
   fclose(fp);
   exit(0);
   */

   /*
   fdebug = (FILE *)fopen("debug.dat", "w");
   index1 = 0;
   for( i = 0; i < windoww; i++) 
   for( j = 0; j < windowh; j++) 
	   fprintf( fdebug, "%d %d %d \n", i, j, xim->data[index1++]);

   */

#ifdef DEBUG
   printf("bits_per_pixel=%d\n",xim->bits_per_pixel);
   printf("depth=%d\n",xim->depth);
   printf("width=%d\n",xim->width);
   printf("height=%d\n",xim->height);
#endif

   /* draw which portion? */
   /* x-axis,width :: y-axis,height :: quad 4 but positive coords */ 
   /* min_i=MAX(0,((int) *x1)); min_j=MAX(0,((int) *y1)); */
   max_i=MIN(windoww,((int) *x2));
   max_j=MIN(windowh,((int) *y2));

   /* buffer to write lines */
   if (!(buf = (unsigned char *) malloc(sizeof(unsigned char)*(max_i+1)*3)))
   {error("grab_window() :: raw rgb buf malloc error!\n");}

   /* scan line method of pixel extraction */
   for (j=MAX(0,((int) *y1));j<max_j;j++)
   {
#ifdef DEBUG
      if (!(j%10)) {printf("row%4d\n",j+1); fflush(stdout);}
#endif
      for (off=buf,i=MAX(0,((int) *x1));i<max_i;i++)
      {
         /* get pixel data */
         pixel=XGetPixel(xim,i,j);

         /* last one the same ==> done */
         if (pixel==pixel_old)
         {
            *off++ = r_map[index];
            *off++ = g_map[index];
            *off++ = b_map[index];
         }
         else
         {
            /* not encountered yet so add to map */
            if ((index=lvec_linear_search(pixel,pixel_map,n_pixels))==-1)
            {
               /* load pixel into pixel and rgb maps */
               pixel_map[n_pixels]=rgb_ans.pixel=pixel;
               XQueryColor(dpy,cmap,&rgb_ans);
               r_map[n_pixels] = *off++ =(unsigned char) rgb_ans.red; 
               g_map[n_pixels] = *off++ =(unsigned char) rgb_ans.green;
               b_map[n_pixels] = *off++ =(unsigned char) rgb_ans.blue;
               index=n_pixels++;
#ifdef DEBUG
               if (n_pixels>=NUM_COLOR_CELLS)
               {error("grab_window() :: too many colors!\n");}
#endif
            }
            else
            {
               *off++ = r_map[index];
               *off++ = g_map[index];
               *off++ = b_map[index];
            }
            pixel=pixel_old;
         }
      }

      /* write scan line */
      if (fwrite(buf,sizeof(unsigned char),off-buf,fp)!=(off-buf))
      {error("grab_window() :: raw rgb file fwrite error!\n");}

#ifdef DEBUG
      if ((off-buf)!=(3*(max_i-MAX(0,((int) *x1)))))
      {error("grab_window() :: bad off %d!\n",off);}
#endif
   }

   /* close byte stream */
   if (fclose(fp))
   {error("grab_window() :: raw rgb file close error!\n");}

   /* free image storage space */
   XDestroyImage(xim);

   /* how many colors in map */
   printf("found %d colors\n",n_pixels);
   free(buf);
   exit(0);

#ifdef NOT
   /* grabbing the raw data is not a good idea */
   char *rgb_data;
   rgb_data=xim->data;
   i=windoww*windowh;
   if (fwrite(rgb_data,sizeof(unsigned char),i,fp)!=i)
   {error("grab_window() :: raw rgb file fwrite error!\n");}

   return;


   /* bitmap ==> 1/0 only!!! */
   XWriteBitmapFile(dpy,"bitmap.hmt",thePixmap,windoww,windowh,-1,-1);
#endif
}

#endif


/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

  xwd gets drawn and visible portion of window only!!!
***********************************xdriver.c**********************************/
void 
grab_window_xwd_(char *filename)
{       
   char catstring[120];


   sprintf(catstring, "xwd -nobdrs -id %d -out %s.xwd",(int)theWindow,filename);

#ifdef DEBUG
   printf("trying to execute %s\n",catstring);
#endif

   system (catstring);
}

/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/

void grab_window_(char *filename,int *index)
{
  char catstring[120];
  char file[120];
  int pid;

  /* printf("theWindow = %d\n",(int) theWindow); */

  sprintf(file,"%s_%.4d",filename,*index);
  sprintf(catstring,"xwd -nobdrs -id %d -out %s.xwd",(int)theWindow,file);
  printf("trying to execute %s\n",catstring);
  system (catstring);
  sprintf(catstring,
  "imcopy %s.xwd -xpos 0 -xsize 640 -ypos 0 -ysize 480 %s.gif",file,file);
  printf("trying to execute %s\n",catstring);
  system (catstring);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void GETWIN(width,height)
#else
void getwin_(width,height)
#endif
   float *width,*height;
{
   *width  = (float) windoww;
   *height = (float) windowh;
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XPLANES(nplanes)
#else
void xplanes_(nplanes)
#endif
   int *nplanes;
{
   *nplanes=XDisplayPlanes(dpy,XDefaultScreen(dpy));
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)     
void XWAITM(button,xmouse,ymouse)
#else
void xwaitm_(button,xmouse,ymouse)
#endif
   int *button;
   float *xmouse,*ymouse;
{
   while(1){
      xmainloop(MOUSE);
      if (mouseRecord.yclicked<=(windowh-text_screenh))
         /* y position is not inside text region,
          * so mouse click was valid. */
         break;
   }
   *ymouse = (float)fabs((double)(mouseRecord.yclicked-
                                  (windowh-text_screenh)));
   *xmouse = (float)mouseRecord.xclicked;
   *button = mouseRecord.buttonNumber;
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void BEEP()
#else
void beep_()
#endif
{
   XBell(dpy,0);
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#ifdef CRAY
void GETS(line)
char *line;
#else
#ifdef TITAN
void GETS(line)
   char **line;
#else
void gets_(line)
   char *line;
#endif
#endif
{
   int i;
   xmainloop(LINE);
   for (i=0;i<strlen(input_line)&&i<80;i++)
#ifdef TITAN      
      line[0][i]=input_line[i];
#else
   line[i]=input_line[i];
#endif
   for (i=0;i<80;i++)
      input_line[i]=' ';
   lineEntered=0;
   input_line_idx=0;
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XDRAWTO(x,y)
#else
void xdrawto_(x,y)
#endif
   float *x,*y;
{
  if (!hold_Window)
    {
      XDrawLine(dpy,theWindow,colorPenGC,
		colorPenPosition.x,colorPenPosition.y, 
		(short) *x, (short) *y);
    }

  if (!hold_Pixmap)
    {
      XDrawLine(dpy,thePixmap,colorPenGC,
                colorPenPosition.x,colorPenPosition.y, 
                (short) *x, (short) *y);
    }

  colorPenPosition.x = (short) *x;
  colorPenPosition.y = (short) *y;
  XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XMOVE(x,y)
#else
void xmove_(x,y)
#endif
   float *x,*y;
{
   colorPenPosition.x = (short) *x;
   colorPenPosition.y = (short) *y;
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XCMAP(red,green,blue,num)
#else
void xcmap_(red,green,blue,num)
#endif
   float red[], green[], blue[];
   int *num;
{
   int i;
   cmap = XDefaultColormap(dpy,XDefaultScreen(dpy));     
   for (i=0;i<=*num;i++)
   {
      if (*num==1)
      {
         colorCells[i].red = (short) 65535;
         colorCells[i].green = (short) 65535;
         colorCells[i].blue = (short) 65535;
      }
      else
      {
         colorCells[i].red = (short) red[i];
         colorCells[i].green = (short) green[i];
         colorCells[i].blue = (short) blue[i];
      }
      colorCells[i].flags = DoRed|DoGreen|DoBlue;
      XAllocColor(dpy,cmap,&colorCells[i]);
   }
   XSync(dpy,False);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void hmt_XCMAP(red,green,blue,num)
#else
void hmt_xcmap_(red,green,blue,num)
#endif
   float red[], green[], blue[];
   int *num;
{
   int i;
   int r,g,b;
   int ct;

   cmap = XDefaultColormap(dpy,XDefaultScreen(dpy));     

   /* paul's 16 colors */
   for (i=0;i<=*num;i++)
   {
      if (*num==1)
      {
         colorCells[i].red = (short) 65535;
         colorCells[i].green = (short) 65535;
         colorCells[i].blue = (short) 65535;
      }
      else
      {
         colorCells[i].red = (short) red[i];
         colorCells[i].green = (short) green[i];
         colorCells[i].blue = (short) blue[i];
      }
      colorCells[i].flags = DoRed|DoGreen|DoBlue;
      printf("(%4u,%4u,%4u)\n",colorCells[i].red,
             colorCells[i].green,colorCells[i].blue);
      XAllocColor(dpy,cmap,&colorCells[i]);
   }

   /* extras */
   Random_Seed(1729);
   for (i=*num+1;i<NUM_COLOR_CELLS;i++, ct++)
   {
      r = Random_Integer(16383,65535);
      g = Random_Integer(00000,65537);
      b = Random_Integer(00000,65535);
      colorCells[i].red = (short) r;
      colorCells[i].green = (short) g;
      colorCells[i].blue = (short) b;
      colorCells[i].flags = DoRed|DoGreen|DoBlue;
      /*
        printf("(%4u,%4u,%4u)\n",colorCells[i].red,
        colorCells[i].green,colorCells[i].blue);
        */
      XAllocColor(dpy,cmap,&colorCells[i]);
   }
   XSync(dpy,False);
   /*
     printf("(sizeof(us)=%d\n",sizeof(unsigned short));
     */
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void HMT_XSETFILL(icolor)
#else
void hmt_xsetfill_(icolor)
#endif    
   int *icolor;
{
   /*    printf("icolor=%d, ng=%d\n",*icolor,*ng%NUM_COLOR_CELLS); */
   /*    gcv.foreground = colorCells[*icolor].pixel; */
   gcv.foreground = colorCells[*icolor%NUM_COLOR_CELLS].pixel;
   XChangeGC(dpy,fillGC,GCForeground,&gcv);
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XSETFILL(icolor)
#else
void xsetfill_(icolor)
#endif    
   int *icolor;
{
   gcv.foreground = colorCells[*icolor].pixel;
   XChangeGC(dpy,fillGC,GCForeground,&gcv);
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XSETPEN(icolor)
#else 
void xsetpen_(icolor)
#endif
   int *icolor;
{
   gcv.foreground = colorCells[*icolor].pixel;
   XChangeGC(dpy,colorPenGC,GCForeground,&gcv);
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XPOLY(xfill,yfill,nvertices,boundary)
#else
void xpoly_(xfill,yfill,nvertices,boundary)
#endif
   float *xfill, *yfill;
   int *nvertices, *boundary;
{
   int i;
   XPoint *vertices;
   vertices = (XPoint *) calloc(*nvertices,sizeof(XPoint));
   for (i=0;i<*nvertices;i++) 
   {
      vertices[i].x = (short) xfill[i];
      vertices[i].y = (short) yfill[i];
   }

   if (!hold_Window)
     {
       XFillPolygon(dpy,theWindow,
		    fillGC,vertices,*nvertices,Nonconvex,CoordModeOrigin);
     }

   if (!hold_Pixmap)
   {
      XFillPolygon(dpy,thePixmap,
                   fillGC,vertices,*nvertices,Nonconvex,CoordModeOrigin);
   }

   if (*boundary)  
   {
     if (!hold_Window)
       {
         XDrawLines(dpy,theWindow,
                    colorPenGC,vertices,*nvertices,CoordModeOrigin);
         XDrawLine(dpy,theWindow,colorPenGC,
                   vertices[*nvertices-1].x,vertices[*nvertices-1].y,
                   vertices[0].x,vertices[0].y);
       }

     if (!hold_Pixmap)
       {
         XDrawLines(dpy,thePixmap,
                    colorPenGC,vertices,*nvertices,CoordModeOrigin);
         XDrawLine(dpy,thePixmap,colorPenGC,
                   vertices[*nvertices-1].x,vertices[*nvertices-1].y,
                   vertices[0].x,vertices[0].y);
       }
   }

   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XCLEAR(i)
#else
void xclear_(i)
#endif
   int *i;
{ 
   XPoint corner[4];
   corner[0].x=0;
   corner[0].y=0;
   corner[1].x=windoww;
   corner[1].y=0;
   corner[2].x=corner[1].x;
   corner[2].y=windowh-text_screenh-line_spacing-line_spacing;
   corner[3].x=0;
   corner[3].y=corner[2].y;
   /*    i=0;   black background  */
   /*    i=1;   white background  */
   /*  i=1;                       */
#if (CRAY|TITAN)
   XSETFILL(i);
#else
   xsetfill_(i);
#endif
   if (!hold_Window)
     {
       XFillPolygon(dpy,theWindow,fillGC,corner,4,Convex,CoordModeOrigin);
     }

   if (!hold_Pixmap)
     {
       XFillPolygon(dpy,thePixmap,fillGC,corner,4,Convex,CoordModeOrigin);
     }
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void XCLEARA()
#else
void xcleara_()
#endif
{
   int i;
   XPoint corner[4];
   corner[0].x=0;
   corner[0].y=0;
   corner[1].x=windoww-(int)((double)windoww/1.3);
   corner[1].y=0;
   corner[2].x=corner[1].x;
   corner[2].y=windowh-text_screenh-line_spacing-line_spacing;
   corner[3].x=0;
   corner[3].y=corner[2].y;
   i=0;
#if (CRAY|TITAN)
   XSETFILL(&i);
#else
   xsetfill_(&i);
#endif
   if (!hold_Window)
     {
       XFillPolygon(dpy,theWindow,fillGC,corner,4,Convex,CoordModeOrigin);
     }

   if (!hold_Pixmap)
     {
       XFillPolygon(dpy,thePixmap,fillGC,corner,4,Convex,CoordModeOrigin);
     }

   XFlush(dpy);
}     



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#ifdef TITAN
void XPUTXT(x,y,fontsize,string,strlength)
float *x,*y,*fontsize;
char **string;
int *strlength;
#else
#ifdef CRAY
void XPUTXT(x,y,fontsize,string,strlength)
   float *x,*y,*fontsize;
   char *string;
   int *strlength;
#else
void xputxt_(x,y,fontsize,string,strlength)
   float *x,*y,*fontsize;
   char *string;
   int *strlength;
#endif
#endif
{
#ifdef TITAN
  if (!hold_Window)
    {
      XDrawString(dpy,theWindow,
                  calligPenGC,(int)*x,(int)*y,*string,*strlength);
    }
  
  if (!hold_Pixmap)
    {
      XDrawString(dpy,thePixmap,
                  calligPenGC,(int)*x,(int)*y,*string,*strlength);
    }
#else
  if (!hold_Window)
    {
      XDrawString(dpy,theWindow,
                  calligPenGC,(int)*x,(int)*y,string,*strlength);
    }

  if (!hold_Pixmap)
    {
      XDrawString(dpy,thePixmap,
                  calligPenGC,(int)*x,(int)*y,string,*strlength);
    }
#endif
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#ifdef TITAN
void PUTS(message,nchars)
char **message;
int *nchars;
#else
#ifdef CRAY
void PUTS(message,nchars)
   char *message;
   int *nchars;
#else
void puts_(message,nchars)
   char *message;
   int *nchars;
#endif
#endif
{
   int i;
#ifdef TITAN
   if (!hold_Window)
     {
       XDrawImageString(dpy,theWindow,fountainPenGC,
			fountainPenPosition.x,fountainPenPosition.y,
			*message,*nchars);
     }

   if (!hold_Pixmap)
     {
       XDrawImageString(dpy,thePixmap,fountainPenGC,
			fountainPenPosition.x,fountainPenPosition.y,
			*message,*nchars);
     }
   
   for (i=0;i < *nchars;i++)
     {Line[active_line][i]= message[0][i];}
#else
   if (!hold_Window)
     {
       XDrawImageString(dpy,theWindow,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,
                       message,*nchars);
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,
                       message,*nchars);
   }
   for (i=0;i < *nchars;i++)
      Line[active_line][i]= message[i];
#endif
   for (i = *nchars;i<80;i++)
      Line[active_line][i]= ' ';
   advanceLine();
   XFlush(dpy); 
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void CLOSEW()
#else
void closew_()
#endif
{
   Quit();
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
  main program for PRENEK and POSTNEK

EXTERNALLY VISIBLE!!!

********************************xdriver.c**********************************/
#if (CRAY|TITAN)
void MAINC (void)
#else
void mainc_(void)
#endif
{
   /* Pop up XWindows */
   SetUpEnv ();

   /* call the Fortran code */
#if (CRAY | TITAN)
   FPREP();
#else
   fprep_ ();      /* run Fortran Main code */
#endif

   Quit ();  
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

EXTERNALLY VISIBLE!!!

***********************************xdriver.c**********************************/
void 
set_ww_wh_ (int *ww, int *wh)
{
   if (*ww<=0||*wh<=0)
   {printf("window width and height must be positive integers!\n"); return;}

   if (*ww>MAX_WINDOWW||*wh>MAX_WINDOWH)
   {printf("window width and/or height > MAX_WINDOWW/H!\n"); return;}

   if (*ww==windoww&&*wh==windowh)
   {printf("window already that size!\n"); return;}

   windoww=*ww;
   windowh=*wh;

   if (dpy==NULL)
   {return;}

   close_env();
   SetUpEnv();
}



/**********************************xdriver.c***********************************
*******************************************************************************
PRIVATE FUNCTIONS!!!
*******************************************************************************
***********************************xdriver.c**********************************/
/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static 
void 
close_env (void)
{
   XFlush(dpy);   
   XSync(dpy,False);
   XFree(colorPenGC);
   XFree(fountainPenGC);
   XFree(fillGC);
   XFree(calligPenGC);
   XFreePixmap(dpy,dimBorder);
   XFreePixmap(dpy,IconPix);
   XFreePixmap(dpy,thePixmap);
   XDestroyWindow(dpy,theWindow);
   XFlush(dpy);   
   XSync(dpy,0);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
SetUpEnv (void) 
{
   char *argvec[1];
   int argcount=1,i,j;  
   XSizeHints szhint;
   long atol();
   XSetWindowAttributes attributes;
   unsigned long  xwindowmask;
   XSizeHints  xsizehints;
   XEvent event;
   Arg arg[5];
   char *font;


   /*  argvec[0]="Nekton";    pff 7/29/92  for photo ops.... */
   /*  argvec[0]="      ";                                   */
   argvec[0]="Nekton";
   for (i=0;i<80;i++)
      input_line[i]=' ';
   for (i=0;i<10;i++)
      for (j=0;j<90;j++)
         Line[i][j]= ' ';
   
   if ((dpy = XOpenDisplay(NULL)) == NULL) 
      error("Can't open display \n Did you setenv DISPLAY to your localHost \n and xhost + on the local display? \n");


   theScreen = DefaultScreen(dpy);
   theDepth  = DefaultDepth( dpy, theScreen);

   szhint.width = szhint.max_width = szhint.min_width = windoww;
   szhint.height = szhint.max_height = szhint.min_height = windowh;
   szhint.x = 10;
   szhint.y = 10;
   szhint.flags = PSize|PMinSize|PMaxSize|USPosition; 
   
   ForeColor = WhitePixel (dpy, theScreen );
   BackColor = BlackPixel (dpy, theScreen );
   dimBorder = XCreatePixmapFromBitmapData(dpy,DefaultRootWindow(dpy),
                                           check_bits,check_width,check_height,
                                           ForeColor,BackColor, theDepth);

   /*  The Following is appended by CSV */
   /*

        attributes.border_pixel
                =WhitePixel(dpy,theScreen);
        attributes.background_pixel
                = BlackPixel(dpy,theScreen);
        attributes.override_redirect = False;

        xwindowmask = CWBackPixel|CWBorderPixel|CWOverrideRedirect;

        theWindow = XCreateWindow( dpy,
                        RootWindow(dpy,theScreen),
                        vfx,vfy,szhint.width,szhint.height,
                        2,theDepth,
                        InputOutput,
                        CopyFromParent,
                        xwindowmask,
                        &attributes);

        XSetNormalHints(dpy,theWindow,&szhint);
    */

    theWindow = XCreateSimpleWindow (dpy,RootWindow(dpy,DefaultScreen(dpy)),
                                    vfx,vfy,szhint.width,szhint.height,2,
                                    ForeColor,BackColor);  
   if (!theWindow) error("Can't Open Window\n");

   thePixmap = XCreatePixmap  (dpy,theWindow,szhint.width,szhint.height, theDepth);

   if (!thePixmap) {error("Can't Open Pixmap\n");}    

   XSelectInput(dpy,theWindow,KeyPressMask|ButtonPressMask);


   IconPix = XCreateBitmapFromData (dpy, DefaultRootWindow(dpy),
                                    icon_bits,icon_width,icon_height);

   /*                         pff 7/29/92  for photo ops.... */
   /* XSetStandardProperties (dpy,theWindow,"      ",NULL,IconPix, */
   /* XSetStandardProperties (dpy,theWindow,"Nekton",NULL,IconPix, */
   XSetStandardProperties (dpy,theWindow,"Nekton",NULL,IconPix,
                           argvec,argcount,&szhint);
   XMapWindow(dpy,theWindow);
   XRaiseWindow(dpy,theWindow);
   XSync(dpy,False);

   cursorPen = XCreateFontCursor (dpy,XC_plus);
   XDefineCursor(dpy, theWindow, cursorPen);
   XFreeCursor(dpy,cursorPen);
   
   gcv.line_style = LineSolid;
   gcv.foreground = WhitePixel (dpy, DefaultScreen(dpy));
   gcv.background = BlackPixel (dpy, DefaultScreen(dpy));


   colorPenGC = XCreateGC(dpy,theWindow,
                          GCLineStyle|GCForeground|GCBackground,&gcv);

   fountainPenGC=XCreateGC(dpy,theWindow,GCForeground|GCBackground,&gcv);

   gcv.fill_style = FillSolid;
   fillGC = XCreateGC(dpy,theWindow,GCFillStyle|GCForeground,&gcv);

   /* standard fonts */
   /* Extra Helvetica font sizes big(--24) to small(--8) */
   /*
     "-Adobe-Helvetica-Medium-R-Normal--24-240-75-75-P-130-ISO8859-1";
     "-Adobe-Helvetica-Medium-R-Normal--18-180-75-75-P-98-ISO8859-1";
     "-Adobe-Helvetica-Medium-R-Normal--14-140-75-75-P-78-ISO8859-1";
     "-Adobe-Helvetica-Medium-R-Normal--12-120-75-75-P-67-ISO8859-1";
     "-Adobe-Helvetica-Medium-R-Normal--10-100-75-75-P-57-ISO8859-1";
     "-Adobe-Helvetica-Medium-R-Normal--8-80-75-75-P-47-ISO8859-1";

     "-*-*-*-*-*--*-*-*-*-*-*-*-*";
     */

   /* use 12pt for big window */
   if (windoww >= 700) 
   {
      font = "-Adobe-Helvetica-Medium-R-Normal--12-120-75-75-P-67-ISO8859-1";
      textFontInfo = XLoadQueryFont(dpy,font);
      if (!textFontInfo)
      {
         font = "-*-Helvetica-Medium-R-Normal--12-120-75-75-P-67-ISO8859-1";
         textFontInfo = XLoadQueryFont(dpy,font);
      }
   }
   /* use 8pt for small window */
   else 
   {
      font = "-Adobe-Helvetica-Medium-R-Normal--8-80-75-75-P-47-ISO8859-1";
      textFontInfo = 	XLoadQueryFont(dpy,font);
      if (!textFontInfo)
      {
         font = "-*-Helvetica-Medium-R-Normal--8-80-75-75-P-47-ISO8859-1";
         textFontInfo = XLoadQueryFont(dpy,font);
      }
   }

   /* no Helvetica-Medium-R-Normal 12||8 ... look for something */
   if (!textFontInfo) 
   {
      font = "-*-Helvetica-Medium-R-Normal--*-*-*-*-*-*-*-*";
      textFontInfo = 	XLoadQueryFont(dpy,font);
   }

   /* ok anything in Medium-R-Normal */
   if (!textFontInfo) 
   {
      font = "-*-*-Medium-R-Normal--*-*-*-*-*-*-*-*";
      textFontInfo = 	XLoadQueryFont(dpy,font);
   }

   /* really desperate now */
   if (!textFontInfo) 
   {
      font = "-*-*-*-*-*--*-*-*-*-*-*-*-*";
      textFontInfo = 	XLoadQueryFont(dpy,font);
   }

   if (!textFontInfo) 
   {error("XLoadQueryFont error in SetUpEnv\n");}

   printf("%s\n",font);

   gcv.font = textFontInfo->fid;

   calligPenGC= XCreateGC(dpy,theWindow,
                          GCFont|GCForeground|GCBackground,&gcv);
   resetXPRINT();

#ifdef TITAN
   while(1){
      XDrawImageString(dpy,theWindow,fountainPenGC,
		       fountainPenPosition.x,fountainPenPosition.y,
		       "Please Hit Any Key to Begin",27);

      if(XCheckMaskEvent(dpy,KeyPressMask|ButtonPressMask,&event)==True)
         break;
   }
   XDrawImageString(dpy,theWindow,fountainPenGC,
                    fountainPenPosition.x,fountainPenPosition.y,
                    "                           ",27);
#endif    

   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static 
void
Random_Seed(unsigned seed)
{
void srand(unsigned seed);

   srand(seed);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static 
int 
Random_Integer(int lb, int ub)
{
   int rand(void);
   double d;
   int k;


   d = rand()/((double) RAND_MAX + 1);
   k = (int) (d*(ub-lb+1));

   return(lb+k);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
advanceLine()
{
   int i;


   if (scroll)	
     {
       top_line = (top_line+1)%num_lines;
       active_line = (active_line+1)%num_lines;

       for (i=0;i<num_lines;i++)  
	 if (!hold_Window)
	   {
	     XDrawImageString(dpy,theWindow,fountainPenGC,
			      left_margin,windowh-text_screenh+i*line_spacing,
			      Line[(top_line+i)%num_lines],80);
	   }

       if (!hold_Pixmap)
	 {
	   XDrawImageString(dpy,thePixmap,fountainPenGC,
			    left_margin,windowh-text_screenh+i*line_spacing,
			    Line[(top_line+i)%num_lines],80);
	 }
       fountainPenPosition.x=left_margin;
       if (!hold_Window)
	 {
	   XDrawImageString(dpy,theWindow,fountainPenGC,
			    fountainPenPosition.x,fountainPenPosition.y,
			    "_                                                                                                                                                                                                                                                              ",150);
	 }

       if (!hold_Pixmap)
	 {
	   XDrawImageString(dpy,thePixmap,fountainPenGC,
			    fountainPenPosition.x,fountainPenPosition.y,
			    "_                                                                                                                                                                                                                                                              ",150);
	 }
       XFlush(dpy);
       return;
     }

   active_line++;
   if (active_line == num_lines) {
      active_line--;
      scroll=1;
      advanceLine();
      return;
   }      
   
   fountainPenPosition.x=left_margin;
   if (fountainPenPosition.y < windowh)
      fountainPenPosition.y += line_spacing;
   if (fountainPenPosition.y >= windowh)
      fountainPenPosition.y =  windowh - text_screenh;
   if (!hold_Window)
   {
     XDrawImageString(dpy,theWindow,fountainPenGC,
		      fountainPenPosition.x,fountainPenPosition.y,
                       "_                                                                                                                                                                                                                                                              ",150);
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,
                       "_                                                                                                                                                                                                                                                              ",150);
   }
   XFlush(dpy); 
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
showChar(char c)
{
   if (!hold_Window)
   {
      XDrawImageString(dpy,theWindow,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,&c,1);
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,&c,1);
   }
   fountainPenPosition.x += char_spacing;
   if (!hold_Window)
   {
      XDrawImageString(dpy,theWindow,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"_",1);
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"_",1);
   }
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
backSpace()
{
   
   if (input_line_idx == 0) return;
   if (!hold_Window)
   {
      XDrawImageString(dpy,theWindow,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"  ",2);
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"  ",2);
   }
   fountainPenPosition.x -= char_spacing;
   if (fountainPenPosition.x < 10) fountainPenPosition.x = 10;
   if (!hold_Window)
   {
   XDrawImageString(dpy,theWindow,fountainPenGC,
                    fountainPenPosition.x,fountainPenPosition.y,"_",1);    
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"_",1);    
   }
   XFlush(dpy);
   input_line[--input_line_idx]=' ';
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
TypeChar(char c)
{
   int i;

   if (c=='\003') Quit();
   switch (c) 
   {
    case '\r':
    case '\n':
      if (!hold_Window)
      {
         XDrawImageString(dpy,theWindow,fountainPenGC,
                          fountainPenPosition.x,fountainPenPosition.y,"  ",2);
      }
      if (!hold_Pixmap)
      {
         XDrawImageString(dpy,thePixmap,fountainPenGC,
                          fountainPenPosition.x,fountainPenPosition.y,"  ",2);
      }
      for (i=0;i<80;i++)
         Line[active_line][i]=input_line[i];
      advanceLine();
      lineEntered=1;
      return;
      
    case '\177':
    case '\010':
      backSpace();
      return;
   }
   
   if (c >= 32 && c <=127)  {  
      if (input_line_idx == 80) 
         XBell(dpy,0);
      else {
         showChar(c);
         input_line[input_line_idx++]=c;
      }
   }
   else XBell(dpy,0);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void
xmainloop (int code)
{
   Window wind;
   static XEvent event;

   XSync(dpy,True);
   switch (code)
   {
    case MOUSE:
   {
      XButtonPressedEvent *but_event; 
      for (;;) {
         XMaskEvent(dpy,ButtonPressMask|KeyPressMask,&event);
         if (event.type==ButtonPress)
            break;
      }
      but_event = (XButtonPressedEvent *) &event;
      wind = but_event->window;
      
      if (wind==theWindow)
      {
         if ((but_event->button & 0xff) == Button1)
            mouseRecord.buttonNumber=1;
         else if ((but_event->button & 0xff) == Button2)
            mouseRecord.buttonNumber=2;
         else if ((but_event->button & 0xff) == Button3)
            mouseRecord.buttonNumber=3;
         mouseRecord.xclicked=but_event->x;
         mouseRecord.yclicked=but_event->y;
      }
   }
   break;
   
 case LINE:
{
   XKeyPressedEvent *key_event;
   char *st, keybuf[10];
   int i;
 keyIn:
   for (;;) {
      XMaskEvent(dpy,ButtonPressMask|KeyPressMask,&event);
      if (event.type==KeyPress)
         break;
   }
   key_event = (XKeyPressedEvent *) &event;
   wind=key_event->window;
   if (wind==theWindow)
   {
      i=XLookupString(key_event,keybuf,sizeof(keybuf),NULL,NULL);
      for (st=keybuf;i>0;i--)
         TypeChar(*st++);
      if (!lineEntered) goto keyIn;
   } 
} 
break;
 default:
   printf("unexpected code in xmainloop %d\n",code);
exit(-1);
break;
}
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
int 
Quit()
{
   XDestroyWindow(dpy,theWindow);
   XFreePixmap(dpy,thePixmap);
   XSync(dpy,0);
   exit(0);
   return(-1);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
error (char *mesg)
{
   fprintf(stderr, mesg);
   exit(0);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
#ifdef CRAY
static
void SYSTEM( string )
char *string;
{
   system( string );
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
void 
RENAME( from, to )
char *from, *to;
{
   rename( from, to );
}
#endif



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  

UNUSED!!!

***********************************xdriver.c**********************************/
static
void
resetXPRINT()  
{
   fountainPenPosition.x = left_margin;
   fountainPenPosition.y = windowh - text_screenh;
   if (!hold_Window)
   {
      XDrawImageString(dpy,theWindow,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"_",1);    
   }
   if (!hold_Pixmap)
   {
      XDrawImageString(dpy,thePixmap,fountainPenGC,
                       fountainPenPosition.x,fountainPenPosition.y,"_",1);    
   }
   XFlush(dpy);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
int
lvec_linear_search(register unsigned long item, register unsigned long *list,
                   register int n) 
{
   register int tmp = n-1;

   while (n--)  {if (*list++ == item) {return(tmp-n);}}
   return(-1);
}



/**********************************xdriver.c***********************************
  Function: 

  Input : 
  Output: 
  Return: 
  Description:  
***********************************xdriver.c**********************************/
static
int
vec_binary_search(register unsigned long item, register unsigned long *list,
                  register int rh) 
{
   register int mid, lh=0;

   rh--;
   while (lh<=rh)
   {
      mid = (lh+rh)>>1;
      if (*(list+mid) == item) 
      {return(mid);}
      if (*(list+mid) > item)  
      {rh = mid-1;}
      else 
      {lh = mid+1;}
   }
   return(-1);
}



/**********************************xdriver.c***********************************
*******************************************************************************
UNUSED FUNCTIONS!!!
*******************************************************************************
***********************************xdriver.c**********************************/
