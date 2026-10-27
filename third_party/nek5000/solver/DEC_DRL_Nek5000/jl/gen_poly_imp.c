#include <math.h>
#include <stdio.h>
#include <gmp.h>

#define PREC_BITS 256
#define DIGITS 50

#define GLL_LAG_FIX_MAX 24

#if 1
#  define STATIC "static "
#else
#  define STATIC ""
#endif


#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

#define DECLARE_1VAR(a)        static int init=0; static mpf_t a; \
                               if(!init) init=1, mpf_init(a)
#define DECLARE_2VARS(a,b)     static int init=0; static mpf_t a,b; \
                               if(!init) init=1, mpf_init(a), mpf_init(b)
#define DECLARE_3VARS(a,b,c)   static int init=0; static mpf_t a,b,c; \
                               if(!init) init=1, mpf_init(a), mpf_init(b), \
                                                 mpf_init(c)
#define DECLARE_4VARS(a,b,c,d) static int init=0; static mpf_t a,b,c,d; \
                               if(!init) init=1, mpf_init(a), mpf_init(b), \
                                                 mpf_init(c), mpf_init(d)
                                                 
static int is_small(const mpf_t x, const mpf_t y) {
  DECLARE_2VARS(xa,ya);
  mpf_abs(xa,x);
  mpf_abs(ya,y);
  mpf_div_2exp(ya,ya,PREC_BITS-mp_bits_per_limb);
  return mpf_cmp(xa,ya) < 0;
}

typedef void fun_3term(mpf_t Pn, int n, const mpf_t x);

#define DECLARE_THREE_TERM(name, i0_init, init_Ps, a_ip1,a_i,a_im1) \
static void name(mpf_t Pn, int n, const mpf_t x) \
{ \
  int i, i0_init; \
  DECLARE_4VARS(a,b,P_im1,P_i); \
  init_Ps; \
  for(i=i0+1; i<n; ++i) { \
    mpf_mul(a, x,P_i); \
    mpf_mul_ui(a, a,a_i); \
    mpf_mul_ui(b, P_im1,a_im1); \
    mpf_sub(a, a,b); \
    mpf_swap(P_im1, P_i); \
    mpf_div_ui(P_i, a,a_ip1); \
  } \
  mpf_set(Pn, n>i0?P_i:P_im1); \
}

DECLARE_THREE_TERM(legendre,    i0=0,(mpf_set_ui(P_im1,1),mpf_set   (P_i,x)),
                   i+1, 2*i+1, i  )
DECLARE_THREE_TERM(legendre_d1, i0=0,(mpf_set_ui(P_im1,0),mpf_set_ui(P_i,1)),
                   i  , 2*i+1, i+1)
DECLARE_THREE_TERM(legendre_d2, i0=1,(mpf_set_ui(P_im1,0),mpf_set_ui(P_i,3)),
                   i-1, 2*i+1, i+2)

static void newton(mpf_t x, double seed,
                   fun_3term *fun, fun_3term *der, int n)
{
  DECLARE_3VARS(ox,f,df);
  mpf_set_d(x, seed);
  do {
    mpf_set(ox, x);
    fun(f, n,x), der(df, n,x), mpf_div(f, f,df), mpf_sub(x, x,f);
  } while(!is_small(f,x));
  fun( f, n,x), der(df, n,x), mpf_div(f, f,df), mpf_sub(x, x,f);
}

static void gauss_node(mpf_t z, int n, int i) {
  if( (n&1) && i==n/2 ) mpf_set_ui(z,0);
  else newton(z, cos( (2*n-2*i-1)*(PI/2)/n ), legendre,legendre_d1,n);
}

static void lobatto_node(mpf_t z, int n, int i) {
  if( (n&1) && i==n/2 ) mpf_set_ui(z,0);
  else if(i==0)   mpf_set_d(z,-(double)1);
  else if(i==n-1) mpf_set_ui(z,1);
  else newton(z, cos( (n-1-i)*PI/(n-1) ), legendre_d1,legendre_d2,n-1);
}

#define PRINT_LIST(i, i0,nline,n, printi,sep,sepline) \
  do { \
    int i; \
    for(i=i0;i<n;++i) { \
      printi; \
      printf("%s",i==n-1?"":((i-i0)%nline==nline-1?sepline:sep)); \
    } \
  } while(0)

static void print_gll_lag_fix(int n)
{
  int i;
  DECLARE_1VAR(z);
  if(n>3) {
    printf("static const double gllz_%02d[%2d] = {\n  ",n,n/2-1);
    for(i=1;i<=n/2-1;++i) {
      lobatto_node(z, n,n-1-i);
      if(i!=1) printf(",\n  ");
      gmp_printf("%.*Fg",DIGITS,z);
    }
    puts("\n};\n");
  }
  printf(STATIC "void gll_lag_%02d(double *restrict p, double *restrict w,\n"
           "                       unsigned n, int d, double xh)\n{\n",n);
  printf("  const double x = xh*2;\n");
  #define PRINT_D(i) do { \
    printf("d%02d=x",i); \
    if(2*i+1==n)    printf("              "); \
    else if(i==0)   printf("+2            "); \
    else if(i==n-1) printf("-2            "); \
    else if(i<n/2)  printf("+2*gllz_%02d[%2d]",n,i-1); \
    else            printf("-2*gllz_%02d[%2d]",n,n-2-i); \
  } while(0)
  printf("%s",                            "  const double ");
  PRINT_LIST(i, 0,3,n, PRINT_D(i),",",",\n               ");
  #undef PRINT_D
  #define PRINT_U0(i) (i==0  ?printf("    1"):printf("u0_%02d",i))
  #define PRINT_V0(i) (i==n-1?printf("    1"):printf("v0_%02d",i))
  #define PRINT_U1(i) (i<=1  ?printf("    %d",i      ):printf("u1_%02d",i))
  #define PRINT_V1(i) (i>=n-2?printf("    %d",n-1-(i)):printf("v1_%02d",i))
  #define PRINT_U2(i) (i<=1  ?printf("    0"): \
                      (i==2  ?printf("    2"):printf("u2_%02d",i)))
  #define PRINT_V2(i) (i>=n-2?printf("    0"): \
                      (i==n-3?printf("    2"):printf("v2_%02d",i)))
  printf("%s",";\n  const double ");
  PRINT_LIST(i, 1,3,n,
    (PRINT_U0(i),putchar('='),PRINT_U0(i-1),printf("*d%02d",i-1)),
    ",",",\n               ");
  printf("%s",";\n  const double ");
  PRINT_LIST(i, 1,3,n,
    (PRINT_V0(n-1-i),putchar('='),printf("d%02d*",n-i),PRINT_V0(n-i)),
    ",",",\n               ");
  printf("%s",";\n  ");
  PRINT_LIST(i, 0,3,n, 
    (printf("p[%2d]=w[%2d]*",i,i),PRINT_U0(i),putchar('*'),
     PRINT_V0(i)),"; ",";\n  ");
  puts(";\n  if(d>0) {");
  if(n>2) {
    printf("%s","    const double ");
    PRINT_LIST(i, 2,2,n,
      (PRINT_U1(i),putchar('='),PRINT_U1(i-1),printf("*d%02d",i-1),
       putchar('+'),PRINT_U0(i-1)),
      ",",",\n                 ");
    printf("%s",";\n    const double ");
    PRINT_LIST(i, 2,2,n,
      (PRINT_V1(n-1-i),putchar('='),printf("d%02d*",n-i),PRINT_V1(n-i),
       putchar('+'),PRINT_V0(n-i)),
      ",",",\n                 ");
    puts(";");
  }
  for(i=0;i<n;++i) {
    printf("    p[%d+%2d]=2*w[%2d]*(",n,i,i);
    if(i==0)        printf("                  "),PRINT_V1(0);
    else if(i==n-1) PRINT_U1(i),printf("                  ");
    else PRINT_U1(i),putchar('*'),PRINT_V0(i),putchar('+'),
         PRINT_U0(i),putchar('*'),PRINT_V1(i);
    puts(");");
  }
  puts("    if(d>1) {");
  if(n>3) {
    printf("%s","      const double ");
    PRINT_LIST(i, 3,2,n,
      (PRINT_U2(i),putchar('='),PRINT_U2(i-1),printf("*d%02d",i-1),
       printf("+2*"),PRINT_U1(i-1)),
      ",",",\n                   ");
    printf("%s",";\n      const double ");
    PRINT_LIST(i, 3,2,n,
      (PRINT_V2(n-1-i),putchar('='),printf("d%02d*",n-i),PRINT_V2(n-i),
       printf("+2*"),PRINT_V1(n-i)),
      ",",",\n                   ");
    puts(";");
  }  
  if(n<3) for(i=0;i<n;++i) printf("      p[2*%d+%2d]=0;\n",n,i);
  else for(i=0;i<n;++i) {
      printf("      p[2*%d+%2d]=4*w[%2d]*(",n,i,i);
      if(i>1)
        PRINT_U2(i),putchar('*'),PRINT_V0(i);
      else printf("           ");
      if(i>0 && i<n-1)
        printf("+2*"),PRINT_U1(i),putchar('*'),PRINT_V1(i);
      else printf("              ");
      if(i<n-2)
        putchar('+'),PRINT_U0(i),putchar('*'),PRINT_V2(i);
      else printf("            ");
      puts(");");
  }
  #undef PRINT_U0
  #undef PRINT_V0
  #undef PRINT_U1
  #undef PRINT_V1
  #undef PRINT_U2
  #undef PRINT_V2
  puts("    }\n  }\n}");
}


int main()
{
  int n;
  mpf_set_default_prec(PREC_BITS);
  puts("/* generated by gen_poly_imp.c */\n");
  printf("#define GLL_LAG_FIX_MAX %d\n\n",GLL_LAG_FIX_MAX);
  /*puts("typedef void gll_lag_fun(double *p, int d, int n, double x);\n");*/
  for(n=2;n<=GLL_LAG_FIX_MAX;++n)
      print_gll_lag_fix(n), puts("");
  printf(STATIC "const double *const gllz_table[%d] = {\n  ",
    GLL_LAG_FIX_MAX-3);
  PRINT_LIST(i, 4,8,(GLL_LAG_FIX_MAX+1),
    printf("gllz_%02d",i), ", ",",\n  ");
  puts("\n};");
  puts("");
  printf(STATIC "lagrange_fun *const gll_lag_table[%d] = {\n  ",
    GLL_LAG_FIX_MAX-1);
  PRINT_LIST(i, 2,6,(GLL_LAG_FIX_MAX+1),
    printf("&gll_lag_%02d",i), ", ",",\n  ");
  puts("\n};");
  puts("");
  return 0;
}
