#include <math.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"

#define EPS (128*DBL_EPSILON)


/* minimizes cancellation error (but not round-off ...) */
static double sum_3(const double a, const double b, const double c)
{
  if     ( (a>=0 && b>=0) || (a<=0 && b<=0) ) return (a+b)+c;
  else if( (a>=0 && c>=0) || (a<=0 && c<=0) ) return (a+c)+b;
  else return a+(b+c);
}

/* solve     c
          - --- + b + a x == 0        with sign(x) = sign
             x
*/
static double rat_root(const double a, const double b, const double c,
                       const double sign)
{
  double bh = (fabs(b) + sqrt(b*b + 4*a*c))/2;
  return sign * (b*sign <= 0 ? bh/a : c/bh);
}

/*
  find d[ri] <= lambda <= d[ri+1]
  such that 0 = lambda - v[0] + \sum_i^n v[i]^2 / (d[i] - lambda)
*/
static double sec_root(double *y, const double *d, const double *v,
                       const int ri, const int n)
{
  double dl = d[ri], dr = d[ri+1], L = dr-dl;
  double x0l = L/2, x0r = -L/2;
  int i;
  double al, ar, bln, blp, brn, brp, cl, cr;
  double fn, fp, lambda0, lambda;
  double tol = L;
  if(fabs(dl)>tol) tol=fabs(dl);
  if(fabs(dr)>tol) tol=fabs(dr);
  tol *= EPS;
  for(;;) {
    if(fabs(x0l)==0 || x0l < 0) { *y=0; return dl; }
    if(fabs(x0r)==0 || x0r > 0) { *y=0; return dr; }
    lambda0 = fabs(x0l) < fabs(x0r) ? dl + x0l : dr + x0r;
    al = ar = cl = cr = bln = blp = brn = brp = 0;
    fn = fp = 0;
    for(i=1;i<=ri;++i) {
      double den = (d[i]-dl)-x0l;
      double fac = v[i]/den;
      double num = sum_3(d[i],-dr,-2*x0r);
      fn += v[i]*fac;
      fac *= fac;
      ar += fac;
      if(num > 0) brp += fac*num; else brn += fac*num;
      bln += fac*(d[i]-dl);
      cl  += fac*x0l*x0l;
    }
    for(i=ri+1;i<=n;++i) {
      double den = (d[i]-dr)-x0r;
      double fac = v[i]/den;
      double num = sum_3(d[i],-dl,-2*x0l);
      fp += v[i]*fac;
      fac *= fac;
      al += fac;
      if(num > 0) blp += fac*num; else bln += fac*num;
      brp += fac*(d[i]-dr);
      cr  += fac*x0r*x0r;
    }
    if(lambda0>0) fp+=lambda0; else fn+=lambda0;
    if(v[0]<0) fp-=v[0],blp-=v[0],brp-=v[0];
          else fn-=v[0],bln-=v[0],brn-=v[0];
    if(fp+fn > 0) { /* go left */
      x0l = rat_root(1+al,sum_3(dl,blp,bln),cl,1);
      lambda = dl + x0l;
      x0r = x0l - L;
    } else { /* go right */
      x0r = rat_root(1+ar,sum_3(dr,brp,brn),cr,-1);
      lambda = dr + x0r;
      x0l = x0r + L;
    }
    if( fabs(lambda-lambda0) < tol ) {
      double ty=0, fac;
      for(i=1;i<=ri;++i) fac = v[i]/((d[i]-dl)-x0l), ty += fac*fac;
      for(i=ri+1;i<=n;++i) fac = v[i]/((d[i]-dr)-x0r), ty += fac*fac;
      *y = 1/sqrt(1+ty);
      return lambda;
    }
  }
}

/*
  find the eigenvalues of
  
  d[1]           v[1]
       d[2]      v[2]
            d[n] v[n]
  v[1] v[2] v[n] v[0]
  
  sets d[0], d[n+1] to Gershgorin bounds
  
  also gives (n+1)th component of each orthonormal eigenvector in y
*/
static void tdeig(double *lambda, double *y, double *d, const double *v,
                  const int n)
{
  int i;
  double v1norm = 0, min=v[0], max=v[0];
  for(i=1;i<=n;++i) {
    double vi = fabs(v[i]), a=d[i]-vi, b=d[i]+vi;
    v1norm += vi;
    if(a<min) min=a;
    if(b>max) max=b;
  }
  d[0]   = v[0] - v1norm < min ? v[0] - v1norm : min;
  d[n+1] = v[0] + v1norm > max ? v[0] + v1norm : max;
  for(i=0;i<=n;++i) lambda[i] = sec_root(&y[i],d,v,i,n);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int n;
  if(nrhs!=3) { mexWarnMsgTxt("Three inputs required."); return; }
  if(nlhs!=2) { mexWarnMsgTxt("Two outputs required."); return; }
  if(mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
    mexWarnMsgTxt("First input not a full, double, real array."); return;
  }
  if(mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
    mexWarnMsgTxt("Second input not a full, double, real array."); return;
  }
  if(mxIsSparse(prhs[2]) || !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])
   ||mxGetM(prhs[2])!=1  || mxGetN(prhs[2])!=1) {
    mexWarnMsgTxt("Third input not a double real scalar."); return;
  }
  n = mxGetPr(prhs[2])[0]+.5;
  if(n<1) { mexWarnMsgTxt("n < 1"); return; }
  if(mxGetM(prhs[0]) < n+2) { mexWarnMsgTxt("length(d) < n+2"); return; }
  if(mxGetM(prhs[1]) < n+1) { mexWarnMsgTxt("length(v) < n+1"); return; }
  plhs[0] = mxCreateDoubleMatrix(n+1,1,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(n+1,1,mxREAL);
  tdeig(mxGetPr(plhs[0]),mxGetPr(plhs[1]),
        mxGetPr(prhs[0]),mxGetPr(prhs[1]),n);
}
