/*
   This subroutine reverts data from base-64 notation to floating point.
*/
float revert_(t) char *t; {

char *alph64="1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-";
static int revmap[256], init=1;
/*
   Make exp[] double so C won't have to convert for each multiply.  This is
   the real exponent minus 5 (i.e., what the mantissa will be multiplied by).
*/
static double exp[64]={
                 1e-36,1e-35,1e-34,1e-33,1e-32,1e-31,1e-30,1e-29,1e-28,1e-27,
                 1e-26,1e-25,1e-24,1e-23,1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,
                 1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-09,1e-08,1e-07,
                 1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1e+00,1e+01,1e+02,1e+03,
                 1e+04,1e+05,1e+06,1e+07,1e+08,1e+09,1e+10,1e+11,1e+12,1e+13,
                 1e+14,1e+15,1e+16,1e+17,1e+18,1e+19,1e+20,1e+21,1e+22,1e+23,
                 1e+24,1e+25,1e+26,1e+27};

register int mantis, i;
float retval;

/*
   Initialize here to maintain character-set independence.
*/
if (init) {
   for (i=0; i<64; i++)  revmap[alph64[i]] = i;
   init = 0;
 }

mantis = ((((revmap[t[0]] << 6) | revmap[t[1]]) << 6) | revmap[t[2]]) - 131072;
retval = (double)mantis * exp[revmap[t[3]]];

return retval;
}
