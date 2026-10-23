#ifndef RAND_ELT_TEST_H
#define RAND_ELT_TEST_H

void rand_elt_2(double *x, double *y,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns);

void rand_elt_3(double *x, double *y, double *z,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns,
                const double *zt, unsigned nt);

void bubble_elt(double *x, double *y, double *z,
                const double *zr, unsigned nr,
                const double *zs, unsigned ns,
                const double *zt, unsigned nt, int type);

#endif
