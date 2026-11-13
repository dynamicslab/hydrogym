CC=mpicc -std=c99 --pedantic
CFLAGS+=-DMPI
CFLAGS+=-DPREFIX=jl_
CFLAGS+=-DNO_NEK_EXITT
CFLAGS+=-DGLOBAL_LONG
LDFLAGS+=-lm

#CFLAGS+=-DPRINT_MALLOCS=1

CFLAGS+=-DUSE_NAIVE_BLAS
#CFLAGS+=-DUSE_CBLAS
#LDFLAGS+=-lcblas

#CFLAGS+=-DAMG_DUMP
CFLAGS+=-DGS_TIMING -DGS_BARRIER

#CFLAGS+=-O0 -g
CFLAGS+=-O3 -march=native

CFLAGS+=-W -Wall -Wno-unused-function -Wno-unused-parameter
#CFLAGS+=-Minform=warn

CCCMD=$(CC) $(G) $(CFLAGS)
LINKCMD=$(CC) $(G) $(LDFLAGS)
#RLINKCMD = $(LD) -r
.PHONY: cmds deps tests clean objects odepinfo

TESTS=sort_test sort_test2 sarray_sort_test spchol_test \
      comm_test crystal_test sarray_transfer_test \
      gs_test gs_test_old gs_unique_test \
      xxt_test xxt_test2 crs_test \
      poly_test poly_test2 lob_bnd_test obbox_test \
      findpts_el_2_test findpts_el_2_test2 \
      findpts_el_3_test findpts_el_3_test2 \
      findpts_local_test findpts_test

CRS=$(AMG)

tests: $(TESTS) ;
clean: ; @$(RM) $(TESTS) *.o *.s

cmds: ; @echo CC = $(CCCMD); echo LINK = $(LINKCMD);

deps: ; ./cdep.py *.c > makefile.cdep;

odepinfo: deps objects; @./odep_info.py *.o

-include makefile.cdep

%.o: %.c ; @echo CC $<; $(CCCMD) -c $< -o $@
%.s: %.c ; @echo CC -S $<; $(CCCMD) -S $< -o $@
objects: $(OBJECTS) ;

poly_imp.h: gen_poly_imp.c
	$(RM) poly_imp.h;
	$(CC) -lgmp -lm gen_poly_imp.c -o gen_poly_imp;
	./gen_poly_imp > poly_imp.h;
	$(RM) gen_poly_imp

GS_OBJECTS=gs.o sort.o sarray_transfer.o sarray_sort.o \
           gs_local.o fail.o crystal.o comm.o tensor.o

XXT=sparse_cholesky.o xxt.o
AMG=amg.o

sort_test: sort.o fail.o comm.o tensor.o gs_local.o sort_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
sort_test2: sort.o fail.o comm.o tensor.o gs_local.o sort_test2.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
sarray_sort_test: sort.o fail.o comm.o tensor.o gs_local.o sarray_sort.o sarray_sort_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
spchol_test: sparse_cholesky.o sort.o fail.o comm.o tensor.o gs_local.o spchol_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
comm_test: fail.o comm.o tensor.o gs_local.o comm_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
crystal_test: fail.o crystal.o comm.o tensor.o gs_local.o crystal_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
sarray_transfer_test: sarray_transfer.o sarray_sort.o sort.o fail.o crystal.o comm.o tensor.o gs_local.o sarray_transfer_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@

gs_test: gs_test.o $(GS_OBJECTS);		@echo LINK $@; $(LINKCMD) $^ -o $@
gs_test_old: gs_test_old.o $(GS_OBJECTS);	@echo LINK $@; $(LINKCMD) $^ -o $@
gs_unique_test: gs_unique_test.o $(GS_OBJECTS);	@echo LINK $@; $(LINKCMD) $^ -o $@
xxt_test: xxt_test.o $(CRS) $(GS_OBJECTS);	@echo LINK $@; $(LINKCMD) $^ -o $@
xxt_test2: xxt_test2.o $(CRS) $(GS_OBJECTS);	@echo LINK $@; $(LINKCMD) $^ -o $@
crs_test: crs_test.o $(CRS) $(GS_OBJECTS);	@echo LINK $@; $(LINKCMD) $^ -o $@

poly_test2: poly.o fail.o comm.o tensor.o gs_local.o poly_test2.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
poly_test: poly.o fail.o comm.o tensor.o gs_local.o poly_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
lob_bnd_test: tensor.o poly.o lob_bnd.o fail.o comm.o gs_local.o lob_bnd_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
obbox_test: rand_elt_test.o poly.o obbox.o tensor.o lob_bnd.o fail.o comm.o gs_local.o obbox_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_el_2_test2: tensor.o rand_elt_test.o lob_bnd.o fail.o comm.o gs_local.o poly.o findpts_el_2.o findpts_el_2_test2.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_el_2_test: poly.o fail.o comm.o tensor.o gs_local.o findpts_el_2.o findpts_el_2_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_el_3_test2: tensor.o rand_elt_test.o lob_bnd.o fail.o comm.o gs_local.o poly.o findpts_el_3.o findpts_el_3_test2.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_el_3_test: poly.o fail.o comm.o tensor.o gs_local.o findpts_el_3.o findpts_el_3_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_local_test: rand_elt_test.o lob_bnd.o fail.o comm.o tensor.o gs_local.o poly.o findpts_local.o sort.o sarray_sort.o obbox.o findpts_el_3.o findpts_el_2.o findpts_local_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@
findpts_test: sarray_transfer.o sort.o rand_elt_test.o lob_bnd.o poly.o findpts.o sarray_sort.o findpts_local.o obbox.o tensor.o findpts_el_3.o findpts_el_2.o fail.o crystal.o comm.o gs_local.o findpts_test.o ; @echo LINK $@; $(LINKCMD) $^ -o $@

