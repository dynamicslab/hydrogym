FLAGS =  $(FFLAGS_IN)
CFLAGS =  $(CFLAGS_IN)
LFLAGS = $(LFLAGS_IN)
prefix = $(bin_nek_tools)

OBJS = int_tp.o speclib.o mxm44f2.o byte.o int_tp_gbox.o

all: int_tp


int_tp:	$(OBJS)
	$(F77) $(FLAGS) -o $(prefix)/int_tp $(OBJS) $(LFLAGS)

clean:
	'rm' *.o

int_tp.o  	: int_tp.f	        ; $(F77) -c $(FLAGS)  int_tp.f 
byte.o          : byte.c                ; $(CC)  -c $(CFLAGS) byte.c
speclib.o  	: ../../core/speclib.f	; $(F77) -c $(FLAGS)  ../../core/speclib.f 
mxm44f2.o  	: mxm44f2.f	        ; $(F77) -c $(FLAGS)  mxm44f2.f 
int_tp_gbox.o   : int_tp_gbox.f	        ; $(F77) -c $(FLAGS)  int_tp_gbox.f 
