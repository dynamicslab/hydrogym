FLAGS =  $(FFLAGS_IN)
CFLAGS = $(CFLAGS_IN)
LFLAGS = $(LFLAGS_IN)
prefix = $(bin_nek_tools)

NOBJ1 = reatore2.o byte.o strings.o
NOBJ2 = re2torea.o byte.o strings.o

all: reatore2 re2torea

reatore2:	$(NOBJ1)
	$(F77) -o $(prefix)/reatore2 $(NOBJ1) 

re2torea:	$(NOBJ2)
	$(F77) -o $(prefix)/re2torea $(NOBJ2) 


clean:
	'rm' *.o

reatore2.o  	: reatore2.f		 ; $(F77) -c $(FLAGS)  reatore2.f 
re2torea.o  	: re2torea.f	 	 ; $(F77) -c $(FLAGS)  re2torea.f 
strings.o  	: strings.f		 ; $(F77) -c $(FLAGS)  strings.f 
byte.o		: byte.c     		 ; $(CC)  -c $(CFLAGS) byte.c
