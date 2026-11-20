FLAGS =  $(FFLAGS_IN)
CFLAGS = $(CFLAGS_IN)
LFLAGS = $(LFLAGS_IN) 
prefix = $(bin_nek_tools)

OBJS = n2to3.o byte.o

all: n2to3


n2to3:	$(OBJS)
	$(F77) $(FLAGS) -o $(prefix)/n2to3 $(OBJS) $(LFLAGS) 

clean:
	'rm' *.o

n2to3.o          : n2to3.f		 ;  $(F77) -c $(FLAGS) n2to3.f 
byte.o           : byte.c      		 ;  $(CC)  -c $(CFLAGS) byte.c
