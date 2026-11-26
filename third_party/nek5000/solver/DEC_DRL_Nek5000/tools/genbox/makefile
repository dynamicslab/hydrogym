FLAGS =  $(FFLAGS_IN)
CFLAGS = $(CFLAGS_IN)
LFLAGS = 
prefix = $(bin_nek_tools)

OBJS = genbox.o byte.o 

all: genbox


genbox:	$(OBJS)
	$(F77) $(FLAGS) -o $(prefix)/genbox $(OBJS) $(LFLAGS) 

clean:
	'rm' *.o

genbox.o       : genbox.f		;  $(F77) -c $(FLAGS) genbox.f
byte.o         : byte.c  		;  $(CC)  -c $(CFLAGS) byte.c
