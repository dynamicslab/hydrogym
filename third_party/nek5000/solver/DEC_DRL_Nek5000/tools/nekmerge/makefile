FLAGS =  $(FFLAGS_IN)
CFLAGS =  $(CFLAGS_IN)
LFLAGS = $(LFLAGS_IN)
prefix = $(bin_nek_tools)

NOBJ1 = nekmerge.o reader.o byte.o strings.o tsort.o

all: nekmerge

nekmerge:	$(NOBJ1)
	$(F77) $(FLAGS) -o  $(prefix)/nekmerge $(NOBJ1) $(LFLAGS)

clean:
	'rm' *.o

nekmerge.o	: nekmerge.f; $(F77) -c $(FLAGS)  nekmerge.f 
reader.o  	: reader.f	; $(F77) -c $(FLAGS)  reader.f 
tsort.o  	: tsort.f	; $(F77) -c $(FLAGS)  tsort.f 
strings.o  	: strings.f	; $(F77) -c $(FLAGS)  strings.f 
byte.o		: byte.c	; $(CC) -c $(CFLAGS)  byte.c
