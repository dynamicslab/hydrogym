$(OBJDIR)/powerit.o         :powerit.f POWERITD;                     $(F77) -c $(FL2) -I./ $< -o $@
