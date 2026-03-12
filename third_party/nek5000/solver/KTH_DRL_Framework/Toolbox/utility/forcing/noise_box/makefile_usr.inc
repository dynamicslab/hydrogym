$(OBJDIR)/noisebx.o         :noisebx.f NOISEBXD;                     $(F77) -c $(FL2) -I./ $< -o $@
