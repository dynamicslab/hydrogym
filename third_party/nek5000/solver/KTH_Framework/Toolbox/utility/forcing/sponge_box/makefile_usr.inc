$(OBJDIR)/spongebx.o        :spongebx.f SPONGEBXD;                   $(F77) -c $(FL2) -I./ $< -o $@
