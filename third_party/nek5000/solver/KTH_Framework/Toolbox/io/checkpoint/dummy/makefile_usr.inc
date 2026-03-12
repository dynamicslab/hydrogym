$(OBJDIR)/chkptd.o          :chkptd.f;                               $(F77) -c $(FL2) -I./ $< -o $@
