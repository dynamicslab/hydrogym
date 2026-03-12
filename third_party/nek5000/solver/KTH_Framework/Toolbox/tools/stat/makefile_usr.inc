$(OBJDIR)/stat.o            :stat.f STATD MAP2D FRAMELP;             $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/stat_IO.o         :stat_IO.f STATD MAP2D FRAMELP;          $(F77) -c $(FL2) -I./ $< -o $@