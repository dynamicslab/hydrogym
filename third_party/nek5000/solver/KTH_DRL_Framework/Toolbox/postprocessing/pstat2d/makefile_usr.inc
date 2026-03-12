$(OBJDIR)/pstat2D.o         :${PSTAT2D_SRC}/pstat2D.f;               $(F77) -c $(FL2) $< -o $@
$(OBJDIR)/pstat2D_IO.o      :${PSTAT2D_SRC}/pstat2D_IO.f;            $(F77) -c $(FL2) $< -o $@