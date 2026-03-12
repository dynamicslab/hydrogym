$(OBJDIR)/pstat3D.o         :${PSTAT3D_SRC}/pstat3D.f;               $(F77) -c $(FL2) $< -o $@
$(OBJDIR)/pstat3D_IO.o      :${PSTAT3D_SRC}/pstat3D_IO.f;            $(F77) -c $(FL2) $< -o $@
