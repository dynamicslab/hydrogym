$(OBJDIR)/io_tools_block.o  :io_tools_block.f IOTOOLD;               $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/io_tools.o        :io_tools.f IOTOOLD;                     $(F77) -c $(FL2) -I./ $< -o $@
