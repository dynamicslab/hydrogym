$(OBJDIR)/rprm_block.o      :rprm_block.f RPRMD;                     $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/rprm.o            :rprm.f RPRMD FRAMELP;                   $(F77) -c $(FL2) -I./ $< -o $@
