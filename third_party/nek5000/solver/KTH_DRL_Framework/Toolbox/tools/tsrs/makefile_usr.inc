$(OBJDIR)/tsrs.o               :tsrs.f TSRSD FRAMELP;                   $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/tsrs_IO.o            :tsrs_IO.f TSRSD;                        $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/pts_redistribute.o   :pts_redistribute.f TSRSD;               $(F77) -c $(FL2) -I./ $< -o $@
