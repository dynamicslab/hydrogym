$(OBJDIR)/sfd.o             :sfd.f SFDD FRAMELP;         $(F77) -c $(FL2) -I./ $< -o $@
