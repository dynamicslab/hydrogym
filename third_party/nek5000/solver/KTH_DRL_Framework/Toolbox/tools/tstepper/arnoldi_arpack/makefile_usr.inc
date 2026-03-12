$(OBJDIR)/arn_arp.o         :arn_arp.f ARN_ARPD TSTEPPERD;           $(F77) -c $(FL2) -I./ $< -o $@
$(OBJDIR)/arn_arp_io.o      :arn_arp_io.f ARN_ARPD TSTEPPERD;        $(F77) -c $(FL2) -I./ $< -o $@
