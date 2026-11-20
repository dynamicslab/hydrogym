c-----------------------------------------------------------------------
      subroutine in_situ_init()
#ifdef VISIT
      call visit_init()
#endif
      end
c-----------------------------------------------------------------------
      subroutine in_situ_check()
#ifdef VISIT
      call visit_check()
#endif
      end
c-----------------------------------------------------------------------
      subroutine in_situ_end()
#ifdef VISIT
      call visit_end()
#endif
      end
c-----------------------------------------------------------------------

