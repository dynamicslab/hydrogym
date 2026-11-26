c-----------------------------------------------------------------------
      subroutine n2tec_field_open

      include 'basics.inc'
      include 'tecio.inc'

      character*1 title(80)

      integer*4   debug,disdouble,visdouble
      character*1 nulchar
      integer*4   zero
      pointer     (nullptr,null)

      debug     = 2
      visdouble = 0
      disdouble = 0
      nulchar   = char(0)
      zero      = 0
      nullptr   = 0
c
c Open field.plt and write the header information.
c 

      call blank(s,80)
      len = ltrunc(date,28)
      call chcopy(title(1),'Nek5000 ',8)
      call chcopy(title(9),date,len)
      len = len+9
      call chcopy(title(len),nulchar,1)

      i = tecini100(title,
     &              'X Y Z VTX P'//nulchar,
     &              'field.plt'//nulchar,
     &              '.'//nulchar,
     &               debug,
     &               visdouble)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_fezone_hdr(ncell,npts)

      include 'basics.inc'
      include 'tecio.inc'

      character*1 nulchar
      pointer     (nullptr,null)

      nulchar   = char(0)
      nullptr   = 0
c
c Write the zone header information for the finite-element zone.
C  

c   ZoneTitle - The title of the zone. Must be null-terminated.
c   ZoneType - The type of the zone: 0=ORDERED, 1=FELINESEG, 2=FETRIANGLE,
c   3=FEQUADRILATERAL, 4=FETETRAHEDRON, 5=FEBRICK

      zonetype = 3
      if (if3d) zonetype = 4

      kmax      = 1  

      i = teczne100('Finite Zone'//nulchar,
     &              zonetype,
     &              npts,
     &              ncell,
     &              kmax,
     &              0,
     &              0,
     &              0,
     &              1,
     &              0,
     &              0,
     &              null,
     &              null,
     &              0)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_fezone_hdr(ncell,npts)

      include 'basics.inc'
      include 'tecio.inc'

      character*1 nulchar
      pointer     (nullptr,null)

      nulchar   = char(0)
      nullptr   = 0
c
c Write the zone header information for the finite-element zone.
c  

c   ZoneTitle - The title of the zone. Must be null-terminated.
c   ZoneType - The type of the zone: 0=ORDERED, 1=FELINESEG, 2=FETRIANGLE,
c   3=FEQUADRILATERAL, 4=FETETRAHEDRON, 5=FEBRICK

      zonetype = 3
      if (if3d) zonetype = 4

      kmax      = 1  

      i = teczne100('Finite Zone'//nulchar,
     &              zonetype,
     &              npts,
     &              ncell,
     &              kmax,
     &              0,
     &              0,
     &              0,
     &              1,
     &              0,
     &              0,
     &              null,
     &              null,
     &              0)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_xyz_out(npts,xm,ym,zm)

      include 'tecio.inc'
      include 'basics.inc'

      real xm(npts),ym(npts),zm(npts)
      integer*4   disdouble

      character*1 nulchar
      pointer     (nullptr,null)

      nulchar   = char(0)
      nullptr   = 0
c
c Write out the field data for the finite-element zone.
C  
      disdouble = 0
      i    = tecdat100(npts,xm,disdouble)
      i    = tecdat100(npts,ym,disdouble)
      if (if3d) i = tecdat100(npts,zm,disdouble)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_scalar_out(npts,scalar)

      include 'tecio.inc'
      include 'basics.inc'

      real scalar(npts)
      integer*4   disdouble

      character*1 nulchar
      pointer     (nullptr,null)

      nulchar   = char(0)
      nullptr   = 0
c
c Write out the field data for the finite-element zone.
C  
      disdouble = 0
      i    = tecdat100(npts,scalar,disdouble)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_cell_out(cell)

      include 'tecio.inc'

      integer cell(1)

      i = tecnod100(cell)

      return
      end
c-----------------------------------------------------------------------
      subroutine n2tec_fclose(cell)

      include 'tecio.inc'

      i = tecend100() 

      return
      end
c-----------------------------------------------------------------------
