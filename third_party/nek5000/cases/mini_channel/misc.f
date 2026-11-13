        subroutine box2channel
C---------------------------------
c  Define variables
C---------------------------------
        implicit none 
        
        include 'SIZE'
        include "GEOM"
        include "INPUT"
        include "SOLN"

        integer i, ntot
        
        real Betax, Betay
        real Lx, Ly, Lz, Wh, H
        common /hill_param/ Lx, Ly, Lz, Wh, H, Betax ,Betay
        real shift, amp
        real xscale, yscale, zscale, yh, xx, yy, zz
        real hill_step,hill_height,xfac,glmax,glmin
        real xmin, xmax, ymin, ymax, zmin, zmax
        save xmin, xmax, ymin, ymax, zmin, zmax
        logical ifminmax
        save ifminmax
        data ifminmax /.false./

C---------------------------------
c Function
C---------------------------------
        ntot = nx1*ny1*nz1*nelt

        if (.not.ifminmax) then
            ifminmax = .true.
            xmin = glmin(xm1,ntot)
            xmax = glmax(xm1,ntot)
            ymin = glmin(ym1,ntot)
            ymax = glmax(ym1,ntot)
            if (if3d) then
            zmin = glmin(zm1,ntot)
            zmax = glmax(zm1,ntot)
            endif
        endif
        
c increase resolution near the wall
        do i=1,ntot
        ym1(i,1,1,1) = 0.5*(tanh(Betay*(2*ym1(i,1,1,1)-1.0))/
     $ tanh(Betay) + 1.0)
        enddo
        
c rescale rectangular domain [0,Lx]x[0,Ly]x[0,Lz]
        xscale = Lx/(xmax-xmin)
        yscale = Ly/(ymax-ymin)
        
        do i=1,ntot
        xx = xm1(i,1,1,1)
        yy = ym1(i,1,1,1)
        xm1(i,1,1,1) = (xx - xmin) * xscale
        ym1(i,1,1,1) = (yy - ymin) * yscale
        enddo
        
        if (if3d) then
        zscale = Lz/(zmax-zmin)
        do i=1,ntot
        zz = zm1(i,1,1,1)
        zm1(i,1,1,1) = (zz - zmin) * zscale
        enddo
        endif 
        
        return 
        end