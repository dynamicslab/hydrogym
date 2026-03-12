c---------------------------------------------------------------------
c VisIt Simulation Code for Nek5000
c
c   Provide a connection to VisIt simulation code. You will be able
c to connect to VisIt and drive the simulation though VisIt or
c have sim code send data to VisIt. 
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c visit_init
c
c   Initialize VisIt. This will create the sim file needed by Visit
c and create the initial VisIt setup.
c---------------------------------------------------------------------
      subroutine visit_init
      implicit none
      include "visitfortransimV2interface.inc"
      include "mpif.h"
c     // local variables
      integer err
c     // SIMSTATE common block
      integer runflag, endflag
      common /SIMSTATE/ runflag, endflag
      save /SIMSTATE/
c     // PARALLEL state common block
      integer par_rank, par_size
      common /PARALLEL/ par_rank, par_size
      save /PARALLEL/

#ifdef VISIT_STOP
c     // The sim will wait for VisIt to connect after first step.
      runflag = 0
#else
c     // Default is to run sim and VisIt can connect any time.
      runflag = 1
#endif
      endflag = 0

c     // Determine the rank and size of this MPI task so we can tell
c     // VisIt's libsim about it.
      call MPI_COMM_RANK(MPI_COMM_WORLD, par_rank, err)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, par_size, err)
      if(par_size.gt.1) then
          err = visitsetparallel(1)
      endif
      err = visitsetparallelrank(par_rank)

      call simulationarguments()
c     // TODO: look for the visitsetupenv2 function.
c     // Has better scaling, but has not been release for fortran.
      err = visitsetupenv()
c     // Have the master process create the sim file.
      if(par_rank.eq.0) then
          err = visitinitializesim("nek5000", 7,
     .       "Nek5000 Simulation", 18,
     .       "/no/useful/path", 15,
     .       VISIT_F77NULLSTRING, VISIT_F77NULLSTRINGLEN,
     .       VISIT_F77NULLSTRING, VISIT_F77NULLSTRINGLEN,
     .       VISIT_F77NULLSTRING, VISIT_F77NULLSTRINGLEN)
      endif

      end

c---------------------------------------------------------------------
c simulationarguments
c     This routine handles command line arguments
c     -dir <VisIt directory> 
c     -options <VisIt Options> 
c     -trace <VisIt trace file>
c---------------------------------------------------------------------
      subroutine simulationarguments()
      implicit none
      character (len=80) str
      integer err, i, N, len
      integer visitsetoptions, visitsetdirectory, visitopentracefile

      N = iargc()
      i = 1
      len = 80
5     if (i.le.N) then
          call getarg(i, str)
          if(str.eq."-dir") then
              call getarg(i+1, str)
              err = visitsetdirectory(str, len)
              i = i + 1
          elseif(str.eq."-options") then
              call getarg(i+1, str)
              err = visitsetoptions(str, len)
              i = i + 1
          elseif(str.eq."-trace") then
              call getarg(i+1, str)
              err = visitopentracefile(str, len)
              i = i + 1
          endif
          i = i + 1
          goto 5
      endif
      end

c---------------------------------------------------------------------
c processvisitcommand
c---------------------------------------------------------------------
      integer function processvisitcommand()
      implicit none
      include "mpif.h"
      include "visitfortransimV2interface.inc"
c     // PARALLEL state common block
      integer par_rank, par_size
      common /PARALLEL/ par_rank, par_size
      integer command, e, doloop, success, ret
      integer VISIT_COMMAND_PROCESS
      integer VISIT_COMMAND_SUCCESS
      integer VISIT_COMMAND_FAILURE
      parameter (VISIT_COMMAND_PROCESS = 0)
      parameter (VISIT_COMMAND_SUCCESS = 1)
      parameter (VISIT_COMMAND_FAILURE = 2)

      if(par_rank.eq.0) then
          success = visitprocessenginecommand()

          if(success.gt.0) then
              command = VISIT_COMMAND_SUCCESS
              ret = 1
          else
              command = VISIT_COMMAND_FAILURE
              ret = 0
          endif

          call MPI_BCAST(command,1,MPI_INTEGER,0,MPI_COMM_WORLD,e)
      else
          doloop = 1
2345      call MPI_BCAST(command,1,MPI_INTEGER,0,MPI_COMM_WORLD,e)
          if(command.eq.VISIT_COMMAND_PROCESS) then
              success = visitprocessenginecommand()
          elseif(command.eq.VISIT_COMMAND_SUCCESS) then
              ret = 1
              doloop = 0
          else
              ret = 0
              doloop = 0
          endif

          if(doloop.ne.0) then
              goto 2345
          endif
      endif
      processvisitcommand = ret
      end

c---------------------------------------------------------------------
c visit_check
c---------------------------------------------------------------------
      subroutine visit_check()
      implicit none
      include "mpif.h"
      include "visitfortransimV2interface.inc"
c     // functions
      integer processvisitcommand
c     // local variables
      integer visitstate, result, blocking, ierr
c     // SIMSTATE common block
      integer runflag, endflag
      common /SIMSTATE/ runflag, endflag
c     // PARALLEL state common block
      integer par_rank, par_size
      common /PARALLEL/ par_rank, par_size

c     // If at the end of sim, lets not force an update.
      if(endflag.eq.0) then
c        // Check if we are connected to VisIt
         result = visitisconnected()
         if(result.eq.1) then
c            // Tell VisIt that the timestep changed
             result = visittimestepchanged()
c            // Tell VisIt to update its plots
             result = visitupdateplots()
         endif
      endif

c     write (6, 2000)
 2000 format('VisIt Check!')
      do 10
c         // If we are running don't block
          if(runflag.eq.1) then
              blocking = 0
          else
              blocking = 1
          endif

c         // Detect input from VisIt on processor 0 and then broadcast
c         // the results of that input to all processors.
          if(par_rank.eq.0) then
              visitstate = visitdetectinput(blocking, -1)
          endif
          call MPI_BCAST(visitstate,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

          if (visitstate.eq.0) then
c             // Okay - Process time step.
              goto 1234
          elseif (visitstate.eq.1) then
c             // Attempt to Connect VisIt
              ierr = runflag
              runflag = 0
              result = visitattemptconnection()
              if (result.eq.1) then
                  write (6, 2001)
 2001             format('VisIt connected!')
              else
                  write (6, 2002)
 2002             format('VisIt did not connected!')
              endif
              flush( 6 )
              runflag = ierr
          elseif (visitstate.eq.2) then
c             // Engine socket input
c             ierr = runflag
c             runflag = 0
              if (processvisitcommand().eq.0) then
                  result = visitdisconnect()
c                 // If VisIt is disconnect lets run.
                  runflag = 1
              endif
c             // Check if user wants to exit sim.
              if(runflag.eq.2) then
                  goto 1234
              endif
          elseif (visitstate.eq.3) then
c             // Console socket input
          elseif (visitstate.lt.0) then
c             // Error
              goto 1234
          endif
10    continue
1234  end

c---------------------------------------------------------------------
c visit_end
c---------------------------------------------------------------------
      subroutine visit_end()
      implicit none
      include "visitfortransimV2interface.inc"
c     // local variables
      integer result
c     // SIMSTATE common block
      integer runflag, endflag
      common /SIMSTATE/ runflag, endflag

c     // Check if we are connected to VisIt
      result = visitisconnected()
      if(result.eq.1) then
c        // Let VisIt exit the sim.

c        // This will tell the visit_check function we are at the end.
         endflag = 1
         runflag = 0

         do 10
            call visit_check()

c           // User asked to finish.
            if (endflag.eq.2) then
               EXIT
            endif
10       continue
      endif

      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c These functions must be defined to satisfy the 
c visitfortransimV2interface lib.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c---------------------------------------------------------------------
c visitcommandcallback
c   Handle User defined functions from VisIt user interface.
c---------------------------------------------------------------------
      subroutine visitcommandcallback (cmd, lcmd, args, largs)
      implicit none
      character*8 cmd, args
      integer     lcmd, largs
      include "visitfortransimV2interface.inc"
c     // SIMSTATE common block
      integer runflag, endflag
      common /SIMSTATE/ runflag, endflag

c     // Handle the commands that we define in visitgetmetadata.
      if(visitstrcmp(cmd, lcmd, "stop", 4).eq.0) then
          runflag = 0
      elseif(visitstrcmp(cmd, lcmd, "step", 4).eq.0) then
          runflag = 2
      elseif(visitstrcmp(cmd, lcmd, "run", 3).eq.0) then
          runflag = 1
      elseif(visitstrcmp(cmd, lcmd, "exit", 4).eq.0) then
          call exitt()
      elseif(visitstrcmp(cmd, lcmd, "finish", 6).eq.0) then
          if(endflag.eq.1) then
              endflag = 2
              runflag = 2
          endif
      endif
      end

c---------------------------------------------------------------------
c visitbroadcastintfunction
c---------------------------------------------------------------------
      integer function visitbroadcastintfunction(value, sender)
      implicit none
      include "mpif.h"
      integer value, sender, ierr
      call MPI_BCAST(value,1,MPI_INTEGER,sender,MPI_COMM_WORLD,ierr)
      visitbroadcastintfunction = 0
      end

c---------------------------------------------------------------------
c visitbroadcaststringfunction
c---------------------------------------------------------------------
      integer function visitbroadcaststringfunction(str, lstr, sender)
      implicit none
      include "mpif.h"
      character*8 str
      integer     lstr, sender, ierr
      call MPI_BCAST(str,lstr,MPI_CHARACTER,sender,MPI_COMM_WORLD,ierr)
      visitbroadcaststringfunction = 0
      end

c---------------------------------------------------------------------
c visitslaveprocesscallback
c---------------------------------------------------------------------
      subroutine visitslaveprocesscallback ()
      implicit none
      include "mpif.h"
      integer c, ierr, VISIT_COMMAND_PROCESS
      parameter (VISIT_COMMAND_PROCESS = 0)
      c = VISIT_COMMAND_PROCESS
      call MPI_BCAST(c, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      end

c---------------------------------------------------------------------
c visitactivatetimestep
c---------------------------------------------------------------------
      integer function visitactivatetimestep()
      implicit none
      include "visitfortransimV2interface.inc"
      visitactivatetimestep = VISIT_OKAY
      end

c---------------------------------------------------------------------
c visitgetmetadata
c   This function tells VisIt about all of the data that this
c   simulation will output. Mesh, variables, expressions and commands.
c---------------------------------------------------------------------
      integer function visitgetmetadata()
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      INCLUDE 'PARALLEL'
      include "visitfortransimV2interface.inc"
c     // SIMSTATE common block
      integer runflag, endflag
      common /SIMSTATE/ runflag, endflag
c     // PARALLEL state common block
      integer par_rank, par_size
      common /PARALLEL/ par_rank, par_size
c     // Local variables
      integer md, mmd, vmd, cmd, emd, err, iDim, k
      character*3 seqstring

      if(visitmdsimalloc(md).eq.VISIT_OKAY) then
        err = visitmdsimsetcycletime(md, ISTEP, TIME)
        if(runflag.eq.1) then
          err = visitmdsimsetmode(md, VISIT_SIMMODE_RUNNING)
        else
          err = visitmdsimsetmode(md, VISIT_SIMMODE_STOPPED)
        endif

c       // TODO: ask why this changes after first save?
c       write(6,*) 'VISIT: IFXYO',IFXYO
c       if(IFXYO) then
c         // Add a mesh
          if(IF3D) then
             iDim = 3
          else
             iDim = 2
          endif

          if(visitmdmeshalloc(mmd).eq.VISIT_OKAY) then
            err = visitmdmeshsetname(mmd, "mesh", 4)
            err = visitmdmeshsetmeshtype(mmd,
     .          VISIT_MESHTYPE_CURVILINEAR)
            err = visitmdmeshsettopologicaldim(mmd, iDim)
            err = visitmdmeshsetspatialdim(mmd, iDim)
            err = visitmdmeshsetnumdomains(mmd, NELGT)
            err = visitmdmeshsetdomaintitle(mmd, "Domains", 7)
            err = visitmdmeshsetdomainpiecename(mmd, "domain", 6)
c           err = visitmdmeshsetxunits(mmd, "cm", 2)
c           err = visitmdmeshsetyunits(mmd, "cm", 2)
            err = visitmdmeshsetxlabel(mmd, "X-Axis", 6)
            err = visitmdmeshsetylabel(mmd, "Y-Axis", 6)
            if(IF3D) then
c             err = visitmdmeshsetzunits(mmd, "cm", 2)
              err = visitmdmeshsetzlabel(mmd, "Z-Axis", 6)
            endif
            err = visitmdsimaddmesh(md, mmd)
          endif
c       endif

        if(IFVO) then
c         // Add a X velocity variable on the mesh.
          if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
            err = visitmdvarsetname(vmd, "x_velocity", 10)
            err = visitmdvarsetmeshname(vmd, "mesh", 4)
c           err = visitmdvarsetunits(vmd, "cm", 2)
            err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
            err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c           err = visitmdvarsettreatasascii(vmd, 0)
            err = visitmdsimaddvariable(md, vmd)
          endif

c         // Add a Y velocity variable on the mesh.
          if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
            err = visitmdvarsetname(vmd, "y_velocity", 10)
            err = visitmdvarsetmeshname(vmd, "mesh", 4)
c           err = visitmdvarsetunits(vmd, "cm", 2)
            err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
            err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c           err = visitmdvarsettreatasascii(vmd, 0)
            err = visitmdsimaddvariable(md, vmd)
          endif

          if(IF3D) then
c           // Add a Z velocity variable on the mesh.
            if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
              err = visitmdvarsetname(vmd, "z_velocity", 10)
              err = visitmdvarsetmeshname(vmd, "mesh", 4)
c             err = visitmdvarsetunits(vmd, "cm", 2)
              err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
              err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c             err = visitmdvarsettreatasascii(vmd, 0)
              err = visitmdsimaddvariable(md, vmd)
            endif
          endif

c         // Add a velocity expression.
          if(visitmdexpralloc(emd).eq.VISIT_OKAY) then
              err = visitmdexprsetname(emd, "velocity", 8)
              if(IF3D) then
                  err = visitmdexprsetdefinition(emd, 
     .                      "{x_velocity, y_velocity, z_velocity}", 36)
              else
                  err = visitmdexprsetdefinition(emd, 
     .                      "{x_velocity, y_velocity}", 24)
              endif
              err = visitmdexprsettype(emd, VISIT_VARTYPE_VECTOR)

              err = visitmdsimaddexpression(md, emd)
          endif

c         // Add a velocity magnitude expression.
          if(visitmdexpralloc(emd).eq.VISIT_OKAY) then
              err = visitmdexprsetname(emd, "velocity_mag", 12)
              err = visitmdexprsetdefinition(emd, 
     .                  "magnitude(velocity)", 19)
              err = visitmdexprsettype(emd, VISIT_VARTYPE_SCALAR)

              err = visitmdsimaddexpression(md, emd)
          endif
        endif

        if(IFPO) then
c         // Add a pressure variable on the mesh.
          if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
            err = visitmdvarsetname(vmd, "pressure", 8)
            err = visitmdvarsetmeshname(vmd, "mesh", 4)
c           err = visitmdvarsetunits(vmd, "cm", 2)
            err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
            err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c           err = visitmdvarsettreatasascii(vmd, 0)
            err = visitmdsimaddvariable(md, vmd)
          endif
        endif

        if(IFTO) then
c         // Add a temperature variable on the mesh.
          if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
            err = visitmdvarsetname(vmd, "temperature", 11)
            err = visitmdvarsetmeshname(vmd, "mesh", 4)
c           err = visitmdvarsetunits(vmd, "cm", 2)
            err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
            err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c           err = visitmdvarsettreatasascii(vmd, 0)
            err = visitmdsimaddvariable(md, vmd)
          endif
        endif

        do k=1,LDIMT-1
c         // Add a user defined variable on the mesh.
          if(IFPSCO(k)) then
            if(visitmdvaralloc(vmd).eq.VISIT_OKAY) then
              write (seqstring,'(I0)') k
              err = visitmdvarsetname(vmd, "s"//trim(seqstring),
     .                                1+len_trim(seqstring))
              err = visitmdvarsetmeshname(vmd, "mesh", 4)
c             err = visitmdvarsetunits(vmd, "cm", 2)
              err = visitmdvarsetcentering(vmd, VISIT_VARCENTERING_NODE)
              err = visitmdvarsettype(vmd, VISIT_VARTYPE_SCALAR)
c             err = visitmdvarsettreatasascii(vmd, 0)
              err = visitmdsimaddvariable(md, vmd)
            endif
          endif
        enddo


c       // Add simulation commands
        err = visitmdcmdalloc(cmd)
        if(err.eq.VISIT_OKAY) then
           err = visitmdcmdsetname(cmd, "stop", 4)
           err = visitmdsimaddgenericcommand(md, cmd)
        endif
        err = visitmdcmdalloc(cmd)
        if(err.eq.VISIT_OKAY) then
           err = visitmdcmdsetname(cmd, "step", 4)
           err = visitmdsimaddgenericcommand(md, cmd)
        endif
        err = visitmdcmdalloc(cmd)
        if(err.eq.VISIT_OKAY) then
           err = visitmdcmdsetname(cmd, "run", 3)
           err = visitmdsimaddgenericcommand(md, cmd)
        endif
        err = visitmdcmdalloc(cmd)
        if(err.eq.VISIT_OKAY) then
           err = visitmdcmdsetname(cmd, "exit", 4)
           err = visitmdsimaddgenericcommand(md, cmd)
        endif
        err = visitmdcmdalloc(cmd)
        if(err.eq.VISIT_OKAY) then
           err = visitmdcmdsetname(cmd, "finish", 6)
           err = visitmdsimaddgenericcommand(md, cmd)
        endif
      endif
      visitgetmetadata = md
      end

c---------------------------------------------------------------------
c visitgetmesh
c    Use this function to return mesh data to VisIt.
c---------------------------------------------------------------------
      integer function visitgetmesh(domain, name, lname)
      include 'SIZE'
      include 'TOTAL'
      character*8 name
      integer     domain, lname
      include "visitfortransimV2interface.inc" 

c     // local variables
      integer vh, x, y, z, err, dl, dmdims(3)

      vh = VISIT_INVALID_HANDLE

c     // Fortran starting index 1, but VisIt is 0
c     // Also need to get local domain index
      domain = GLLEL(domain + 1)

      if(visitstrcmp(name, lname, "mesh", 4).eq.0) then
        if(visitcurvmeshalloc(vh).eq.VISIT_OKAY) then
          err = visitvardataalloc(x)
          err = visitvardataalloc(y)
          if(IF3D) err = visitvardataalloc(z)

          dl = lx1 * ly1 * lz1
          err = visitvardatasetd(x, VISIT_OWNER_SIM, 1, dl, 
     .                           XM1(1,1,1,domain))
          err = visitvardatasetd(y, VISIT_OWNER_SIM, 1, dl,
     .                           YM1(1,1,1,domain))
          if(IF3D) then
            err = visitvardatasetd(z, VISIT_OWNER_SIM, 1, dl,
     .                             ZM1(1,1,1,domain))
          endif

          dmdims(1) = lx1
          dmdims(2) = ly1
          dmdims(3) = lz1
          if(IF3D) then
            err = visitcurvmeshsetcoordsxyz(vh, dmdims, x, y, z)
          else
            err = visitcurvmeshsetcoordsxy(vh, dmdims, x, y)
          endif
        endif
      endif

      visitgetmesh = vh
      end

c---------------------------------------------------------------------
c visitgetmaterial
c    Use this function to return material data to VisIt.
c---------------------------------------------------------------------
      integer function visitgetmaterial(domain, name, lname)
      implicit none
      character*8 name
      integer     domain, lname
      include "visitfortransimV2interface.inc"
      visitgetmaterial = VISIT_INVALID_HANDLE
      end

c---------------------------------------------------------------------
c visitgetvariable
c    Use this function to return variable data to VisIt.
c---------------------------------------------------------------------
      integer function visitgetvariable(gdomain, name, lname)
c     implicit none
      include 'SIZE'
      include 'TOTAL'
      character*8 name
      integer     gdomain, lname
      include "visitfortransimV2interface.inc"
c     // local vars
      integer h, nvals, err, domain, k

      nvals = lx1 * ly1 * lz1

      h = VISIT_INVALID_HANDLE

c     // Fortran starting index 1, but VisIt is 0
c     // Also need to get local domain index
      gdomain = gdomain + 1
      domain = GLLEL(gdomain)

      if(visitvardataalloc(h).eq.VISIT_OKAY) then
          if(visitstrcmp(name, lname, "temperature", 11).eq.0) then
              err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                               T(1,1,1,domain,1))
          elseif(visitstrcmp(name, lname, "pressure", 8).eq.0) then
            if(gdomain.le.NELGV) then
              err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                               PR(1,1,1,domain))
            endif
          elseif(visitstrcmp(name, lname, "x_velocity", 10).eq.0) then
            if(gdomain.le.NELGV) then
              err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                               VX(1,1,1,domain))
            endif
          elseif(visitstrcmp(name, lname, "y_velocity", 10).eq.0) then
            if(gdomain.le.NELGV) then
              err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                               VY(1,1,1,domain))
            endif
          elseif(visitstrcmp(name, lname, "z_velocity", 10).eq.0) then
            if(gdomain.le.NELGV) then
              err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                               VZ(1,1,1,domain))
            endif
          elseif(visitstrcmp("s", 1, name, 1).eq.0) then
c             // Handle the user define variables.
              read( name(2:lname), '(i10)' ) k
              if(IFPSCO(k)) then
                  k = k + 1
                  err = visitvardatasetd(h, VISIT_OWNER_SIM, 1, nvals,
     .                                   T(1,1,1,domain,k))
              endif
          endif
      endif

      visitgetvariable = h
      end

c---------------------------------------------------------------------
c visitgetcurve
c    Use this function to return curve data to VisIt.
c---------------------------------------------------------------------
      integer function visitgetcurve(handle, name, lname)
      implicit none
      character*8 name
      integer     handle, lname
      include "visitfortransimV2interface.inc"
      visitgetcurve = VISIT_INVALID_HANDLE
      end

c---------------------------------------------------------------------
c visitgetdomainlist
c    This function returns a list of domains owned by this process.
c---------------------------------------------------------------------
      integer function visitgetdomainlist()
      INCLUDE 'SIZE'
      INCLUDE 'PARALLEL'
      include "visitfortransimV2interface.inc"
c     // local vars
      integer h, dl, err

      h = VISIT_INVALID_HANDLE

      if(visitdomainlistalloc(h).eq.VISIT_OKAY) then
          if(visitvardataalloc(dl).eq.VISIT_OKAY) then
c             // Hack to work around the index difference between
c             // fortran and VisIt C. Temp change and make copy.
              do i=1,NELT
                  LGLEL(i) = LGLEL(i) - 1
              enddo
              err = visitvardataseti(dl, VISIT_OWNER_COPY,1,NELT,LGLEL)
              err = visitdomainlistsetdomains(h, NELGT, dl)
c             // restore correct fortran values.
              do i=1,NELT
                  LGLEL(i) = LGLEL(i) + 1
              enddo
          endif
      endif

      visitgetdomainlist = h
      end

c---------------------------------------------------------------------
c visitgetdomainbounds
c    This function allows VisIt to create ghost zones between domains.
c---------------------------------------------------------------------
      integer function visitgetdomainbounds(name, lname)
      implicit none
      character*8 name
      integer     lname
      include "visitfortransimV2interface.inc"
      visitgetdomainbounds = VISIT_INVALID_HANDLE
      end

c---------------------------------------------------------------------
c visitgetdomainnesting
c    This is used to tell VisIt how AMR patches are nested.
c---------------------------------------------------------------------
      integer function visitgetdomainnesting(name, lname)
      implicit none
      character*8 name
      integer     lname
      include "visitfortransimV2interface.inc"
      visitgetdomainnesting = VISIT_INVALID_HANDLE
      end

c---------------------------------------------------------------------
c     visitgetmixedvariable
c     This is needed to run with newer version of Visit (2.12.0 tested).
c     TODO: The comment should be changed
c---------------------------------------------------------------------
      integer function visitgetmixedvariable(domain, name, lname)
      implicit none
      character*8 name
      integer     domain, lname
      include "visitfortransimV2interface.inc"
      visitgetmixedvariable = VISIT_INVALID_HANDLE
      end
