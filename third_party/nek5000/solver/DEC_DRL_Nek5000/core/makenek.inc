# This include file is used by the makenek script
# to automatically create a makefile for Nek5000 
# (c) 2008,2009,2010 UCHICAGO ARGONNE, LLC
# ------------------------------------------------

echo "makenek - automatic build tool for Nek5000"

if [ "$PPLIST" == "?" ]; then
  echo "available pre-processor symbols:" 
  echo "  MPIIO     use MPI-IO I/O kernel (experimental)"
  echo "  BGQ       use BGQ optimized mxm"
  echo "  XSMM      use libxsmm for mxm"
  echo "  BLAS_MXM  disable nek-supplied BLAS routines"
  echo "  CVODE     enable CVODE solver for scalars"
  echo "  MOAB      enable MOAB/CUBIT support (experimental)"
  echo "  NO_LAPACK disable nek-supplied LAPACK routines"
  echo "  EXTBAR    adds underscore to exit call(for BGQ)"
  echo "  CMTNEK    CMT-nek, a DG compressible-flow solver"
  exit 1
fi

# first do some checks ...
if [ $# -eq 0 ]; then
  echo ""
  echo "usage: makenek [.usr filename | clean] < -nocompile >"
  echo ""
  exit 1
fi

if [ "$mver" == "" ]; then
  echo "ERROR: Your makenek is too old!"
  echo "Please upgrade to last version shipped with the Nek source."
  exit 1
else
  if [ $mver -lt 1 ]; then
    echo "ERROR: Your makenek is too old!"
    echo "Please upgrade to last version shipped with the Nek source."
    exit 1
  fi
fi

if [ $1 = "clean" ]; then
  make clean
  exit 0
fi

NOCOMPILE=0
if [ "$2" == "-nocompile" ]; then
  NOCOMPILE=1
fi 

CASENAME=$1
CASEDIR=`pwd`
APATH_SRC=`cd $SOURCE_ROOT_CORE; pwd`
SOURCE_ROOT_CORE=$APATH_SRC
APATH_SRC=`cd $SOURCE_ROOT_JL; pwd`
SOURCE_ROOT_JL=$APATH_SRC
echo "JL root" $SOURCE_ROOT_JL
APATH_SRC=`cd $SOURCE_ROOT_CMT; pwd`
SOURCE_ROOT_CMT=$APATH_SRC
echo "CMT root" $SOURCE_ROOT_CMT

IFMOAB=false
IFNEKNEK=false

# do some basic checks
if [ "$CASEDIR" == "$SOURCE_ROOT_CORE" ]; then
   echo "FATAL ERROR: Working directory has to be different from the source!"
   exit 1
fi

if [ ! -f ${CASENAME}'.usr' ]; then
   echo "FATAL ERROR: Cannot find $1.usr!"
   exit 1
fi

if [ -f SIZEu ]; then
   if [ -f SIZE ]; then
      echo "FATAL ERROR: found SIZEu and SIZE!"
      exit 1
   fi
   NEK_WARN="WARNING: SIZEu will be deprecated in a future release. Please rename to SIZE!"
   export NEK_WARN 
   mv -v SIZEu SIZE
fi

if [ ! -f SIZE ]; then
   echo "FATAL ERROR: Cannot find SIZE!"
   exit 1
fi

if [ ! -f $SOURCE_ROOT_CORE/makefile.template ]; then
  echo "FATAL ERROR: Cannot find $SOURCE_ROOT_CORE/makefile.template!"
  exit 1
fi

# test F77 compiler
which `echo $F77 | awk '{print $1}'` 1>/dev/null
if [ $? -ne 0 ]; then
  echo "FATAL ERROR: Cannot find $F77!"
  exit 1
fi
\rm test_f77.o 2>/dev/null
# basic compiler test
cat > test_f77.f << _ACEOF
      subroutine test
      end
_ACEOF
$F77 -c test_f77.f >/dev/null
if [ ! -f test_f77.o ]; then
  echo "FATAL ERROR: Basic compiler test for $F77 failed!"
  exit 1
fi
\rm test_f77.* 2>/dev/null

# test C compiler
which `echo $CC | awk '{print $1}'` 1>/dev/null
if [ $? -ne 0 ]; then
  echo "FATAL ERROR: Cannot find $CC!"
  exit 1
fi
\rm test_cc.o 2>/dev/null
cat > test_cc.c << _ACEOF
      void function(){}
_ACEOF
$CC -c test_cc.c 1>/dev/null
if [ ! -f test_cc.o ]; then
  echo "FATAL ERROR: Basic compiler test for $CC failed!"
  exit 1
fi
\rm test_cc.* 2>/dev/null

# initial clean-up
rm -f nek5000 2>/dev/null
rm -f ./obj/subuser.o        2>/dev/null

# Check ptr size
cat > tmp.c << _ACEOF
#include <stdlib.h>
#include <stdio.h>
int main()
{
  int *p;printf("%li\n",sizeof(p));
}
_ACEOF
$CC tmp.c 2>&1>/dev/null
ptrSize=`./a.out`
rm tmp.c a.out
if [ "$ptrSize" == "8" ]
then
  PPLIST="${PPLIST} PTRSIZE8"
fi

# Check if the compiler adds an underscore to external functions
UNDERSCORE=false
cat > test_underscore.f << _ACEOF
      subroutine underscore_test
        call byte_rewind
      end
_ACEOF
$F77 -c test_underscore.f 2>&1 >/dev/null 
nm test_underscore.o | grep byte_rewind_ 1>/dev/null
if [ $? -eq 0 ] 
then
  UNDERSCORE=true
fi
\rm test_underscore.* 2>/dev/null


# trying to figure out which compiler we're using 
F77ok=0

F77comp_=`$F77 -showme 2>/dev/null 1>.tmp`
F77comp=`cat .tmp | awk '{print $1}' | awk -F/ '{print $NF}'`
if [ -f "`which $F77comp 2>/dev/null`" ]; then
  F77ok=1
fi

if [ $F77ok -eq 0 ]; then
  F77comp_=`$F77 -show 2>/dev/null 1>.tmp`
  F77comp=`cat .tmp | awk '{print $1}' | awk -F/ '{print $NF}'`
  if [ -f "`which $F77comp 2>/dev/null`" ]; then
    F77ok=1
  fi
fi

if [ $F77ok -eq 0 ]; then
  F77comp_=`$F77 -craype-verbose 2>/dev/null 1>.tmp`
  F77comp=`cat .tmp | awk '{print $1}' | awk -F/ '{print $NF}'`
  if [ -f "`which $F77comp 2>/dev/null`" ]; then
    F77ok=1
  fi
fi

if [ $F77ok -eq 0 ]; then
  F77comp=`echo $F77 | awk '{print $1}'`
  if [ -f "`which $F77comp 2>/dev/null`" ]; then
    F77ok=1
  fi
fi

\rm -f .tmp
if [ $F77ok -eq 0 ]; then
  F77comp="unknown"
fi

# assign F77 compiler specific flags
case $F77comp in
  *pgf*)        P="-r8 -Mpreprocess"
               ;;
  *gfortran*)   P="-fdefault-real-8 -fdefault-double-8 -x f77-cpp-input"
               ;;
  *ifort*)      P="-r8 -fpconstant -fpp"
               ;;
  *pathf*)      P="-r8 -cpp -fno-second-underscore"
               ;;
#  *ftn*)        P="-s default64 -eF"
  *ftn*)        P="-r8 -fpconstant -fpp"
               ;;
  *xlf*)       P="-qrealsize=8 -qdpc=e -qsuffix=cpp=f"
               PPPO="-WF,"
               F77="${F77} -qsuppress=cmpmsg"
               ;;
  *sunf*)       P="-r8const -xtypemap=real:64 -fpp"
               ;;
  *open*)       P="-r8 -cpp -fno-second-underscore"
               ;;
  *)  echo "ERROR: Unable to detect compiler!"
      echo "        - don't know how to promote datatype REAL to 8 bytes"
      echo "        - don't know how to invoke the C pre-processor (CPP) before compilation"
      echo "       Please edit the makefile and specify the requested compiler flags using the P variable."
      echo ""
      P="<specify your compiler flags here>"
      NOCOMPILE=1
      read;;
esac
export PPPO

# set preprocessor symbols
if [ "$IFMPI" == "false" -o "$IFMPI" == "no" ]; then
  IFMPI=false
else
  # default
  IFMPI=true
  PPLIST="${PPLIST} MPI"
fi
export IFMPI

# Check size of long int
cat > tmp.c << _ACEOF
#include <stdlib.h>
#include <stdio.h>
int main()
{
  int i;
  i=sizeof(long int);
  printf("%i\n",i);
}
_ACEOF
$CC tmp.c 2>&1>/dev/null
longIntTest=`./a.out`
rm tmp.c a.out
if [ "$longIntTest" == "8" ]
then
  PPLIST="${PPLIST} LONGINT8"
fi

if [ "$UNDERSCORE" == "true" ]; then
  PPLIST="${PPLIST} UNDERSCORE"
fi 

if [ "$IFPROFILING" == "true" -o "$IFPROFILING" == "yes" ]; then
  if [ "$IFMPI" == "true" -o "IFMPI" == "yes" ]; then
    PPLIST="${PPLIST} TIMER MPITIMER"
  else
    PPLIST="${PPLIST} TIMER"
  fi
fi

PPLIST="${PPLIST} GLOBAL_LONG_LONG"

# add user defined preprocessor symbols
IFCMT=false
for i in $PPLIST
do
   if [ "$i" == "MOAB" ]; then
     IFMOAB=true
     echo $F77comp | grep 'gfort' >/dev/null 
     if [ $? -eq 0 ]; then
       P="${P} -fcray-pointer"
     fi
     if [ "$MOAB_DIR" == "" ]; then
       echo "ABORT: Please specify path to MOAB in MOAB_DIR!" 
       exit 1
     else
       if [ ! -d $MOAB_DIR ]; then
         echo "ABORT: Cannot find " $MOAB_DIR 
         exit 1
       fi 
       if [ ! -f $MOAB_DIR/lib/iMesh-Defs.inc ]; then
         echo "ABORT: Cannot find iMesh-Defs.inc in" $MOAB_DIR"/lib; make sure MOAB is installed."
	 echo "       (to install, use 'make install' in" $MOAB_DIR")"
         exit 1
       fi 
     fi
   fi

   if [ "$i" == "NEKNEK" ]; then
     IFNEKNEK=true
   fi
   if [ "$i" == "CMTNEK" ]; then
     IFCMT=true
   fi
done
export IFCMT


echo $PPLIST | grep 'CVODE' >/dev/null 
if [ $? -eq 0 ]; then
   IFCVODE=true 
else
   IFCVODE=false
fi

# AMG setup
if [ "$IFAMG_DUMP" == "true" ]; then
    if [ "$IFAMG" != "true" -a "$IFAMG" != "yes" ]; then
      echo "ABORT: Cannot dump AMG files if AMG is disabled!"
      exit 1
    fi
    IFAMG_DUMP=true
else
    IFAMG_DUMP=false
fi
export IFAMG_DUMP

if [ "$IFAMG" == "true" -o "$IFAMG" == "yes" ]; then
  IFAMG=true
else
  # default
  IFAMG=false
fi

# MXM-handling
if [ "$MXM_USER" != "" ]; then
    echo "ABORT: makenek version is too old! Update with latest from source!"
    exit 1 
fi
MXM_USER="mxm_std.o blas.o"
echo $PPLIST | grep 'BGQ' >/dev/null 
if [ $? -eq 0 ]; then
   MXM_USER="mxm_std.o mxm_bgq.o " 
   OPT_FLAGS_STD="-O3"
   OPT_FLAGS_MAG="-O3"
fi
echo $PPLIST | grep 'XSMM' >/dev/null 
if [ $? -eq 0 ]; then
   MXM_USER="mxm_std.o blas.o " 
fi
echo $PPLIST | grep 'BLAS_MXM' >/dev/null 
if [ $? -eq 0 ]; then
   MXM_USER="mxm_std.o" 
fi
LPACK="dsygv.o ssygv.o"
echo $PPLIST | grep 'NO_LAPACK' >/dev/null 
if [ $? -eq 0 ]; then
   LPACK=" "
fi

# set optimization flags
L0="\$(G) -O0"
L2="\$(G) -O2"
L3="\$(G) -O3" 
L4="\$(L3)"

# user specified opt flags
if [ "$OPT_FLAGS_STD" != "" ]; then
  echo $OPT_FLAGS_STD | grep "\-O." 1>/dev/null
  if [ $? -eq 0 ]; then
    L2="\$(G) $OPT_FLAGS_STD"
    L3="\$(G) $OPT_FLAGS_STD" 
  else
    L2="\$(G) -O2 $OPT_FLAGS_STD"
    L3="\$(G) -O3 $OPT_FLAGS_STD"
  fi
fi

if [ "$OPT_FLAGS_MAG" != "" ]; then
    L4="\$(G) $OPT_FLAGS_MAG"
fi

if [ "$USR_LIB" != "" ]; then
    USR_LFLAGS="${USR_LFLAGS} ${USR_LIB}"
fi

# tweak makefile template 
echo "generating makefile ..."
rm -rf makefile 2>/dev/null

sed -e "s:^F77[ ]*=.*:F77\:=$F77:" \
-e "s:^CC[ ]*=.*:CC\:=$CC:" \
-e "s:^G[ ]*=.*:G\:=$G:" \
-e "s:^OPT_FLAGS[ ]*=.*:OPT_FLAGS\:=$OPT_FLAGS:" \
-e "s/^P[ ]*=.*/P:=$P/" \
-e "s/^L0[ ]*=.*/L0=$L0/" \
-e "s/^L2[ ]*=.*/L2=$L2/" \
-e "s/^L3[ ]*=.*/L3=$L3/" \
-e "s/^L4[ ]*=.*/L4=$L4/" \
-e "s/^PPPO[ ]*=.*/PPPO=$PPPO/" \
-e "s/^PPS[ ]*=.*/PPS=$PPLIST/" \
-e "s:^MXM[ ]*=.*:MXM=$MXM_USER:" \
-e "s:^LPACK[ ]*=.*:LPACK=$LPACK:" \
-e "s/^IFAMG[ ]*=.*/IFAMG:=$IFAMG/" \
-e "s/^IFAMG_DUMP[ ]*=.*/IFAMG_DUMP:=$IFAMG_DUMP/" \
-e "s/^IFMPI[ ]*=.*/IFMPI:=$IFMPI/" \
-e "s/^IFCMT[ ]*=.*/IFCMT:=$IFCMT/" \
-e "s/^IFMOAB[ ]*=.*/IFMOAB:=$IFMOAB/" \
-e "s:^MOAB_DIR[ ]*=.*:MOAB_DIR\:=${MOAB_DIR}:" \
-e "s/^IFNEKNEK[ ]*=.*/IFNEKNEK:=$IFNEKNEK/" \
-e "s:^USR[ ]*=.*:USR\:=$USR:" \
-e "s:^USR_LFLAGS[ ]*=.*:USR_LFLAGS\:=$USR_LFLAGS:" \
-e "s:^S[ ]*=.*:S\:=${SOURCE_ROOT_CORE}:" \
-e "s:^S2[ ]*=.*:S2\:=${SOURCE_ROOT_CMT}:" \
-e "s:^J[ ]*=.*:J\:=${SOURCE_ROOT_JL}:" $SOURCE_ROOT_CORE/makefile.template >.makefile

#echo $G | grep '\-g' 1>/dev/null
#if [ $? -eq 0 ]; then
#  sed 's/-O[1-4]/-O0/g' .makefile > .makefile.tmp
#  mv .makefile.tmp .makefile
#  echo "Activate DEBUG mode"
#fi

if [ "$IFVISIT" == "true" -o "$IFVISIT" == "yes" ]; then
  echo ""
  echo "Activate VisIt Sim Mode"
  IFVISIT=true

  if [ -z "$VISIT_INC" ]; then
    VISIT_INC=${VISIT_INSTALL}/libsim/V2/include
  fi
  echo "  VISIT_INC = ($VISIT_INC)"
  if [ -z "$VISIT_LIB" ]; then
    VISIT_LIB=${VISIT_INSTALL}/libsim/V2/lib
  fi
  echo "  VISIT_LIB = ($VISIT_LIB)"
  if [ -n "$VISIT_STOP" ]; then
    VISIT_STOP="VISIT_STOP"
    echo "  VISIT_STOP is defined. Nek5000 will wait for VisIt to Connect."
  fi

  sed -e "s/^IFVISIT[ ]*=.*/IFVISIT:=${IFVISIT}/" \
      -e "s:^VISIT_IFLAGS[ ]*=.*:VISIT_IFLAGS\:=-I${VISIT_INC}:" \
      -e "s:^VISIT_LFLAGS[ ]*=.*:VISIT_LFLAGS\:=-L${VISIT_LIB} -lsimV2 -lsimV2f -ldl -lstdc++:" \
      -e "s/^PPS[ ]*=.*/PPS=${PPLIST} VISIT ${VISIT_STOP}/" \
      .makefile > .makefile.tmp

  mv .makefile.tmp .makefile
  echo ""
fi

if [ "$USR" != "" ]; then
  echo "###########################################################" >> makefile
  echo "include makefile_usr.inc" >> .makefile
fi

if [ -f .makefile ]; then
  sed -e "1i\\
### makefile automatically created by makenek `date +"%m/%d/%Y %T"` ###" \
-e "s:^CASEDIR[ ]*=.*:CASEDIR\:=${CASEDIR}:" \
-e "s:^CASENAME[ ]*=.*:CASENAME\:=${CASENAME}:" .makefile > makefile 
else
  echo "ERROR: Nek Makefile could not be created!"
  exit 1 
fi
\rm .makefile 2>/dev/null

# add plugins
for i in $PLUGIN_LIST
do
  if [ ! -f makenek_plugin_$i ]; then
     echo "ABORT: cannot find plug-in config file" makenek_plugin_$i
     exit 1
  fi
  source makenek_plugin_$i

  plugindir="SRC_PLUGIN"
  plugindir=`echo ${!plugindir}`

  if [ ! -f $plugindir/makenek.inc ]; then
     echo "ABORT: Cannot find plug-in include file file" $plugindir/makenek.inc
     exit 1
  fi
  source $plugindir/makenek.inc
  if [ $ierr -ne 0 ]; then
     echo "ABORT: plug-in config failed!"
     exit 1
  else
    echo "found plug-in" $i in $plugindir
  fi
done

# tweak SIZE file
cat SIZE | grep 'SIZE.inc' >/dev/null
if [ $? -ne 0 ];  then
  cat SIZE | grep -i 'optlevel' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      integer optlevel,loglevel' >>SIZE
     echo '      common /lolevels/ optlevel,loglevel' >>SIZE
  fi
  cat SIZE | grep -i 'lxo' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      parameter(lxo   = lx1) ! max output grid size (lxo>=lx1)' >>SIZE
  fi
  cat SIZE | grep -i 'lpart' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      parameter(lpart = 1  ) ! max number of particles/proc' >>SIZE
  fi
  cat SIZE | grep -i 'ax1' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      integer ax1,ay1,az1,ax2,ay2,az2' >> SIZE
     echo '      parameter (ax1=lx1,ay1=ly1,az1=lz1,ax2=lx2,ay2=ly2,az2=lz2) ! running averages' >> SIZE
  fi
  cat SIZE | grep -i 'lys=lxs' >/dev/null
  if [ $? -ne 0 ]; then
     cat SIZE | grep -iv lxs > SIZE.x; mv SIZE.x SIZE  # Clean existing SIZE file of old version
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      parameter (lxs=1,lys=lxs,lzs=(lxs-1)*(ldim-2)+1) !New Pressure Preconditioner' >> SIZE

  fi
  cat SIZE | grep -i 'lcvx1' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      integer lcvx1,lcvy1,lcvz1,lcvelt' >> SIZE
     echo '      parameter (lcvx1=1,lcvy1=1,lcvz1=1,lcvelt=1) ! cvode arrays' >> SIZE
  fi
  cat SIZE | grep -i 'lfdm' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      parameter (lfdm=0)  ! == 1 for fast diagonalization method' >> SIZE
  fi
  cat SIZE | grep -i 'nio' >/dev/null
  if [ $? -ne 0 ]; then
     echo >>SIZE
     echo 'c automatically added by makenek' >>SIZE
     echo '      common/IOFLAG/nio  ! for logfile verbosity control' >> SIZE
  fi

  cat SIZE | grep -i 'toteq' >/dev/null
  if [ $? -ne 0 ]; then
  if [ "$IFCMT" == "true" ]; then
      echo >>SIZE
      echo 'c automatically added by makenek' >>SIZE
      echo '      integer toteq' >> SIZE
      echo '      parameter(toteq = 5  ) ! Number of conserved variables '  >>SIZE
      echo 'c IFCMT=TRUE  then toteq=5'  >>SIZE
  else
      echo >>SIZE
      echo 'c automatically added by makenek' >>SIZE
      echo '      integer toteq' >> SIZE
      echo '      parameter(toteq = 1  ) ! Number of conserved variables '  >>SIZE
      echo 'c IFCMT=FALSE  then toteq=1'  >>SIZE
  fi
  fi
fi

# tweak .usr (actually, .f ) file
rm -f $CASENAME.f
cp -pv $CASENAME.usr $CASENAME.f

cat $CASENAME.f | grep -i "subroutine.*usrsetvert" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end
_ACEOF
fi

cat $CASENAME.f | grep -i "subroutine.*userqtl" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
      subroutine userqtl(flag)
      logical flag

      call qthermal_ig(flag)

      return
      end
_ACEOF
fi

if [ "$IFCVODE" == "true" -o "$IFCVODE" == "yes" ]; then

cat $CASENAME.f | grep -i "^#include.*cvode_aux.*\.h" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
#include "cvode_aux.h"
_ACEOF
fi

cat $CASENAME.f | grep -i "^#include.*cvode_preco.*\.h" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
#include "cvode_preco_dummy.h"
_ACEOF
fi

fi

if [ "$IFCMT" == "true" -o "$IFCMT" == "yes" ]; then
cat $CASENAME.f | grep -i "subroutine.*cmt_usrflt" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
      subroutine cmt_usrflt(rmult) ! user defined filter
      include 'SIZE'
      real rmult(lx1)
      call rone(rmult,lx1)
      return
      end
_ACEOF
fi

cat $CASENAME.f | grep -i "subroutine.*cmt_userflux" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
      subroutine cmt_userflux ! user defined flux
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'CMTDATA'
      real fluxout(lx1*lz1)
      return
      end
_ACEOF
fi

cat $CASENAME.f | grep -i "subroutine.*cmt_userEOS" >/dev/null
if [ $? -ne 0 ]; then
cat >> $CASENAME.f << _ACEOF

c automatically added by makenek
      subroutine cmt_userEOS ! user defined EOS 
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'CMTDATA'

      return
      end
_ACEOF
fi
if [ -e "cmtparticles.usrp" ]
then
echo 'Particles found CMT'
else
cat >> $CASENAME.f << _ACEOF
c
c automatically added by makenek
      subroutine usr_particles_init ! used for particles
      return
      end
c
c automatically added by makenek
      subroutine usr_particles_solver ! used for particles
      return
      end
c
c automatically added by makenek
      subroutine usr_particles_io(istep) ! used for particles
      integer istep
      return
      end

_ACEOF
fi
fi


if [ $NOCOMPILE -eq 1 ]; then
  exit 0
fi 
