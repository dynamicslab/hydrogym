#!/bin/sh
# mpif_sizeof_gfortran.sh <mpif-sizeof.h>: make HPC-X's mpif.h usable with gfortran.
#
# The HPC-X of the aarch64 NVIDIA HPC SDK is built with nvfortran, which has
# REAL*2 (half precision), so its mpif-sizeof.h declares MPI_SIZEOF (and PMPI_SIZEOF) for REAL*2
# and COMPLEX*4. gfortran rejects those declarations ("Old-style type
# declaration REAL*2 not supported"), so every gfortran build that includes
# mpif.h fails: Nek5000, NekRS (CMake's FindMPI test, and the case files that
# NekRS compiles at run time). This drops exactly those MPI_SIZEOF procedures
# and keeps everything else (the x86_64 HPC-X has none of them).
set -eu
f="$1"
awk '
/^ *SUBROUTINE P?MPI_Sizeof_/ { blk = $0 ORS; inb = 1; drop = 0; next }
inb {
    blk = blk $0 ORS
    if ($0 ~ /REAL\*2|COMPLEX\*4/) drop = 1
    if ($0 ~ /END SUBROUTINE/) { if (!drop) printf "%s", blk; inb = 0 }
    next
}
{ print }
' "$f" > "$f.gfortran"
cat "$f.gfortran" > "$f" && rm -f "$f.gfortran"
! grep -q -E 'REAL\*2|COMPLEX\*4' "$f"
