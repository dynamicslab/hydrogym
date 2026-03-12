#!/bin/bash
# Bash script to export all toolbox paths

# Change a possible relative path to an absolute one
# This is just a hack
cd ${TOOLBOX_SRC}
TOOLBOX_SRC=`pwd`
cd -

# Framework paths
export DRIVER_SRC=${TOOLBOX_SRC}"/driver"
export IO_SRC=${TOOLBOX_SRC}"/io"
export POSTPR_SRC=${TOOLBOX_SRC}"/postprocessing"
export TOOLS_SRC=${TOOLBOX_SRC}"/tools"
export UTILITY_SRC=${TOOLBOX_SRC}"/utility"

export COMM_SRC=${DRIVER_SRC}"/comm"
TOOLBOX_INC=" -I"${COMM_SRC}
export FRAME_SRC=${DRIVER_SRC}"/frame"
TOOLBOX_INC+=" -I"${FRAME_SRC}

export MONITOR_SRC=${FRAME_SRC}"/monitor"
TOOLBOX_INC+=" -I"${MONITOR_SRC}
export RUNPARAM_SRC=${FRAME_SRC}"/runparam"
TOOLBOX_INC+=" -I"${RUNPARAM_SRC}

export CHKPT_SRC=${IO_SRC}"/checkpoint"
TOOLBOX_INC+=" -I"${CHKPT_SRC}
export CHKPTDMM_SRC=${CHKPT_SRC}"/dummy"
export CHKPTMS_SRC=${CHKPT_SRC}"/mstep"
TOOLBOX_INC+=" -I"${CHKPTMS_SRC}

export IO_TOOLS_SRC=${IO_SRC}"/io_tools"
TOOLBOX_INC+=" -I"${IO_TOOLS_SRC}

# Statistics post-processing include file has to be copied to setup directory
export PSTAT2D_SRC=${POSTPR_SRC}"/pstat2d"
export PSTAT3D_SRC=${POSTPR_SRC}"/pstat3d"

export BASEFLOW_SRC=${TOOLS_SRC}"/baseflow"
# Statistics include file has to be copied to setup directory
export STAT_SRC=${TOOLS_SRC}"/stat"
# Time series include file has to be copied to setup directory
export TSRS_SRC=${TOOLS_SRC}"/tsrs"
export TSTEPPER_SRC=${TOOLS_SRC}"/tstepper"
TOOLBOX_INC+=" -I"${TSTEPPER_SRC}

export SFD_SRC=${BASEFLOW_SRC}"/sfd"
TOOLBOX_INC+=" -I"${SFD_SRC}

# Arnoldi include file has to be copied to setup directory
export ARNARP_SRC=${TSTEPPER_SRC}"/arnoldi_arpack"
export POWERIT_SRC=${TSTEPPER_SRC}"/powerit"
TOOLBOX_INC+=" -I"${POWERIT_SRC}

export BCND_SRC=${UTILITY_SRC}"/bcnd"
export CONHT_SRC=${UTILITY_SRC}"/conht"
TOOLBOX_INC+=" -I"${CONHT_SRC}
export FORCING_SRC=${UTILITY_SRC}"/forcing"
export GRID_SRC=${UTILITY_SRC}"/grid"
export MATH_SRC=${UTILITY_SRC}"/math"
TOOLBOX_INC+=" -I"${MATH_SRC}

# Generalised synthetic eddy method include file has to be copied to setup directory
export GSYEM_SRC=${BCND_SRC}"/gsyem"

export NOISE_BOX_SRC=${FORCING_SRC}"/noise_box"
TOOLBOX_INC+=" -I"${NOISE_BOX_SRC}
export SPONGE_BOX_SRC=${FORCING_SRC}"/sponge_box"
TOOLBOX_INC+=" -I"${SPONGE_BOX_SRC}
# Tripping line include file has to be copied to setup directory
export TRIP_LINE_SRC=${FORCING_SRC}"/trip_line"

export MAP2D_SRC=${GRID_SRC}"/map2D"
TOOLBOX_INC+=" -I"${MAP2D_SRC}

# Include path
export TOOLBOX_INC

