#!/bin/bash 
## Compile for mini-channel, small-wing and NACA0012 wing cases 

## First of all, we need unpack the solvers 
BASE_PATH=$(pwd)
solver_path=$BASE_PATH/solver/
cd $solver_path
tar -xvf DEC_DRL_Nek5000.tar.gz   
echo "[SYS] NEK - Wing - DRL"
tar -vxf DEC_Nek5000.tar   
echo "[SYS] NEK - Wing - RAW"
tar -xvf KTH_DRL_Framework.tar.gz   
echo "[SYS] NEK - TCF - DRL"
tar -xvf KTH_Framework.tar.gz 
echo "[SYS] NEK - TCF - RAW"
cd $BASE_PATH


# With the correct Solver, lets compile case by case 
case_name=mini_channel 
target_path=$BASE_PATH/cases/${case_name}
cd ${target_path} 
printf '\n' | ./compile_script --clean && ./compile_script --all > $BASE_PATH/log.mini_channel 2>&1
cd $BASE_PATH
echo "[INFO] DONE ${target_path}"

case_name=large_channel 
target_path=$BASE_PATH/cases/${case_name}
cd ${target_path} 
printf '\n' | ./compile_script --clean && ./compile_script --all > $BASE_PATH/log.large_channel 2>&1
cd $BASE_PATH
echo "[INFO] DONE ${target_path}"


case_name=small_wing 
target_path=$BASE_PATH/cases/${case_name}
cd ${target_path} 
./compile_script clean && ./compile_script small_wing > $BASE_PATH/log.small_wing 2>&1
cd $BASE_PATH
echo "[INFO] DONE ${target_path}"

case_name=naca0012_200k 
target_path=$BASE_PATH/cases/${case_name}
cd ${target_path} 
./compile_script clean && ./compile_script naca_wing > $BASE_PATH/log.naca_wing 2>&1
cd $BASE_PATH
echo "[INFO] DONE ${target_path}"
