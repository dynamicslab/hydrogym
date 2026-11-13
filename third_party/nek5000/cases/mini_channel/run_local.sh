#!/bin/bash 

# bash compile_script --all

rm logrun.out

casename=phill
rm  -f $casename.sch
echo $casename > SESSION.NAME
echo $PWD/ >> SESSION.NAME

mpirun -n 16 ./nek5000 | tee logrun.out
