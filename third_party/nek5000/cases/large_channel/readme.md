# Compiling the NEK5000 

This folder is used for compiling the code run for Nek5000 simulation. 

## Get started
1. To setup the compiler, please check [compile_script](./compile_script) at LINE 24 and 25: 

        export FC="mpifort "${TOOLBOX_INC}
        
        export CC="mpicc "

Indicates the F77 compiler is the *mpifort* and C compiler is *mpicc*
Please, adopt them to the compiler you have on PC/Cluter!!

2. Parameter of Channel flow: please check [TCF_PARAM](./TCF_PARAM), where geometries and initialization are defined. NOTE that the domain height (wall-normal) is $2.0$

3. General parameter used for simulations are in [SIZE](./SIZE), where you can find comments

4. Once you setup everything, please compile the code via 

        ./compile_script --all 

    You can decide to clean the 3rd-party dependices, but usually NOT NECESSARY!

    You can clean all the current dependencies via 

        ./compile_scipt --clean

## Mesh? 

  Go to the folder [msh](./msh)

    cd ./msh

  Then modify the *Nx, Ny, Nz* as we will design, and then 

    bash mesh.sh 

  Now it is asking the file to input:

    channel_flow.box

  If everything goes well, then input name for output 

    box
  
  Finally give an value below than *0.2*:

    0.00001 


## How it works? 
All the user-defined subroutines are placed in [tcf.usr](./tcf.usr), which WORKS ONLY IF COMPILED!

