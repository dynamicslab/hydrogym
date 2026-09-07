# Simulation complie on a small wing coarse mesh 


## File Notation
### From SmallWing 
+ *small_wing.ma2*
+ *small_wing.re2*
+ *small_wing.rea*
+ *small_wing.restart*
+ *SIZE*
+ *rs8small_wing0.f00001*
+ *rs8small_wing0.f00002*
+ *rs8small_wing0.f00003*
+ *rs8small_wing0.f00004*

### From NACA
+ *naca_wing.map*
+ *naca_wing.usr*
+ *naca_wing.wall*
+ *naca_wing.f*
+ Everything else

## Process of running a simulation 

1. Check the parameter definiation 
    
    + *SIZE*: The basic dimension and run-time paramters 
    + *.rea*: The NEK parameters in Physics and for simulation setup 
    + *.usr*: Set the boundary and initial condition in function
    + *.re2*: Geometry file of geometry 
    + *.map*: Geometry file: The No.nodes  
    + *.ma2*: Geometry file of mesh  
    + *.wall*: Wall definiation and global elements 
    + *.restart*: A flag file to have the restart file

2. Comilpe: 

        ./makenek clean

    Then: 

        ./makenek small_wing 

3. Run simulation: 

        bash lcRun.sh


4. To visualize the results, modify the *.nek5000* file, then run: 

        cd la2_data

   then open in ParaView

        paraview --data=small_wing.nek5000



## Logs

### Oct 5 
Solved:
1. In *makenek* file, for genertic compiler flags, use the commands:
    
        G="-I./inc_src -mcmodel=large -std=legacy"

2. 


### Oct 6 

Solved:
1. Correct the restart issue, please use *small_wing* for the name when compiling. 

To DO: 
1. Debugging the error of statistics showing the layer 

