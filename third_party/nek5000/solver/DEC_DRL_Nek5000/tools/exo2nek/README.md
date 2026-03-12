Reads a exodus II (.exo) file and generates a .re2 file containing the grid,
curved sides and sideset ids. 

   - Requires the 3rd-party Exodus library (and NetCDF with modifications see below) 
   - So far tested with Cubit, but any other package exporting exodusII should work, too
   - Supported are HEX27 (3D) and QUAD9 (2D)
   - For 2d meshes use z=0
   - Sideset ids are stored in the 5th argument of the fluid bc array
   - The "real" BC's have to be specified in the .usr file using the sideset ids 
   - Periodic BC's are not supported yet
   - Conjugate heat transfer (v and t-mesh) is not supported

## BUILD INSTRUCTIONS
- Download latest netcfd-c release from http://www.unidata.ucar.edu/downloads/netcdf/index.jsp
-  Modify the following defines in include/netcdf.h (recommended for use in exodus)
```
#define NC_MAX_DIMS     65536 
#define NC_MAX_VARS     524288
#define NC_MAX_VAR_DIMS 8
```
-  ```./configure --prefix=$HOME/lib/netcdf CC=gcc --disable-netcdf-4 --disable-fsync --disable-dap```
-  ```make && make install```
- Download latest exodus release from https://github.com/gsjaardema/seacas/archive/exodus.zip
- Ensure ```NetCDF_DIR:PATH``` in cmake-exodus points to the netcdf source
- ```mkdir build; cd build;```
- ```../cmake-exodus```  
- ```make && make install```
- Add ```-I.../seacas/libraries/exoIIv2for32/include``` to the ```F77``` variable in maketools 
- Specify location of exodus and netcdf library using ```USR_LFLAGS``` in maketools e.g. ```-L$HOME/lib```
- Run ```maketools exo2nek```

