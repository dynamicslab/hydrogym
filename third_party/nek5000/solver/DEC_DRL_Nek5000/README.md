# Nek5000 
![https://travis-ci.org/Nek5000/Nek5000](https://travis-ci.org/Nek5000/Nek5000.svg?branch=develop)
[![GPLv3 licensed](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://raw.githubusercontent.com/Nek5000/nek5000/develop/LICENSE)

Nek5000 is an open source, highly scalable and portable spectral element code
designed to simulate:
* unsteady Stokes
* unsteady incompressible Navier-Stokes
* low Mach-number flows
* heat transfer and species transport
* incompressible magnetohydrodynamics (MHD)

Written in Fortran 77 and C, it relies on MPI for parallelization and on a
subset of LAPACK routines for eigenvalue computations, depending on the
particular solver employed.  Nek5000 output formats can be read by the parallel
visualization package VisIt developed at LLNL/LBNL. VisIt is mandatory for
large problems (e.g. more than 100,000 spectral elements).


## Features

* Scales to over a million processes
* High-order spatial discretization using spectral elements
* High-order semi-implicit timestepping
* Incompressible + low Mach number (variable density) flows
* Efficient preconditioners (multigrid + scalable coarse grid solves)
* Highly optimized computational kernels (e.g. matrix-matrix multiply)
* Low memory footprint and scalable memory design
* High performance parallel I/O
* ALE / moving meshes and free surface flow
* Accurate Lagrangian particle tracking
* Conjugate fluid-solid heat transfer
* Scalable communication kernels
* Built-in profiling analysis
* Interface to VisIt for parallel data analysis and visualization


## Download

You can download the latest release of nek5000 as 
[a zip](https://github.com/Nek5000/nek5000/archive/master.zip) or 
[a tarball](https://github.com/Nek5000/nek5000/archive/master.tar.gz).

You can also clone the repository with git:
```
git clone https://github.com/Nek5000/nek5000.git -b master
```
or even check it out with svn:
```
svn co https://github.com/Nek5000/nek5000.git/branches/master nek5000
```


## User Guide

For more information, see the [user guide](https://nek5000.mcs.anl.gov/documentation/).


## Notes for users of legacy SVN repo

With the move to Git, the Nek5000 source directories have been reorganized to
improve modularity.  Most of the contents of `trunk/nek/` (including the
all-important `makenek`) are now located in the `core/` directory.  Scripts for
running Nek5000 (such as `nekmpi` and `nekb`) are located in the `bin/`
directory.  Tools such as `genbox` and `genmap` are located in `tools/`.  A
small subset of the examples (useful for quick regression tests) are located in
`short_tests/`.  All examples are kept in a separate repository,
[NekExamples](https://github.com/Nek5000/NekExamples), to keep this one
light-weight. 

Nek5000 works the same way it used to: build cases with `core/makenek` and run them with a script, e.g. `bin/nekmpi`.
