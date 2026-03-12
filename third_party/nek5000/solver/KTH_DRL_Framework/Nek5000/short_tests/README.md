Short Test
========


The NekUnitTests.py module contains the Nek5000 short (verification) tests, implmented 
using the Python standard-library unittests framework.  The modules require 
Python 2.7 or higher.  


Before running the tests, several environment variables may be optionally defined.

* `NEK_SOURCE_ROOT`: Points to the top-level Nek5000 repository (default: this repository).
* `FC`: The Fortran 77 compiler (default mpif77).
* `CC`: The C compiler (default: mpicc).
* `PPLIST`: List of pre-processor symbols (default: none)
* `USR_LFLAGS`: Linking flags (default: none)
* `MPI=[0|1]`: If 1, run tests with MPI. (default: 1)

Setting the following in your enviroment will affect the execution

* `PARALLEL_PROCS`: The number of processes to use when running with MPI
  (default: 4)
* `LOG_ROOT`: If defined, move complted logs into this directory.  If not defined,
  leave logs in the case folders.  (default: undefined)
* `VERBOSE_TESTS=[true|false]`: If true, display standard output from compiler and
   Nek5000 to terminal window.  Standard output will always be recorded in
   logfiles, whether VERBOSE_TESTS is true or false.  (default: false)


To run all the tests, first `cd` into this directory and then run:
`$ python -m 'unittest' NekTests >log`

If you wish to run tests for one short run e.g.:
`$ python -m 'unittest' NekTests.Eddy_EddyUv.test_PnPn2_Parallel`
