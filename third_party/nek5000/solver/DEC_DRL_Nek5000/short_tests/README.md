NekTests
========
Nek unittests
-------------

The NekUnitTests.py module contains the Nek5000 verification tests, implmented 
using the Python standard-library unittests framework.  The modules require 
Python 2.7 or higher.  

### Quick Start

NekTests may be run as a standalone executable or as a unittest module.
The standalone executable may be run as:

`$ ./NekTests.py [-h] [--f77 F77] [--cc CC] [--ifmpi {true,false}] [--nprocs NPROCS] [-v]
`

with the following optional arguments:
optional arguments:
*  -h, --help: show this help message and exit
*  --f77:  The Fortran 77 compiler to use (default: mpif77)
*  --cc:  The C compiler to use (default: mpicc_
*  --ifmpi {true,false}:  Enable/disable parallel tests with MPI (default: true)
*  --nprocs NPROCS:  Number of processes to use for MPI tests (default: 4)
*  -v, --verbose:  Enable verbose output

The unittest module itself is described below.

### Module Contents

The module contains a separate class for each test problem.  The classes are:
* Axi
* Benard_Ray9
* Benard_RayDD
* Benard_RayDN
* Benard_RayNN
* Eddy_EddyUv
* KovStState
* LowMachTest
* VarVis

Each class also contains four methods for different formulations and
parallelization modes:
* test_PnPn_Serial
* test_PnPn_Parallel
* test_PnPn2_Serial
* test_PnPn2_Parallel

### Running unittest Modules

The tests may be run using the Python standard-library 'unittest' module, which
requires no additional dependencies.  The tests may also be run with any
third-party testing tool compatable with unittest, such as nose, py.test,
TwistedTrial, and others.  

#### Environment

Before running the tests, several environment variables may be optionally defined.

Setting the following in your environement (e.g. `export SOURCE_ROOT=$HOME/repos/Nek5000`)
will set the corresponding variables in makenek.  They affect compilation of Nek5000.
* `SOURCE_ROOT`: Points to the top-level Nek5000 repository (default: this repository).
* `F77`: The Fortran 77 compiler (default mpif77).
* `CC`: The C compiler (default: mpicc).
* `G`: Generic compiler flags, such as -g (default: none)
* `PPLIST`: List of pre-processor symbols (default: none)
* `USR_LFLAGS`: Linking flags (default: none)
* `IFMPI=[true|false]`: If true, run tests with MPI. (default: true)

Setting the following in your enviroment will affect the execution of NekTests
* `PARALLEL_PROCS`: The number of processes to use when running with MPI
  (default: 4)
* `EXAMPLES_ROOT`: Points to an alternate Nek5000 examples directory (default: this directory)
* `TOOLS_ROOT`: Points to an alternate directory for Nek5000 tools. (default: $SOURCE_ROOT/tools)
* `TOOLS_BIN`: If defined, compile tools in this directory. (default: `$TOOLS_ROOT/bin`)
* `LOG_ROOT`: If defined, move complted logs into this directory.  If not defined,
  leave logs in the example folders.  (default: undefined)
* `VERBOSE_TESTS=[true|false]`: If true, display standard output from compiler and
   Nek5000 to terminal window.  Standard output will always be recorded in
   logfiles, whether VERBOSE_TESTS is true or false.  (default: false)

#### unittest

To run all the tests, first `cd` into this directory and then run:
`$ python -m 'unittest' NekUnitTests`

If you wish to run tests for one example problem (for example, "TurbChannel"), run:
`$ python -m 'unittest' NekUnitTests.TurbChannel`

If you wish to run tests for one example problem and one
formulation/parallelization (for example, test_PnPn_Serial), run:
`$ python -m 'unittest' NekUnitTests.TurbChannel.test_PnPn_Serial`
