/*---------------------------------------------------------------------------*\
   Copyright (c) 2019-2025, UCHICAGO ARGONNE, LLC.

   The UChicago Argonne, LLC as Operator of Argonne National
   Laboratory holds copyright in the Software. The copyright holder
   reserves all rights except those expressly granted to licensees,
   and U.S. Government license rights.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the disclaimer below.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the disclaimer (as noted below)
   in the documentation and/or other materials provided with the
   distribution.

   3. Neither the name of ANL nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
   UCHICAGO ARGONNE, LLC, THE U.S. DEPARTMENT OF
   ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   Additional BSD Notice
   ---------------------
   1. This notice is required to be provided under our contract with
   the U.S. Department of Energy (DOE). This work was produced at
   Argonne National Laboratory under Contract
   No. DE-AC02-06CH11357 with the DOE.

   2. Neither the United States Government nor UCHICAGO ARGONNE,
   LLC nor any of their employees, makes any warranty,
   express or implied, or assumes any liability or responsibility for the
   accuracy, completeness, or usefulness of any information, apparatus,
   product, or process disclosed, or represents that its use would not
   infringe privately-owned rights.

   3. Also, reference herein to any specific commercial products, process,
   or services by trade name, trademark, manufacturer or otherwise does
   not necessarily constitute or imply its endorsement, recommendation,
   or favoring by the United States Government or UCHICAGO ARGONNE LLC.
   The views and opinions of authors expressed
   herein do not necessarily state or reflect those of the United States
   Government or UCHICAGO ARGONNE, LLC, and shall
   not be used for advertising or product endorsement purposes.

\*---------------------------------------------------------------------------*/


#include "driver.hpp"

int main(int argc, char** argv)
{
  const auto timeStart = std::chrono::high_resolution_clock::now();

  if (const char* env_val = std::getenv("NEKRS_SIGNUM_BACKTRACE")) {
    std::signal(std::atoi(env_val), signalHandlerBacktrace);  
  }

  if (const char* env_val = std::getenv("NEKRS_SIGNUM_TERM")) {
    std::signal(std::atoi(env_val), signalHandlerLastStep);  
  }

  if (const char* env_val = std::getenv("NEKRS_SIGNUM_UPD")) {
    std::signal(std::atoi(env_val), signalHandlerUpdateFile);  
  }

  {
    int request = MPI_THREAD_SINGLE;
    const char* env_val = std::getenv ("NEKRS_MPI_THREAD_MULTIPLE");
    if (env_val)
      if (std::stoi(env_val)) request = MPI_THREAD_MULTIPLE;

    int provided;
    int retval =  MPI_Init_thread(&argc, &argv, request, &provided);
    if (retval != MPI_SUCCESS) {
      std::cout << "FATAL ERROR: Cannot initialize MPI!" << "\n";
      exit(EXIT_FAILURE);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  }

  // HydroGym MPMD launch (mpirun -n 1 python ... : -n N nekrs-mpmd ...): the controller splits MPI_COMM_WORLD by
  // color (0 = controller, 1 = solver, hydrogym/nek/env.py mpi_split), so the solver must take part in the same
  // split instead of using MPI_COMM_WORLD. The UDF creates the intercommunicator (tag 99) on the solver communicator.
  const char* mpmdColor = std::getenv("NEKRS_MPMD_COLOR");

  MPI_Comm commGlobal;
  if (mpmdColor) {
    MPI_Comm_split(MPI_COMM_WORLD, std::atoi(mpmdColor), worldRank, &commGlobal);
    MPI_Barrier(commGlobal);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &commGlobal);
  }
  const double time0 = MPI_Wtime();

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  if(!getenv("NEKRS_HOME")) {
    std::cout << "FATAL ERROR: Cannot find env variable NEKRS_HOME!" << "\n";
    fflush(stdout);
    MPI_Abort(commGlobal, EXIT_FAILURE);
  }

  cmdOptions* cmdOpt = processCmdLineOptions(argc, argv, commGlobal);

  MPI_Comm comm = setupSession(cmdOpt, commGlobal);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (cmdOpt->attach) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    fprintf(stderr, "rank %d on %s: pid<%d>\n", rank, hostname, getpid());

    if (rank == 0) {
      fprintf(stderr, "Attach debugger, then send <SIGCONT> to rank0 to continue\n");

      sigset_t signalSet;
      sigemptyset(&signalSet);
      sigaddset(&signalSet, SIGCONT);
      int signal;
      sigwait(&signalSet, &signal); 
    }

    MPI_Barrier(comm); // block until signal on rank0 is received 
  }

  auto abort = [&](const std::string& txt)
  {
    if (cmdOpt->debug) throw; 

    if (!txt.empty()) {
      std::cerr << txt << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // terminates all processes
     
    } else {
      MPI_Barrier(mpmdColor ? commGlobal : MPI_COMM_WORLD); // waiting to be terminated
    } 
  };

  try {
    { // change working dir
      const size_t last_slash = cmdOpt->setupFile.rfind('/') + 1;
      const std::string casepath = cmdOpt->setupFile.substr(0,last_slash);
      chdir(casepath.c_str());
      const std::string casename = cmdOpt->setupFile.substr(last_slash, cmdOpt->setupFile.length() - last_slash);
      if(casepath.length() > 0) chdir(casepath.c_str());
      cmdOpt->setupFile.assign(casename);
    }

    auto parKeyValuePairs = readPar(cmdOpt->setupFile, comm);

    {
      if (!cmdOpt->redirectOutput.empty()) {
        auto outputFile = cmdOpt->redirectOutput;
        if (rank == 0) std::cout << "redirecting output to " << outputFile << " ...\n";
        const int fd = open(outputFile.c_str(), O_WRONLY|O_CREAT|O_APPEND, S_IWUSR|S_IRUSR);
        dup2(fd, fileno(stderr));
        dup2(fd, fileno(stdout));
      } else {
        auto it = parKeyValuePairs.find("general");
        if (it != parKeyValuePairs.end()) {
          auto jt = it->second.find("redirectoutputto");
          if (jt != it->second.end()) {
            auto outputFile = jt->second;
            if (rank == 0) std::cout << "redirecting output to " << outputFile << " ...\n";
            const int fd = open(outputFile.c_str(), O_WRONLY|O_CREAT|O_APPEND, S_IWUSR|S_IRUSR);
            dup2(fd, fileno(stderr));
            dup2(fd, fileno(stdout));
          }
        }
      }
    }

    if (rank == 0) {
       time_t now = time(0);
       tm *gmtm = gmtime(&now);
       char *dt = asctime(gmtm);
       std::cout << "UTC time: " << dt << std::endl;
    }

    const auto multiSession = [&]()
    {
      int retVal;
      MPI_Comm_compare(commGlobal, comm, &retVal);
      return (retVal == MPI_IDENT) ? false : true;
    }(); 

    nekrs::setup(commGlobal, 
                 comm, 
      	         cmdOpt->buildOnly, 
                 cmdOpt->sizeTarget,
                 cmdOpt->ciMode, 
                 parKeyValuePairs,
                 cmdOpt->setupFile,
                 cmdOpt->backend, 
                 cmdOpt->deviceID,
                 cmdOpt->nSessions, 
                 cmdOpt->sessionID,
                 cmdOpt->debug);
 
    if (cmdOpt->buildOnly) {
      nekrs::finalize();
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
 
    double time = nekrs::startTime();
 
    double elapsedTime = 0;
    {
      MPI_Barrier(comm);
      const auto timeStop = std::chrono::high_resolution_clock::now();
      elapsedTime += std::chrono::duration<double, std::milli>(timeStop - timeStart).count() / 1e3;
      MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, comm);
      nekrs::updateTimer("setup", elapsedTime);
      if (rank == 0)
        std::cout << "initialization took " << elapsedTime << " s" << std::endl;
    }
 
    int tStep = 0;
    int isLastStep = 0;
    nekrs::lastStep(isLastStep);
 
    nekrs::udfExecuteStep(time, tStep, /* checkpointStep */ 0);
    nekrs::resetTimer("udfExecuteStep");
 
    double elapsedStepSum = 0;
 
    double tSolveStepMin = std::numeric_limits<double>::max();
    double tSolveStepMax = std::numeric_limits<double>::min();

    if (nekrs::endTime() > nekrs::startTime()) {
      if (rank == 0) std::cout << "\ntimestepping to time " << nekrs::endTime() << " ...\n";
    } else if (nekrs::numSteps() > tStep) {
      if (rank == 0) std::cout << "\ntimestepping for " << nekrs::numSteps() << " steps ...\n";
    } else {
      isLastStep = 1;
      if (rank == 0) std::cout << "endTime or numSteps reached already -> skip timestepping\n"; 
    }

    if (rank == 0) std::cout << std::endl;

    fflush(stdout);
    MPI_Pcontrol(1);
    while (!isLastStep) {
      MPI_Barrier(comm);
      const double timeStartStep = MPI_Wtime();
 
      ++tStep;
      auto [dtSubStep, dt] = nekrs::dt(tStep);

      const double timeNew = time + dt;
 
      isLastStep = nekrs::lastStep(timeNew, tStep, elapsedTime);
      if (sig_terminate) isLastStep = 1;

      if (isLastStep && nekrs::endTime() > 0) {
        dtSubStep = nekrs::finalTimeStepSize(time);
      }

      int checkpointStep = nekrs::checkpointStep(timeNew, tStep);
      if (nekrs::writeInterval() == 0) checkpointStep = 0;
      if (isLastStep) checkpointStep = 1;
      if (nekrs::writeInterval() < 0) checkpointStep = 0;
      nekrs::checkpointStep(checkpointStep);

      // all sessions run the same number of global steps 
      // but some may do sub-stepping
      {
        nekrs::initStep(time, dtSubStep, tStep);
        int corrector = 1;
        bool stepConverged = false;
        do {
          stepConverged = nekrs::runStep(corrector++);
          if (nekrs::printStepInfoFreq() && !stepConverged) {
            nekrs::printStepInfo(timeNew, tStep, true, true);
          }
        } while (!stepConverged);
        nekrs::finishStep();
      }

      if(sig_processUpdFile) {
        nekrs::processUpdFile();
        sig_processUpdFile = 0;
      }

      // print solver stats only 
      if (nekrs::printStepInfoFreq()) {
        if (tStep % nekrs::printStepInfoFreq() == 0)
          nekrs::printStepInfo(timeNew, tStep, false, true);
      }
 
      if (checkpointStep) nekrs::writeCheckpoint(timeNew);
 
      MPI_Barrier(comm);
      const double elapsedStep = MPI_Wtime() - timeStartStep;
      tSolveStepMin = std::min(elapsedStep, tSolveStepMin);
      tSolveStepMax = std::max(elapsedStep, tSolveStepMax);
      nekrs::updateTimer("minSolveStep", tSolveStepMin);
      nekrs::updateTimer("maxSolveStep", tSolveStepMax);
 
      elapsedStepSum += elapsedStep;
      elapsedTime += elapsedStep;
      nekrs::updateTimer("elapsedStep", elapsedStep);
      nekrs::updateTimer("elapsedStepSum", elapsedStepSum);
      nekrs::updateTimer("elapsed", elapsedTime);

      if (nekrs::printStepInfoFreq()) {
        if (tStep % nekrs::printStepInfoFreq() == 0) {
          nekrs::printStepInfo(timeNew, tStep, true, false);
          if (rank == 0) std::cout << "::" << std::endl; 
        }
      }
 
      if(nekrs::runTimeStatFreq()) {
        if (tStep % nekrs::runTimeStatFreq() == 0 || isLastStep) {
          nekrs::printRuntimeStatistics(tStep);
        }
      }
 
      time = timeNew;

      if (tStep % 100 == 0) fflush(stdout);
    }
    MPI_Pcontrol(0);
  }
  catch (const std::overflow_error& e)
  {
    abort(std::string(e.what()));
  }
  catch (const std::underflow_error& e)
  {
    abort(std::string(e.what()));
  }
  catch (const std::invalid_argument& e)
  {
    abort(std::string(e.what()));
  }
#ifdef CPPTRACE_ENABLED
  catch (const cpptrace::exception& e)
  {
    const auto msg = "Aborting in " 
                     + e.trace().frames.begin()->filename 
                     + ":" + std::string(e.trace().frames.begin()->symbol) + "\n"
                     + e.message();
    abort(!std::string(e.message()).empty() ? msg : std::string(""));
  }
#endif
  catch (const std::runtime_error& e)
  {
    abort(std::string(e.what()));
  }
  catch (const std::exception& e)
  {
    abort(std::string(e.what()));
  }
  catch (...)
  {
    std::cerr << "unknown exception occured" << std::endl;
    throw;
  }

  const int exitValue = nekrs::finalize();

  MPI_Barrier(commGlobal);
  MPI_Finalize();

  if(exitValue)
    return EXIT_FAILURE;
  else
    return EXIT_SUCCESS;
}
