#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <sstream>
#include <cstring>
#include <getopt.h>
#include <cfenv>
#include <limits>
#include <math.h>
#include <unistd.h>
#include <libgen.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fcntl.h>
#include <chrono>
#include <csignal>

#include "inipp.hpp"

#ifdef CPPTRACE_ENABLED
#include <cpptrace/cpptrace.hpp>
#endif

#include "nekrs.hpp"

namespace {

int worldRank;
volatile sig_atomic_t sig_processUpdFile = 0;

volatile sig_atomic_t sig_writeStackTrace = 0;

volatile sig_atomic_t sig_terminate = 0;

std::fexcept_t flag;

struct cmdOptions
{
  int buildOnly = 0;
  int ciMode = 0;
  int attach = 0;
  int debug = 0;
  int sizeTarget = 0;
  std::string multiSessionFile;
  std::string setupFile;
  std::string deviceID;
  std::string backend;
  int nSessions = 1;
  int sessionID = 0;
  std::string redirectOutput;
};

struct session_t {
  int size;
  std::string setupFile;
};


bool file_exist(const std::string& fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

long int file_size(const std::string& filename)
{
    FILE *p_file = fopen(filename.c_str(),"rb");
    fseek(p_file,0,SEEK_END);
    auto size = ftell(p_file);
    fclose(p_file);
    return size;
}

cmdOptions* processCmdLineOptions(int argc, char** argv, const MPI_Comm &comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  cmdOptions* cmdOpt = new cmdOptions();

  int err = 0;
  int printHelp = 0;
  std::string helpCat;

  if (rank == 0) {
    while(1) {
      static struct option long_options[] =
      {
        {"setup", required_argument, 0, 's'},
        {"cimode", required_argument, 0, 'c'},
        {"build-only", optional_argument, 0, 'b'},
        {"debug", no_argument, 0, 'd'},
        {"attach", no_argument, 0, 'a'},
        {"backend", required_argument, 0, 't'},
        {"device-id", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'r'},
        {"help", optional_argument, 0, 'h'},
        {0, 0, 0, 0}
      };
      int option_index = 0;
      int c = getopt_long (argc, argv, "", long_options, &option_index);

      if (c == -1)
        break;

      switch(c) {
      case 's':
        cmdOpt->setupFile.assign(optarg);
        if (cmdOpt->setupFile.find(".par") != std::string::npos)
          cmdOpt->setupFile.erase(cmdOpt->setupFile.find(".par"), std::string::npos);
        if (cmdOpt->setupFile.substr(cmdOpt->setupFile.find_last_of(".") + 1) == "sess") {
          cmdOpt->multiSessionFile = cmdOpt->setupFile;
          cmdOpt->setupFile.clear();
        }
        break;
      case 'b':
        cmdOpt->buildOnly = 1;
        cmdOpt->sizeTarget = size;
        if(!optarg && argv[optind] != NULL && argv[optind][0] != '-')
          cmdOpt->sizeTarget = std::stoi(argv[optind++]);
        break;
      case 'c':
        cmdOpt->ciMode = atoi(optarg);
        if (cmdOpt->ciMode < 0) {
          std::cout << "ERROR: ci test id has to be >= 0!\n";
          printHelp = 1;
        }
        break;
      case 'd':
        cmdOpt->debug = 1;
        break;
      case 'a':
        cmdOpt->attach = 1;
        cmdOpt->debug = 0;
        break;
      case 'i':
        cmdOpt->deviceID.assign(optarg);
        break;
      case 't':
        cmdOpt->backend.assign(optarg);
        break;
      case 'r':
        cmdOpt->redirectOutput.assign(optarg);
        break;
      case 'h':
        printHelp++;
        if(!optarg && argv[optind] != NULL && argv[optind][0] != '-')
          helpCat.assign(argv[optind++]);
        break;
      default:
        err = 1;
      }
    }
  }

  for(auto opt: {&cmdOpt->multiSessionFile, &cmdOpt->setupFile, &cmdOpt->deviceID, &cmdOpt->backend, &cmdOpt->redirectOutput})
  {
    int bufSize = opt->size() + 1;
    MPI_Bcast(&bufSize, 1, MPI_INT, 0, comm);
    auto buf = (char*) std::calloc(bufSize, sizeof(char));
    if(rank == 0) std::strcpy(buf, opt->c_str());
    MPI_Bcast(buf, bufSize, MPI_BYTE, 0, comm);
    opt->assign(buf);
    free(buf);
  }

  MPI_Bcast(&cmdOpt->buildOnly, sizeof(cmdOpt->buildOnly), MPI_BYTE, 0, comm);
  MPI_Bcast(&cmdOpt->sizeTarget, sizeof(cmdOpt->sizeTarget), MPI_BYTE, 0, comm);
  MPI_Bcast(&cmdOpt->ciMode, sizeof(cmdOpt->ciMode), MPI_BYTE, 0, comm);
  MPI_Bcast(&cmdOpt->debug, sizeof(cmdOpt->debug), MPI_BYTE, 0, comm);
  MPI_Bcast(&cmdOpt->attach, sizeof(cmdOpt->attach), MPI_BYTE, 0, comm);

  if (cmdOpt->debug) {
    std::feclearexcept(FE_ALL_EXCEPT);
    std::fegetexceptflag(&flag, FE_ALL_EXCEPT);
  }

  if(cmdOpt->setupFile.empty() && cmdOpt->multiSessionFile.empty())
    printHelp++;

  MPI_Bcast(&printHelp, sizeof(printHelp), MPI_BYTE, 0, comm);
  MPI_Bcast(&err, sizeof(err), MPI_BYTE, 0, comm);
  if (err | printHelp) {
    if (rank == 0) {
      auto print = [&](const std::string& txtFile)
      {
        std::string installDir;
        installDir.assign(getenv("NEKRS_HOME"));
        std::ifstream f(installDir + "/doc/" + txtFile);
        if (f.is_open()) std::cout << f.rdbuf();
        f.close();
      };

      if (helpCat == "par") {
        print("parHelp.txt");
      } else if (helpCat == "env") {
        print("envHelp.txt");
      } else {
        std::cout << "usage: ./nekrs "
                  << "[ --help <par|env> ] "
                  << "--setup <par|sess file> "
                  << "[ --build-only <#procs> ] [ --cimode <id> ] [ --debug ] [ --attach ] "
                  << "[ --backend <CPU|CUDA|HIP|DPCPP|OPENCL> ] [ --device-id <id|LOCAL-RANK> ] "
                  << "[ --output <local-path-to-logfile>]"
                  << "\n";
      }
    }
    MPI_Finalize();
    exit((err) ? EXIT_FAILURE : EXIT_SUCCESS);
  }

  return cmdOpt;
}

std::map<std::string, std::map<std::string, std::string>> readPar(const std::string &_setupFile, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  const auto setupFile = _setupFile + ".par";

  if (rank == 0) 
   std::cout << "reading " << setupFile << std::endl; 

  int err = 0;
  if (rank == 0) {
    if (!file_exist(setupFile)) {
      std::cerr << "Cannot find setup file " << setupFile << std::endl;
      err++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, comm);
  if (err) {
    MPI_Abort(comm, EXIT_FAILURE);
  }

  long fsize;
  if (rank == 0) {
    fsize = file_size(setupFile);  
  }
  MPI_Bcast(&fsize, sizeof(fsize), MPI_BYTE, 0, comm);
  auto fileBuf = (char *) std::malloc(fsize * sizeof(char));
  if (rank == 0) {
    std::ifstream input(setupFile, std::ifstream::binary);
    input.read(fileBuf, fsize);
    input.close();
  }
  MPI_Bcast(fileBuf, fsize, MPI_CHAR, 0, comm);

  auto par = new inipp::Ini();

  std::stringstream is;
  is.write(fileBuf, fsize);

  par->parse(is);
  par->interpolate();

  return par->sections;
}

MPI_Comm setupSession(cmdOptions *cmdOpt, const MPI_Comm &comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Comm newComm = comm;

  if(cmdOpt->multiSessionFile.size()) {
    std::string multiSessionFileContent;

    if(rank == 0) {
      std::ifstream f(cmdOpt->multiSessionFile);
      if (!f) {
        std::cout << "FATAL ERROR: Cannot find sess file "
                  << cmdOpt->multiSessionFile << "!\n";
        fflush(stdout);
        MPI_Abort(comm, EXIT_FAILURE);
      }
      std::ostringstream ss;
      if (f.is_open()) ss << f.rdbuf();
      f.close();
      multiSessionFileContent = ss.str();
    }
    int bufSize = multiSessionFileContent.size() + 1;
    MPI_Bcast(&bufSize, sizeof(bufSize), MPI_BYTE, 0, comm);
    char* buf = (char*) malloc(bufSize * sizeof(char));
    strcpy(buf, multiSessionFileContent.c_str());
    MPI_Bcast(buf, bufSize * sizeof(char), MPI_BYTE, 0, comm);
    multiSessionFileContent = std::string(buf);
    free(buf);

    auto serializeString = [](const std::string sin, char dlim)
    {
      std::vector<std::string> slist;
      std::string s(sin);
      s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
      std::stringstream ss;
      ss.str(s);
      while (ss.good()) {
        std::string substr;
        std::getline(ss, substr, dlim);
        if(!substr.empty()) slist.push_back(substr);
      }
      return slist;
    };

    auto list = serializeString(multiSessionFileContent, ';');
    auto sessionList = new session_t[list.size()];

    int nSessions = 0;
    int rankSum = 0;
    for(std::string s : list) {
      auto items = serializeString(s,':');
      if(items.size() != 2) {
        if(rank == 0) std::cout << "FATAL ERROR: invalid sess file entry!\n";
        fflush(stdout);
        MPI_Abort(comm, EXIT_FAILURE);
      }
      sessionList[nSessions].setupFile = items[0];
      sessionList[nSessions].size = std::stoi(items[1]);
      rankSum += sessionList[nSessions].size;
      nSessions++;
    }

    int err = 0;
    if(rankSum != size) err = 1;
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, comm);
    if(err) {
      if(rank == 0) std::cout << "FATAL ERROR: size of sub-communicators does not match parent!\n";
      fflush(stdout);
      MPI_Abort(comm, EXIT_FAILURE);
    }

    int color = MPI_UNDEFINED;
    int rankOffsetSession = 0;
    for(int i = 0; i < nSessions; i++) {
      if(rank - rankOffsetSession < sessionList[i].size) {
        color = i;
        break;
      }
      rankOffsetSession += sessionList[i].size;
    }

    int rankGlobal, sizeGlobal;
    MPI_Comm_rank(comm, &rankGlobal);
    MPI_Comm_size(comm, &sizeGlobal);

    MPI_Comm_split(comm, color, rankGlobal, &newComm);

    MPI_Comm_rank(newComm, &rank);
    MPI_Comm_size(newComm, &size);

    cmdOpt->setupFile = sessionList[color].setupFile;
    cmdOpt->sizeTarget = size;
    cmdOpt->sessionID = color;
    cmdOpt->nSessions = nSessions;

    delete[] sessionList;

    if(cmdOpt->debug) {
      std::cout << "globalRank:" << rankGlobal << " localRank: " << rank << " pid: " << ::getpid()
                << " commSize: " << size << " setupFile:" << cmdOpt->setupFile << "\n";
    }
    fflush(stdout);
    MPI_Barrier(comm);

    const std::string outputFile = 
      cmdOpt->setupFile.substr(0, cmdOpt->setupFile.rfind('/', cmdOpt->setupFile.length())) + "/logfile";
    if(rank == 0) {  
      std::cout << "redirecting session" << cmdOpt->sessionID << " output to " << outputFile << " ...\n";
    }
    const int fd = open(outputFile.c_str(), O_WRONLY|O_CREAT|O_APPEND, S_IWUSR|S_IRUSR);
    dup2(fd, fileno(stderr));
    dup2(fd, fileno(stdout));
  }
  return newComm;
}

void writeStackTraceToFile()
{
#ifdef CPPTRACE_ENABLED
  std::cerr << "generating stacktrace ...\n";
  const std::string fileName = "stacktrace." + std::to_string(worldRank);
  std::ofstream outfile;
  outfile.open(fileName, std::ios::out | std::ios::trunc );
  outfile << cpptrace::generate_trace();
  outfile.close(); 
  sig_writeStackTrace = 0;
#endif
}

void signalHandlerBacktrace(int signum) 
{
   // stricly speaking not async-safe but often it works ok
   sig_writeStackTrace = 1;
   writeStackTraceToFile();
}

void signalHandlerUpdateFile(int signum) 
{
  sig_processUpdFile = 1;
}

void signalHandlerLastStep(int signum)
{
  sig_terminate = 1;
}


} // namespace
