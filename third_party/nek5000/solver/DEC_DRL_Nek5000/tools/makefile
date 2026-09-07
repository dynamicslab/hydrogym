SHELL = /bin/bash

all:
	@for i in $(MODULES) ; do           \
	if test -f $${i}/makefile ; then    \
        export FFLAGS_IN=" ";               \
        export CFLAGS_IN=${US} ;            \
        if [ "$$i" == "exo2nek" ]; then     \
           export FFLAGS_IN=${R8} ;         \
           if [ ! -z $(DOALL) ]; then       \
              continue;                     \
           fi ;                             \
        fi ;                                \
        if [ "$$i" == "genmap" ]; then      \
           export FFLAGS_IN=${R8} ;         \
        fi ;                                \
        if [ "$$i" == "genbox" ]; then      \
           export FFLAGS_IN=${R8} ;         \
        fi ;                                \
        if [ "$$i" == "prenek" ]; then      \
           export CFLAGS_IN="$(US) -Dr8";   \
        fi ;                                \
        echo "----------------------";      \
        echo "Make $${i}..." ;              \
        echo "----------------------";      \
        cd $${i} ;                          \
        ${MAKE} ;                           \
        if test  $$? -ne 0 ; then           \
          exit 1 ;                          \
        fi ;                                \
        cd .. ;                             \
        echo "" ;                           \
	fi ;                                \
	done ;                              \

clean:
	@for i in $(MODULES) ; do           \
      if test -f $${i}/makefile ;  then     \
        cd $${i} ;                   \
        ${MAKE} clean ;              \
        cd .. ;                      \
      fi ;                           \
    done ;                           
