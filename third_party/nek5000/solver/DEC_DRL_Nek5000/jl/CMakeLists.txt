cmake_minimum_required (VERSION 2.6)

set(CMAKE_BUILD_TYPE Release)

set(FUN_PREFIX "nek_" CACHE STRING "prefix for exported symbols")

option(LOCAL_LONG "use long for local index type (instead of int)" ON)
option(LOCAL_LONG_LONG "use long long for local index type (instead of int)" OFF)

add_definitions(-DPREFIX=${FUN_PREFIX} -DUSE_NAIVE_BLAS -DNO_NEK_EXITT -DUNDERSCORE)

if(LOCAL_LONG)
  add_definitions(-DUSE_LONG)
  set(NEK_UINT "unsigned long")
elseif(LOCAL_LONG_LONG)
  add_definitions(-DUSE_LONG_LONG)
  set(NEK_UINT "unsigned long long")
else()
  set(NEK_UINT "unsigned")
endif()

configure_file (
  "${PROJECT_SOURCE_DIR}/ext/nek_config.h.in"
  "${PROJECT_BINARY_DIR}/nek_config.h"
  )

add_library (nek_intp sort.c sarray_sort.c obbox.c tensor.c poly.c lob_bnd.c findpts_el_3.c findpts_el_2.c fail.c comm.c gs_local.c findpts_local.c ext/findpts_local_ext.c)

install (TARGETS nek_intp DESTINATION lib)
install (FILES "${PROJECT_BINARY_DIR}/nek_config.h" DESTINATION include/nek)
install (FILES "${PROJECT_SOURCE_DIR}/ext/findpts_local.h" DESTINATION include/nek)
