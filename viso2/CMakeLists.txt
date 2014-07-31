# project
cmake_minimum_required (VERSION 2.6)
project (viso2)

set( CMAKE_VERBOSE_MAKEFILE 1) # to see all make process

include_directories( ${PROJECT_SOURCE_DIR}/src )

FILE(GLOB LIBELAS_SRC_FILES  "${PROJECT_SOURCE_DIR}/src/*.cpp" )
MESSAGE( STATUS "LIBVISO2_SRC_FILES: " ${LIBVISO2_SRC_FILES} )

add_library ( viso2 OBJECT ${LIBELAS_SRC_FILES} )

