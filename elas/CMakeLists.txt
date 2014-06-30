cmake_minimum_required(VERSION 2.8)

project( elas )

set( CMAKE_VERBOSE_MAKEFILE 1) # to see all make process


include_directories( ${PROJECT_SOURCE_DIR}/include )

FILE(GLOB LIBELAS_SRC_FILES  "${PROJECT_SOURCE_DIR}/src/*.cpp" )
MESSAGE( STATUS "LIBELAS_SRC_FILES: " ${LIBELAS_SRC_FILES} )

add_library ( elas OBJECT ${LIBELAS_SRC_FILES} )

