cmake_minimum_required(VERSION 2.8)

project( LSD3DR )

set( CMAKE_VERBOSE_MAKEFILE 1) # to see all make process


set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -msse3 -pg -std=c++11 -Wall -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse3 -pg -std=c++11 -Wall -O3    ")


add_subdirectory( elas )
add_subdirectory( LSD3DR_main )

add_executable( ${PROJECT_NAME} $<TARGET_OBJECTS:elas> $<TARGET_OBJECTS:LSD3DR_main>)




# build-in packages

find_package(Eigen3)
if(EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(Boost 1.47 COMPONENTS system )
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

find_package(OpenGL)
if(OPENGL_FOUND)
  include_directories(${OPENGL_INCLUDE_DIR})
  target_link_libraries( ${PROJECT_NAME} ${OPENGL_LIBRARIES})
endif()

find_package(CUDA 6.0)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()

find_package(VTK 5.8)
if(VTK_FOUND)
  include_directories(${VTK_INCLUDE_DIRS})
  target_link_libraries( ${PROJECT_NAME} ${VTK_LIBRARIES})
endif()



# Point Grey FlyCapture2 SDK
include_directories("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)
target_link_libraries( ${PROJECT_NAME} flycapture )


# OpenNI used by pcl
#link_directories(/usr/lib/OpenNI2/Drivers)
#target_link_libraries(realtimeDisparity  OpenNI2 OpenNI flann_cpp)
find_library(OPENNI2 OpenNI2)
find_library(OPENNI OpenNI)
find_library(FLANN_CPP flann_cpp)




#
# use locally-installed opencv and pcl (which are latest from github)
#
set (LOCAL_INSTALL_DIR "/home/zhao/localinstall")
include_directories("${LOCAL_INSTALL_DIR}/include/" )
include_directories("${LOCAL_INSTALL_DIR}/include/opencv" )
include_directories("${LOCAL_INSTALL_DIR}/include/pcl-1.7" )
# link all *.so files directory
FILE(GLOB LIBOPENCV_FILES "${LOCAL_INSTALL_DIR}/lib/libopencv*.so" )
FILE(GLOB LIBPCL_FILES "${LOCAL_INSTALL_DIR}/lib/libpcl*.so" )
target_link_libraries( ${PROJECT_NAME}  ${LIBOPENCV_FILES} ${LIBPCL_FILES} )


