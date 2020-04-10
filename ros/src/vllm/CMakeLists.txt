cmake_minimum_required(VERSION 3.1)
project(vllm LANGUAGES CXX C)

# == Set default build type to release ==
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# == Check C++17 ==
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17\n")
endif()
# == C++17 ==
set(CMAKE_CXX_STANDARD 17)
message("Compiler:\n\t${CMAKE_CXX_COMPILER} (using C++17)")

# == Clear "CMAKE_CXX_FLAGS" ==
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs")

#################
# Find packages #
#################

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport tf)

# OpenCV
find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Point Cloud Library
find_package(PCL QUIET REQUIRED COMPONENTS common io registration visualization kdtree)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
message(STATUS "PCL version:\n\t${PCL_VERSION}")

# popl
set(popl_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd/openvslam/3rd/popl/include)
include_directories(${popl_INCLUDE_DIR})

# vllm
set(vllm_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
include_directories(${vllm_INCLUDE_DIR})

# Pangolin
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

#################################
# catkin specific configuration #
#################################
catkin_package(CATKIN_DEPENDS cv_bridge image_transport tf)

#########
# Build #
#########
# ROOT
get_filename_component(VLLM_ROOT ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(VLLM_ROOT ${VLLM_ROOT} DIRECTORY)
get_filename_component(VLLM_ROOT ${VLLM_ROOT} DIRECTORY)
message(STATUS "Root directory of VLLM: ${VLLM_ROOT}")

# OpenVSLAM
set(OpenVSLAM_ROOT ${VLLM_ROOT}/3rd/openvslam)
message(STATUS "Root directory of OpenVSLAM: ${OpenVSLAM_ROOT}")
set(OpenVSLAM_INCLUDE_DIR
    ${OpenVSLAM_ROOT}/src ${OpenVSLAM_ROOT}/3rd/json/include
    ${OpenVSLAM_ROOT}/3rd/popl/include ${OpenVSLAM_ROOT}/3rd/spdlog/include)
message(STATUS "OpenVSLAM_INCLUDE_DIR: ${OpenVSLAM_INCLUDE_DIR}")

# VLLM
set(VLLM_LIB_DIR ${VLLM_ROOT}/build/)
message(STATUS "Library directory of VLLM: ${VLLM_LIB_DIR}")
find_library(VLLM_LIB vllm HINTS ${VLLM_LIB_DIR})
message(STATUS "Library of VLLM: ${VLLM_LIB}")

include_directories(${catkin_INCLUDE_DIRS} ${OpenVSLAM_INCLUDE_DIR} include)

# Executable
add_executable(vllm_node src/vllm_node.cpp src/decorator.cpp)

target_link_libraries(
  vllm_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  pangolin
  pthread
  ${VLLM_LIB})
target_compile_definitions(vllm_node PUBLIC USE_DBOW2)
