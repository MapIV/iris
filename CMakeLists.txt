cmake_minimum_required(VERSION 3.11)
project(trafic-light-detection)

# Check C++17
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++17\n")
endif()
# C++17
set(CMAKE_CXX_STANDARD 17)
message("Compiler:\n\t${CMAKE_CXX_COMPILER} (using C++17)")

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# clear "CMAKE_CXX_FLAGS"
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -fopenmp ")

# set warning flags
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas)
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()

# print "CMAKE_CXX_FLAGS"
message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

# g2o
find_package(
    g2o REQUIRED
    COMPONENTS g2o::core
               g2o::stuff
               g2o::types_sba
               g2o::types_sim3
               g2o::types_icp
               g2o::solver_dense
               g2o::solver_eigen
               g2o::solver_csparse
               g2o::csparse_extension
    OPTIONAL_COMPONENTS g2o::csparse)

# Eigen3
find_package(Eigen3 REQUIRED)
message("Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# header & source
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# =======================================
# Exe
add_executable(icp icp_sim3.cpp)
target_link_libraries(
    icp
    PUBLIC g2o::core
           g2o::stuff
           g2o::types_sba
           g2o::types_sim3
           g2o::solver_dense
           g2o::solver_eigen
           g2o::solver_csparse
           g2o::types_icp
           g2o::csparse_extension
           ${CXSPARSE_LIBRARIES}
           ${SUITESPARSE_LIBRARIES})
