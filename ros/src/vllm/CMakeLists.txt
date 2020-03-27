cmake_minimum_required(VERSION 3.1)
project(vllm LANGUAGES CXX C)

##################
# Set build type #
##################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

###########################
# Set application options #
###########################

set(USE_SANITIZER
    OFF
    CACHE BOOL "Enable Address/Memory sanitizer (set env as ASAN_OPTIONS=detect_leaks=1)")

if(USE_SANITIZER)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  message(STATUS "Address/Memory sanitizer: ENABLED")
else()
  message(STATUS "Address/Memory sanitizer: DISABLED")
endif()

########################
# Set compiler options #
########################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math")

set(BUILD_WITH_MARCH_NATIVE
    OFF
    CACHE BOOL "Enable architecture-aware optimization (-march=native)")

if(BUILD_WITH_MARCH_NATIVE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
  message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
  message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
endif()

#######################
# Check C++11 support #
#######################

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  message(STATUS "C++11 support: OK (-std=c++17)")
elseif(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  message(STATUS "C++11 support: OK (-std=c++14)")
elseif(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  message(STATUS "C++11 support: OK (-std=c++11)")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  message(STATUS "C++11 support: OK (-std=c++0x)")
else()
  message(
    FATAL_ERROR
      "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler."
  )
endif()

#################
# Find packages #
#################

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport)

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

catkin_package(CATKIN_DEPENDS cv_bridge image_transport)

#########
# Build #
#########

# find_library(OpenVSLAM_LIB openvslam HINTS ${OpenVSLAM_LIB_DIR})

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
add_executable(vllm_node src/vllm_node.cpp)
target_link_libraries(vllm_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES}
                      pangolin ${VLLM_LIB})
target_compile_definitions(vllm_node PUBLIC USE_DBOW2)

add_executable(imu_node src/imu_node.cpp src/ekf.cpp)
target_link_libraries(imu_node ${catkin_LIBRARIES})
