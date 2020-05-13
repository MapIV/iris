cmake_minimum_required(VERSION 3.1)
project(vllm_ros)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

# == Set default build type to release ==
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message(STATUS "Build Type:\n\t${CMAKE_BUILD_TYPE}")

# == Clear "CMAKE_CXX_FLAGS" ==
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs -mfpmath=both -mtune=native")

# # == Set warning flags ==
# set(CXX_WARNING_FLAGS
#     -Wall
#     -Wextra
#     -Wconversion
#     -Wswitch-default
#     -Wdisabled-optimization
#     -Wformat
#     -Winit-self
#     -Woverloaded-virtual
#     -Wfloat-equal
#     -Wno-old-style-cast
#     -Wno-pragmas)
# foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
# endforeach()
# message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

#################
# Find packages #
#################

# == Point Cloud Library ==
find_package(PCL QUIET REQUIRED COMPONENTS common io registration visualization kdtree)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
message(STATUS "PCL version:\n\t${PCL_VERSION}")

# == OpenMP ==
set(USE_OPENMP OFF)
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP: ENABLED")
  else()
    message(STATUS "OpenMP: DISABLED")
  endif()
else()
  message(STATUS "OpenMP: DISABLED")
endif()

# == OpenCV ==
# specify OpenCV directory because my PC has two OpenCV
find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# == Eigen ==
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")

# == Find catkin macros and libraries ==
# cmake-format: off
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge 
  image_transport 
  tf 
  visualization_msgs 
  dynamic_reconfigure)
# cmake-format: on

# == popl ==
set(popl_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd/openvslam/3rd/popl/include)
include_directories(${popl_INCLUDE_DIR})

# == json ==
set(json_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd/openvslam/3rd/json/include)
include_directories(${json_INCLUDE_DIR})

# == spdlog ==
set(spdlog_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd/openvslam/3rd/spdlog/include)
include_directories(${spdlog_INCLUDE_DIR})

# == Pangolin ==
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

# == OpenGL==
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# == vllm ==
set(vllm_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
include_directories(${vllm_INCLUDE_DIR})

# # == g2o ==
find_package(
  g2o REQUIRED
  COMPONENTS g2o::core
             g2o::stuff
             g2o::types_sba
             g2o::types_sim3
             g2o::solver_dense
             g2o::solver_eigen
             g2o::solver_csparse
             g2o::csparse_extension
  OPTIONAL_COMPONENTS g2o::csparse)

# Check first if CSparse is built from g2o
if(TARGET g2o::csparse)
  set(${CXSPARSE_LIBRARIES} g2o::csparse)
else()
  # CXSparse for g2o
  set(CXSPARSE_INCLUDE_DIR "/usr/include/suitesparse")
  set(CXSPARSE_LIBRARY "/usr/lib/x86_64-linux-gnu/libcxsparse.so")
  # find_package(CXSparse)
  # if(CXSPARSE_FOUND)
  include_directories(${CXSPARSE_INCLUDE_DIRS})
  # endif()
  # SuiteSparse for g2o
  set(SuiteSparse_DIR "/usr/lib/x86_64-linux-gnu")
  find_package(SuiteSparse)
  if(SUITESPARSE_FOUND)
    include_directories(${SUITESPARSE_INCLUDE_DIRS})
  endif()
endif()

message(STATUS "SUITESPARSE_CONFIG_INCLUDE_DIR: ${SUITESPARSE_CONFIG_INCLUDE_DIR}")
message(STATUS "SUITESPARSE_CONFIG_LIBRARY: ${SUITESPARSE_CONFIG_LIBRARY}")
message(STATUS "CXSPARSE_INCLUDE_DIR: ${CXSPARSE_INCLUDE_DIR}")
message(STATUS "CXSPARSE_LIBRARY: ${CXSPARSE_LIBRARY}")

set(G2O_LIBS
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::types_sim3
    g2o::solver_dense
    g2o::solver_eigen
    g2o::solver_csparse
    g2o::csparse_extension
    ${CXSPARSE_LIBRARIES}
    ${SUITESPARSE_LIBRARIES})
message(STATUS "g2o version:\n\t${g2o_VERSION}")

##################
# dynamic_config #
##################
generate_dynamic_reconfigure_options(cfg/dynamic.cfg)

#################################
# catkin specific configuration #
#################################
# cmake-format: off
catkin_package(CATKIN_DEPENDS 
  cv_bridge 
  image_transport 
  tf 
  visualization_msgs 
  dynamic_reconfigure)
# cmake-format: on

# == UNKNOWN==
set(USE_SANITIZER OFF)
if(USE_SANITIZER)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  message(STATUS "Address/Memory sanitizer: ENABLED")
else()
  message(STATUS "Address/Memory sanitizer: DISABLED")
endif()

# ----- Set compiler options -----

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -ffast-math")

# set(BUILD_WITH_MARCH_NATIVE
#     OFF
#     CACHE BOOL "Enable architecture-aware optimization (-march=native)")
# if(BUILD_WITH_MARCH_NATIVE)
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
#   set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
#   message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
# else()
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
#   set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
#   message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
# endif()

set(USE_SSE_ORB
    OFF
    CACHE BOOL "Enable SSE3 instruction for ORB extraction (-msse3)")
if(USE_SSE_ORB)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_SSE_ORB)
  message(STATUS "SSE3 for ORB extraction (-msse3): ENABLED")
else()
  message(STATUS "SSE3 for ORB extraction (-msse3): DISABLED")
endif()

set(USE_SSE_FP_MATH
    OFF
    CACHE BOOL "Enable SSE instruction for floating-point operation (-mfpmath=sse)")
if(USE_SSE_FP_MATH)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpmath=sse")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse")
  message(STATUS "SSE for floating-point operation (-mfpmath=sse): ENABLED")
else()
  message(STATUS "SSE for floating-point operation (-mfpmath=sse): DISABLED")
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
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#########
# Build #
#########
# == set ROOT ==
get_filename_component(VLLM_ROOT ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(VLLM_ROOT ${VLLM_ROOT} DIRECTORY)
get_filename_component(VLLM_ROOT ${VLLM_ROOT} DIRECTORY)
message(STATUS "Root directory of VLLM: ${VLLM_ROOT}")

# == OpenVSLAM ==
set(OpenVSLAM_ROOT ${VLLM_ROOT}/3rd/openvslam)
message(STATUS "Root directory of OpenVSLAM: ${OpenVSLAM_ROOT}")
set(OpenVSLAM_INCLUDE_DIR ${OpenVSLAM_ROOT}/src)
message(STATUS "OpenVSLAM_INCLUDE_DIR: ${OpenVSLAM_INCLUDE_DIR}")

# == VLLM ==
set(VLLM_LIB_DIR ${VLLM_ROOT}/build/lib/)
find_library(VLLM_LIB vllm HINTS ${VLLM_LIB_DIR})
message(STATUS "Library directory of VLLM: ${VLLM_LIB_DIR}")
message(STATUS "Library of VLLM: ${VLLM_LIB}")

# == OpenVSLAM ==
set(VSLAM_LIB_DIR ${VLLM_ROOT}/build/3rd/openvslam/lib/)
find_library(VSLAM_LIB openvslam HINTS ${VSLAM_LIB_DIR})
message(STATUS "Library directory of VSLAM: ${VSLAM_LIB_DIR}")
message(STATUS "Library of VSLAM: ${VSLAM_LIB}")

# == include ==
include_directories(${catkin_INCLUDE_DIRS} ${OpenVSLAM_INCLUDE_DIR} include)

# === VLLM_NODE ===
add_executable(vllm_node src/vllm_node.cpp src/communication.cpp)
target_compile_definitions(vllm_node PUBLIC USE_DBOW2)
target_link_libraries(
  vllm_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  pangolin
  pthread
  ${VLLM_LIB}
  ${VSLAM_LIB}
  ${G2O_LIBS}
  ${CXSPARSE_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${LAPACK_LIBRARIES})

# === VSLAM_NODE ===
add_executable(vslam_node src/vslam_node.cpp src/communication.cpp)
target_compile_definitions(vslam_node PUBLIC USE_DBOW2)
target_link_libraries(
  vslam_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  pthread
  ${VLLM_LIB}
  ${VSLAM_LIB}
  ${G2O_LIBS}
  ${CXSPARSE_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${LAPACK_LIBRARIES})
